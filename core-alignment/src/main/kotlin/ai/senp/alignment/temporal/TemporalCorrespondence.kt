package ai.senp.alignment.temporal

import ai.senp.core.contracts.MotionStructureClass
import ai.senp.core.contracts.MotionUnit
import ai.senp.core.contracts.MotionUnitCorrespondence
import ai.senp.core.contracts.SynchronizationScope
import ai.senp.core.contracts.SynchronizationSemantics
import ai.senp.core.contracts.TemporalStructure
import ai.senp.core.contracts.TimestampCorrespondence
import ai.senp.core.contracts.UnmatchedReason
import ai.senp.core.contracts.UnitBoundaryStatus
import ai.senp.core.contracts.UnitCompleteness
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

internal data class CorrespondenceOutcome(
    val correspondences: List<MotionUnitCorrespondence>,
    val matchedCount: Int,
    val confidence: Double,
    val ambiguity: Double,
    val hasUnmatchedUnits: Boolean,
    val hasUnmatchedTimestamps: Boolean,
)

internal class TemporalCorrespondenceSolver(
    private val config: TemporalSynchronizationConfig,
    private val space: TemporalDescriptorSpace,
    private val source: PreparedTemporalSequence,
    private val reference: PreparedTemporalSequence,
    private val stats: MutableTemporalStats,
) {
    fun solve(
        semantics: SynchronizationSemantics,
        sourceStructure: TemporalStructure,
        referenceStructure: TemporalStructure,
    ): CorrespondenceOutcome {
        stats.sourceUnits = sourceStructure.motionUnits.size
        stats.referenceUnits = referenceStructure.motionUnits.size
        val candidates = buildCandidates(semantics, sourceStructure.motionUnits, referenceStructure.motionUnits)
        val assignments = assignUnits(semantics, sourceStructure.motionUnits, referenceStructure.motionUnits, candidates)
        val output = mutableListOf<MotionUnitCorrespondence>()
        val matchedReferenceIds = linkedSetOf<String>()

        sourceStructure.motionUnits.forEach { sourceUnit ->
            val assignment = assignments[sourceUnit.unitId]
            if (assignment == null) {
                output += MotionUnitCorrespondence.SourceUnmatchedUnit(
                    sourceUnit.unitId,
                    UnmatchedReason.EXTRA_ACTION,
                    0.84,
                )
                return@forEach
            }
            val referenceUnit = referenceStructure.motionUnits.first { it.unitId == assignment.referenceUnitId }
            val fine = fineAlign(sourceUnit, referenceUnit, assignment)
            if (fine != null) {
                stats.bestFineMatchedFraction = maxOf(stats.bestFineMatchedFraction ?: 0.0, fine.matchedFraction)
                stats.bestFineDecisionConfidence = maxOf(stats.bestFineDecisionConfidence ?: 0.0, fine.decisionConfidence)
            }
            val crossClass = sourceUnit.structureClass != referenceUnit.structureClass
            val crossClassAmbiguous = crossClass &&
                assignment.ambiguity > config.maximumCrossClassAmbiguityForMatch
            val crossClassLowConfidence = crossClass && fine != null &&
                fine.decisionConfidence < config.minimumCrossClassMatchedUnitConfidence
            if (
                fine == null || fine.matchedFraction < config.minimumMatchedTimestampFraction ||
                fine.decisionConfidence < config.minimumMatchedUnitConfidence || crossClassAmbiguous || crossClassLowConfidence
            ) {
                output += MotionUnitCorrespondence.SourceUnmatchedUnit(
                    sourceUnit.unitId,
                    if (assignment.ambiguity > config.maximumAmbiguityForConfidentMatch) {
                        UnmatchedReason.AMBIGUOUS
                    } else {
                        UnmatchedReason.NO_COMPATIBLE_COUNTERPART
                    },
                    (1.0 - assignment.ambiguity * 0.5).coerceIn(0.25, 0.9),
                )
            } else {
                output += MotionUnitCorrespondence.MatchedUnit(
                    sourceUnitId = sourceUnit.unitId,
                    referenceUnitId = referenceUnit.unitId,
                    timeline = fine.timeline,
                    decisionConfidence = fine.decisionConfidence,
                    ambiguity = assignment.ambiguity,
                )
                matchedReferenceIds += referenceUnit.unitId
            }
        }
        referenceStructure.motionUnits.filter { it.unitId !in matchedReferenceIds }.forEach { referenceUnit ->
            output += MotionUnitCorrespondence.ReferenceUnmatchedUnit(
                referenceUnit.unitId,
                if (referenceUnit.structureClass == MotionStructureClass.ACYCLIC) {
                    UnmatchedReason.MISSING_REFERENCE_STEP
                } else {
                    UnmatchedReason.NO_COMPATIBLE_COUNTERPART
                },
                0.84,
            )
        }
        val matched = output.filterIsInstance<MotionUnitCorrespondence.MatchedUnit>()
        return CorrespondenceOutcome(
            correspondences = output,
            matchedCount = matched.size,
            confidence = matched.map(MotionUnitCorrespondence.MatchedUnit::decisionConfidence).averageOrZero(),
            ambiguity = matched.map(MotionUnitCorrespondence.MatchedUnit::ambiguity).averageOrZero(),
            hasUnmatchedUnits = output.any {
                it is MotionUnitCorrespondence.SourceUnmatchedUnit || it is MotionUnitCorrespondence.ReferenceUnmatchedUnit
            },
            hasUnmatchedTimestamps = matched.any { unit -> hasMaterialTimestampHole(unit) },
        )
    }

    private fun hasMaterialTimestampHole(unit: MotionUnitCorrespondence.MatchedUnit): Boolean {
        val matchedIndices = unit.timeline.mapIndexedNotNull { index, decision ->
            index.takeIf { decision is TimestampCorrespondence.Matched }
        }
        if (matchedIndices.isEmpty()) return unit.timeline.any { it is TimestampCorrespondence.UnmatchedSource }
        val firstMatched = matchedIndices.first()
        val lastMatched = matchedIndices.last()
        return unit.timeline.withIndex().any { (index, decision) ->
            decision is TimestampCorrespondence.UnmatchedSource && index in firstMatched..lastMatched
        }
    }

    private fun buildCandidates(
        semantics: SynchronizationSemantics,
        sourceUnits: List<MotionUnit>,
        referenceUnits: List<MotionUnit>,
    ): Map<Pair<String, String>, UnitCandidate> {
        val output = linkedMapOf<Pair<String, String>, UnitCandidate>()
        sourceUnits.forEachIndexed { sourceIndex, sourceUnit ->
            val ranked = referenceUnits.mapIndexedNotNull { referenceIndex, referenceUnit ->
                if (!compatibleUnits(sourceUnit, referenceUnit)) return@mapIndexedNotNull null
                stats.coarseComparisons += 1L
                coarseCompare(semantics, sourceUnit, referenceUnit)?.let { candidate ->
                    val contextualCost = candidate.cost + unitPositionPrior(
                        sourceIndex,
                        sourceUnits.size,
                        referenceIndex,
                        referenceUnits.size,
                    )
                    val previousBest = stats.bestCoarseCandidateCost
                    if (previousBest == null || contextualCost < previousBest) stats.bestCoarseCandidateCost = contextualCost
                    referenceUnit.unitId to candidate.copy(cost = contextualCost)
                }
            }.sortedBy { it.second.cost }
            val best = ranked.firstOrNull()?.second
            val second = ranked.getOrNull(1)?.second
            ranked.take(config.coarseShortlistSize).forEach { (referenceId, candidate) ->
                if (candidate.cost <= config.coarseMaximumCost) {
                    val ambiguity = if (candidate === best) ambiguity(best, second) else {
                        min(1.0, candidate.cost / config.coarseMaximumCost)
                    }
                    output[sourceUnit.unitId to referenceId] = candidate.copy(ambiguity = ambiguity)
                    stats.coarseAcceptedCandidateCount += 1
                }
            }
        }
        return output
    }

    private fun coarseCompare(
        semantics: SynchronizationSemantics,
        sourceUnit: MotionUnit,
        referenceUnit: MotionUnit,
    ): UnitCandidate? {
        val sourceFrames = frames(source, sourceUnit)
        val referenceFrames = frames(reference, referenceUnit)
        if (sourceFrames.isEmpty() || referenceFrames.isEmpty()) return null
        var bestCost = Double.POSITIVE_INFINITY
        var bestShift = 0
        var bestCoverage = 0.0
        val maximumShiftSamples = when {
            semantics.scope == SynchronizationScope.FULL_SEQUENCE -> 0
            sourceUnit.structureClass == MotionStructureClass.CYCLIC &&
                referenceUnit.structureClass == MotionStructureClass.CYCLIC -> config.coarseShiftSamples
            sourceUnit.completeness == UnitCompleteness.PARTIAL ||
                referenceUnit.completeness == UnitCompleteness.PARTIAL -> min(config.coarseShiftSamples, 2)
            else -> 0
        }
        val shifts = -maximumShiftSamples..maximumShiftSamples
        for (shift in shifts) {
            var cost = 0.0
            var coverage = 0.0
            var count = 0
            repeat(config.coarseSamplesPerUnit) { sample ->
                val sourceFraction = sample.toDouble() / (config.coarseSamplesPerUnit - 1).toDouble()
                val referenceFraction = sourceFraction + shift.toDouble() / config.coarseSamplesPerUnit.toDouble()
                if (referenceFraction !in 0.0..1.0) return@repeat
                val distance = space.distance(
                    frameAtFraction(sourceFrames, sourceFraction),
                    frameAtFraction(referenceFrames, referenceFraction),
                )
                cost += distance.cost
                coverage += distance.coverage
                count += 1
            }
            if (count < config.coarseSamplesPerUnit / 2) continue
            val meanCoverage = coverage / count.toDouble()
            val meanCost = cost / count.toDouble() + (1.0 - meanCoverage) * 0.35
            if (meanCoverage >= config.descriptorMinimumCoverage && meanCost < bestCost) {
                bestCost = meanCost
                bestShift = shift
                bestCoverage = meanCoverage
            }
        }
        return if (bestCost.isFinite()) {
            UnitCandidate(referenceUnit.unitId, bestCost, bestShift, bestCoverage, 0.0)
        } else {
            null
        }
    }

    private fun assignUnits(
        semantics: SynchronizationSemantics,
        sourceUnits: List<MotionUnit>,
        referenceUnits: List<MotionUnit>,
        candidates: Map<Pair<String, String>, UnitCandidate>,
    ): Map<String, UnitCandidate> {
        val output = linkedMapOf<String, UnitCandidate>()
        if (semantics.allowReferenceUnitReuse) {
            sourceUnits.filter { it.structureClass == MotionStructureClass.CYCLIC }.forEach { sourceUnit ->
                candidates.filterKeys { it.first == sourceUnit.unitId }.values.minByOrNull(UnitCandidate::cost)?.let { best ->
                    output[sourceUnit.unitId] = best
                }
            }
        }
        val orderedSource = sourceUnits.filter { it.unitId !in output }
        if (orderedSource.isEmpty()) return output
        val orderedReference = referenceUnits.filter { referenceUnit ->
            orderedSource.any { compatibleUnits(it, referenceUnit) }
        }
        output += orderedAssignment(semantics, orderedSource, orderedReference, candidates)
        return output
    }

    private fun orderedAssignment(
        semantics: SynchronizationSemantics,
        sourceUnits: List<MotionUnit>,
        referenceUnits: List<MotionUnit>,
        candidates: Map<Pair<String, String>, UnitCandidate>,
    ): Map<String, UnitCandidate> {
        if (sourceUnits.isEmpty() || referenceUnits.isEmpty()) return emptyMap()
        val costs = Array(sourceUnits.size + 1) { DoubleArray(referenceUnits.size + 1) { Double.POSITIVE_INFINITY } }
        val moves = Array(sourceUnits.size + 1) { ByteArray(referenceUnits.size + 1) }
        costs[0][0] = 0.0
        for (i in 0..sourceUnits.size) {
            for (j in 0..referenceUnits.size) {
                val base = costs[i][j]
                if (!base.isFinite()) continue
                if (i < sourceUnits.size && semantics.allowUnmatchedSource) {
                    update(costs, moves, i + 1, j, base + config.unitSkipCost, SOURCE_SKIP)
                }
                if (j < referenceUnits.size && semantics.allowUnmatchedReference) {
                    update(costs, moves, i, j + 1, base + config.unitSkipCost, REFERENCE_SKIP)
                }
                if (i < sourceUnits.size && j < referenceUnits.size) {
                    candidates[sourceUnits[i].unitId to referenceUnits[j].unitId]?.let { candidate ->
                        update(costs, moves, i + 1, j + 1, base + candidate.cost, MATCH)
                    }
                }
            }
        }
        if (!costs[sourceUnits.size][referenceUnits.size].isFinite()) return emptyMap()
        val output = linkedMapOf<String, UnitCandidate>()
        var i = sourceUnits.size
        var j = referenceUnits.size
        while (i > 0 || j > 0) {
            when (moves[i][j]) {
                MATCH -> {
                    val candidate = candidates[sourceUnits[i - 1].unitId to referenceUnits[j - 1].unitId]
                    if (candidate != null) output[sourceUnits[i - 1].unitId] = candidate
                    i -= 1
                    j -= 1
                }
                SOURCE_SKIP -> i -= 1
                REFERENCE_SKIP -> j -= 1
                else -> return emptyMap()
            }
        }
        return output
    }

    private fun fineAlign(
        sourceUnit: MotionUnit,
        referenceUnit: MotionUnit,
        candidate: UnitCandidate,
    ): FineAlignment? {
        val sourceFrames = frames(source, sourceUnit)
        val referenceFrames = frames(reference, referenceUnit)
        if (sourceFrames.isEmpty() || referenceFrames.isEmpty()) return null
        stats.fineAlignmentCount += 1
        val n = sourceFrames.size
        val m = referenceFrames.size
        val band = max(2, ceil(max(n, m) * config.fineBandFraction).toInt())
        val maximumReferenceAdvance = maximumReferenceAdvance(sourceFrames, referenceFrames)
        stats.maximumFineBandWidth = max(stats.maximumFineBandWidth, band * 2 + 1)
        val back = Array(n) { hashMapOf<Int, Int>() }
        var previous = hashMapOf<Int, Double>()
        for (i in 0 until n) {
            val expected = expectedReferenceIndex(sourceFrames[i], sourceFrames, referenceFrames, candidate.shift)
            val current = hashMapOf<Int, Double>()
            for (j in max(0, expected - band)..min(m - 1, expected + band)) {
                stats.fineCells += 1L
                val distance = space.distance(sourceFrames[i], referenceFrames[j])
                val localCost = distance.cost + if (distance.oppositeReliable) config.oppositeDirectionCellPenalty else 0.0
                if (i == 0) {
                    current[j] = localCost
                    back[i][j] = j
                    continue
                }
                var bestCost = Double.POSITIVE_INFINITY
                var bestPrevious = -1
                for (previousJ in max(0, j - maximumReferenceAdvance)..j) {
                    val previousCost = previous[previousJ] ?: continue
                    val proposed = previousCost + timestampStepPenalty(
                        sourceFrames[i - 1],
                        sourceFrames[i],
                        referenceFrames[previousJ],
                        referenceFrames[j],
                    ) + localCost
                    if (proposed < bestCost) {
                        bestCost = proposed
                        bestPrevious = previousJ
                    }
                }
                if (bestPrevious >= 0) {
                    current[j] = bestCost
                    back[i][j] = bestPrevious
                }
            }
            if (current.isEmpty()) return null
            previous = current
        }
        val endReference = previous.minByOrNull { it.value }?.key ?: return null
        val mapping = IntArray(n)
        var referenceIndex = endReference
        for (i in n - 1 downTo 0) {
            mapping[i] = referenceIndex
            referenceIndex = back[i][referenceIndex] ?: referenceIndex
        }
        val slopeValid = slopeValidity(sourceFrames, referenceFrames, mapping)
        val frozenValid = frozenValidity(sourceFrames, mapping)
        val timeline = sourceFrames.mapIndexed { index, frame ->
            val target = referenceFrames[mapping[index]]
            val distance = space.distance(frame, target)
            stats.finePathTimestamps += 1L
            val confidenceValid = frame.confidence >= config.minimumFrameConfidence && target.confidence >= config.minimumFrameConfidence
            val coverageValid = distance.coverage >= config.descriptorMinimumCoverage
            val directionValid = !distance.oppositeReliable
            val costValid = distance.cost <= config.fineMaximumCellCost
            val warpValid = slopeValid[index] && frozenValid[index]
            if (!confidenceValid) stats.fineRejectedConfidence += 1L
            if (!coverageValid) stats.fineRejectedCoverage += 1L
            if (!directionValid) stats.fineRejectedOppositeDirection += 1L
            if (!costValid) stats.fineRejectedCost += 1L
            if (!warpValid) stats.fineRejectedWarp += 1L
            val defensible = confidenceValid && coverageValid && directionValid && costValid && warpValid
            if (defensible) {
                stats.fineAcceptedTimestamps += 1L
                TimestampCorrespondence.Matched(
                    frame.timestamp,
                    target.timestamp,
                    confidenceFromCost(distance.cost, distance.coverage, candidate.ambiguity),
                )
            } else {
                TimestampCorrespondence.UnmatchedSource(
                    frame.timestamp,
                    when {
                        frame.discontinuityBefore || target.discontinuityBefore -> UnmatchedReason.DISCONTINUITY
                        frame.confidence < config.minimumFrameConfidence || target.confidence < config.minimumFrameConfidence -> UnmatchedReason.INSUFFICIENT_DATA
                        distance.oppositeReliable -> UnmatchedReason.NO_COMPATIBLE_COUNTERPART
                        candidate.ambiguity > config.maximumAmbiguityForConfidentMatch -> UnmatchedReason.AMBIGUOUS
                        else -> UnmatchedReason.NO_COMPATIBLE_COUNTERPART
                    },
                    (1.0 - min(1.0, distance.cost / 2.0)).coerceIn(0.05, 0.8),
                )
            }
        }
        val matched = timeline.filterIsInstance<TimestampCorrespondence.Matched>()
        if (matched.isEmpty()) return null
        val matchedFraction = matched.size.toDouble() / timeline.size.toDouble()
        val meanConfidence = matched.map(TimestampCorrespondence.Matched::decisionConfidence).averageOrZero()
        val pathCost = previous.getValue(endReference) / n.toDouble()
        return FineAlignment(
            timeline,
            matchedFraction,
            (meanConfidence * matchedFraction * exp(-0.25 * pathCost) * (1.0 - 0.35 * candidate.ambiguity))
                .coerceIn(0.0, 1.0),
        )
    }

    private fun slopeValidity(
        sourceFrames: List<PreparedTemporalFrame>,
        referenceFrames: List<PreparedTemporalFrame>,
        mapping: IntArray,
    ): BooleanArray {
        val valid = BooleanArray(sourceFrames.size) { true }
        for (index in sourceFrames.indices) {
            var lower = index
            var upper = index
            while (lower > 0 && sourceFrames[index].timestamp.value - sourceFrames[lower].timestamp.value < config.slopeWindowMs) lower -= 1
            while (upper < sourceFrames.lastIndex && sourceFrames[upper].timestamp.value - sourceFrames[index].timestamp.value < config.slopeWindowMs) upper += 1
            if (upper == lower) continue
            val sourceDelta = sourceFrames[upper].timestamp.value - sourceFrames[lower].timestamp.value
            val referenceDelta = referenceFrames[mapping[upper]].timestamp.value - referenceFrames[mapping[lower]].timestamp.value
            valid[index] = sourceDelta > 0L && (
                referenceDelta == 0L || referenceDelta.toDouble() / sourceDelta.toDouble() in
                    config.minimumWarpSlope..config.maximumWarpSlope
                )
        }
        return valid
    }

    private fun frozenValidity(sourceFrames: List<PreparedTemporalFrame>, mapping: IntArray): BooleanArray {
        val valid = BooleanArray(sourceFrames.size) { true }
        var start = 0
        while (start < mapping.size) {
            var end = start
            while (end + 1 < mapping.size && mapping[end + 1] == mapping[start]) end += 1
            if (sourceFrames[end].timestamp.value - sourceFrames[start].timestamp.value > config.maximumFrozenMappingMs) {
                for (index in start..end) valid[index] = false
            }
            start = end + 1
        }
        return valid
    }

    private fun frames(sequence: PreparedTemporalSequence, unit: MotionUnit): List<PreparedTemporalFrame> =
        sequence.framesInside(unit.range.start.value, unit.range.endExclusive.value)

    private fun unitPositionPrior(
        sourceIndex: Int,
        sourceCount: Int,
        referenceIndex: Int,
        referenceCount: Int,
    ): Double {
        if (sourceCount <= 1 || referenceCount <= 1) return 0.0
        val sourcePosition = sourceIndex.toDouble() / (sourceCount - 1).toDouble()
        val referencePosition = referenceIndex.toDouble() / (referenceCount - 1).toDouble()
        return 0.06 * abs(sourcePosition - referencePosition)
    }

    private fun ambiguity(best: UnitCandidate?, second: UnitCandidate?): Double {
        if (best == null || second == null) return 0.0
        return (1.0 - (second.cost - best.cost).coerceAtLeast(0.0) / max(1e-6, second.cost)).coerceIn(0.0, 1.0)
    }

    private fun compatibleUnits(sourceUnit: MotionUnit, referenceUnit: MotionUnit): Boolean {
        if (sourceUnit.structureClass == referenceUnit.structureClass) return true
        val cyclicAcyclicPair = setOf(sourceUnit.structureClass, referenceUnit.structureClass) ==
            setOf(MotionStructureClass.CYCLIC, MotionStructureClass.ACYCLIC)
        if (!cyclicAcyclicPair) return false
        val fragment = if (sourceUnit.structureClass == MotionStructureClass.ACYCLIC) sourceUnit else referenceUnit
        return fragment.completeness == UnitCompleteness.PARTIAL &&
            (fragment.startBoundary == UnitBoundaryStatus.OPEN || fragment.endBoundary == UnitBoundaryStatus.OPEN)
    }

    private fun expectedReferenceIndex(
        sourceFrame: PreparedTemporalFrame,
        sourceFrames: List<PreparedTemporalFrame>,
        referenceFrames: List<PreparedTemporalFrame>,
        shift: Int,
    ): Int {
        if (sourceFrames.size <= 1 || referenceFrames.size <= 1) return 0
        val sourceStart = sourceFrames.first().timestamp.value
        val sourceEnd = sourceFrames.last().timestamp.value
        val sourceDuration = sourceEnd - sourceStart
        val elapsedFraction = if (sourceDuration <= 0L) 0.0 else {
            (sourceFrame.timestamp.value - sourceStart).toDouble() / sourceDuration.toDouble()
        }
        val shiftedFraction = elapsedFraction + shift.toDouble() / config.coarseSamplesPerUnit.toDouble()
        val referenceStart = referenceFrames.first().timestamp.value
        val referenceEnd = referenceFrames.last().timestamp.value
        val targetTimestamp = referenceStart +
            (shiftedFraction.coerceIn(0.0, 1.0) * (referenceEnd - referenceStart).toDouble()).toLong()
        return nearestTimestampIndex(referenceFrames, targetTimestamp)
    }

    private fun confidenceFromCost(cost: Double, coverage: Double, ambiguity: Double): Double =
        (exp(-cost) * coverage * (1.0 - 0.35 * ambiguity)).coerceIn(0.0, 1.0)

    private fun frameAtFraction(frames: List<PreparedTemporalFrame>, fraction: Double): PreparedTemporalFrame {
        if (frames.size == 1) return frames.first()
        val start = frames.first().timestamp.value
        val end = frames.last().timestamp.value
        val target = start + (fraction.coerceIn(0.0, 1.0) * (end - start).toDouble()).toLong()
        return frames[nearestTimestampIndex(frames, target)]
    }

    private fun nearestTimestampIndex(frames: List<PreparedTemporalFrame>, targetTimestamp: Long): Int {
        var low = 0
        var high = frames.lastIndex
        while (low <= high) {
            val middle = (low + high) ushr 1
            val timestamp = frames[middle].timestamp.value
            when {
                timestamp < targetTimestamp -> low = middle + 1
                timestamp > targetTimestamp -> high = middle - 1
                else -> return middle
            }
        }
        if (low <= 0) return 0
        if (low >= frames.size) return frames.lastIndex
        val left = frames[low - 1].timestamp.value
        val right = frames[low].timestamp.value
        return if (targetTimestamp - left <= right - targetTimestamp) low - 1 else low
    }

    private fun maximumReferenceAdvance(
        sourceFrames: List<PreparedTemporalFrame>,
        referenceFrames: List<PreparedTemporalFrame>,
    ): Int {
        val sourceStepMs = medianPositiveStepMs(sourceFrames)
        val referenceStepMs = medianPositiveStepMs(referenceFrames)
        return ceil(config.maximumWarpSlope * sourceStepMs.toDouble() / referenceStepMs.toDouble())
            .toInt()
            .coerceAtLeast(1)
    }

    private fun medianPositiveStepMs(frames: List<PreparedTemporalFrame>): Long {
        val deltas = frames.zipWithNext().mapNotNull { (left, right) ->
            (right.timestamp.value - left.timestamp.value).takeIf { it > 0L }?.toDouble()
        }
        return quantile(deltas, 0.5).toLong().coerceAtLeast(1L)
    }

    private fun timestampStepPenalty(
        sourcePrevious: PreparedTemporalFrame,
        sourceCurrent: PreparedTemporalFrame,
        referencePrevious: PreparedTemporalFrame,
        referenceCurrent: PreparedTemporalFrame,
    ): Double {
        val sourceDeltaMs = (sourceCurrent.timestamp.value - sourcePrevious.timestamp.value).coerceAtLeast(1L)
        val referenceDeltaMs = referenceCurrent.timestamp.value - referencePrevious.timestamp.value
        val slope = referenceDeltaMs.toDouble() / sourceDeltaMs.toDouble()
        return config.fineStepPenalty * abs(slope - 1.0).coerceAtMost(config.maximumWarpSlope)
    }

    private fun update(
        costs: Array<DoubleArray>,
        moves: Array<ByteArray>,
        row: Int,
        column: Int,
        proposed: Double,
        movement: Byte,
    ) {
        if (proposed < costs[row][column]) {
            costs[row][column] = proposed
            moves[row][column] = movement
        }
    }

    private data class UnitCandidate(
        val referenceUnitId: String,
        val cost: Double,
        val shift: Int,
        val coverage: Double,
        val ambiguity: Double,
    )

    private data class FineAlignment(
        val timeline: List<TimestampCorrespondence>,
        val matchedFraction: Double,
        val decisionConfidence: Double,
    )

    private companion object {
        const val MATCH: Byte = 1
        const val SOURCE_SKIP: Byte = 2
        const val REFERENCE_SKIP: Byte = 3
    }
}

internal data class MutableTemporalStats(
    val sourceFrames: Int,
    val referenceFrames: Int,
    var sourceUnits: Int = 0,
    var referenceUnits: Int = 0,
    var coarseComparisons: Long = 0L,
    var coarseAcceptedCandidateCount: Int = 0,
    var bestCoarseCandidateCost: Double? = null,
    var fineCells: Long = 0L,
    var finePathTimestamps: Long = 0L,
    var fineAcceptedTimestamps: Long = 0L,
    var fineRejectedOppositeDirection: Long = 0L,
    var fineRejectedCost: Long = 0L,
    var fineRejectedCoverage: Long = 0L,
    var fineRejectedConfidence: Long = 0L,
    var fineRejectedWarp: Long = 0L,
    var bestFineMatchedFraction: Double? = null,
    var bestFineDecisionConfidence: Double? = null,
    var maximumFineBandWidth: Int = 0,
    var fineAlignmentCount: Int = 0,
) {
    fun freeze(): TemporalComputationStats = TemporalComputationStats(
        sourceFrameCount = sourceFrames,
        referenceFrameCount = referenceFrames,
        sourceUnitCount = sourceUnits,
        referenceUnitCount = referenceUnits,
        coarseUnitComparisons = coarseComparisons,
        coarseAcceptedCandidateCount = coarseAcceptedCandidateCount,
        bestCoarseCandidateCost = bestCoarseCandidateCost,
        fineCellsEvaluated = fineCells,
        finePathTimestampCount = finePathTimestamps,
        fineAcceptedTimestampCount = fineAcceptedTimestamps,
        fineRejectedOppositeDirectionCount = fineRejectedOppositeDirection,
        fineRejectedCostCount = fineRejectedCost,
        fineRejectedCoverageCount = fineRejectedCoverage,
        fineRejectedConfidenceCount = fineRejectedConfidence,
        fineRejectedWarpCount = fineRejectedWarp,
        bestFineMatchedFraction = bestFineMatchedFraction,
        bestFineDecisionConfidence = bestFineDecisionConfidence,
        maximumFineBandWidth = maximumFineBandWidth,
        fineAlignmentCount = fineAlignmentCount,
    )
}

private fun List<Double>.averageOrZero(): Double = if (isEmpty()) 0.0 else average()
