package ai.senp.alignment.temporal

import ai.senp.core.contracts.ActivitySegment
import ai.senp.core.contracts.ActivitySegmentKind
import ai.senp.core.contracts.MotionStructureClass
import ai.senp.core.contracts.MotionUnit
import ai.senp.core.contracts.TemporalStructure
import ai.senp.core.contracts.TimestampMs
import ai.senp.core.contracts.TimestampRange
import ai.senp.core.contracts.UnitBoundaryStatus
import ai.senp.core.contracts.UnitCompleteness
import ai.senp.core.contracts.VideoRole
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

internal data class DetectedTemporalStructure(
    val structure: TemporalStructure,
    val analyzableFraction: Double,
)

internal class TemporalStructureDetector(
    private val config: TemporalSynchronizationConfig,
) {
    fun detect(prepared: PreparedTemporalSequence): DetectedTemporalStructure {
        val sequence = prepared.signal
        if (sequence.duration.value <= 0L || prepared.frames.isEmpty()) {
            return DetectedTemporalStructure(
                TemporalStructure(
                    role = sequence.role,
                    duration = sequence.duration,
                    classification = MotionStructureClass.UNKNOWN,
                    activitySegments = emptyList(),
                    motionUnits = emptyList(),
                    confidence = 0.0,
                ),
                0.0,
            )
        }

        val cells = buildCells(prepared)
        val analyzableDuration = cells.filter { it.reliable && !it.discontinuity }.sumOf(Cell::durationMs)
        val analyzableFraction = (analyzableDuration.toDouble() / sequence.duration.value.toDouble()).coerceIn(0.0, 1.0)
        val reliableMotion = cells.filter(Cell::reliable).map(Cell::motion).filter { it > 1e-9 }
        val activeThreshold = max(
            config.minimumNormalizedActiveSpeedPerSecond,
            if (reliableMotion.isEmpty()) config.minimumNormalizedActiveSpeedPerSecond else {
                quantile(reliableMotion, config.activeVelocityQuantile) * config.activeVelocityMultiplier
            },
        )
        cells.forEach { cell -> cell.active = cell.reliable && !cell.discontinuity && cell.motion >= activeThreshold }
        bridgeShortQuietGaps(cells)
        removeShortActiveRuns(cells)
        val anyActive = cells.any(Cell::active)
        classifyCells(cells, anyActive)
        val activitySegments = mergeCells(cells)
        val motionUnits = buildMotionUnits(prepared, activitySegments)
        val classification = classifyStructure(motionUnits, activitySegments, analyzableFraction)
        val unitConfidence = motionUnits.map(MotionUnit::confidence).averageOrZero()
        val segmentConfidence = activitySegments.map(ActivitySegment::confidence).averageOrZero()
        val confidence = if (activitySegments.isEmpty()) 0.0 else {
            (0.45 * analyzableFraction + 0.30 * segmentConfidence + 0.25 * unitConfidence).coerceIn(0.0, 1.0)
        }
        return DetectedTemporalStructure(
            TemporalStructure(
                role = sequence.role,
                duration = sequence.duration,
                classification = classification,
                activitySegments = activitySegments,
                motionUnits = motionUnits,
                confidence = confidence,
            ),
            analyzableFraction,
        )
    }

    private fun buildCells(prepared: PreparedTemporalSequence): MutableList<Cell> {
        val frames = prepared.frames
        val smoothedMotion = smoothedMotion(frames)
        val output = mutableListOf<Cell>()
        if (frames.first().timestamp.value > 0L) {
            output += Cell(
                startMs = 0L,
                endMs = frames.first().timestamp.value,
                motion = 0.0,
                confidence = 0.0,
                reliable = false,
                discontinuity = false,
            )
        }
        frames.forEachIndexed { index, frame ->
            val end = if (index < frames.lastIndex) frames[index + 1].timestamp.value else prepared.signal.duration.value
            if (end <= frame.timestamp.value) return@forEachIndexed
            val nextHasDiscontinuity = frames.getOrNull(index + 1)?.discontinuityBefore == true
            val unsupportedTail = index == frames.lastIndex && end - frame.timestamp.value > config.discontinuityGapMs
            output += Cell(
                startMs = frame.timestamp.value,
                endMs = end,
                motion = smoothedMotion[index],
                confidence = if (unsupportedTail) 0.0 else frame.confidence,
                reliable = !unsupportedTail && frame.confidence >= config.minimumFrameConfidence,
                discontinuity = nextHasDiscontinuity || frame.discontinuityBefore,
            )
        }
        return output
    }

    private fun smoothedMotion(frames: List<PreparedTemporalFrame>): DoubleArray {
        if (config.motionSmoothingRadiusMs <= 0L) {
            return DoubleArray(frames.size) { index -> frames[index].motionMagnitude }
        }
        val prefix = DoubleArray(frames.size + 1)
        frames.indices.forEach { index -> prefix[index + 1] = prefix[index] + frames[index].motionMagnitude }
        val output = DoubleArray(frames.size)
        var left = 0
        var right = 0
        for (index in frames.indices) {
            val timestamp = frames[index].timestamp.value
            while (left < index && timestamp - frames[left].timestamp.value > config.motionSmoothingRadiusMs) left += 1
            if (right < index) right = index
            while (
                right + 1 < frames.size &&
                frames[right + 1].timestamp.value - timestamp <= config.motionSmoothingRadiusMs
            ) {
                right += 1
            }
            output[index] = (prefix[right + 1] - prefix[left]) / (right - left + 1).toDouble()
        }
        return output
    }

    private fun bridgeShortQuietGaps(cells: MutableList<Cell>) {
        var index = 0
        while (index < cells.size) {
            if (cells[index].active || !cells[index].reliable || cells[index].discontinuity) {
                index += 1
                continue
            }
            val start = index
            while (index < cells.size && !cells[index].active && cells[index].reliable && !cells[index].discontinuity) index += 1
            val end = index - 1
            val duration = cells[end].endMs - cells[start].startMs
            val surrounded = start > 0 && index < cells.size && cells[start - 1].active && cells[index].active
            if (surrounded && duration <= config.bridgeQuietGapMs) {
                for (candidate in start..end) cells[candidate].active = true
            }
        }
    }

    private fun removeShortActiveRuns(cells: MutableList<Cell>) {
        var index = 0
        while (index < cells.size) {
            if (!cells[index].active) {
                index += 1
                continue
            }
            val start = index
            while (index < cells.size && cells[index].active) index += 1
            val end = index - 1
            if (cells[end].endMs - cells[start].startMs < config.minimumActiveRunMs) {
                for (candidate in start..end) cells[candidate].active = false
            }
        }
    }

    private fun classifyCells(cells: MutableList<Cell>, anyActive: Boolean) {
        if (!anyActive) {
            cells.forEach { cell ->
                cell.kind = when {
                    cell.discontinuity -> ActivitySegmentKind.DISCONTINUITY
                    !cell.reliable -> ActivitySegmentKind.UNRELIABLE
                    else -> ActivitySegmentKind.HOLD
                }
            }
            return
        }
        val firstActive = cells.indexOfFirst(Cell::active)
        val lastActive = cells.indexOfLast(Cell::active)
        cells.forEachIndexed { index, cell ->
            cell.kind = when {
                cell.discontinuity -> ActivitySegmentKind.DISCONTINUITY
                !cell.reliable -> ActivitySegmentKind.UNRELIABLE
                cell.active -> ActivitySegmentKind.ACTIVE
                index < firstActive -> ActivitySegmentKind.SETUP
                index > lastActive -> ActivitySegmentKind.IDLE
                else -> ActivitySegmentKind.HOLD
            }
        }
        var index = firstActive + 1
        while (index < lastActive) {
            if (cells[index].kind != ActivitySegmentKind.HOLD) {
                index += 1
                continue
            }
            val start = index
            while (index <= lastActive && cells[index].kind == ActivitySegmentKind.HOLD) index += 1
            val end = index - 1
            val duration = cells[end].endMs - cells[start].startMs
            val replacement = when {
                duration >= config.restGapMs -> ActivitySegmentKind.REST
                duration >= config.minimumHoldMs -> ActivitySegmentKind.HOLD
                else -> ActivitySegmentKind.ACTIVE
            }
            for (candidate in start..end) cells[candidate].kind = replacement
        }
    }

    private fun mergeCells(cells: List<Cell>): List<ActivitySegment> {
        if (cells.isEmpty()) return emptyList()
        val output = mutableListOf<ActivitySegment>()
        var start = 0
        for (index in 1..cells.size) {
            val boundary = index == cells.size || cells[index].kind != cells[start].kind
            if (!boundary) continue
            val rangeStart = cells[start].startMs
            val rangeEnd = cells[index - 1].endMs
            if (rangeEnd > rangeStart) {
                val confidence = cells.subList(start, index).map(Cell::confidence).averageOrZero().coerceIn(0.0, 1.0)
                output += ActivitySegment(
                    TimestampRange(TimestampMs(rangeStart), TimestampMs(rangeEnd)),
                    cells[start].kind,
                    confidence,
                )
            }
            start = index
        }
        return output
    }

    private fun buildMotionUnits(
        prepared: PreparedTemporalSequence,
        segments: List<ActivitySegment>,
    ): List<MotionUnit> {
        if (segments.isEmpty()) return emptyList()
        val units = mutableListOf<UnitCandidate>()
        val activeLike = setOf(ActivitySegmentKind.ACTIVE, ActivitySegmentKind.HOLD)
        fun belongsToMotionBlock(segment: ActivitySegment): Boolean =
            segment.kind in activeLike || (
                segment.kind == ActivitySegmentKind.UNRELIABLE &&
                    config.maximumBridgeUnreliableGapMs > 0L &&
                    segment.range.duration().value <= config.maximumBridgeUnreliableGapMs
                )
        var index = 0
        while (index < segments.size) {
            if (!belongsToMotionBlock(segments[index])) {
                index += 1
                continue
            }
            val start = index
            while (index < segments.size && belongsToMotionBlock(segments[index])) index += 1
            val block = segments.subList(start, index)
            val blockStart = block.first().range.start.value
            val blockEnd = block.last().range.endExclusive.value
            val activeDuration = block.filter { it.kind == ActivitySegmentKind.ACTIVE }.sumOf { it.range.duration().value }
            if (activeDuration == 0L) {
                if (blockEnd - blockStart >= config.minimumHoldMs) {
                    units += UnitCandidate(blockStart, blockEnd, MotionStructureClass.ISOMETRIC, 0.82)
                }
                continue
            }
            val cyclic = cyclicUnits(prepared, blockStart, blockEnd)
            if (cyclic.size >= 2 || cyclic.singleOrNull()?.complete == true) {
                units += cyclic
            } else {
                block.forEach { segment ->
                    val duration = segment.range.duration().value
                    if (duration < min(config.minimumMotionUnitMs, config.minimumHoldMs)) return@forEach
                    units += UnitCandidate(
                        segment.range.start.value,
                        segment.range.endExclusive.value,
                        if (segment.kind == ActivitySegmentKind.HOLD) MotionStructureClass.ISOMETRIC else MotionStructureClass.ACYCLIC,
                        if (segment.kind == ActivitySegmentKind.HOLD) 0.78 else 0.72,
                    )
                }
            }
        }
        return units.filter { it.endMs - it.startMs >= 1L }.mapIndexed { unitIndex, candidate ->
            val clipStartsMoving = candidate.startMs <= prepared.frames.first().timestamp.value &&
                prepared.frames.first().motionMagnitude >= config.minimumNormalizedActiveSpeedPerSecond
            val lastFrame = prepared.frames.last()
            val clipEndsMoving = candidate.endMs >= prepared.signal.duration.value &&
                lastFrame.motionMagnitude >= config.minimumNormalizedActiveSpeedPerSecond
            val openStart = candidate.openStart ||
                (clipStartsMoving && candidate.structureClass != MotionStructureClass.CYCLIC)
            val openEnd = candidate.openEnd ||
                (clipEndsMoving && candidate.structureClass != MotionStructureClass.CYCLIC)
            val complete = !openStart && !openEnd
            MotionUnit(
                unitId = "${prepared.signal.role.name.lowercase()}-u${unitIndex.toString().padStart(4, '0')}",
                range = TimestampRange(TimestampMs(candidate.startMs), TimestampMs(candidate.endMs)),
                structureClass = candidate.structureClass,
                completeness = if (complete) UnitCompleteness.COMPLETE else UnitCompleteness.PARTIAL,
                startBoundary = if (openStart) UnitBoundaryStatus.OPEN else UnitBoundaryStatus.CLOSED,
                endBoundary = if (openEnd) UnitBoundaryStatus.OPEN else UnitBoundaryStatus.CLOSED,
                confidence = candidate.confidence.coerceIn(0.0, 1.0),
            )
        }
    }

    private fun cyclicUnits(
        prepared: PreparedTemporalSequence,
        startMs: Long,
        endMs: Long,
    ): List<UnitCandidate> {
        val frames = prepared.framesInside(startMs, endMs)
        if (frames.size < 5 || endMs - startMs < config.minimumCycleMs) return emptyList()
        val feature = dominantFeature(frames) ?: return emptyList()
        val values = frames.mapNotNull { frame -> frame.values[feature]?.value }
        if (values.size < frames.size * 0.7) return emptyList()
        val valueRange = quantile(values, 0.95) - quantile(values, 0.05)
        if (valueRange <= 1e-7) return emptyList()
        val turns = mutableListOf<Turn>()
        var previousSign = 0
        for (index in 1 until frames.size) {
            val derivative = frames[index].derivatives[feature] ?: continue
            val sign = when {
                derivative > 0.04 -> 1
                derivative < -0.04 -> -1
                else -> 0
            }
            if (sign != 0 && previousSign != 0 && sign != previousSign) {
                val turnIndex = (index - 1).coerceAtLeast(0)
                val isMinimum = previousSign < 0 && sign > 0
                turns += Turn(turnIndex, isMinimum)
            }
            if (sign != 0) previousSign = sign
        }
        val candidatesByKind = listOf(true, false).map { minimum ->
            turns.filter { it.minimum == minimum }.map { frames[it.index].timestamp.value }
        }.filter { it.size >= 2 }
        if (candidatesByKind.isEmpty()) {
            val startValue = frames.first().values[feature]?.value
            val endValue = frames.last().values[feature]?.value
            val closesLoop = startValue != null && endValue != null &&
                abs(startValue - endValue) <= valueRange * config.singleCycleEndpointToleranceFraction
            return if (turns.isNotEmpty() && closesLoop && endMs - startMs >= config.minimumCycleMs) {
                listOf(
                    UnitCandidate(
                        startMs = startMs,
                        endMs = endMs,
                        structureClass = MotionStructureClass.CYCLIC,
                        confidence = 0.84,
                        complete = true,
                    ),
                )
            } else {
                emptyList()
            }
        }
        val boundaryCore = candidatesByKind.minByOrNull { timestamps ->
            val gaps = timestamps.zipWithNext().map { (a, b) -> (b - a).toDouble() }
            if (gaps.isEmpty()) Double.POSITIVE_INFINITY else variation(gaps)
        }.orEmpty()
        if (boundaryCore.size < 2) return emptyList()
        val periods = boundaryCore.zipWithNext().map { (a, b) -> (b - a).toDouble() }
        val period = quantile(periods, 0.5).toLong()
        if (period < config.minimumCycleMs) return emptyList()
        val boundaryValues = boundaryCore.mapNotNull { timestamp -> nearestFrame(frames, timestamp).values[feature]?.value }
        val targetValue = boundaryValues.averageOrZero()
        val startValue = frames.first().values[feature]?.value
        val onePeriodLater = nearestFrame(frames, startMs + period)
        val durationInPeriods = (endMs - startMs).toDouble() / period.toDouble()
        val nearestWholePeriods = durationInPeriods.roundToInt()
        val durationSupportsEndpointPhase = nearestWholePeriods >= 2 &&
            abs(durationInPeriods - nearestWholePeriods.toDouble()) <= 0.15
        val endpointPhaseRepeats = durationSupportsEndpointPhase || (
            startValue != null && endpointMatches(frames.first(), feature, targetValue, valueRange) &&
                abs(onePeriodLater.timestamp.value - (startMs + period)) <= max(120L, period / 5L) &&
                onePeriodLater.values[feature]?.value?.let { repeated -> abs(repeated - startValue) <= valueRange * 0.20 } == true
            )
        val boundaries = if (endpointPhaseRepeats) {
            buildList {
                add(startMs)
                var boundary = startMs + period
                while (boundary < endMs) {
                    add(boundary)
                    boundary += period
                }
                if (endMs - last() >= (period * 0.55).toLong()) add(endMs)
            }.toMutableList()
        } else {
            boundaryCore.toMutableList()
        }
        if (!endpointPhaseRepeats && endpointMatches(frames.first(), feature, targetValue, valueRange) &&
            boundaries.first() - startMs <= (period * 0.25).toLong()
        ) {
            boundaries[0] = startMs
        } else if (!endpointPhaseRepeats && boundaries.first() - startMs >= (period * 0.55).toLong()) {
            boundaries.add(0, startMs)
        }
        if (endpointMatches(frames.last(), feature, targetValue, valueRange) &&
            endMs - boundaries.last() <= (period * 0.30).toLong()
        ) {
            boundaries[boundaries.lastIndex] = endMs
        } else if (endMs - boundaries.last() >= (period * 0.55).toLong()) {
            boundaries += endMs
        }
        val distinct = boundaries.distinct().sorted()
        if (distinct.size < 2) return emptyList()
        return distinct.zipWithNext().mapNotNull { (start, end) ->
            if (end - start < config.minimumMotionUnitMs) return@mapNotNull null
            val nearPeriod = abs((end - start).toDouble() - period.toDouble()) / period.toDouble() <= 0.45
            val openStart = start == startMs && !endpointMatches(frames.first(), feature, targetValue, valueRange)
            val openEnd = end == endMs && !endpointMatches(frames.last(), feature, targetValue, valueRange)
            UnitCandidate(
                startMs = start,
                endMs = end,
                structureClass = MotionStructureClass.CYCLIC,
                confidence = if (nearPeriod) 0.88 else 0.62,
                complete = nearPeriod && !openStart && !openEnd,
                openStart = openStart,
                openEnd = openEnd,
            )
        }
    }

    private fun dominantFeature(frames: List<PreparedTemporalFrame>): String? {
        val keys = frames.flatMap { it.values.keys }.toSet()
        return keys.maxByOrNull { key ->
            val values = frames.mapNotNull { it.values[key]?.takeIf { sample -> sample.confidence >= config.minimumFeatureConfidence }?.value }
            if (values.size < 3) 0.0 else quantile(values, 0.90) - quantile(values, 0.10)
        }
    }

    private fun endpointMatches(
        frame: PreparedTemporalFrame,
        feature: String,
        targetValue: Double,
        valueRange: Double,
    ): Boolean {
        val value = frame.values[feature]?.value ?: return false
        return abs(value - targetValue) <= valueRange * 0.20
    }

    private fun classifyStructure(
        units: List<MotionUnit>,
        segments: List<ActivitySegment>,
        analyzableFraction: Double,
    ): MotionStructureClass {
        if (analyzableFraction < config.minimumAnalyzableFraction) return MotionStructureClass.UNKNOWN
        if (units.isEmpty()) return MotionStructureClass.UNKNOWN
        val classes = units.map(MotionUnit::structureClass).toSet()
        val hasSustainedHold = segments.any {
            it.kind == ActivitySegmentKind.HOLD && it.range.duration().value >= config.minimumHoldMs
        }
        return when {
            classes.size > 1 -> MotionStructureClass.MIXED
            classes.single() == MotionStructureClass.CYCLIC && hasSustainedHold -> MotionStructureClass.MIXED
            else -> classes.single()
        }
    }

    private fun nearestFrame(frames: List<PreparedTemporalFrame>, timestamp: Long): PreparedTemporalFrame =
        frames.minBy { abs(it.timestamp.value - timestamp) }

    private fun variation(values: List<Double>): Double {
        if (values.isEmpty()) return Double.POSITIVE_INFINITY
        val mean = values.average()
        if (mean <= 1e-9) return Double.POSITIVE_INFINITY
        return values.map { abs(it - mean) }.average() / mean
    }

    private data class Turn(val index: Int, val minimum: Boolean)

    private data class UnitCandidate(
        val startMs: Long,
        val endMs: Long,
        val structureClass: MotionStructureClass,
        val confidence: Double,
        val complete: Boolean = true,
        val openStart: Boolean = false,
        val openEnd: Boolean = false,
    )

    private data class Cell(
        val startMs: Long,
        val endMs: Long,
        val motion: Double,
        val confidence: Double,
        val reliable: Boolean,
        val discontinuity: Boolean,
        var active: Boolean = false,
        var kind: ActivitySegmentKind = ActivitySegmentKind.IDLE,
    ) {
        val durationMs: Long get() = endMs - startMs
    }
}

private fun List<Double>.averageOrZero(): Double = if (isEmpty()) 0.0 else average()
