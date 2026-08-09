package ai.senp.alignment.temporal

import ai.senp.core.contracts.ActivitySegmentKind
import ai.senp.core.contracts.CanonicalObservationSequence
import ai.senp.core.contracts.DurationMs
import ai.senp.core.contracts.MotionStructureClass
import ai.senp.core.contracts.MotionUnitCorrespondence
import ai.senp.core.contracts.ObservationSampling
import ai.senp.core.contracts.SpatialSynchronizationDiagnostics
import ai.senp.core.contracts.SynchronizationRefusalReason
import ai.senp.core.contracts.SynchronizationRequest
import ai.senp.core.contracts.SynchronizationRequirements
import ai.senp.core.contracts.SynchronizationSemantics
import ai.senp.core.contracts.SynchronizationStatus
import ai.senp.core.contracts.TimestampCorrespondence
import ai.senp.core.contracts.TimestampMs
import ai.senp.core.contracts.UnitBoundaryStatus
import ai.senp.core.contracts.UnitCompleteness
import ai.senp.core.contracts.VideoRole
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class TemporalSynchronizationV2Test {
    private val engine = TemporalSynchronizationEngine()

    @Test
    fun selfIdentityIsNearlyIdentityAndHighConfidence() {
        val source = cyclic(VideoRole.SOURCE, cycles = 4)
        val reference = cyclic(VideoRole.REFERENCE, cycles = 4)
        val result = run(source, reference).result

        assertNotEquals(SynchronizationStatus.REFUSED, result.status)
        assertEquals(MotionStructureClass.CYCLIC, result.sourceTemporalStructure.classification)
        assertTrue(result.diagnostics.correspondenceConfidence > 0.70)
        val matches = result.correspondences.filterIsInstance<MotionUnitCorrespondence.MatchedUnit>()
            .flatMap { it.timeline }
            .filterIsInstance<TimestampCorrespondence.Matched>()
        assertTrue(matches.isNotEmpty())
        assertTrue(matches.count { abs(it.sourceTimestamp.value - it.referenceTimestamp.value) <= 100L } >= matches.size * 9 / 10)
    }

    @Test
    fun oneReferenceUnitIsReusableByTenSourceUnitsWithoutGlobalStretching() {
        val source = cyclic(VideoRole.SOURCE, cycles = 10)
        val reference = cyclic(VideoRole.REFERENCE, cycles = 1)
        val result = run(source, reference).result
        val matches = result.correspondences.filterIsInstance<MotionUnitCorrespondence.MatchedUnit>()

        assertEquals(10, result.sourceTemporalStructure.motionUnits.size)
        assertEquals(1, result.referenceTemporalStructure.motionUnits.size)
        assertEquals(10, matches.size)
        assertEquals(1, matches.map { it.referenceUnitId }.distinct().size)
        assertTrue(matches.zipWithNext().any { (left, right) ->
            val leftEnd = left.timeline.filterIsInstance<TimestampCorrespondence.Matched>().last().referenceTimestamp.value
            val rightStart = right.timeline.filterIsInstance<TimestampCorrespondence.Matched>().first().referenceTimestamp.value
            rightStart < leftEnd
        })
    }

    @Test
    fun tenReferenceUnitsVersusOneSourceLeavesReferenceUnitsExplicitlyUnmatched() {
        val source = cyclic(VideoRole.SOURCE, cycles = 1)
        val reference = cyclic(VideoRole.REFERENCE, cycles = 10)
        val result = run(source, reference).result

        assertNotEquals(SynchronizationStatus.REFUSED, result.status)
        assertEquals(1, result.correspondences.filterIsInstance<MotionUnitCorrespondence.MatchedUnit>().size)
        assertEquals(9, result.correspondences.filterIsInstance<MotionUnitCorrespondence.ReferenceUnmatchedUnit>().size)
        assertTrue(result.diagnostics.correspondenceAmbiguity > 0.80)
    }

    @Test
    fun reliableOppositeDirectionSamePoseIsNeverPaired() {
        val source = ramp(VideoRole.SOURCE, increasing = true)
        val reference = ramp(VideoRole.REFERENCE, increasing = false)
        val result = run(source, reference).result

        assertEquals(SynchronizationStatus.REFUSED, result.status)
        assertEquals(SynchronizationRefusalReason.NO_COMMON_MOTION, result.refusal?.reason)
        assertTrue(result.correspondences.none { it is MotionUnitCorrespondence.MatchedUnit })
    }

    @Test
    fun partialStartAndEndProduceOpenUnitsWithoutWholeVideoEndpointRequirement() {
        val source = cyclic(VideoRole.SOURCE, cycles = 3, startPhase = 0.23)
        val reference = cyclic(VideoRole.REFERENCE, cycles = 4)
        val result = run(source, reference).result
        val partial = result.sourceTemporalStructure.motionUnits.filter { it.completeness == UnitCompleteness.PARTIAL }

        assertTrue(partial.isNotEmpty())
        assertTrue(partial.any {
            it.startBoundary == UnitBoundaryStatus.OPEN || it.endBoundary == UnitBoundaryStatus.OPEN
        })
        assertNotEquals(SynchronizationStatus.REFUSED, result.status)
    }

    @Test
    fun multipleSetsPreserveRestAndSeparateActivityBlocks() {
        val source = cyclicSets(VideoRole.SOURCE)
        val reference = cyclicSets(VideoRole.REFERENCE)
        val result = run(source, reference).result
        val segments = result.sourceTemporalStructure.activitySegments

        assertTrue(segments.any { it.kind == ActivitySegmentKind.REST })
        assertTrue(segments.count { it.kind == ActivitySegmentKind.ACTIVE } >= 2)
        val rest = segments.first { it.kind == ActivitySegmentKind.REST }.range
        assertTrue(result.sourceTemporalStructure.motionUnits.none { unit ->
            unit.range.start < rest.start && unit.range.endExclusive > rest.endExclusive
        })
    }

    @Test
    fun staticClipIsIsometricHoldRatherThanMotionFailure() {
        val source = staticSignal(VideoRole.SOURCE)
        val reference = staticSignal(VideoRole.REFERENCE)
        val result = run(source, reference).result

        assertNotEquals(SynchronizationStatus.REFUSED, result.status)
        assertEquals(MotionStructureClass.ISOMETRIC, result.sourceTemporalStructure.classification)
        assertTrue(result.sourceTemporalStructure.activitySegments.all { it.kind == ActivitySegmentKind.HOLD })
        assertEquals(MotionStructureClass.ISOMETRIC, result.sourceTemporalStructure.motionUnits.single().structureClass)
    }

    @Test
    fun insufficientObservationsRefuseInsteadOfFabricatingMapping() {
        val source = ramp(VideoRole.SOURCE, increasing = true, confidence = 0.08)
        val reference = ramp(VideoRole.REFERENCE, increasing = true, confidence = 0.08)
        val result = run(source, reference).result

        assertEquals(SynchronizationStatus.REFUSED, result.status)
        assertEquals(SynchronizationRefusalReason.INSUFFICIENT_OBSERVATIONS, result.refusal?.reason)
        assertEquals(0.0, result.diagnostics.correspondenceConfidence)
    }

    @Test
    fun unobservedPrefixAndTailDoNotBecomeFabricatedIsometricHold() {
        val source = sparseLateStaticSignal(VideoRole.SOURCE)
        val reference = sparseLateStaticSignal(VideoRole.REFERENCE)
        val result = run(source, reference).result

        assertEquals(SynchronizationStatus.REFUSED, result.status)
        assertEquals(SynchronizationRefusalReason.INSUFFICIENT_OBSERVATIONS, result.refusal?.reason)
        assertTrue(result.sourceTemporalStructure.activitySegments.any { it.kind == ActivitySegmentKind.UNRELIABLE })
        assertTrue(result.diagnostics.sourceAnalyzableFraction < 0.10)
    }

    @Test
    fun acyclicMotionUsesOrderedLocalCorrespondence() {
        val source = ramp(VideoRole.SOURCE, increasing = true)
        val reference = ramp(VideoRole.REFERENCE, increasing = true, durationMs = 2_800L)
        val result = run(source, reference).result

        assertNotEquals(SynchronizationStatus.REFUSED, result.status)
        assertEquals(MotionStructureClass.ACYCLIC, result.sourceTemporalStructure.classification)
        val matched = result.correspondences.filterIsInstance<MotionUnitCorrespondence.MatchedUnit>().single()
        val referenceTimes = matched.timeline.filterIsInstance<TimestampCorrespondence.Matched>().map { it.referenceTimestamp }
        assertTrue(referenceTimes.zipWithNext().all { (left, right) -> left <= right })
    }

    @Test
    fun editedSpliceCreatesDiscontinuityAndNoUnitCrossesIt() {
        val source = cyclic(VideoRole.SOURCE, cycles = 4, discontinuityAtMs = 2_000L)
        val reference = cyclic(VideoRole.REFERENCE, cycles = 4)
        val result = run(source, reference).result

        assertTrue(result.sourceTemporalStructure.activitySegments.any { it.kind == ActivitySegmentKind.DISCONTINUITY })
        assertTrue(result.sourceTemporalStructure.motionUnits.none { unit ->
            unit.range.start.value < 2_000L && unit.range.endExclusive.value > 2_000L
        })
        assertNotEquals(SynchronizationStatus.SYNCHRONIZED, result.status)
    }

    @Test
    fun requiredObjectChannelWithoutEvidenceIsTypedRefusal() {
        val source = cyclic(VideoRole.SOURCE, cycles = 2)
        val reference = cyclic(VideoRole.REFERENCE, cycles = 2)
        val request = request(source, reference).copy(
            requirements = SynchronizationRequirements(setOf("object_state")),
        )
        val result = engine.synchronize(
            request,
            testSpatialDiagnostics(),
            sourceSignal = source,
            referenceSignal = reference,
        )

        assertEquals(SynchronizationStatus.REFUSED, result.status)
        assertEquals(SynchronizationRefusalReason.REQUIRED_CHANNEL_MISSING, result.refusal?.reason)
        assertEquals(setOf("object_state"), result.refusal?.missingRequiredChannelSemanticTypes)
    }

    @Test
    fun nonlinearVariableSpeedKeepsUnitLocalMonotonicWarp() {
        val reference = warpedCyclic(VideoRole.REFERENCE, cycles = 3, durationMs = 3_000L, exponent = 1.0)
        val source = warpedCyclic(VideoRole.SOURCE, cycles = 3, durationMs = 3_900L, exponent = 1.18)
        val result = run(source, reference).result

        assertNotEquals(SynchronizationStatus.REFUSED, result.status)
        result.correspondences.filterIsInstance<MotionUnitCorrespondence.MatchedUnit>().forEach { unit ->
            val timestamps = unit.timeline.filterIsInstance<TimestampCorrespondence.Matched>().map { it.referenceTimestamp }
            assertTrue(timestamps.zipWithNext().all { (left, right) -> left <= right })
        }
    }

    @Test
    fun twoReferenceUnitsCanServeSevenSourceUnitsIndependently() {
        val source = cyclic(VideoRole.SOURCE, cycles = 7)
        val reference = cyclic(VideoRole.REFERENCE, cycles = 2)
        val result = run(source, reference).result
        val matches = result.correspondences.filterIsInstance<MotionUnitCorrespondence.MatchedUnit>()

        assertNotEquals(SynchronizationStatus.REFUSED, result.status)
        assertEquals(7, matches.size)
        assertTrue(matches.map { it.referenceUnitId }.distinct().size <= 2)
    }

    @Test
    fun differentSamplingCadencesStillAlignByTimestampedMotion() {
        val source = cyclic(VideoRole.SOURCE, cycles = 3, sampleMs = 67L)
        val reference = cyclic(VideoRole.REFERENCE, cycles = 3, sampleMs = 100L)
        val result = run(source, reference).result

        assertNotEquals(SynchronizationStatus.REFUSED, result.status)
        assertTrue(result.correspondences.filterIsInstance<MotionUnitCorrespondence.MatchedUnit>().isNotEmpty())
    }

    @Test
    fun irregularSamplingDensityUsesTimestampFractionsRatherThanSampleIndices() {
        val durationMs = 3_000L
        val gaps = longArrayOf(35L, 165L, 60L, 140L, 45L, 155L)
        val timestamps = buildList {
            var timestamp = 0L
            var gapIndex = 0
            while (timestamp < durationMs) {
                add(timestamp)
                timestamp += gaps[gapIndex % gaps.size]
                gapIndex += 1
            }
        }
        val source = cyclicAtTimestamps(VideoRole.SOURCE, durationMs, timestamps)
        val reference = cyclic(VideoRole.REFERENCE, cycles = 3, sampleMs = 100L)
        val result = run(source, reference).result

        val irregularMatches = result.correspondences.filterIsInstance<MotionUnitCorrespondence.MatchedUnit>()
            .flatMap { it.timeline }
            .filterIsInstance<TimestampCorrespondence.Matched>()
        val medianTimestampErrorMs = quantile(
            irregularMatches.map { abs(it.sourceTimestamp.value - it.referenceTimestamp.value).toDouble() },
            0.5,
        )
        assertNotEquals(SynchronizationStatus.REFUSED, result.status)
        assertTrue(irregularMatches.isNotEmpty())
        assertTrue(medianTimestampErrorMs < 80.0)
        assertTrue(result.diagnostics.correspondenceConfidence > 0.35)
    }

    @Test
    fun descriptorHistoryAndLookaheadNeverCrossDiscontinuity() {
        val source = signal(VideoRole.SOURCE, 700L, 50L) { timestamp ->
            FeatureVector(timestamp.toDouble() / 700.0, timestamp.toDouble() / 1_400.0, timestamp == 350L)
        }
        val reference = signal(VideoRole.REFERENCE, 700L, 50L) { timestamp ->
            FeatureVector(timestamp.toDouble() / 700.0, timestamp.toDouble() / 1_400.0)
        }
        val prepared = TemporalDescriptorSpace(source, reference, TemporalSynchronizationConfig()).prepare(source)
        val before = prepared.frames.single { it.timestamp.value == 300L }
        val after = prepared.frames.single { it.timestamp.value == 350L }

        assertTrue(before.lookaheadDelta.isEmpty())
        assertTrue(after.historyDelta.isEmpty())
    }

    @Test
    fun closedBoundarySemanticsTruthfullyRejectPartialClip() {
        val source = cyclic(VideoRole.SOURCE, cycles = 3, startPhase = 0.23)
        val reference = cyclic(VideoRole.REFERENCE, cycles = 3)
        val request = request(source, reference).copy(
            semantics = SynchronizationSemantics(
                allowOpenSourceBegin = false,
                allowOpenSourceEnd = false,
            ),
        )
        val result = engine.synchronize(
            request,
            testSpatialDiagnostics(),
            sourceSignal = source,
            referenceSignal = reference,
        )

        assertEquals(SynchronizationStatus.REFUSED, result.status)
        assertEquals(SynchronizationRefusalReason.UNRELIABLE_CORRESPONDENCE, result.refusal?.reason)
    }

    @Test
    fun noCommonMotionReturnsTypedRefusal() {
        val source = ramp(VideoRole.SOURCE, increasing = true)
        val reference = staticSignal(VideoRole.REFERENCE)
        val result = run(source, reference).result

        assertEquals(SynchronizationStatus.REFUSED, result.status)
        assertEquals(SynchronizationRefusalReason.NO_COMMON_MOTION, result.refusal?.reason)
        assertTrue(result.correspondences.none { it is MotionUnitCorrespondence.MatchedUnit })
    }

    @Test
    fun reversedCyclicClipIsNotGloballyWarpedIntoSuccess() {
        val source = cyclic(VideoRole.SOURCE, cycles = 3)
        val reference = reversedCyclic(VideoRole.REFERENCE, cycles = 3)
        val result = run(source, reference).result

        assertNotEquals(SynchronizationStatus.SYNCHRONIZED, result.status)
        if (result.status == SynchronizationStatus.PARTIAL) {
            assertTrue(
                result.correspondences.any { it is MotionUnitCorrespondence.SourceUnmatchedUnit } ||
                    result.correspondences.filterIsInstance<MotionUnitCorrespondence.MatchedUnit>()
                        .any { unit -> unit.timeline.any { it is TimestampCorrespondence.UnmatchedSource } },
            )
        }
    }

    @Test
    fun orderedAcyclicMissingStepLeavesReferenceUnitUnmatched() {
        val source = acyclicBlocks(VideoRole.SOURCE, blocks = 1)
        val reference = acyclicBlocks(VideoRole.REFERENCE, blocks = 2)
        val result = run(source, reference).result

        assertNotEquals(SynchronizationStatus.REFUSED, result.status)
        assertEquals(MotionStructureClass.ACYCLIC, result.sourceTemporalStructure.classification)
        assertTrue(result.referenceTemporalStructure.motionUnits.size >= 2)
        assertTrue(result.correspondences.any { it is MotionUnitCorrespondence.ReferenceUnmatchedUnit })
    }

    @Test
    fun deliberatePauseIsExplicitHoldWithinMixedTemporalStructure() {
        val source = rampWithPause(VideoRole.SOURCE)
        val reference = rampWithPause(VideoRole.REFERENCE)
        val result = run(source, reference).result

        assertNotEquals(SynchronizationStatus.REFUSED, result.status)
        assertTrue(result.sourceTemporalStructure.activitySegments.any { it.kind == ActivitySegmentKind.HOLD })
        assertEquals(MotionStructureClass.MIXED, result.sourceTemporalStructure.classification)
    }

    @Test
    fun boundedFineSearchDoesNotAllocateWholeVideoMatrix() {
        val source = cyclic(VideoRole.SOURCE, cycles = 40, sampleMs = 67L)
        val reference = cyclic(VideoRole.REFERENCE, cycles = 40, sampleMs = 67L)
        val started = System.nanoTime()
        val run = run(source, reference)
        val elapsedMs = (System.nanoTime() - started) / 1_000_000L
        println(
            "TEMPORAL_BENCHMARK frames=${run.stats.sourceFrameCount}+${run.stats.referenceFrameCount} " +
                "units=${run.stats.sourceUnitCount}+${run.stats.referenceUnitCount} " +
                "coarse=${run.stats.coarseUnitComparisons} fineCells=${run.stats.fineCellsEvaluated} " +
                "naiveCells=${run.stats.naiveWholeVideoCellCount} maxBand=${run.stats.maximumFineBandWidth} " +
                "elapsedMs=$elapsedMs",
        )

        assertNotEquals(SynchronizationStatus.REFUSED, run.result.status)
        assertTrue(run.stats.fineCellsEvaluated > 0L)
        assertTrue(run.stats.fineCellsEvaluated < run.stats.naiveWholeVideoCellCount / 5L)
        assertTrue(run.stats.maximumFineBandWidth < 20)
    }

    private fun run(source: TemporalSignalSequence, reference: TemporalSignalSequence): TemporalSynchronizationRun =
        engine.synchronizeDetailed(
            request(source, reference),
            testSpatialDiagnostics(),
            sourceSignal = source,
            referenceSignal = reference,
        )

    private fun testSpatialDiagnostics(): SpatialSynchronizationDiagnostics =
        SpatialSynchronizationDiagnostics(aggregateConfidence = 1.0)

    private fun request(source: TemporalSignalSequence, reference: TemporalSignalSequence): SynchronizationRequest =
        SynchronizationRequest(
            source = emptyCanonical(VideoRole.SOURCE, source.duration),
            reference = emptyCanonical(VideoRole.REFERENCE, reference.duration),
        )

    private fun emptyCanonical(role: VideoRole, duration: DurationMs): CanonicalObservationSequence =
        CanonicalObservationSequence(
            role = role,
            duration = duration,
            sampling = ObservationSampling(analysisFramesPerSecond = 20.0),
            observations = emptyList(),
        )

    private fun cyclic(
        role: VideoRole,
        cycles: Int,
        cycleMs: Long = 1_000L,
        sampleMs: Long = 50L,
        startPhase: Double = 0.0,
        discontinuityAtMs: Long? = null,
    ): TemporalSignalSequence {
        val duration = cycles * cycleMs
        return signal(role, duration, sampleMs) { timestamp ->
            val phase = timestamp.toDouble() / cycleMs.toDouble() + startPhase
            val angle = 2.0 * PI * phase
            FeatureVector(-cos(angle), 0.65 * sin(angle), discontinuityAtMs == timestamp)
        }
    }

    private fun cyclicAtTimestamps(
        role: VideoRole,
        durationMs: Long,
        timestamps: List<Long>,
        cycleMs: Long = 1_000L,
    ): TemporalSignalSequence {
        val frames = timestamps.filter { it < durationMs }.map { timestamp ->
            val angle = 2.0 * PI * timestamp.toDouble() / cycleMs.toDouble()
            TemporalSignalFrame(
                timestamp = TimestampMs(timestamp),
                features = mapOf(
                    "articulation.primary" to TemporalFeatureSample(-cos(angle), 0.98),
                    "articulation.secondary" to TemporalFeatureSample(0.65 * sin(angle), 0.98),
                ),
                confidence = 0.98,
            )
        }
        return TemporalSignalSequence(role, DurationMs(durationMs), frames)
    }

    private fun reversedCyclic(
        role: VideoRole,
        cycles: Int,
        cycleMs: Long = 1_000L,
    ): TemporalSignalSequence {
        val duration = cycles * cycleMs
        return signal(role, duration, 50L) { timestamp ->
            val phase = cycles.toDouble() - timestamp.toDouble() / cycleMs.toDouble()
            val angle = 2.0 * PI * phase
            FeatureVector(-cos(angle), 0.65 * sin(angle))
        }
    }

    private fun acyclicBlocks(role: VideoRole, blocks: Int): TemporalSignalSequence {
        val motionMs = 1_000L
        val restMs = 1_200L
        val duration = blocks * motionMs + (blocks - 1).coerceAtLeast(0) * restMs
        return signal(role, duration, 50L) { timestamp ->
            val blockWidth = motionMs + restMs
            val block = minOf(blocks - 1, (timestamp / blockWidth).toInt())
            val within = timestamp - block * blockWidth
            val base = block.toDouble()
            val value = if (within < motionMs) base + within.toDouble() / motionMs.toDouble() else base + 1.0
            FeatureVector(value, value * 0.35)
        }
    }

    private fun rampWithPause(role: VideoRole): TemporalSignalSequence = signal(role, 1_900L, 50L) { timestamp ->
        val value = when {
            timestamp < 700L -> 0.5 * timestamp.toDouble() / 700.0
            timestamp < 1_200L -> 0.5
            else -> 0.5 + 0.5 * (timestamp - 1_200L).toDouble() / 700.0
        }
        FeatureVector(value, value * 0.4)
    }

    private fun cyclicSets(role: VideoRole): TemporalSignalSequence {
        val firstEnd = 2_000L
        val restEnd = 3_300L
        val duration = 5_300L
        return signal(role, duration, 50L) { timestamp ->
            when {
                timestamp < firstEnd -> {
                    val angle = 2.0 * PI * timestamp.toDouble() / 1_000.0
                    FeatureVector(-cos(angle), 0.65 * sin(angle))
                }
                timestamp < restEnd -> FeatureVector(-1.0, 0.0)
                else -> {
                    val angle = 2.0 * PI * (timestamp - restEnd).toDouble() / 1_000.0
                    FeatureVector(-cos(angle), 0.65 * sin(angle))
                }
            }
        }
    }

    private fun warpedCyclic(
        role: VideoRole,
        cycles: Int,
        durationMs: Long,
        exponent: Double,
    ): TemporalSignalSequence = signal(role, durationMs, 50L) { timestamp ->
        val elapsed = timestamp.toDouble() / durationMs.toDouble()
        val phase = cycles * java.lang.Math.pow(elapsed, exponent)
        val angle = 2.0 * PI * phase
        FeatureVector(-cos(angle), 0.65 * sin(angle))
    }

    private fun ramp(
        role: VideoRole,
        increasing: Boolean,
        durationMs: Long = 2_000L,
        confidence: Double = 0.98,
    ): TemporalSignalSequence = signal(role, durationMs, 50L, confidence) { timestamp ->
        val fraction = timestamp.toDouble() / durationMs.toDouble()
        val value = if (increasing) fraction else 1.0 - fraction
        FeatureVector(value, value * 0.4)
    }

    private fun staticSignal(role: VideoRole): TemporalSignalSequence = signal(role, 2_000L, 50L) {
        FeatureVector(0.25, -0.15)
    }

    private fun sparseLateStaticSignal(role: VideoRole): TemporalSignalSequence = TemporalSignalSequence(
        role = role,
        duration = DurationMs(2_000L),
        frames = listOf(1_000L, 1_050L).map { timestamp ->
            TemporalSignalFrame(
                timestamp = TimestampMs(timestamp),
                features = mapOf(
                    "articulation.primary" to TemporalFeatureSample(0.25, 0.98),
                    "articulation.secondary" to TemporalFeatureSample(-0.15, 0.98),
                ),
                confidence = 0.98,
            )
        },
    )

    private fun signal(
        role: VideoRole,
        durationMs: Long,
        sampleMs: Long,
        confidence: Double = 0.98,
        values: (Long) -> FeatureVector,
    ): TemporalSignalSequence {
        val frames = generateSequence(0L) { it + sampleMs }
            .takeWhile { it < durationMs }
            .map { timestamp ->
                val vector = values(timestamp)
                TemporalSignalFrame(
                    timestamp = TimestampMs(timestamp),
                    features = mapOf(
                        "articulation.primary" to TemporalFeatureSample(vector.primary, confidence),
                        "articulation.secondary" to TemporalFeatureSample(vector.secondary, confidence),
                    ),
                    confidence = confidence,
                    discontinuityBefore = vector.discontinuity,
                )
            }.toList()
        return TemporalSignalSequence(role, DurationMs(durationMs), frames)
    }

    private data class FeatureVector(
        val primary: Double,
        val secondary: Double,
        val discontinuity: Boolean = false,
    )
}
