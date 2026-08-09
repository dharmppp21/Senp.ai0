package ai.senp.sync.validation

import ai.senp.core.contracts.ActivitySegment
import ai.senp.core.contracts.ActivitySegmentKind
import ai.senp.core.contracts.BodySideHypothesis
import ai.senp.core.contracts.CanonicalObservationSequence
import ai.senp.core.contracts.DurationMs
import ai.senp.core.contracts.MirrorHypothesis
import ai.senp.core.contracts.MotionUnit
import ai.senp.core.contracts.MotionUnitCorrespondence
import ai.senp.core.contracts.MotionStructureClass
import ai.senp.core.contracts.RelativeViewHypothesis
import ai.senp.core.contracts.SpatialDiagnosticReason
import ai.senp.core.contracts.SpatialReliabilitySegment
import ai.senp.core.contracts.SpatialReliabilityStatus
import ai.senp.core.contracts.SpatialSynchronizationDiagnostics
import ai.senp.core.contracts.SynchronizationDiagnostics
import ai.senp.core.contracts.SynchronizationRefusal
import ai.senp.core.contracts.SynchronizationRefusalReason
import ai.senp.core.contracts.SynchronizationResult
import ai.senp.core.contracts.SynchronizationStatus
import ai.senp.core.contracts.TemporalStructure
import ai.senp.core.contracts.TimestampCorrespondence
import ai.senp.core.contracts.TimestampMs
import ai.senp.core.contracts.TimestampRange
import ai.senp.core.contracts.UnitBoundaryStatus
import ai.senp.core.contracts.UnitCompleteness
import ai.senp.core.contracts.UnmatchedReason
import ai.senp.core.contracts.VideoRole

internal object ValidationTestFixtures {
    fun resultFor(
        scenario: SyntheticScenarioBundle,
        oppositeFirst: Boolean = false,
    ): SynchronizationResult {
        val sourceTemporal = temporal(scenario, VideoRole.SOURCE)
        val referenceTemporal = temporal(scenario, VideoRole.REFERENCE)
        val spatial = spatial(scenario)
        val status = when {
            SynchronizationStatus.REFUSED in scenario.expectedOutcome.allowedStatuses -> SynchronizationStatus.REFUSED
            SynchronizationStatus.SYNCHRONIZED in scenario.expectedOutcome.allowedStatuses -> SynchronizationStatus.SYNCHRONIZED
            else -> SynchronizationStatus.PARTIAL
        }
        val correspondences = if (status == SynchronizationStatus.REFUSED) {
            sourceTemporal.motionUnits.map {
                MotionUnitCorrespondence.SourceUnmatchedUnit(it.unitId, UnmatchedReason.INSUFFICIENT_DATA, 0.9)
            } + referenceTemporal.motionUnits.map {
                MotionUnitCorrespondence.ReferenceUnmatchedUnit(it.unitId, UnmatchedReason.INSUFFICIENT_DATA, 0.9)
            }
        } else {
            correspondences(scenario, sourceTemporal, referenceTemporal, oppositeFirst)
        }
        val refusal = if (status == SynchronizationStatus.REFUSED) {
            val reason = scenario.expectedOutcome.allowedRefusalReasons.firstOrNull()
                ?: SynchronizationRefusalReason.UNRELIABLE_CORRESPONDENCE
            SynchronizationRefusal(
                reason = reason,
                message = "validation stub refusal",
                missingRequiredChannelSemanticTypes = if (reason == SynchronizationRefusalReason.REQUIRED_CHANNEL_MISSING) setOf("object_pose") else emptySet(),
            )
        } else null
        val limitedCoverage = scenario.scenarioId in setOf("poor_pose_coverage", "long_occlusion")
        val diagnostics = SynchronizationDiagnostics(
            overallConfidence = if (status == SynchronizationStatus.REFUSED) 0.2 else 0.9,
            spatialConfidence = spatial.aggregateConfidence,
            temporalConfidence = if (status == SynchronizationStatus.REFUSED) 0.3 else 0.9,
            correspondenceConfidence = if (status == SynchronizationStatus.REFUSED) 0.2 else 0.9,
            sourceAnalyzableFraction = if (limitedCoverage) 0.42 else 0.95,
            referenceAnalyzableFraction = 0.95,
            correspondenceAmbiguity = if (scenario.expectedOutcome.ambiguityMustBeExplicit) 0.55 else 0.05,
        )
        return SynchronizationResult(
            status = status,
            sourceTemporalStructure = sourceTemporal,
            referenceTemporalStructure = referenceTemporal,
            spatialDiagnostics = spatial,
            correspondences = correspondences,
            diagnostics = diagnostics,
            refusal = refusal,
        )
    }

    private fun correspondences(
        scenario: SyntheticScenarioBundle,
        source: TemporalStructure,
        reference: TemporalStructure,
        oppositeFirst: Boolean,
    ): List<MotionUnitCorrespondence> {
        val expectedSourceUnmatched = scenario.expectedOutcome.expectedSourceUnmatchedUnitIds.size
        val expectedReferenceUnmatched = scenario.expectedOutcome.expectedReferenceUnmatchedUnitIds.size
        val sourceToMatch = source.motionUnits.dropLast(expectedSourceUnmatched)
        val referenceToMatchCount = (reference.motionUnits.size - expectedReferenceUnmatched).coerceAtLeast(1)
        val matchedReferenceIds = mutableSetOf<String>()
        val result = mutableListOf<MotionUnitCorrespondence>()
        sourceToMatch.forEachIndexed { index, sourceUnit ->
            var referenceIndex = when (scenario.scenarioId) {
                "non_cyclic_activity" -> if (index == 0) 0 else 2
                else -> index % referenceToMatchCount
            }
            referenceIndex = referenceIndex.coerceAtMost(reference.motionUnits.lastIndex)
            val referenceUnit = reference.motionUnits[referenceIndex]
            matchedReferenceIds += referenceUnit.unitId
            val preferredSourceMs = sourceUnit.range.start.value + minOf(200L, sourceUnit.range.duration().value / 2)
            val sourceTimestamp = TimestampMs(
                scenario.sourceTruth.samples
                    .filter { it.reliable && sourceUnit.range.contains(TimestampMs(it.timestampMs)) }
                    .minByOrNull { kotlin.math.abs(it.timestampMs - preferredSourceMs) }
                    ?.timestampMs
                    ?: preferredSourceMs,
            )
            val sourcePhase = scenario.sourceTruth.samples
                .minByOrNull { kotlin.math.abs(it.timestampMs - sourceTimestamp.value) }
                ?.phaseProgress
                ?: 0.2
            val preferredReferenceMs = referenceUnit.range.start.value + minOf(200L, referenceUnit.range.duration().value / 2)
            var referenceTimestamp = TimestampMs(
                scenario.referenceTruth.samples
                    .filter { it.reliable && referenceUnit.range.contains(TimestampMs(it.timestampMs)) }
                    .minByOrNull { kotlin.math.abs(it.phaseProgress - sourcePhase) }
                    ?.timestampMs
                    ?: preferredReferenceMs,
            )
            if (oppositeFirst && index == 0 && referenceUnit.range.duration().value >= 800L) {
                referenceTimestamp = TimestampMs(referenceUnit.range.start.value + 700L)
            }
            val timeline = mutableListOf<ai.senp.core.contracts.TimestampCorrespondence>(
                TimestampCorrespondence.Matched(sourceTimestamp, referenceTimestamp, 0.9),
            )
            if (scenario.expectedOutcome.expectedSourceUnmatchedRanges.isNotEmpty() && index == 1) {
                val gap = scenario.expectedOutcome.expectedSourceUnmatchedRanges.first()
                val unmatched = gap.start
                if (sourceUnit.range.contains(unmatched) && unmatched != sourceTimestamp) {
                    timeline += TimestampCorrespondence.UnmatchedSource(unmatched, UnmatchedReason.OCCLUSION, 0.95)
                    timeline.sortBy { it.sourceTimestamp.value }
                }
            }
            result += MotionUnitCorrespondence.MatchedUnit(
                sourceUnitId = sourceUnit.unitId,
                referenceUnitId = referenceUnit.unitId,
                timeline = timeline,
                decisionConfidence = 0.9,
                ambiguity = if (scenario.expectedOutcome.ambiguityMustBeExplicit) 0.5 else 0.05,
            )
        }
        source.motionUnits.drop(sourceToMatch.size).forEach {
            result += MotionUnitCorrespondence.SourceUnmatchedUnit(it.unitId, UnmatchedReason.EXTRA_ACTION, 0.9)
        }
        reference.motionUnits.filter { it.unitId !in matchedReferenceIds }.forEach {
            result += MotionUnitCorrespondence.ReferenceUnmatchedUnit(it.unitId, UnmatchedReason.MISSING_REFERENCE_STEP, 0.9)
        }
        return result
    }

    fun temporal(scenario: SyntheticScenarioBundle, role: VideoRole): TemporalStructure {
        val truth = if (role == VideoRole.SOURCE) scenario.sourceTruth else scenario.referenceTruth
        val duration = if (role == VideoRole.SOURCE) scenario.request.source.duration else scenario.request.reference.duration
        val units = truth.units.map {
            val partial = it.openBegin || it.openEnd
            MotionUnit(
                unitId = it.unitId,
                range = it.range,
                structureClass = it.structureClass,
                completeness = if (partial) UnitCompleteness.PARTIAL else UnitCompleteness.COMPLETE,
                startBoundary = if (it.openBegin) UnitBoundaryStatus.OPEN else UnitBoundaryStatus.CLOSED,
                endBoundary = if (it.openEnd) UnitBoundaryStatus.OPEN else UnitBoundaryStatus.CLOSED,
                confidence = 0.9,
            )
        }
        val activity = activitySegments(scenario, role, duration)
        return TemporalStructure(role, duration, truth.classification, activity, units, 0.9)
    }

    private fun activitySegments(
        scenario: SyntheticScenarioBundle,
        role: VideoRole,
        duration: DurationMs,
    ): List<ActivitySegment> {
        if (scenario.scenarioId == "multiple_sets_rests") {
            return listOf(
                ActivitySegment(range(0, 2000), ActivitySegmentKind.ACTIVE, 0.9),
                ActivitySegment(range(2000, 3000), ActivitySegmentKind.REST, 0.9),
                ActivitySegment(range(3000, duration.value), ActivitySegmentKind.ACTIVE, 0.9),
            )
        }
        if (scenario.scenarioId == "static_isometric") {
            return listOf(ActivitySegment(range(0, duration.value), ActivitySegmentKind.HOLD, 0.9))
        }
        if (scenario.scenarioId == "pause_hold") {
            val truth = if (role == VideoRole.SOURCE) scenario.sourceTruth else scenario.referenceTruth
            val holdSamples = truth.samples.filter { it.state == "HOLD" }
            if (holdSamples.isNotEmpty()) {
                val period = truth.samples.zipWithNext().map { (left, right) -> right.timestampMs - left.timestampMs }.minOrNull() ?: 100L
                val holdStart = holdSamples.first().timestampMs
                val holdEnd = minOf(duration.value, holdSamples.last().timestampMs + period)
                return buildList {
                    if (holdStart > 0) add(ActivitySegment(range(0, holdStart), ActivitySegmentKind.ACTIVE, 0.9))
                    add(ActivitySegment(range(holdStart, holdEnd), ActivitySegmentKind.HOLD, 0.9))
                    if (holdEnd < duration.value) add(ActivitySegment(range(holdEnd, duration.value), ActivitySegmentKind.ACTIVE, 0.9))
                }
            }
        }
        val gap = if (role == VideoRole.SOURCE) scenario.spatialTruth.expectedDiscontinuities.firstOrNull() else null
        if (gap != null && gap.endExclusive.value <= duration.value) {
            return buildList {
                if (gap.start.value > 0) add(ActivitySegment(range(0, gap.start.value), ActivitySegmentKind.ACTIVE, 0.9))
                val gapKind = if (scenario.scenarioId == "long_occlusion") {
                    ActivitySegmentKind.UNRELIABLE
                } else {
                    ActivitySegmentKind.DISCONTINUITY
                }
                add(ActivitySegment(gap, gapKind, 0.9))
                if (gap.endExclusive.value < duration.value) add(ActivitySegment(range(gap.endExclusive.value, duration.value), ActivitySegmentKind.ACTIVE, 0.9))
            }
        }
        return listOf(ActivitySegment(range(0, duration.value), ActivitySegmentKind.ACTIVE, 0.9))
    }

    fun spatialOutput(scenario: SyntheticScenarioBundle): SpatialHarnessOutput {
        val sourceScale = bodyScale(scenario.request.source)
        val referenceScale = bodyScale(scenario.request.reference)
        val factor = if (sourceScale > 1e-9 && referenceScale > 1e-9) referenceScale / sourceScale else 1.0
        return SpatialHarnessOutput(
            canonicalSource = scaleHumanPose(scenario.request.source, factor),
            canonicalReference = scenario.request.reference,
            diagnostics = spatial(scenario),
        )
    }

    private fun scaleHumanPose(sequence: CanonicalObservationSequence, factor: Double): CanonicalObservationSequence = sequence.copy(
        observations = sequence.observations.map { observation ->
            observation.copy(
                channels = observation.channels.map { channel ->
                    if (channel.semanticType != "human_pose") {
                        channel
                    } else {
                        channel.copy(
                            values = channel.values.map { value ->
                                value.copy(values = value.values.map { component -> component?.times(factor) })
                            },
                        )
                    }
                },
            )
        },
    )

    private fun bodyScale(sequence: CanonicalObservationSequence): Double {
        val channel = sequence.observations.asSequence()
            .flatMap { it.channels.asSequence() }
            .first { it.semanticType == "human_pose" && it.values.all { value -> value.values.all { component -> component != null } } }
        fun point(key: String): List<Double> = channel.values.single { it.key == key }.values.map { requireNotNull(it) }
        fun distance(left: List<Double>, right: List<Double>): Double = kotlin.math.sqrt(
            left.indices.sumOf { axis ->
                val delta = left[axis] - right[axis]
                delta * delta
            },
        )
        val pelvis = point("pelvis")
        val leftAnkle = point("left_ankle")
        val rightAnkle = point("right_ankle")
        val ankleMid = leftAnkle.indices.map { axis -> (leftAnkle[axis] + rightAnkle[axis]) / 2.0 }
        return distance(pelvis, ankleMid)
    }

    fun spatial(scenario: SyntheticScenarioBundle): SpatialSynchronizationDiagnostics {
        val sourceRange = range(0, scenario.request.source.duration.value)
        val referenceRange = range(0, scenario.request.reference.duration.value)
        val view = RelativeViewHypothesis(
            sourceRange = sourceRange,
            referenceRange = referenceRange,
            relativeYawDegrees = scenario.spatialTruth.relativeYawDegrees,
            relativeElevationDegrees = scenario.spatialTruth.relativeElevationDegrees,
            mirror = scenario.spatialTruth.mirror,
            selectedBodySide = scenario.spatialTruth.selectedSide,
            confidence = 0.9,
            sideSelectionStability = if (scenario.spatialTruth.expectedStableSide) 0.95 else 0.3,
        )
        val reliability = if (scenario.spatialTruth.expectedDiscontinuities.isEmpty()) {
            listOf(
                SpatialReliabilitySegment(
                    role = VideoRole.SOURCE,
                    range = sourceRange,
                    status = SpatialReliabilityStatus.COMPATIBLE,
                    confidence = 0.95,
                ),
                SpatialReliabilitySegment(
                    role = VideoRole.REFERENCE,
                    range = referenceRange,
                    status = SpatialReliabilityStatus.COMPATIBLE,
                    confidence = 0.95,
                ),
            )
        } else {
            scenario.spatialTruth.expectedDiscontinuities.map {
                SpatialReliabilitySegment(
                    role = VideoRole.SOURCE,
                    range = it,
                    status = SpatialReliabilityStatus.DISCONTINUITY,
                    confidence = 0.95,
                    reasons = setOf(SpatialDiagnosticReason.CAMERA_DISCONTINUITY),
                )
            }
        }
        return SpatialSynchronizationDiagnostics(
            relativeViewHypotheses = listOf(view),
            reliabilitySegments = reliability,
            aggregateConfidence = 0.9,
        )
    }

    private fun range(start: Long, end: Long): TimestampRange = TimestampRange(TimestampMs(start), TimestampMs(end))
}
