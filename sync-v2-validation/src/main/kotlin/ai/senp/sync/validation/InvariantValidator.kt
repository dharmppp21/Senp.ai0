package ai.senp.sync.validation

import ai.senp.core.contracts.ActivitySegmentKind
import ai.senp.core.contracts.BodySideHypothesis
import ai.senp.core.contracts.CanonicalObservationSequence
import ai.senp.core.contracts.MirrorHypothesis
import ai.senp.core.contracts.MotionStructureClass
import ai.senp.core.contracts.MotionUnitCorrespondence
import ai.senp.core.contracts.SpatialReliabilityStatus
import ai.senp.core.contracts.SynchronizationRefusalReason
import ai.senp.core.contracts.SynchronizationResult
import ai.senp.core.contracts.SynchronizationStatus
import ai.senp.core.contracts.TimestampCorrespondence
import ai.senp.core.contracts.TimestampMs
import kotlin.math.abs

object InvariantValidator {
    /** Exact frozen invariant vocabulary. Keeping this exhaustive prevents a fixture addition from silently bypassing validation. */
    val supportedFrozenInvariants: Set<String> = setOf(
        "ACYCLIC_SUPPORTED",
        "AMBIGUITY_NOT_SILENT",
        "ANALYZABLE_COVERAGE_EXPLICIT",
        "CONFIDENCE_REQUIRED",
        "DISCONTINUITY_EXPLICIT",
        "FORM_DIFFERENCE_PRESERVED",
        "FPS_LENGTH_INDEPENDENT",
        "HOLD_EXPLICIT",
        "IDENTITY_NOT_REFUSED_WHEN_OBSERVABLE",
        "ISOMETRIC_SUPPORTED",
        "LOCAL_ALIGNMENT_ALLOWED",
        "LOCAL_WARP_ALLOWED",
        "MASKS_EXPLICIT",
        "MIRROR_HYPOTHESIS_EXPLICIT",
        "NO_CONFIDENT_MAPPING_ACROSS_UNRELIABLE_GAP",
        "NO_CYCLE_REQUIREMENT",
        "NO_EQUAL_REP_COUNT_ASSUMPTION",
        "NO_FABRICATED_SUCCESS",
        "NO_FORCED_PAIR",
        "NO_FORCED_UNMATCHED",
        "NO_NONRIGID_FORM_ERASURE",
        "OPEN_BOUNDARY_ALLOWED",
        "PARTIAL_UNIT_ALLOWED",
        "REFERENCE_UNIT_REUSE_ALLOWED",
        "REFERENCE_UNMATCHED_EXPLICIT",
        "REFUSAL_OR_PARTIAL_REQUIRED",
        "REFUSAL_REQUIRED",
        "REQUIRED_CHANNELS_EXPLICIT",
        "REST_SETUP_EXPLICIT",
        "SIDE_STABILITY_EXPLICIT",
        "SOURCE_TIMESTAMP_UNMATCHED_ALLOWED",
        "SOURCE_UNMATCHED_EXPLICIT",
        "SPATIAL_DIAGNOSTICS_EXPLICIT",
        "SPATIAL_DISCONTINUITY_EXPLICIT",
        "SPATIAL_NUISANCE_CANONICALIZED",
        "SPATIAL_RELIABILITY_EXPLICIT",
        "SPATIAL_TRANSFORM_ONLY_NUISANCE",
        "SUBJECT_AMBIGUITY_EXPLICIT",
        "SUBJECT_IDENTITY_NOT_ASSUMED",
        "SUBSEQUENCE_ALLOWED",
        "TIMESTAMP_FIRST",
        "TRANSPORT_METADATA_NOT_CORRESPONDENCE",
        "UNIT_LOCAL_WARP",
        "UNRELIABLE_SEGMENT_EXPLICIT",
        "VIEW_HYPOTHESIS_EXPLICIT",
    )

    fun validate(scenario: SyntheticScenarioBundle, result: SynchronizationResult): ScenarioValidationReport {
        val findings = mutableListOf<ValidationFinding>()
        fun check(invariant: String, condition: Boolean, message: String) {
            findings += ValidationFinding(invariant, condition, message)
        }

        val unknownInvariants = scenario.expectedInvariants - supportedFrozenInvariants
        check(
            "FROZEN_INVARIANT_VOCABULARY",
            unknownInvariants.isEmpty(),
            "unsupported frozen invariants=$unknownInvariants",
        )
        check(
            "EXPECTED_STATUS",
            result.status in scenario.expectedOutcome.allowedStatuses,
            "status=" + result.status + "; allowed=" + scenario.expectedOutcome.allowedStatuses,
        )

        if ("TIMESTAMP_FIRST" in scenario.expectedInvariants) {
            val ordered = listOf(scenario.request.source, scenario.request.reference).all { sequence ->
                sequence.observations.zipWithNext().all { (left, right) -> left.timestamp < right.timestamp }
            }
            check("TIMESTAMP_FIRST", ordered, "source/reference observations must use strictly increasing explicit timestamps")
        }
        if ("CONFIDENCE_REQUIRED" in scenario.expectedInvariants) {
            val confidenceValues = listOf(
                result.diagnostics.overallConfidence,
                result.diagnostics.spatialConfidence,
                result.diagnostics.temporalConfidence,
                result.diagnostics.correspondenceConfidence,
                result.spatialDiagnostics.aggregateConfidence,
            )
            check(
                "CONFIDENCE_REQUIRED",
                confidenceValues.all { it.isFinite() && it in 0.0..1.0 },
                "synchronization confidence channels must remain explicit finite probabilities",
            )
        }
        if ("IDENTITY_NOT_REFUSED_WHEN_OBSERVABLE" in scenario.expectedInvariants) {
            check(
                "IDENTITY_NOT_REFUSED_WHEN_OBSERVABLE",
                result.status != SynchronizationStatus.REFUSED,
                "observable identity input must not be refused",
            )
        }
        if ("REFUSAL_REQUIRED" in scenario.expectedInvariants) {
            check("REFUSAL_REQUIRED", result.status == SynchronizationStatus.REFUSED, "fixture requires truthful refusal")
        }
        if ("REFUSAL_OR_PARTIAL_REQUIRED" in scenario.expectedInvariants) {
            check(
                "REFUSAL_OR_PARTIAL_REQUIRED",
                result.status == SynchronizationStatus.REFUSED || result.status == SynchronizationStatus.PARTIAL,
                "evidence-limited fixture must refuse or remain partial",
            )
        }
        if ("NO_FABRICATED_SUCCESS" in scenario.expectedInvariants) {
            check(
                "NO_FABRICATED_SUCCESS",
                result.status != SynchronizationStatus.SYNCHRONIZED,
                "missing/common-motion evidence must not be promoted to synchronized success",
            )
        }
        if ("ANALYZABLE_COVERAGE_EXPLICIT" in scenario.expectedInvariants) {
            check(
                "ANALYZABLE_COVERAGE_EXPLICIT",
                result.diagnostics.sourceAnalyzableFraction < 1.0,
                "coverage-limited source must report a sub-unity analyzable fraction",
            )
        }
        if ("SPATIAL_DIAGNOSTICS_EXPLICIT" in scenario.expectedInvariants) {
            check(
                "SPATIAL_DIAGNOSTICS_EXPLICIT",
                result.spatialDiagnostics.relativeViewHypotheses.isNotEmpty() ||
                    result.spatialDiagnostics.reliabilitySegments.isNotEmpty() ||
                    result.spatialDiagnostics.sourceTransforms.isNotEmpty() ||
                    result.spatialDiagnostics.referenceTransforms.isNotEmpty(),
                "spatial normalization fixture requires inspectable diagnostics",
            )
        }
        if ("SPATIAL_RELIABILITY_EXPLICIT" in scenario.expectedInvariants) {
            val roles = result.spatialDiagnostics.reliabilitySegments.map { it.role }.toSet()
            check(
                "SPATIAL_RELIABILITY_EXPLICIT",
                roles.containsAll(setOf(ai.senp.core.contracts.VideoRole.SOURCE, ai.senp.core.contracts.VideoRole.REFERENCE)),
                "viewpoint fixture requires explicit source and reference spatial reliability coverage",
            )
        }
        if ("MASKS_EXPLICIT" in scenario.expectedInvariants) {
            val hasExplicitMissingMask = scenario.request.source.observations
                .flatMap { it.channels }
                .flatMap { it.values }
                .any { value -> value.mask.zip(value.values).any { (available, component) -> !available && component == null } }
            check("MASKS_EXPLICIT", hasExplicitMissingMask, "occluded synthetic observations must carry explicit component masks")
        }
        if ("TRANSPORT_METADATA_NOT_CORRESPONDENCE" in scenario.expectedInvariants) {
            val explicitTimestampTruth = scenario.request.source.observations.isNotEmpty() && scenario.request.reference.observations.isNotEmpty()
            check(
                "TRANSPORT_METADATA_NOT_CORRESPONDENCE",
                explicitTimestampTruth,
                "codec/edit metadata is fixture metadata only; correspondence truth remains timestamped observations",
            )
        }
        if ("FPS_LENGTH_INDEPENDENT" in scenario.expectedInvariants) {
            val cadenceIndependent = scenario.request.source.sampling.inputNominalFramesPerSecond != scenario.request.source.sampling.analysisFramesPerSecond ||
                scenario.request.reference.sampling.inputNominalFramesPerSecond != scenario.request.reference.sampling.analysisFramesPerSecond ||
                scenario.request.source.duration != scenario.request.reference.duration
            check("FPS_LENGTH_INDEPENDENT", cadenceIndependent, "fixture must distinguish transport cadence/duration from analysis timing")
        }
        if ("SUBJECT_AMBIGUITY_EXPLICIT" in scenario.expectedInvariants) {
            check(
                "SUBJECT_AMBIGUITY_EXPLICIT",
                result.refusal?.reason == SynchronizationRefusalReason.SUBJECT_AMBIGUITY || result.diagnostics.correspondenceAmbiguity > 0.0,
                "multi-subject input must expose subject ambiguity",
            )
        }
        if ("HOLD_EXPLICIT" in scenario.expectedInvariants) {
            check(
                "HOLD_EXPLICIT",
                result.sourceTemporalStructure.activitySegments.any { it.kind == ActivitySegmentKind.HOLD },
                "pause/hold input must retain an explicit HOLD activity segment",
            )
        }
        if ("NO_CYCLE_REQUIREMENT" in scenario.expectedInvariants) {
            check(
                "NO_CYCLE_REQUIREMENT",
                result.sourceTemporalStructure.classification != MotionStructureClass.CYCLIC,
                "isometric/acyclic input must not be forced into a cyclic classification",
            )
        }
        if ("UNRELIABLE_SEGMENT_EXPLICIT" in scenario.expectedInvariants) {
            val temporalUnreliable = result.sourceTemporalStructure.activitySegments.any {
                it.kind == ActivitySegmentKind.UNRELIABLE
            }
            val spatialUnreliable = result.spatialDiagnostics.reliabilitySegments.any {
                it.role == ai.senp.core.contracts.VideoRole.SOURCE &&
                    it.status in setOf(SpatialReliabilityStatus.UNRELIABLE, SpatialReliabilityStatus.DISCONTINUITY)
            }
            check(
                "UNRELIABLE_SEGMENT_EXPLICIT",
                temporalUnreliable || spatialUnreliable,
                "long occlusion must remain an explicit unreliable/discontinuity segment",
            )
        }
        if ("NO_CONFIDENT_MAPPING_ACROSS_UNRELIABLE_GAP" in scenario.expectedInvariants) {
            val gaps = scenario.spatialTruth.expectedDiscontinuities
            val confidentMatchesInGap = result.correspondences
                .filterIsInstance<MotionUnitCorrespondence.MatchedUnit>()
                .flatMap { it.timeline }
                .filterIsInstance<TimestampCorrespondence.Matched>()
                .count { decision ->
                    decision.decisionConfidence >= 0.5 && gaps.any { it.contains(decision.sourceTimestamp) }
                }
            check(
                "NO_CONFIDENT_MAPPING_ACROSS_UNRELIABLE_GAP",
                confidentMatchesInGap == 0,
                "unreliable/discontinuous source spans cannot contain confident timestamp matches; count=$confidentMatchesInGap",
            )
        }
        if ("SUBJECT_IDENTITY_NOT_ASSUMED" in scenario.expectedInvariants) {
            val subjectIds = scenario.sourceTruth.samples.mapNotNull { it.subjectId }.toSet()
            val discontinuityExplicit = result.sourceTemporalStructure.activitySegments.any {
                it.kind == ActivitySegmentKind.DISCONTINUITY
            } || result.spatialDiagnostics.reliabilitySegments.any {
                it.role == ai.senp.core.contracts.VideoRole.SOURCE && it.status == SpatialReliabilityStatus.DISCONTINUITY
            }
            check(
                "SUBJECT_IDENTITY_NOT_ASSUMED",
                subjectIds.size > 1 && (result.status == SynchronizationStatus.REFUSED || discontinuityExplicit),
                "leave/re-enter fixture must expose the subject transition instead of assuming identity continuity",
            )
        }

        if (result.status == SynchronizationStatus.REFUSED) {
            val reason = result.refusal?.reason
            val allowed = scenario.expectedOutcome.allowedRefusalReasons
            check(
                "TRUTHFUL_REFUSAL",
                allowed.isEmpty() || reason in allowed,
                "refusal=$reason; allowed=$allowed",
            )
            if (scenario.scenarioId == "object_required_pose_only") {
                check(
                    "REQUIRED_CHANNELS_EXPLICIT",
                    reason == SynchronizationRefusalReason.REQUIRED_CHANNEL_MISSING &&
                        result.refusal?.missingRequiredChannelSemanticTypes?.contains("object_pose") == true,
                    "pose-only object-dependent input must identify object_pose as missing",
                )
            }
            if (scenario.scenarioId == "poor_pose_coverage" || scenario.scenarioId == "long_occlusion") {
                check(
                    "ANALYZABLE_COVERAGE_EXPLICIT",
                    result.diagnostics.sourceAnalyzableFraction < 1.0,
                    "coverage-limited refusal must retain explicit analyzable coverage",
                )
            }
            return report(scenario, result, findings)
        }

        val matched = result.correspondences.filterIsInstance<MotionUnitCorrespondence.MatchedUnit>()
        val sourceUnmatched = result.correspondences.filterIsInstance<MotionUnitCorrespondence.SourceUnmatchedUnit>()
        val referenceUnmatched = result.correspondences.filterIsInstance<MotionUnitCorrespondence.ReferenceUnmatchedUnit>()

        if (scenario.scenarioId == "same_video_self") {
            val timestampMatches = matched.flatMap { it.timeline }.filterIsInstance<TimestampCorrespondence.Matched>()
            check("IDENTITY_MAPPING", timestampMatches.isNotEmpty(), "identity case must contain matched timestamps")
            check(
                "IDENTITY_MAPPING",
                timestampMatches.all { abs(it.sourceTimestamp.value - it.referenceTimestamp.value) <= 1L },
                "identity mapping must preserve timestamps within 1 ms",
            )
            check(
                "NO_FORCED_UNMATCHED",
                sourceUnmatched.isEmpty() && referenceUnmatched.isEmpty() && matched.none { unit -> unit.timeline.any { it is TimestampCorrespondence.UnmatchedSource } },
                "identity case must not introduce unmatched material",
            )
        }

        val expectedSourceUnmatchedCount = scenario.expectedOutcome.expectedSourceUnmatchedUnitIds.size
        if (expectedSourceUnmatchedCount > 0) {
            check(
                "SOURCE_UNMATCHED_EXPLICIT",
                sourceUnmatched.size >= expectedSourceUnmatchedCount,
                "expected at least " + expectedSourceUnmatchedCount + " explicit source-unmatched units; actual=" + sourceUnmatched.size,
            )
        }
        val expectedReferenceUnmatchedCount = scenario.expectedOutcome.expectedReferenceUnmatchedUnitIds.size
        if (expectedReferenceUnmatchedCount > 0) {
            check(
                "REFERENCE_UNMATCHED_EXPLICIT",
                referenceUnmatched.size >= expectedReferenceUnmatchedCount,
                "expected at least " + expectedReferenceUnmatchedCount + " explicit reference-unmatched units; actual=" + referenceUnmatched.size,
            )
        }
        scenario.expectedOutcome.expectedSourceUnmatchedRanges.forEach { expectedRange ->
            val representedByTimestamp = matched.flatMap { it.timeline }
                .filterIsInstance<TimestampCorrespondence.UnmatchedSource>()
                .any { expectedRange.contains(it.sourceTimestamp) }
            val unmatchedUnitIds = sourceUnmatched.mapTo(mutableSetOf()) { it.sourceUnitId }
            val representedByWholeUnit = result.sourceTemporalStructure.motionUnits.any { unit ->
                unit.unitId in unmatchedUnitIds &&
                    unit.range.start.value < expectedRange.endExclusive.value &&
                    expectedRange.start.value < unit.range.endExclusive.value
            }
            check(
                "SOURCE_TIMESTAMP_UNMATCHED_ALLOWED",
                representedByTimestamp || representedByWholeUnit,
                "expected the unreliable source range $expectedRange to be represented by an unmatched timestamp or an overlapping source-unmatched unit",
            )
        }

        if (scenario.expectedOutcome.referenceReuseRequired) {
            val counts = matched.groupingBy(MotionUnitCorrespondence.MatchedUnit::referenceUnitId).eachCount()
            check(
                "REFERENCE_UNIT_REUSE_ALLOWED",
                counts.values.any { it > 1 },
                "expected at least one reference motion unit to be independently reused; counts=$counts",
            )
        }

        if (scenario.expectedOutcome.ambiguityMustBeExplicit) {
            check(
                "AMBIGUITY_NOT_SILENT",
                result.diagnostics.correspondenceAmbiguity > 0.0 || matched.any { it.ambiguity > 0.0 },
                "ambiguous fixture requires non-zero correspondence ambiguity",
            )
        }

        if (scenario.expectedOutcome.zeroInteriorOppositeDirectionPairs) {
            val opposite = matched.flatMap { unit -> unit.timeline.filterIsInstance<TimestampCorrespondence.Matched>() }
                .count { decision -> isReliableInteriorOpposite(scenario, decision) }
            check(
                "ZERO_INTERIOR_OPPOSITE_DIRECTION_PAIRINGS",
                opposite == 0,
                "reliable interior mappings with opposite synthetic direction=$opposite",
            )
        }

        if (scenario.expectedInvariants.contains("OPEN_BOUNDARY_ALLOWED")) {
            val wantsBegin = scenario.sourceTruth.units.any { it.openBegin }
            val wantsEnd = scenario.sourceTruth.units.any { it.openEnd }
            if (wantsBegin) {
                check(
                    "OPEN_BEGIN_PARTIAL",
                    result.sourceTemporalStructure.motionUnits.any { it.startBoundary.name == "OPEN" },
                    "source mid-motion start requires an open-begin unit",
                )
            }
            if (wantsEnd) {
                check(
                    "OPEN_END_PARTIAL",
                    result.sourceTemporalStructure.motionUnits.any { it.endBoundary.name == "OPEN" },
                    "source mid-motion end requires an open-end unit",
                )
            }
        }
        if (scenario.scenarioId == "multiple_sets_rests") {
            check(
                "REST_SETUP_EXPLICIT",
                result.sourceTemporalStructure.activitySegments.any { it.kind == ActivitySegmentKind.REST },
                "multiple sets must retain an explicit REST segment",
            )
        }
        if (scenario.scenarioId == "static_isometric") {
            check(
                "ISOMETRIC_SUPPORTED",
                result.sourceTemporalStructure.classification == MotionStructureClass.ISOMETRIC &&
                    result.sourceTemporalStructure.motionUnits.any { it.structureClass == MotionStructureClass.ISOMETRIC } &&
                    result.sourceTemporalStructure.activitySegments.any { it.kind == ActivitySegmentKind.HOLD },
                "static hold must remain an isometric motion unit with HOLD semantics",
            )
        }
        if (scenario.scenarioId == "non_cyclic_activity") {
            check(
                "ACYCLIC_SUPPORTED",
                result.sourceTemporalStructure.classification == MotionStructureClass.ACYCLIC &&
                    result.referenceTemporalStructure.classification == MotionStructureClass.ACYCLIC,
                "ordered non-cyclic fixture must remain ACYCLIC",
            )
        }
        if (scenario.spatialTruth.expectedDiscontinuities.isNotEmpty()) {
            val temporalDiscontinuity = result.sourceTemporalStructure.activitySegments.any { it.kind == ActivitySegmentKind.DISCONTINUITY }
            val spatialDiscontinuity = result.spatialDiagnostics.reliabilitySegments.any {
                it.status == SpatialReliabilityStatus.DISCONTINUITY
            }
            check(
                "DISCONTINUITY_EXPLICIT",
                temporalDiscontinuity || spatialDiscontinuity,
                "known camera/edit/subject gap must be explicit in temporal or spatial diagnostics",
            )
        }
        if (scenario.expectedInvariants.contains("MIRROR_HYPOTHESIS_EXPLICIT")) {
            check(
                "MIRROR_HYPOTHESIS_EXPLICIT",
                result.spatialDiagnostics.relativeViewHypotheses.any { it.mirror == scenario.spatialTruth.mirror },
                "mirror fixture requires the expected explicit mirror hypothesis=" + scenario.spatialTruth.mirror,
            )
        }
        if (scenario.expectedInvariants.contains("VIEW_HYPOTHESIS_EXPLICIT")) {
            check(
                "VIEW_HYPOTHESIS_EXPLICIT",
                result.spatialDiagnostics.relativeViewHypotheses.any { hypothesis ->
                    val yaw = hypothesis.relativeYawDegrees
                    val elevation = hypothesis.relativeElevationDegrees
                    yaw != null && elevation != null &&
                        abs(yaw - scenario.spatialTruth.relativeYawDegrees) <= 5.0 &&
                        abs(elevation - scenario.spatialTruth.relativeElevationDegrees) <= 5.0
                },
                "viewpoint fixture requires yaw/elevation evidence consistent with the synthetic oracle",
            )
        }
        if (scenario.expectedInvariants.contains("SIDE_STABILITY_EXPLICIT")) {
            check(
                "SIDE_STABILITY_EXPLICIT",
                result.spatialDiagnostics.relativeViewHypotheses.any { hypothesis ->
                    val expectedStable = scenario.spatialTruth.expectedStableSide
                    val stabilityConsistent = if (expectedStable) {
                        hypothesis.sideSelectionStability >= 0.5
                    } else {
                        hypothesis.sideSelectionStability < 0.5
                    }
                    stabilityConsistent &&
                        (scenario.spatialTruth.selectedSide == BodySideHypothesis.UNKNOWN ||
                            hypothesis.selectedBodySide == scenario.spatialTruth.selectedSide) &&
                        (scenario.spatialTruth.mirror == MirrorHypothesis.AMBIGUOUS ||
                            hypothesis.mirror == scenario.spatialTruth.mirror)
                },
                "side-selection fixture requires explicit stability consistent with the synthetic oracle",
            )
        }
        if (scenario.scenarioId == "poor_pose_coverage" || scenario.scenarioId == "long_occlusion") {
            check(
                "ANALYZABLE_COVERAGE_EXPLICIT",
                result.diagnostics.sourceAnalyzableFraction < 1.0,
                "coverage-limited source must not report fully analyzable coverage",
            )
        }

        return report(scenario, result, findings)
    }

    fun validateSpatialOutput(scenario: SyntheticScenarioBundle, output: SpatialHarnessOutput): ScenarioValidationReport {
        val findings = mutableListOf<ValidationFinding>()
        if ("SPATIAL_NUISANCE_CANONICALIZED" in scenario.expectedInvariants) {
            val sourceScale = meanBodyScale(output.canonicalSource)
            val referenceScale = meanBodyScale(output.canonicalReference)
            val ratio = if (sourceScale > 1e-9 && referenceScale > 1e-9) sourceScale / referenceScale else Double.NaN
            findings += ValidationFinding(
                "SPATIAL_NUISANCE_CANONICALIZED",
                ratio.isFinite() && ratio in 0.98..1.02,
                "canonical source/reference body scale must agree within 2%; ratio=$ratio",
            )
        }
        val expectedDelta = scenario.spatialTruth.expectedTrueFormDelta
        if (expectedDelta > 0.0) {
            val actualDelta = abs(meanForm(output.canonicalSource) - meanForm(output.canonicalReference))
            findings += ValidationFinding(
                "FORM_DIFFERENCE_PRESERVED",
                actualDelta >= expectedDelta * 0.5,
                "expected non-rigid form delta to remain visible; expected=$expectedDelta actual=$actualDelta",
            )
        }
        if (scenario.spatialTruth.expectedDiscontinuities.isNotEmpty()) {
            findings += ValidationFinding(
                "SPATIAL_DISCONTINUITY_EXPLICIT",
                output.diagnostics.reliabilitySegments.any { it.status == SpatialReliabilityStatus.DISCONTINUITY },
                "spatial adapter must expose discontinuity rather than stretching through it",
            )
        }
        return ScenarioValidationReport(scenario.scenarioId, "SPATIAL", findings)
    }

    fun validateTemporalOutput(scenario: SyntheticScenarioBundle, structure: ai.senp.core.contracts.TemporalStructure): ScenarioValidationReport {
        val findings = mutableListOf<ValidationFinding>()
        if (scenario.sourceTruth.classification == MotionStructureClass.ISOMETRIC) {
            findings += ValidationFinding(
                "ISOMETRIC_SUPPORTED",
                structure.classification == MotionStructureClass.ISOMETRIC && structure.activitySegments.any { it.kind == ActivitySegmentKind.HOLD },
                "temporal adapter must represent a static hold without inventing a cycle",
            )
        }
        if (scenario.sourceTruth.classification == MotionStructureClass.ACYCLIC) {
            findings += ValidationFinding(
                "ACYCLIC_SUPPORTED",
                structure.classification == MotionStructureClass.ACYCLIC,
                "temporal adapter must preserve ordered acyclic structure",
            )
        }
        if (scenario.sourceTruth.activityKinds.contains(ActivitySegmentKind.REST)) {
            findings += ValidationFinding(
                "REST_SETUP_EXPLICIT",
                structure.activitySegments.any { it.kind == ActivitySegmentKind.REST },
                "rest between sets must remain explicit",
            )
        }
        return ScenarioValidationReport(scenario.scenarioId, "TEMPORAL", findings)
    }

    private fun isReliableInteriorOpposite(
        scenario: SyntheticScenarioBundle,
        decision: TimestampCorrespondence.Matched,
    ): Boolean {
        val source = nearestTruth(scenario.sourceTruth.samples, decision.sourceTimestamp)
        val reference = nearestTruth(scenario.referenceTruth.samples, decision.referenceTimestamp)
        return source.reliable && reference.reliable && source.interior && reference.interior &&
            source.direction != 0 && reference.direction != 0 && source.direction != reference.direction
    }

    private fun nearestTruth(samples: List<SyntheticTimestampTruth>, timestamp: TimestampMs): SyntheticTimestampTruth =
        samples.minBy { abs(it.timestampMs - timestamp.value) }

    private fun meanForm(sequence: CanonicalObservationSequence): Double {
        val ratios = sequence.observations.mapNotNull { observation ->
            val channel = observation.channels.firstOrNull {
                it.semanticType == "human_pose" && it.componentAxes == listOf("x", "y", "z")
            } ?: return@mapNotNull null
            fun point(key: String): List<Double>? {
                val values = channel.values.firstOrNull { it.key == key }?.values ?: return null
                if (values.any { it == null }) return null
                return values.map { requireNotNull(it) }
            }
            fun distance(left: List<Double>, right: List<Double>): Double = kotlin.math.sqrt(
                left.indices.sumOf { axis ->
                    val delta = left[axis] - right[axis]
                    delta * delta
                },
            )
            val leftShoulder = point("left_shoulder") ?: return@mapNotNull null
            val rightShoulder = point("right_shoulder") ?: return@mapNotNull null
            val pelvis = point("pelvis") ?: return@mapNotNull null
            val leftAnkle = point("left_ankle") ?: return@mapNotNull null
            val rightAnkle = point("right_ankle") ?: return@mapNotNull null
            val ankleMid = leftAnkle.indices.map { axis -> (leftAnkle[axis] + rightAnkle[axis]) / 2.0 }
            val bodyLength = distance(pelvis, ankleMid)
            if (bodyLength <= 1e-9) return@mapNotNull null
            distance(leftShoulder, rightShoulder) / bodyLength
        }
        return if (ratios.isEmpty()) 0.0 else ratios.average()
    }

    private fun meanBodyScale(sequence: CanonicalObservationSequence): Double {
        val scales = sequence.observations.mapNotNull { observation ->
            val channel = observation.channels.firstOrNull {
                it.semanticType == "human_pose" && it.componentAxes == listOf("x", "y", "z")
            } ?: return@mapNotNull null
            fun point(key: String): List<Double>? {
                val values = channel.values.firstOrNull { it.key == key }?.values ?: return null
                if (values.any { it == null }) return null
                return values.map { requireNotNull(it) }
            }
            fun distance(left: List<Double>, right: List<Double>): Double = kotlin.math.sqrt(
                left.indices.sumOf { axis ->
                    val delta = left[axis] - right[axis]
                    delta * delta
                },
            )
            val pelvis = point("pelvis") ?: return@mapNotNull null
            val leftAnkle = point("left_ankle") ?: return@mapNotNull null
            val rightAnkle = point("right_ankle") ?: return@mapNotNull null
            val ankleMid = leftAnkle.indices.map { axis -> (leftAnkle[axis] + rightAnkle[axis]) / 2.0 }
            distance(pelvis, ankleMid)
        }
        return if (scales.isEmpty()) 0.0 else scales.average()
    }

    private fun report(
        scenario: SyntheticScenarioBundle,
        result: SynchronizationResult,
        findings: List<ValidationFinding>,
    ) = ScenarioValidationReport(scenario.scenarioId, result.status.name, findings)
}
