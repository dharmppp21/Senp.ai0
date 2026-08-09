package ai.senp.sync.validation

import ai.senp.core.contracts.ActivitySegmentKind
import ai.senp.core.contracts.BodySideHypothesis
import ai.senp.core.contracts.CanonicalObservation
import ai.senp.core.contracts.CanonicalObservationSequence
import ai.senp.core.contracts.ChannelAvailability
import ai.senp.core.contracts.DurationMs
import ai.senp.core.contracts.MirrorHypothesis
import ai.senp.core.contracts.MotionStructureClass
import ai.senp.core.contracts.ObservationChannel
import ai.senp.core.contracts.ObservationSampling
import ai.senp.core.contracts.ObservationValue
import ai.senp.core.contracts.SynchronizationRefusalReason
import ai.senp.core.contracts.SynchronizationRequirements
import ai.senp.core.contracts.SynchronizationRequest
import ai.senp.core.contracts.SynchronizationStatus
import ai.senp.core.contracts.TimestampMs
import ai.senp.core.contracts.TimestampRange
import ai.senp.core.contracts.VideoRole
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.pow
import kotlin.random.Random

object SyntheticScenarioGenerator {
    const val DEFAULT_SEED: Long = 20_260_808L

    private data class Generated(
        val sequence: CanonicalObservationSequence,
        val truth: SyntheticTemporalTruth,
    )

    private data class PoseView(
        val yawDegrees: Double = 0.0,
        val elevationDegrees: Double = 0.0,
        val mirrored: Boolean = false,
        val uniformScale: Double = 1.0,
        val sideAsymmetry: Double = 0.0,
    )

    fun generate(seed: Long = DEFAULT_SEED): SyntheticSuiteManifest {
        val entries = frozenAcceptanceEntries()
        val scenarios = entries.mapIndexed { index, (id, invariants) ->
            buildScenario(id, invariants, seed + index * 7_919L)
        }
        return SyntheticSuiteManifest(
            seed = seed,
            scenarioCount = scenarios.size,
            scenarios = scenarios,
            coverage = scenarios.map { scenario ->
                ScenarioCoverage(
                    scenarioId = scenario.scenarioId,
                    integrationLanes = lanesFor(scenario.expectedInvariants),
                )
            },
        )
    }

    private fun frozenAcceptanceEntries(): List<Pair<String, Set<String>>> {
        val text = checkNotNull(javaClass.getResource("/fixtures/sync-v2-adversarial-acceptance-v1.json")) {
            "Frozen Sync-v2 adversarial acceptance fixture is not on the validation classpath"
        }.readText()
        return Json.parseToJsonElement(text).jsonObject.getValue("scenarios").jsonArray.map { element ->
            val scenario = element.jsonObject
            scenario.getValue("id").jsonPrimitive.content to
                scenario.getValue("expectedInvariants").jsonArray.map { it.jsonPrimitive.content }.toSet()
        }
    }

    private fun lanesFor(invariants: Set<String>): Set<String> = buildSet {
        if (invariants.any { it.contains("SPATIAL") || it.contains("MIRROR") || it.contains("VIEW") || it.contains("FORM") || it.contains("SIDE") }) add("spatial")
        if (invariants.any { it.contains("TEMPORAL") || it.contains("HOLD") || it.contains("CYCLE") || it.contains("ACYCLIC") || it.contains("REST") || it.contains("DISCONTINUITY") || it.contains("OPEN_BOUNDARY") }) add("temporal")
        add("correspondence")
    }

    private fun buildScenario(id: String, invariants: Set<String>, seed: Long): SyntheticScenarioBundle {
        val rng = Random(seed)
        var source = cyclic(VideoRole.SOURCE, 3, jitter = rng.nextDouble(-0.01, 0.01))
        var reference = cyclic(VideoRole.REFERENCE, 3)
        var spatial = SyntheticSpatialTruth()
        var outcome = synchronized()
        val tags = linkedMapOf<String, String>()
        var requirements = SynchronizationRequirements()

        when (id) {
            "same_video_self" -> {
                source = cyclic(VideoRole.SOURCE, 3, noise = 0.0)
                reference = source.withRole(VideoRole.REFERENCE)
                tags["control"] = "identity"
            }
            "different_fps" -> {
                source = cyclic(VideoRole.SOURCE, 3, samplePeriodMs = 100, inputFps = 60.0, analysisFps = 10.0)
                reference = cyclic(VideoRole.REFERENCE, 3, samplePeriodMs = 67, inputFps = 24.0, analysisFps = 15.0)
            }
            "different_resolution" -> {
                tags["source_resolution"] = "1920x1080"; tags["reference_resolution"] = "640x360"
            }
            "different_codec" -> {
                tags["source_codec"] = "h264"; tags["reference_codec"] = "hevc"
            }
            "rotation_metadata" -> {
                tags["source_rotation_degrees"] = "90"; tags["reference_rotation_degrees"] = "0"
            }
            "yaw_elevation_viewpoint" -> {
                source = cyclic(VideoRole.SOURCE, 3, poseView = PoseView(yawDegrees = 68.0, elevationDegrees = 23.0))
                reference = cyclic(VideoRole.REFERENCE, 3)
                spatial = spatial.copy(relativeYawDegrees = 68.0, relativeElevationDegrees = 23.0)
            }
            "mirror" -> {
                source = cyclic(VideoRole.SOURCE, 3, poseView = PoseView(mirrored = true, sideAsymmetry = 1.0))
                reference = cyclic(VideoRole.REFERENCE, 3, poseView = PoseView(sideAsymmetry = 1.0))
                spatial = spatial.copy(mirror = MirrorHypothesis.MIRRORED, selectedSide = BodySideHypothesis.LEFT)
            }
            "side_selection_stability" -> {
                source = cyclic(VideoRole.SOURCE, 3, alternatingSideCoverage = true)
                spatial = spatial.copy(
                    mirror = MirrorHypothesis.AMBIGUOUS,
                    selectedSide = BodySideHypothesis.UNKNOWN,
                    expectedStableSide = false,
                )
            }
            "camera_movement_discontinuity" -> {
                val gap = range(1400, 1900)
                source = cyclic(
                    VideoRole.SOURCE,
                    3,
                    unreliable = listOf(gap),
                    discontinuities = listOf(gap),
                    poseViewAfterMs = gap.endExclusive.value,
                    poseViewAfter = PoseView(yawDegrees = 36.0, elevationDegrees = 8.0, uniformScale = 1.05),
                )
                spatial = spatial.copy(expectedDiscontinuities = listOf(gap))
                outcome = partial()
            }
            "start_mid_motion" -> source = cyclic(VideoRole.SOURCE, 3, openBegin = true)
            "end_mid_motion" -> source = cyclic(VideoRole.SOURCE, 3, openEnd = true)
            "one_reference_ten_source" -> {
                source = cyclic(VideoRole.SOURCE, 10); reference = cyclic(VideoRole.REFERENCE, 1)
                outcome = synchronized(referenceReuse = true)
            }
            "ten_reference_one_source" -> {
                source = cyclic(VideoRole.SOURCE, 1); reference = cyclic(VideoRole.REFERENCE, 10)
                outcome = partial(referenceUnmatched = (1 until 10).map { "reference-u$it" }.toSet())
            }
            "two_reference_seven_source" -> {
                source = cyclic(VideoRole.SOURCE, 7); reference = cyclic(VideoRole.REFERENCE, 2)
                outcome = synchronized(referenceReuse = true)
            }
            "multiple_sets_rests" -> {
                source = multipleSets(VideoRole.SOURCE); reference = multipleSets(VideoRole.REFERENCE)
            }
            "extra_source_action" -> {
                source = cyclicWithExtraAction(VideoRole.SOURCE)
                reference = cyclic(VideoRole.REFERENCE, 2)
                outcome = partial(sourceUnmatched = setOf("source-extra"))
            }
            "missing_source_action" -> {
                source = cyclic(VideoRole.SOURCE, 2); reference = cyclic(VideoRole.REFERENCE, 3)
                outcome = partial(referenceUnmatched = setOf("reference-u2"))
            }
            "repeated_identical_phase" -> {
                reference = cyclic(VideoRole.REFERENCE, 6)
                outcome = partial(ambiguity = true)
            }
            "variable_speed" -> source = cyclic(VideoRole.SOURCE, 3, phaseExponent = 1.65)
            "pause_hold" -> {
                val hold = range(1300, 1700)
                source = cyclic(VideoRole.SOURCE, 3, holds = listOf(hold))
                reference = cyclic(VideoRole.REFERENCE, 3, holds = listOf(range(1350, 1550)))
            }
            "very_slow" -> source = cyclic(VideoRole.SOURCE, 3, unitMs = 2200)
            "very_fast" -> source = cyclic(VideoRole.SOURCE, 3, unitMs = 450, samplePeriodMs = 50, analysisFps = 20.0)
            "static_isometric" -> {
                source = isometric(VideoRole.SOURCE); reference = isometric(VideoRole.REFERENCE)
            }
            "no_common_motion" -> {
                source = cyclic(VideoRole.SOURCE, 2, form = 0.8)
                reference = isometric(VideoRole.REFERENCE, form = -0.7)
                outcome = refused(SynchronizationRefusalReason.NO_COMMON_MOTION)
            }
            "poor_pose_coverage" -> {
                val gap = range(300, 2700)
                source = cyclic(VideoRole.SOURCE, 3, unreliable = listOf(gap))
                outcome = refusedOrPartial(SynchronizationRefusalReason.INSUFFICIENT_OBSERVATIONS)
            }
            "short_occlusion" -> {
                val gap = range(1400, 1700)
                source = cyclic(VideoRole.SOURCE, 3, unreliable = listOf(gap))
                outcome = partial(sourceUnmatchedRanges = listOf(gap))
            }
            "long_occlusion" -> {
                val gap = range(800, 2300)
                source = cyclic(VideoRole.SOURCE, 3, unreliable = listOf(gap), discontinuities = listOf(gap))
                spatial = spatial.copy(expectedDiscontinuities = listOf(gap))
                outcome = refusedOrPartial(SynchronizationRefusalReason.INSUFFICIENT_OBSERVATIONS)
            }
            "person_leaves_reenters" -> {
                val gap = range(900, 1900)
                source = cyclic(VideoRole.SOURCE, 3, unreliable = listOf(gap), discontinuities = listOf(gap), reenterAfterMs = 1900)
                spatial = spatial.copy(expectedDiscontinuities = listOf(gap))
                outcome = partial()
            }
            "multiple_people_subject_ambiguity" -> {
                source = cyclic(VideoRole.SOURCE, 3, multipleSubjects = true)
                spatial = spatial.copy(subjectAmbiguous = true)
                outcome = refusedOrPartial(SynchronizationRefusalReason.SUBJECT_AMBIGUITY, ambiguity = true)
            }
            "different_body_proportions" -> {
                source = cyclic(VideoRole.SOURCE, 3, form = 0.22, poseView = PoseView(uniformScale = 1.32))
                reference = cyclic(VideoRole.REFERENCE, 3, form = 0.0)
                spatial = spatial.copy(nuisanceUniformScale = 1.32, expectedTrueFormDelta = 0.22)
            }
            "true_form_difference" -> {
                source = cyclic(VideoRole.SOURCE, 3, form = 0.34)
                reference = cyclic(VideoRole.REFERENCE, 3, form = 0.0)
                spatial = spatial.copy(expectedTrueFormDelta = 0.34)
            }
            "reversed_video" -> {
                source = cyclic(VideoRole.SOURCE, 3, reversed = true)
                outcome = refusedOrPartial(SynchronizationRefusalReason.UNRELIABLE_CORRESPONDENCE, ambiguity = true)
            }
            "edited_spliced_video" -> {
                val cut = range(1400, 1800)
                source = cyclic(
                    VideoRole.SOURCE,
                    3,
                    unreliable = listOf(cut),
                    discontinuities = listOf(cut),
                    phaseJumpAfterMs = cut.endExclusive.value,
                    poseViewAfterMs = cut.endExclusive.value,
                    poseViewAfter = PoseView(yawDegrees = 58.0, elevationDegrees = 9.0, uniformScale = 1.04),
                )
                spatial = spatial.copy(expectedDiscontinuities = listOf(cut))
                outcome = partial()
            }
            "slow_motion_edit" -> source = cyclic(VideoRole.SOURCE, 3, unitMs = 2000, inputFps = 30.0, analysisFps = 10.0)
            "non_cyclic_activity" -> {
                source = acyclic(VideoRole.SOURCE, listOf("step-a", "step-c"))
                reference = acyclic(VideoRole.REFERENCE, listOf("step-a", "step-b", "step-c"))
                outcome = partial(referenceUnmatched = setOf("reference-step-b"))
            }
            "object_required_pose_only" -> {
                requirements = SynchronizationRequirements(setOf("human_pose", "object_pose"))
                outcome = refused(SynchronizationRefusalReason.REQUIRED_CHANNEL_MISSING)
            }
            else -> error("Frozen scenario has no deterministic validation generator: $id")
        }

        if ("AMBIGUITY_NOT_SILENT" in invariants) {
            outcome = outcome.copy(ambiguityMustBeExplicit = true)
        }

        return SyntheticScenarioBundle(
            scenarioId = id,
            seed = seed,
            expectedInvariants = invariants,
            tags = tags,
            request = SynchronizationRequest(source = source.sequence, reference = reference.sequence, requirements = requirements),
            sourceTruth = source.truth,
            referenceTruth = reference.truth,
            spatialTruth = spatial,
            expectedOutcome = outcome,
        )
    }

    private fun synchronized(referenceReuse: Boolean = false) = ExpectedSynchronizationOutcome(
        allowedStatuses = setOf(SynchronizationStatus.SYNCHRONIZED),
        referenceReuseRequired = referenceReuse,
    )

    private fun partial(
        sourceUnmatched: Set<String> = emptySet(),
        referenceUnmatched: Set<String> = emptySet(),
        sourceUnmatchedRanges: List<TimestampRange> = emptyList(),
        ambiguity: Boolean = false,
    ) = ExpectedSynchronizationOutcome(
        allowedStatuses = setOf(SynchronizationStatus.PARTIAL),
        expectedSourceUnmatchedUnitIds = sourceUnmatched,
        expectedReferenceUnmatchedUnitIds = referenceUnmatched,
        expectedSourceUnmatchedRanges = sourceUnmatchedRanges,
        ambiguityMustBeExplicit = ambiguity,
    )

    private fun refused(reason: SynchronizationRefusalReason) = ExpectedSynchronizationOutcome(
        allowedStatuses = setOf(SynchronizationStatus.REFUSED),
        allowedRefusalReasons = setOf(reason),
    )

    private fun refusedOrPartial(reason: SynchronizationRefusalReason, ambiguity: Boolean = false) = ExpectedSynchronizationOutcome(
        allowedStatuses = setOf(SynchronizationStatus.REFUSED, SynchronizationStatus.PARTIAL),
        allowedRefusalReasons = setOf(reason),
        ambiguityMustBeExplicit = ambiguity,
    )

    private fun cyclic(
        role: VideoRole,
        unitCount: Int,
        unitMs: Long = 1000,
        samplePeriodMs: Long = 100,
        inputFps: Double = 30.0,
        analysisFps: Double = 10.0,
        form: Double = 0.0,
        noise: Double = 0.003,
        jitter: Double = 0.0,
        reversed: Boolean = false,
        phaseExponent: Double = 1.0,
        openBegin: Boolean = false,
        openEnd: Boolean = false,
        unreliable: List<TimestampRange> = emptyList(),
        discontinuities: List<TimestampRange> = emptyList(),
        holds: List<TimestampRange> = emptyList(),
        multipleSubjects: Boolean = false,
        reenterAfterMs: Long? = null,
        phaseJumpAfterMs: Long? = null,
        poseView: PoseView = PoseView(),
        poseViewAfterMs: Long? = null,
        poseViewAfter: PoseView? = null,
        formByUnit: Map<Int, Double> = emptyMap(),
        alternatingSideCoverage: Boolean = false,
    ): Generated {
        require(unitCount > 0)
        val openBeginOffsetMs = if (openBegin) (unitMs * 0.23).toLong() else 0L
        val openEndTrimMs = if (openEnd) (unitMs * 0.23).toLong() else 0L
        val holdDurationMs = holds.sumOf { it.duration().value }
        val duration = unitCount * unitMs - openBeginOffsetMs - openEndTrimMs + holdDurationMs
        val samples = mutableListOf<SyntheticTimestampTruth>()
        val observations = mutableListOf<CanonicalObservation>()
        var t = 0L
        while (t < duration) {
            val elapsedHoldMs = holds.sumOf { hold ->
                when {
                    t <= hold.start.value -> 0L
                    t >= hold.endExclusive.value -> hold.duration().value
                    else -> t - hold.start.value
                }
            }
            val virtualTime = t - elapsedHoldMs + openBeginOffsetMs
            val unitIndex = (virtualTime / unitMs).toInt().coerceIn(0, unitCount - 1)
            val unitStart = unitIndex.toLong() * unitMs
            var phase = ((virtualTime - unitStart).toDouble() / unitMs).coerceIn(0.0, 0.999999)
            phase = phase.powSafe(phaseExponent)
            val activeHold = holds.firstOrNull { it.contains(TimestampMs(t)) }
            if (phaseJumpAfterMs != null && t >= phaseJumpAfterMs) phase = (phase + 0.38) % 1.0
            if (reversed) phase = 1.0 - phase
            val inHold = activeHold != null
            val reliable = unreliable.none { it.contains(TimestampMs(t)) }
            val direction = when {
                !reliable || inHold -> 0
                sin(2.0 * PI * phase) > 0.08 -> if (reversed) -1 else 1
                sin(2.0 * PI * phase) < -0.08 -> if (reversed) 1 else -1
                else -> 0
            }
            val state = when {
                !reliable -> "UNRELIABLE"
                inHold -> "HOLD"
                direction > 0 -> "POSITIVE"
                direction < 0 -> "NEGATIVE"
                else -> "TURN"
            }
            val subject = if (reenterAfterMs != null && t >= reenterAfterMs) "subject-a-reentry" else "subject-a"
            val frameForm = formByUnit[unitIndex] ?: form
            val truth = SyntheticTimestampTruth(t, state, direction, phase, reliable, frameForm, subjectId = subject)
            samples += truth
            val activeView = if (poseViewAfterMs != null && t >= poseViewAfterMs) poseViewAfter ?: poseView else poseView
            observations += CanonicalObservation(
                TimestampMs(t),
                channelsFor(truth, noise + jitter, multipleSubjects, activeView, alternatingSideCoverage),
            )
            t += samplePeriodMs
        }
        val units = (0 until unitCount).mapNotNull { index ->
            val observedStart = maxOf(0L, index * unitMs - openBeginOffsetMs)
            val observedEnd = minOf(duration, (index + 1L) * unitMs - openBeginOffsetMs)
            if (observedEnd <= observedStart) return@mapNotNull null
            SyntheticUnitTruth(
                unitId = "${role.name.lowercase()}-u$index",
                range = range(observedStart, observedEnd),
                structureClass = MotionStructureClass.CYCLIC,
                openBegin = openBegin && index == 0,
                openEnd = openEnd && index == unitCount - 1,
            )
        }
        val activity = buildList {
            add(ActivitySegmentKind.ACTIVE)
            if (holds.isNotEmpty()) add(ActivitySegmentKind.HOLD)
            if (unreliable.isNotEmpty()) add(ActivitySegmentKind.UNRELIABLE)
            if (discontinuities.isNotEmpty()) add(ActivitySegmentKind.DISCONTINUITY)
        }
        return Generated(
            sequence = CanonicalObservationSequence(role, DurationMs(duration), ObservationSampling(inputFps, analysisFps), observations),
            truth = SyntheticTemporalTruth(MotionStructureClass.CYCLIC, samples, units, activity),
        )
    }

    private fun isometric(role: VideoRole, form: Double = 0.0): Generated {
        val duration = 3000L
        val samples = (0L until duration step 100).map { t ->
            SyntheticTimestampTruth(t, "HOLD", 0, 0.5, true, form)
        }
        return Generated(
            CanonicalObservationSequence(
                role,
                DurationMs(duration),
                ObservationSampling(30.0, 10.0),
                samples.map { CanonicalObservation(TimestampMs(it.timestampMs), channelsFor(it, 0.0, false)) },
            ),
            SyntheticTemporalTruth(
                MotionStructureClass.ISOMETRIC,
                samples,
                listOf(SyntheticUnitTruth("${role.name.lowercase()}-u0", range(0, duration), MotionStructureClass.ISOMETRIC)),
                listOf(ActivitySegmentKind.HOLD),
            ),
        )
    }

    private fun multipleSets(role: VideoRole): Generated {
        val unitRanges = listOf(range(0, 1000), range(1000, 2000), range(3000, 4000), range(4000, 5000))
        val duration = 5000L
        val samples = (0L until duration step 100).map { t ->
            val rest = t in 2000 until 3000
            val phase = if (rest) 0.0 else ((t % 1000).toDouble() / 1000.0)
            val direction = if (rest) 0 else if (phase < 0.5) 1 else -1
            SyntheticTimestampTruth(t, if (rest) "REST" else if (direction > 0) "POSITIVE" else "NEGATIVE", direction, phase, true, 0.0, "subject-a")
        }
        return Generated(
            CanonicalObservationSequence(role, DurationMs(duration), ObservationSampling(30.0, 10.0), samples.map { CanonicalObservation(TimestampMs(it.timestampMs), channelsFor(it, 0.0, false)) }),
            SyntheticTemporalTruth(
                MotionStructureClass.CYCLIC,
                samples,
                unitRanges.mapIndexed { index, r -> SyntheticUnitTruth("${role.name.lowercase()}-u$index", r, MotionStructureClass.CYCLIC) },
                listOf(ActivitySegmentKind.ACTIVE, ActivitySegmentKind.REST),
            ),
        )
    }

    private fun cyclicWithExtraAction(role: VideoRole): Generated {
        val duration = 3800L
        val samples = mutableListOf<SyntheticTimestampTruth>()
        val observations = mutableListOf<CanonicalObservation>()
        (0L until duration step 100L).forEach { t ->
            val truth = when {
                t < 2000L -> {
                    val phase = (t % 1000L).toDouble() / 1000.0
                    val sine = sin(2.0 * PI * phase)
                    val direction = when {
                        sine > 0.08 -> 1
                        sine < -0.08 -> -1
                        else -> 0
                    }
                    SyntheticTimestampTruth(
                        timestampMs = t,
                        state = when {
                            direction > 0 -> "POSITIVE"
                            direction < 0 -> "NEGATIVE"
                            else -> "TURN"
                        },
                        direction = direction,
                        phaseProgress = phase,
                        reliable = true,
                        formSignature = 0.0,
                    )
                }
                t < 2800L -> SyntheticTimestampTruth(t, "REST", 0, 0.0, true, 0.0)
                else -> {
                    val phase = ((t - 2800L).toDouble() / 1000.0).coerceIn(0.0, 0.999999)
                    SyntheticTimestampTruth(t, "STEP-EXTRA", 1, phase, true, 0.85)
                }
            }
            samples += truth
            observations += CanonicalObservation(TimestampMs(t), channelsFor(truth, 0.0, false))
        }
        return Generated(
            CanonicalObservationSequence(
                role,
                DurationMs(duration),
                ObservationSampling(30.0, 10.0),
                observations,
            ),
            SyntheticTemporalTruth(
                MotionStructureClass.MIXED,
                samples,
                listOf(
                    SyntheticUnitTruth("${role.name.lowercase()}-u0", range(0, 1000), MotionStructureClass.CYCLIC),
                    SyntheticUnitTruth("${role.name.lowercase()}-u1", range(1000, 2000), MotionStructureClass.CYCLIC),
                    SyntheticUnitTruth("${role.name.lowercase()}-extra", range(2800, 3800), MotionStructureClass.ACYCLIC),
                ),
                listOf(ActivitySegmentKind.ACTIVE, ActivitySegmentKind.REST),
            ),
        )
    }

    private fun acyclic(role: VideoRole, steps: List<String>): Generated {
        val duration = steps.size * 1000L
        val samples = mutableListOf<SyntheticTimestampTruth>()
        val observations = mutableListOf<CanonicalObservation>()
        steps.forEachIndexed { index, step ->
            (0L until 1000L step 100L).forEach { offset ->
                val t = index * 1000L + offset
                val phase = offset / 1000.0
                val stableStepSignature = ((step.hashCode().toLong() and 0x7fffffffL) % 17L).toDouble() / 20.0
                val truth = SyntheticTimestampTruth(t, step.uppercase(), 1, phase, true, stableStepSignature)
                samples += truth
                observations += CanonicalObservation(TimestampMs(t), channelsFor(truth, 0.0, false))
            }
        }
        return Generated(
            CanonicalObservationSequence(role, DurationMs(duration), ObservationSampling(30.0, 10.0), observations),
            SyntheticTemporalTruth(
                MotionStructureClass.ACYCLIC,
                samples,
                steps.mapIndexed { index, step -> SyntheticUnitTruth("${role.name.lowercase()}-$step", range(index * 1000L, (index + 1) * 1000L), MotionStructureClass.ACYCLIC) },
                listOf(ActivitySegmentKind.ACTIVE),
            ),
        )
    }

    private fun channelsFor(
        truth: SyntheticTimestampTruth,
        noise: Double,
        multipleSubjects: Boolean,
        poseView: PoseView = PoseView(),
        alternatingSideCoverage: Boolean = false,
    ): List<ObservationChannel> {
        val reliable = truth.reliable
        val confidence = if (reliable) 0.98 else 0.0
        val motionTruth = ObservationChannel(
            channelId = "synthetic-motion-truth",
            schemaVersion = 1,
            semanticType = "synthetic_motion_truth",
            coordinateSpace = "synthetic_body",
            subjectId = truth.subjectId ?: "subject-a",
            componentAxes = listOf("position", "direction", "form", "phase"),
            values = listOf(
                ObservationValue(
                    "body_state",
                    if (reliable) listOf(
                        sin(2.0 * PI * truth.phaseProgress) + noise,
                        truth.direction.toDouble(),
                        truth.formSignature,
                        truth.phaseProgress,
                    ) else listOf(null, null, null, null),
                    List(4) { reliable },
                    confidence,
                ),
            ),
            availability = if (reliable) ChannelAvailability.OBSERVED else ChannelAvailability.MISSING,
            confidence = confidence,
        )

        fun humanPose(channelId: String, subjectId: String, subjectOffsetX: Double): ObservationChannel {
            val acyclicStep = truth.state.startsWith("STEP-")
            val movement = if (acyclicStep) {
                2.0 * truth.phaseProgress - 1.0 + 1.6 * truth.formSignature
            } else {
                -cos(2.0 * PI * truth.phaseProgress)
            }
            val secondaryMovement = if (acyclicStep) {
                truth.phaseProgress * truth.phaseProgress + 0.9 * truth.formSignature
            } else {
                sin(2.0 * PI * truth.phaseProgress)
            }
            val shoulderHalfWidth = 0.42 * (1.0 + truth.formSignature)
            val ankleLength = 1.0
            val base = linkedMapOf(
                "pelvis" to doubleArrayOf(0.0, 0.0, 0.0),
                "left_hip" to doubleArrayOf(-0.20, -0.05, 0.0),
                "right_hip" to doubleArrayOf(0.20, -0.05, 0.0),
                "left_shoulder" to doubleArrayOf(-shoulderHalfWidth, 0.95, 0.0),
                "right_shoulder" to doubleArrayOf(shoulderHalfWidth, 0.95, 0.0),
                "left_elbow" to doubleArrayOf(
                    -0.56 - 0.01 * poseView.sideAsymmetry,
                    0.69 + (0.20 + 0.01 * poseView.sideAsymmetry) * secondaryMovement,
                    (0.10 + 0.01 * poseView.sideAsymmetry) * movement,
                ),
                "right_elbow" to doubleArrayOf(
                    0.56 - 0.02 * poseView.sideAsymmetry,
                    0.69 - 0.02 * poseView.sideAsymmetry + (0.20 - 0.04 * poseView.sideAsymmetry) * secondaryMovement,
                    (0.10 - 0.02 * poseView.sideAsymmetry) * movement,
                ),
                "left_wrist" to doubleArrayOf(
                    -0.66 - 0.03 * poseView.sideAsymmetry,
                    0.42 - 0.01 * poseView.sideAsymmetry + (0.32 + 0.02 * poseView.sideAsymmetry) * movement,
                    (0.16 + 0.01 * poseView.sideAsymmetry) * movement + 0.045 * poseView.sideAsymmetry,
                ),
                "right_wrist" to doubleArrayOf(
                    0.66 - 0.03 * poseView.sideAsymmetry,
                    0.42 + 0.03 * poseView.sideAsymmetry + (0.32 - 0.04 * poseView.sideAsymmetry) * movement,
                    (0.16 - 0.03 * poseView.sideAsymmetry) * movement - 0.005 * poseView.sideAsymmetry,
                ),
                "left_knee" to doubleArrayOf(-0.20, -0.55 + (0.08 + 0.01 * poseView.sideAsymmetry) * secondaryMovement, (0.05 + 0.005 * poseView.sideAsymmetry) * movement),
                "right_knee" to doubleArrayOf(0.20, -0.55 - 0.01 * poseView.sideAsymmetry + (0.08 - 0.02 * poseView.sideAsymmetry) * secondaryMovement, (0.05 - 0.01 * poseView.sideAsymmetry) * movement),
                "left_ankle" to doubleArrayOf(-0.20, -ankleLength, 0.0),
                "right_ankle" to doubleArrayOf(0.20, -ankleLength, 0.0),
                "left_foot" to doubleArrayOf(-0.20, -ankleLength - 0.16, 0.18),
                "right_foot" to doubleArrayOf(0.20, -ankleLength - 0.16, 0.18),
            )
            return ObservationChannel(
                channelId = channelId,
                schemaVersion = 1,
                semanticType = "human_pose",
                coordinateSpace = "camera_3d",
                subjectId = subjectId,
                componentAxes = listOf("x", "y", "z"),
                values = base.map { (key, point) ->
                    val dynamicSideLandmark = key.endsWith("elbow") || key.endsWith("wrist") ||
                        key.endsWith("knee") || key.endsWith("ankle")
                    val alternatingBucket = (truth.timestampMs / 300L) % 2L
                    val hiddenByAlternation = alternatingSideCoverage && dynamicSideLandmark && when {
                        key.startsWith("left_") -> alternatingBucket == 1L
                        key.startsWith("right_") -> alternatingBucket == 0L
                        else -> false
                    }
                    val observed = reliable && !hiddenByAlternation
                    ObservationValue(
                        key,
                        if (observed) transformPoint(point, poseView, subjectOffsetX).map { it + noise } else listOf(null, null, null),
                        List(3) { observed },
                        if (observed) confidence else 0.0,
                    )
                },
                availability = when {
                    !reliable -> ChannelAvailability.MISSING
                    alternatingSideCoverage -> ChannelAvailability.PARTIAL
                    else -> ChannelAvailability.OBSERVED
                },
                confidence = confidence,
            )
        }

        return buildList {
            add(motionTruth)
            add(humanPose("human-primary", truth.subjectId ?: "subject-a", 0.0))
            if (multipleSubjects) add(humanPose("human-secondary", "subject-b", 1.35))
        }
    }

    private fun transformPoint(point: DoubleArray, view: PoseView, offsetX: Double): List<Double> {
        val x = if (view.mirrored) -point[0] else point[0]
        val yaw = view.yawDegrees * PI / 180.0
        val elevation = view.elevationDegrees * PI / 180.0
        val yawX = cos(yaw) * x + sin(yaw) * point[2]
        val yawZ = -sin(yaw) * x + cos(yaw) * point[2]
        val y = cos(elevation) * point[1] - sin(elevation) * yawZ
        val z = sin(elevation) * point[1] + cos(elevation) * yawZ
        return listOf(
            yawX * view.uniformScale + offsetX,
            y * view.uniformScale,
            z * view.uniformScale,
        )
    }

    private fun Generated.withRole(role: VideoRole): Generated = Generated(
        sequence.copy(role = role),
        truth.copy(units = truth.units.map { it.copy(unitId = it.unitId.replaceBefore("-", role.name.lowercase())) }),
    )

    private fun Double.powSafe(exponent: Double): Double = this.pow(exponent)

    private fun range(start: Number, end: Number): TimestampRange = TimestampRange(TimestampMs(start.toLong()), TimestampMs(end.toLong()))
}
