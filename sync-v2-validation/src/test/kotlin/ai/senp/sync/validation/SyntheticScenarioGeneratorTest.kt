package ai.senp.sync.validation

import ai.senp.core.contracts.ActivitySegmentKind
import ai.senp.core.contracts.MotionStructureClass
import ai.senp.core.contracts.SynchronizationRefusalReason
import ai.senp.core.contracts.SynchronizationStatus
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

class SyntheticScenarioGeneratorTest {
    private val suite = SyntheticScenarioGenerator.generate()

    @Test
    fun frozenMatrixIsCoveredExactlyAndDeterministically() {
        val expected = setOf(
            "same_video_self", "different_fps", "different_resolution", "different_codec", "rotation_metadata",
            "yaw_elevation_viewpoint", "mirror", "side_selection_stability", "camera_movement_discontinuity",
            "start_mid_motion", "end_mid_motion", "one_reference_ten_source", "ten_reference_one_source",
            "two_reference_seven_source", "multiple_sets_rests", "extra_source_action", "missing_source_action",
            "repeated_identical_phase", "variable_speed", "pause_hold", "very_slow", "very_fast", "static_isometric",
            "no_common_motion", "poor_pose_coverage", "short_occlusion", "long_occlusion", "person_leaves_reenters",
            "multiple_people_subject_ambiguity", "different_body_proportions", "true_form_difference", "reversed_video",
            "edited_spliced_video", "slow_motion_edit", "non_cyclic_activity", "object_required_pose_only",
        )
        assertEquals(36, suite.scenarioCount)
        assertEquals(expected, suite.scenarios.map { it.scenarioId }.toSet())
        assertTrue(suite.coverage.all { it.fixtureGeneration == CoverageState.EXECUTABLE })
        assertTrue(suite.coverage.all { it.invariantValidation == CoverageState.EXECUTABLE })
        assertTrue(suite.coverage.all { it.productionIntegration == CoverageState.EXECUTABLE })
        val json = Json { encodeDefaults = true; classDiscriminator = "type" }
        assertEquals(json.encodeToString(suite), json.encodeToString(SyntheticScenarioGenerator.generate()))
        assertNotEquals(json.encodeToString(suite), json.encodeToString(SyntheticScenarioGenerator.generate(42L)))
    }

    @Test
    fun everyGeneratedObservationSequenceIsTimestampFirstAndMaskedTruthfully() {
        suite.scenarios.forEach { scenario ->
            listOf(scenario.request.source, scenario.request.reference).forEach { sequence ->
                assertTrue(sequence.observations.zipWithNext().all { (left, right) -> left.timestamp < right.timestamp })
                sequence.observations.forEach { observation ->
                    observation.channels.forEach { channel ->
                        channel.values.forEach { value ->
                            value.values.zip(value.mask).forEach { (component, available) ->
                                assertEquals(component != null, available)
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun cadenceMetadataDoesNotBecomeTimestampTruth() {
        val scenario = scenario("different_fps")
        assertNotEquals(scenario.request.source.sampling.inputNominalFramesPerSecond, scenario.request.source.sampling.analysisFramesPerSecond)
        assertNotEquals(scenario.request.source.sampling.analysisFramesPerSecond, scenario.request.reference.sampling.analysisFramesPerSecond)
        assertNotEquals(
            scenario.request.source.observations.map { it.timestamp },
            scenario.request.reference.observations.map { it.timestamp },
        )
    }

    @Test
    fun adversarialTemporalOraclesEncodeRequiredSemantics() {
        assertEquals(10, scenario("one_reference_ten_source").sourceTruth.units.size)
        assertEquals(1, scenario("one_reference_ten_source").referenceTruth.units.size)
        assertTrue(scenario("one_reference_ten_source").expectedOutcome.referenceReuseRequired)
        assertEquals(9, scenario("ten_reference_one_source").expectedOutcome.expectedReferenceUnmatchedUnitIds.size)
        assertEquals(1, scenario("extra_source_action").expectedOutcome.expectedSourceUnmatchedUnitIds.size)
        assertTrue(scenario("start_mid_motion").sourceTruth.units.first().openBegin)
        assertTrue(scenario("end_mid_motion").sourceTruth.units.last().openEnd)
        assertTrue(ActivitySegmentKind.REST in scenario("multiple_sets_rests").sourceTruth.activityKinds)
        assertTrue(ActivitySegmentKind.HOLD in scenario("pause_hold").sourceTruth.activityKinds)
        assertEquals(MotionStructureClass.ISOMETRIC, scenario("static_isometric").sourceTruth.classification)
        assertEquals(MotionStructureClass.ACYCLIC, scenario("non_cyclic_activity").sourceTruth.classification)
    }

    @Test
    fun truthfulnessOraclesNeverTurnMissingEvidenceIntoSuccess() {
        assertEquals(setOf(SynchronizationStatus.REFUSED), scenario("no_common_motion").expectedOutcome.allowedStatuses)
        assertTrue(SynchronizationRefusalReason.NO_COMMON_MOTION in scenario("no_common_motion").expectedOutcome.allowedRefusalReasons)
        assertTrue(SynchronizationStatus.REFUSED in scenario("poor_pose_coverage").expectedOutcome.allowedStatuses)
        assertTrue(SynchronizationStatus.PARTIAL in scenario("poor_pose_coverage").expectedOutcome.allowedStatuses)
        assertTrue(SynchronizationRefusalReason.SUBJECT_AMBIGUITY in scenario("multiple_people_subject_ambiguity").expectedOutcome.allowedRefusalReasons)
        val objectRequired = scenario("object_required_pose_only")
        assertEquals(setOf("human_pose", "object_pose"), objectRequired.request.requirements.requiredChannelSemanticTypes)
        assertTrue(objectRequired.request.source.observations.flatMap { it.channels }.none { it.semanticType == "object_pose" })
    }

    @Test
    fun spatialOraclesKeepNuisanceAndTrueFormSeparate() {
        val viewpointScenario = scenario("yaw_elevation_viewpoint")
        val viewpoint = viewpointScenario.spatialTruth
        assertTrue(viewpoint.relativeYawDegrees != 0.0)
        assertTrue(viewpoint.relativeElevationDegrees != 0.0)
        val sourcePose = poseValues(viewpointScenario.request.source)
        val referencePose = poseValues(viewpointScenario.request.reference)
        assertNotEquals(sourcePose.getValue("left_shoulder"), referencePose.getValue("left_shoulder"))
        assertEquals(
            distance(sourcePose.getValue("left_shoulder"), sourcePose.getValue("right_shoulder")),
            distance(referencePose.getValue("left_shoulder"), referencePose.getValue("right_shoulder")),
            absoluteTolerance = 1e-9,
        )

        val mirrorScenario = scenario("mirror")
        val mirrored = poseValues(mirrorScenario.request.source).getValue("left_shoulder")[0]
        val ordinary = poseValues(mirrorScenario.request.reference).getValue("left_shoulder")[0]
        assertTrue(mirrored > 0.0 && ordinary < 0.0)

        val proportions = scenario("different_body_proportions").spatialTruth
        assertTrue(proportions.nuisanceUniformScale != 1.0)
        assertTrue(proportions.expectedTrueFormDelta > 0.0)
        assertTrue(scenario("true_form_difference").spatialTruth.expectedTrueFormDelta > proportions.expectedTrueFormDelta)
        assertTrue(scenario("camera_movement_discontinuity").spatialTruth.expectedDiscontinuities.isNotEmpty())
    }

    @Test
    fun isometricAndPauseHoldPoseGeometryActuallyStopsMoving() {
        val isometric = scenario("static_isometric").request.source.observations.map { observation ->
            observation.channels.single { it.channelId == "human-primary" }.values.map { it.values }
        }
        assertTrue(isometric.drop(1).all { it == isometric.first() })

        val pause = scenario("pause_hold").request.source.observations.filter { it.timestamp.value in 1300L until 1700L }.map { observation ->
            observation.channels.single { it.channelId == "human-primary" }.values.map { it.values }
        }
        assertTrue(pause.isNotEmpty())
        assertTrue(pause.drop(1).all { it == pause.first() })
    }

    @Test
    fun humanPoseChannelUsesConcreteThreeDimensionalLandmarks() {
        suite.scenarios.forEach { scenario ->
            listOf(scenario.request.source, scenario.request.reference).forEach { sequence ->
                val observed = sequence.observations.first().channels.single { it.channelId == "human-primary" }
                assertEquals("human_pose", observed.semanticType)
                assertEquals(listOf("x", "y", "z"), observed.componentAxes)
                assertTrue(setOf("pelvis", "left_shoulder", "right_shoulder", "left_ankle", "right_ankle").all { key ->
                    observed.values.any { it.key == key }
                })
            }
        }
    }

    private fun poseValues(sequence: ai.senp.core.contracts.CanonicalObservationSequence): Map<String, List<Double>> =
        sequence.observations.first().channels.single { it.channelId == "human-primary" }.values.associate { value ->
            value.key to value.values.map { requireNotNull(it) }
        }

    private fun distance(left: List<Double>, right: List<Double>): Double = kotlin.math.sqrt(
        left.indices.sumOf { axis ->
            val delta = left[axis] - right[axis]
            delta * delta
        },
    )

    private fun scenario(id: String): SyntheticScenarioBundle = suite.scenarios.single { it.scenarioId == id }
}
