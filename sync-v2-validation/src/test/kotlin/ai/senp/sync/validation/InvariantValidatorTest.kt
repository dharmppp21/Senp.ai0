package ai.senp.sync.validation

import ai.senp.core.contracts.SynchronizationStatus
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class InvariantValidatorTest {
    private val suite = SyntheticScenarioGenerator.generate()

    @Test
    fun allFrozenScenariosHaveExecutableInvariantValidation() {
        val frozenVocabulary = suite.scenarios.flatMap { it.expectedInvariants }.toSet()
        assertEquals(frozenVocabulary, InvariantValidator.supportedFrozenInvariants)
        assertEquals(36, suite.scenarios.size)
        suite.scenarios.forEach { scenario ->
            val report = InvariantValidator.validate(scenario, ValidationTestFixtures.resultFor(scenario))
            assertTrue(report.passed, scenario.scenarioId + ": " + report.findings.filterNot { it.passed })
        }
    }

    @Test
    fun representativeGoodCandidatesSatisfyInvariantChecks() {
        val ids = listOf(
            "same_video_self",
            "one_reference_ten_source",
            "ten_reference_one_source",
            "extra_source_action",
            "missing_source_action",
            "repeated_identical_phase",
            "short_occlusion",
            "multiple_sets_rests",
            "static_isometric",
            "non_cyclic_activity",
            "camera_movement_discontinuity",
            "mirror",
            "yaw_elevation_viewpoint",
        )
        ids.forEach { id ->
            val scenario = scenario(id)
            val report = InvariantValidator.validate(scenario, ValidationTestFixtures.resultFor(scenario))
            assertTrue(report.passed, id + ": " + report.findings.filterNot { it.passed })
        }
    }

    @Test
    fun reliableOppositeDirectionPairingIsRejected() {
        val scenario = scenario("biceps-like-reliable")
        val report = InvariantValidator.validate(scenario, ValidationTestFixtures.resultFor(scenario, oppositeFirst = true))
        assertFalse(report.passed)
        assertTrue(report.findings.any { it.invariant == "ZERO_INTERIOR_OPPOSITE_DIRECTION_PAIRINGS" && !it.passed })
    }

    @Test
    fun truthfulRefusalsAreAcceptedForEvidenceLimitedCases() {
        listOf(
            "no_common_motion",
            "poor_pose_coverage",
            "long_occlusion",
            "multiple_people_subject_ambiguity",
            "reversed_video",
            "object_required_pose_only",
        ).forEach { id ->
            val scenario = suite.scenarios.single { it.scenarioId == id }
            val result = ValidationTestFixtures.resultFor(scenario)
            assertTrue(result.status == SynchronizationStatus.REFUSED)
            val report = InvariantValidator.validate(scenario, result)
            assertTrue(report.passed, id + ": " + report.findings.filterNot { it.passed })
        }
    }

    @Test
    fun spatialInspectionSeamPreservesTrueFormDifference() {
        listOf("different_body_proportions", "true_form_difference").forEach { id ->
            val scenario = suite.scenarios.single { it.scenarioId == id }
            val report = InvariantValidator.validateSpatialOutput(
                scenario,
                ValidationTestFixtures.spatialOutput(scenario),
            )
            assertTrue(report.passed, id + ": " + report.findings)
        }
    }

    @Test
    fun spatialScaleNuisanceMustBeRemovedBeforeFormValidationPasses() {
        val scenario = suite.scenarios.single { it.scenarioId == "different_body_proportions" }
        val raw = SpatialHarnessOutput(
            canonicalSource = scenario.request.source,
            canonicalReference = scenario.request.reference,
            diagnostics = ValidationTestFixtures.spatial(scenario),
        )
        val rawReport = InvariantValidator.validateSpatialOutput(scenario, raw)
        assertFalse(rawReport.passed)
        assertTrue(rawReport.findings.any { it.invariant == "SPATIAL_NUISANCE_CANONICALIZED" && !it.passed })

        val canonicalReport = InvariantValidator.validateSpatialOutput(
            scenario,
            ValidationTestFixtures.spatialOutput(scenario),
        )
        assertTrue(canonicalReport.passed, canonicalReport.findings.toString())
    }

    @Test
    fun temporalInspectionSeamCoversHoldAcyclicAndRestSemantics() {
        listOf("static_isometric", "non_cyclic_activity", "multiple_sets_rests").forEach { id ->
            val scenario = suite.scenarios.single { it.scenarioId == id }
            val report = InvariantValidator.validateTemporalOutput(
                scenario,
                ValidationTestFixtures.temporal(scenario, ai.senp.core.contracts.VideoRole.SOURCE),
            )
            assertTrue(report.passed, id + ": " + report.findings)
        }
    }

    private fun scenario(id: String): SyntheticScenarioBundle {
        if (id != "biceps-like-reliable") return suite.scenarios.single { it.scenarioId == id }
        return suite.scenarios.single { it.scenarioId == "same_video_self" }.copy(scenarioId = id)
    }
}
