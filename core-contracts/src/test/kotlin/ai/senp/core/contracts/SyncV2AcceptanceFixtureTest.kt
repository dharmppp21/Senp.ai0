package ai.senp.core.contracts

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SyncV2AcceptanceFixtureTest {
    @Test
    fun `adversarial acceptance fixture covers the frozen scenario matrix with invariant expectations`() {
        val fixtureText = checkNotNull(javaClass.getResource("/fixtures/sync-v2-adversarial-acceptance-v1.json"))
            .readText()
        val root = Json.parseToJsonElement(fixtureText).jsonObject
        assertEquals(1, root.getValue("schemaVersion").jsonPrimitive.content.toInt())
        assertEquals("sync-v2", root.getValue("contract").jsonPrimitive.content)

        val scenarios = root.getValue("scenarios").jsonArray
        val byId = scenarios.associateBy { it.jsonObject.getValue("id").jsonPrimitive.content }
        val requiredScenarioIds = setOf(
            "same_video_self",
            "different_fps",
            "different_resolution",
            "different_codec",
            "rotation_metadata",
            "yaw_elevation_viewpoint",
            "mirror",
            "side_selection_stability",
            "camera_movement_discontinuity",
            "start_mid_motion",
            "end_mid_motion",
            "one_reference_ten_source",
            "ten_reference_one_source",
            "two_reference_seven_source",
            "multiple_sets_rests",
            "extra_source_action",
            "missing_source_action",
            "repeated_identical_phase",
            "variable_speed",
            "pause_hold",
            "very_slow",
            "very_fast",
            "static_isometric",
            "no_common_motion",
            "poor_pose_coverage",
            "short_occlusion",
            "long_occlusion",
            "person_leaves_reenters",
            "multiple_people_subject_ambiguity",
            "different_body_proportions",
            "true_form_difference",
            "reversed_video",
            "edited_spliced_video",
            "slow_motion_edit",
            "non_cyclic_activity",
            "object_required_pose_only",
        )
        assertEquals(requiredScenarioIds, byId.keys)

        val allowedInvariants = setOf(
            "TIMESTAMP_FIRST",
            "IDENTITY_NOT_REFUSED_WHEN_OBSERVABLE",
            "NO_FORCED_UNMATCHED",
            "FPS_LENGTH_INDEPENDENT",
            "LOCAL_WARP_ALLOWED",
            "SPATIAL_NUISANCE_CANONICALIZED",
            "FORM_DIFFERENCE_PRESERVED",
            "TRANSPORT_METADATA_NOT_CORRESPONDENCE",
            "SPATIAL_DIAGNOSTICS_EXPLICIT",
            "VIEW_HYPOTHESIS_EXPLICIT",
            "SPATIAL_RELIABILITY_EXPLICIT",
            "MIRROR_HYPOTHESIS_EXPLICIT",
            "AMBIGUITY_NOT_SILENT",
            "SIDE_STABILITY_EXPLICIT",
            "SPATIAL_DISCONTINUITY_EXPLICIT",
            "NO_CONFIDENT_MAPPING_ACROSS_UNRELIABLE_GAP",
            "OPEN_BOUNDARY_ALLOWED",
            "PARTIAL_UNIT_ALLOWED",
            "SUBSEQUENCE_ALLOWED",
            "REFERENCE_UNIT_REUSE_ALLOWED",
            "UNIT_LOCAL_WARP",
            "NO_EQUAL_REP_COUNT_ASSUMPTION",
            "REFERENCE_UNMATCHED_EXPLICIT",
            "NO_FORCED_PAIR",
            "REST_SETUP_EXPLICIT",
            "SOURCE_UNMATCHED_EXPLICIT",
            "CONFIDENCE_REQUIRED",
            "HOLD_EXPLICIT",
            "ISOMETRIC_SUPPORTED",
            "NO_CYCLE_REQUIREMENT",
            "REFUSAL_REQUIRED",
            "NO_FABRICATED_SUCCESS",
            "ANALYZABLE_COVERAGE_EXPLICIT",
            "REFUSAL_OR_PARTIAL_REQUIRED",
            "MASKS_EXPLICIT",
            "SOURCE_TIMESTAMP_UNMATCHED_ALLOWED",
            "UNRELIABLE_SEGMENT_EXPLICIT",
            "DISCONTINUITY_EXPLICIT",
            "SUBJECT_IDENTITY_NOT_ASSUMED",
            "SUBJECT_AMBIGUITY_EXPLICIT",
            "NO_NONRIGID_FORM_ERASURE",
            "SPATIAL_TRANSFORM_ONLY_NUISANCE",
            "LOCAL_ALIGNMENT_ALLOWED",
            "ACYCLIC_SUPPORTED",
            "REQUIRED_CHANNELS_EXPLICIT",
        )

        scenarios.forEach { scenarioElement ->
            val scenario = scenarioElement.jsonObject
            assertTrue(scenario.getValue("setup").jsonPrimitive.content.isNotBlank())
            val invariants = scenario.getValue("expectedInvariants").jsonArray.map { it.jsonPrimitive.content }
            assertTrue(invariants.isNotEmpty())
            assertTrue(invariants.all(allowedInvariants::contains))
        }

        val normalized = fixtureText.lowercase()
        listOf("kabsch", "mediapipe", "dtw", "biceps", "squat", "push-up", "leg_raise").forEach { forbidden ->
            assertFalse(normalized.contains(forbidden), "acceptance fixture must remain implementation and activity agnostic: $forbidden")
        }
    }
}
