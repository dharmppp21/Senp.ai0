package ai.senp.sync.validation

import ai.senp.core.contracts.ActivitySegmentKind
import ai.senp.core.contracts.BodySideHypothesis
import ai.senp.core.contracts.CanonicalObservationSequence
import ai.senp.core.contracts.MirrorHypothesis
import ai.senp.core.contracts.MotionStructureClass
import ai.senp.core.contracts.SpatialSynchronizationDiagnostics
import ai.senp.core.contracts.SynchronizationRefusalReason
import ai.senp.core.contracts.SynchronizationRequest
import ai.senp.core.contracts.SynchronizationResult
import ai.senp.core.contracts.SynchronizationStatus
import ai.senp.core.contracts.TemporalStructure
import ai.senp.core.contracts.TimestampRange
import kotlinx.serialization.Serializable

@Serializable
enum class CoverageState {
    EXECUTABLE,
    STAGED,
}

@Serializable
data class ScenarioCoverage(
    val scenarioId: String,
    val fixtureGeneration: CoverageState = CoverageState.EXECUTABLE,
    val invariantValidation: CoverageState = CoverageState.EXECUTABLE,
    val productionIntegration: CoverageState = CoverageState.EXECUTABLE,
    val integrationLanes: Set<String>,
)

@Serializable
data class SyntheticTimestampTruth(
    val timestampMs: Long,
    val state: String,
    val direction: Int,
    val phaseProgress: Double,
    val reliable: Boolean,
    val formSignature: Double,
    val setId: String? = null,
    val subjectId: String? = "subject-a",
) {
    init {
        require(direction in -1..1) { "synthetic direction must be -1, 0, or 1" }
        require(phaseProgress in 0.0..1.0) { "synthetic phase progress must be in [0, 1]" }
    }

    val interior: Boolean get() = phaseProgress in 0.12..0.88
}

@Serializable
data class SyntheticUnitTruth(
    val unitId: String,
    val range: TimestampRange,
    val structureClass: MotionStructureClass,
    val openBegin: Boolean = false,
    val openEnd: Boolean = false,
)

@Serializable
data class SyntheticTemporalTruth(
    val classification: MotionStructureClass,
    val samples: List<SyntheticTimestampTruth>,
    val units: List<SyntheticUnitTruth>,
    val activityKinds: List<ActivitySegmentKind>,
)

@Serializable
data class SyntheticSpatialTruth(
    val mirror: MirrorHypothesis = MirrorHypothesis.NOT_MIRRORED,
    val selectedSide: BodySideHypothesis = BodySideHypothesis.BILATERAL,
    val relativeYawDegrees: Double = 0.0,
    val relativeElevationDegrees: Double = 0.0,
    val expectedStableSide: Boolean = true,
    val expectedDiscontinuities: List<TimestampRange> = emptyList(),
    val expectedTrueFormDelta: Double = 0.0,
    val nuisanceUniformScale: Double = 1.0,
    val subjectAmbiguous: Boolean = false,
)

@Serializable
data class ExpectedSynchronizationOutcome(
    val allowedStatuses: Set<SynchronizationStatus>,
    val allowedRefusalReasons: Set<SynchronizationRefusalReason> = emptySet(),
    val expectedSourceUnmatchedUnitIds: Set<String> = emptySet(),
    val expectedReferenceUnmatchedUnitIds: Set<String> = emptySet(),
    val expectedSourceUnmatchedRanges: List<TimestampRange> = emptyList(),
    val referenceReuseRequired: Boolean = false,
    val ambiguityMustBeExplicit: Boolean = false,
    val zeroInteriorOppositeDirectionPairs: Boolean = true,
)

@Serializable
data class SyntheticScenarioBundle(
    val schemaVersion: Int = 1,
    val scenarioId: String,
    val seed: Long,
    val expectedInvariants: Set<String>,
    val tags: Map<String, String> = emptyMap(),
    val request: SynchronizationRequest,
    val sourceTruth: SyntheticTemporalTruth,
    val referenceTruth: SyntheticTemporalTruth,
    val spatialTruth: SyntheticSpatialTruth,
    val expectedOutcome: ExpectedSynchronizationOutcome,
)

@Serializable
data class ValidationFinding(
    val invariant: String,
    val passed: Boolean,
    val message: String,
)

@Serializable
data class ScenarioValidationReport(
    val scenarioId: String,
    val status: String,
    val findings: List<ValidationFinding>,
) {
    val passed: Boolean get() = findings.all(ValidationFinding::passed)
}

@Serializable
data class SyntheticSuiteManifest(
    val schemaVersion: Int = 1,
    val seed: Long,
    val scenarioCount: Int,
    val scenarios: List<SyntheticScenarioBundle>,
    val coverage: List<ScenarioCoverage>,
)

/** Local validation seam. A future integration adapter may wrap the concrete Sync-v2 orchestrator. */
fun interface SynchronizationHarnessAdapter {
    fun synchronize(request: SynchronizationRequest): SynchronizationResult
}

/** Local seam for inspecting spatial canonicalization without changing the frozen public contracts. */
fun interface SpatialHarnessAdapter {
    fun synchronize(source: CanonicalObservationSequence, reference: CanonicalObservationSequence): SpatialHarnessOutput
}

@Serializable
data class SpatialHarnessOutput(
    val canonicalSource: CanonicalObservationSequence,
    val canonicalReference: CanonicalObservationSequence,
    val diagnostics: SpatialSynchronizationDiagnostics,
)

/** Local seam for independently exercising a temporal implementation after its branch is integrated. */
fun interface TemporalHarnessAdapter {
    fun analyze(sequence: CanonicalObservationSequence): TemporalStructure
}
