package ai.senp.core.contracts

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
enum class SynchronizationScope {
    FULL_SEQUENCE,
    SUBSEQUENCE,
    LOCAL,
}

@Serializable
data class SynchronizationSemantics(
    val scope: SynchronizationScope = SynchronizationScope.SUBSEQUENCE,
    val allowOpenSourceBegin: Boolean = true,
    val allowOpenSourceEnd: Boolean = true,
    val allowOpenReferenceBegin: Boolean = true,
    val allowOpenReferenceEnd: Boolean = true,
    val allowUnmatchedSource: Boolean = true,
    val allowUnmatchedReference: Boolean = true,
    val allowReferenceUnitReuse: Boolean = true,
)

@Serializable
data class SynchronizationRequirements(
    val requiredChannelSemanticTypes: Set<String> = emptySet(),
) {
    init {
        require(requiredChannelSemanticTypes.all { it.isNotBlank() && it.length <= 128 }) {
            "required channel semantic types must be non-blank and at most 128 characters"
        }
    }
}

@Serializable
data class SynchronizationRequest(
    val contractSchemaVersion: Int = CURRENT_SCHEMA_VERSION,
    val source: CanonicalObservationSequence,
    val reference: CanonicalObservationSequence,
    val semantics: SynchronizationSemantics = SynchronizationSemantics(),
    val requirements: SynchronizationRequirements = SynchronizationRequirements(),
) {
    init {
        require(contractSchemaVersion > 0) { "synchronization contract schema version must be positive" }
        require(source.role == VideoRole.SOURCE) { "synchronization source sequence must have SOURCE role" }
        require(reference.role == VideoRole.REFERENCE) { "synchronization reference sequence must have REFERENCE role" }
    }

    companion object {
        const val CURRENT_SCHEMA_VERSION: Int = 1
    }
}

@Serializable
enum class UnmatchedReason {
    NO_COMPATIBLE_COUNTERPART,
    EXTRA_ACTION,
    MISSING_REFERENCE_STEP,
    REST_OR_SETUP,
    OCCLUSION,
    DISCONTINUITY,
    AMBIGUOUS,
    INSUFFICIENT_DATA,
    OTHER,
}

@Serializable
sealed interface TimestampCorrespondence {
    val sourceTimestamp: TimestampMs
    val decisionConfidence: Double

    @Serializable
    @SerialName("matched")
    data class Matched(
        override val sourceTimestamp: TimestampMs,
        val referenceTimestamp: TimestampMs,
        override val decisionConfidence: Double,
    ) : TimestampCorrespondence {
        init {
            requireProbability(decisionConfidence, "timestamp-match confidence")
        }
    }

    @Serializable
    @SerialName("unmatched_source")
    data class UnmatchedSource(
        override val sourceTimestamp: TimestampMs,
        val reason: UnmatchedReason,
        override val decisionConfidence: Double,
    ) : TimestampCorrespondence {
        init {
            requireProbability(decisionConfidence, "unmatched-source timestamp confidence")
        }
    }
}

@Serializable
sealed interface MotionUnitCorrespondence {
    val decisionConfidence: Double

    @Serializable
    @SerialName("matched_unit")
    data class MatchedUnit(
        val sourceUnitId: String,
        val referenceUnitId: String,
        val timeline: List<TimestampCorrespondence>,
        override val decisionConfidence: Double,
        val ambiguity: Double,
    ) : MotionUnitCorrespondence {
        init {
            requireSyncIdentifier(sourceUnitId, "source motion-unit ID")
            requireSyncIdentifier(referenceUnitId, "reference motion-unit ID")
            require(timeline.isNotEmpty()) { "matched unit must contain timestamp correspondence decisions" }
            require(timeline.zipWithNext().all { (left, right) -> left.sourceTimestamp < right.sourceTimestamp }) {
                "matched-unit source timestamps must be strictly increasing"
            }
            val matchedReferenceTimestamps = timeline.mapNotNull {
                (it as? TimestampCorrespondence.Matched)?.referenceTimestamp
            }
            require(matchedReferenceTimestamps.zipWithNext().all { (left, right) -> left <= right }) {
                "matched-unit reference timestamps must be monotonic"
            }
            require(timeline.any { it is TimestampCorrespondence.Matched }) {
                "matched unit must contain at least one matched timestamp"
            }
            requireProbability(decisionConfidence, "matched-unit confidence")
            requireProbability(ambiguity, "matched-unit ambiguity")
        }
    }

    @Serializable
    @SerialName("source_unmatched_unit")
    data class SourceUnmatchedUnit(
        val sourceUnitId: String,
        val reason: UnmatchedReason,
        override val decisionConfidence: Double,
    ) : MotionUnitCorrespondence {
        init {
            requireSyncIdentifier(sourceUnitId, "source motion-unit ID")
            requireProbability(decisionConfidence, "unmatched-source unit confidence")
        }
    }

    @Serializable
    @SerialName("reference_unmatched_unit")
    data class ReferenceUnmatchedUnit(
        val referenceUnitId: String,
        val reason: UnmatchedReason,
        override val decisionConfidence: Double,
    ) : MotionUnitCorrespondence {
        init {
            requireSyncIdentifier(referenceUnitId, "reference motion-unit ID")
            requireProbability(decisionConfidence, "unmatched-reference unit confidence")
        }
    }
}

@Serializable
data class SynchronizationDiagnostics(
    val overallConfidence: Double,
    val spatialConfidence: Double,
    val temporalConfidence: Double,
    val correspondenceConfidence: Double,
    val sourceAnalyzableFraction: Double,
    val referenceAnalyzableFraction: Double,
    val correspondenceAmbiguity: Double,
) {
    init {
        requireProbability(overallConfidence, "overall synchronization confidence")
        requireProbability(spatialConfidence, "spatial synchronization confidence")
        requireProbability(temporalConfidence, "temporal synchronization confidence")
        requireProbability(correspondenceConfidence, "correspondence confidence")
        requireProbability(sourceAnalyzableFraction, "source analyzable fraction")
        requireProbability(referenceAnalyzableFraction, "reference analyzable fraction")
        requireProbability(correspondenceAmbiguity, "correspondence ambiguity")
    }
}

@Serializable
enum class SynchronizationStatus {
    SYNCHRONIZED,
    PARTIAL,
    REFUSED,
}

@Serializable
enum class SynchronizationRefusalReason {
    INSUFFICIENT_OBSERVATIONS,
    REQUIRED_CHANNEL_MISSING,
    NO_COMMON_MOTION,
    SPATIAL_INCOMPATIBILITY,
    SUBJECT_AMBIGUITY,
    TEMPORAL_AMBIGUITY,
    DISCONTINUITY,
    UNRELIABLE_CORRESPONDENCE,
}

@Serializable
data class SynchronizationRefusal(
    val reason: SynchronizationRefusalReason,
    val message: String,
    val missingRequiredChannelSemanticTypes: Set<String> = emptySet(),
) {
    init {
        require(message.isNotBlank()) { "synchronization refusal message must not be blank" }
        require(message.length <= 512) { "synchronization refusal message must be at most 512 characters" }
        require(missingRequiredChannelSemanticTypes.all { it.isNotBlank() && it.length <= 128 }) {
            "missing required channel semantic types must be non-blank and at most 128 characters"
        }
        require(
            reason == SynchronizationRefusalReason.REQUIRED_CHANNEL_MISSING || missingRequiredChannelSemanticTypes.isEmpty(),
        ) { "missing required channels may only be reported for REQUIRED_CHANNEL_MISSING" }
        if (reason == SynchronizationRefusalReason.REQUIRED_CHANNEL_MISSING) {
            require(missingRequiredChannelSemanticTypes.isNotEmpty()) {
                "REQUIRED_CHANNEL_MISSING refusal must identify at least one missing channel semantic type"
            }
        }
    }
}

@Serializable
data class SynchronizationResult(
    val resultSchemaVersion: Int = CURRENT_SCHEMA_VERSION,
    val status: SynchronizationStatus,
    val semantics: SynchronizationSemantics = SynchronizationSemantics(),
    val sourceTemporalStructure: TemporalStructure,
    val referenceTemporalStructure: TemporalStructure,
    val spatialDiagnostics: SpatialSynchronizationDiagnostics,
    val correspondences: List<MotionUnitCorrespondence>,
    val diagnostics: SynchronizationDiagnostics,
    val refusal: SynchronizationRefusal? = null,
) {
    init {
        require(resultSchemaVersion > 0) { "synchronization result schema version must be positive" }
        require(sourceTemporalStructure.role == VideoRole.SOURCE) { "source temporal structure must have SOURCE role" }
        require(referenceTemporalStructure.role == VideoRole.REFERENCE) { "reference temporal structure must have REFERENCE role" }
        require(spatialDiagnostics.aggregateConfidence == diagnostics.spatialConfidence) {
            "spatial diagnostic confidence must agree with synchronization diagnostics"
        }

        if (status == SynchronizationStatus.REFUSED) {
            require(refusal != null) { "refused synchronization result requires a refusal reason" }
        } else {
            require(refusal == null) { "non-refused synchronization result cannot contain a refusal" }
            require(correspondences.any { it is MotionUnitCorrespondence.MatchedUnit }) {
                "non-refused synchronization result requires at least one matched motion unit"
            }
        }

        validateUnitCoverageAndMappings()
    }

    private fun validateUnitCoverageAndMappings() {
        val sourceUnits = sourceTemporalStructure.motionUnits.associateBy(MotionUnit::unitId)
        val referenceUnits = referenceTemporalStructure.motionUnits.associateBy(MotionUnit::unitId)

        val matched = correspondences.filterIsInstance<MotionUnitCorrespondence.MatchedUnit>()
        val sourceUnmatched = correspondences.filterIsInstance<MotionUnitCorrespondence.SourceUnmatchedUnit>()
        val referenceUnmatched = correspondences.filterIsInstance<MotionUnitCorrespondence.ReferenceUnmatchedUnit>()

        if (status == SynchronizationStatus.SYNCHRONIZED) {
            require(sourceUnmatched.isEmpty() && referenceUnmatched.isEmpty()) {
                "SYNCHRONIZED result cannot contain unmatched motion units"
            }
        }
        if (!semantics.allowUnmatchedSource) {
            require(sourceUnmatched.isEmpty()) { "synchronization semantics disallow unmatched source units" }
            require(matched.none { unit -> unit.timeline.any { it is TimestampCorrespondence.UnmatchedSource } }) {
                "synchronization semantics disallow unmatched source timestamps"
            }
        }
        if (!semantics.allowUnmatchedReference) {
            require(referenceUnmatched.isEmpty()) { "synchronization semantics disallow unmatched reference units" }
        }
        if (!semantics.allowReferenceUnitReuse) {
            require(matched.map(MotionUnitCorrespondence.MatchedUnit::referenceUnitId).distinct().size == matched.size) {
                "synchronization semantics disallow reference motion-unit reuse"
            }
        }

        val sourceDecisionIds = matched.map(MotionUnitCorrespondence.MatchedUnit::sourceUnitId) +
            sourceUnmatched.map(MotionUnitCorrespondence.SourceUnmatchedUnit::sourceUnitId)
        require(sourceDecisionIds.distinct().size == sourceDecisionIds.size) {
            "each source motion unit must have at most one unit-level correspondence decision"
        }
        require(sourceDecisionIds.toSet() == sourceUnits.keys) {
            "every source motion unit must be explicitly matched or unmatched"
        }

        require(referenceUnmatched.map(MotionUnitCorrespondence.ReferenceUnmatchedUnit::referenceUnitId).distinct().size == referenceUnmatched.size) {
            "reference-unmatched motion-unit decisions must be unique"
        }
        val matchedReferenceIds = matched.map(MotionUnitCorrespondence.MatchedUnit::referenceUnitId).toSet()
        val unmatchedReferenceIds = referenceUnmatched.map(MotionUnitCorrespondence.ReferenceUnmatchedUnit::referenceUnitId).toSet()
        require(matchedReferenceIds.intersect(unmatchedReferenceIds).isEmpty()) {
            "a reference motion unit cannot be both matched and unmatched"
        }
        require(matchedReferenceIds + unmatchedReferenceIds == referenceUnits.keys) {
            "every reference motion unit must be matched at least once or explicitly unmatched"
        }

        matched.forEach { correspondence ->
            val sourceUnit = requireNotNull(sourceUnits[correspondence.sourceUnitId]) {
                "matched source motion-unit ID does not exist"
            }
            val referenceUnit = requireNotNull(referenceUnits[correspondence.referenceUnitId]) {
                "matched reference motion-unit ID does not exist"
            }
            correspondence.timeline.forEach { decision ->
                require(sourceUnit.range.contains(decision.sourceTimestamp)) {
                    "timestamp correspondence must lie inside its source motion unit"
                }
                if (decision is TimestampCorrespondence.Matched) {
                    require(referenceUnit.range.contains(decision.referenceTimestamp)) {
                        "timestamp correspondence must lie inside its reference motion unit"
                    }
                }
            }
        }

        sourceUnmatched.forEach {
            require(sourceUnits.containsKey(it.sourceUnitId)) { "unmatched source motion-unit ID does not exist" }
        }
        referenceUnmatched.forEach {
            require(referenceUnits.containsKey(it.referenceUnitId)) { "unmatched reference motion-unit ID does not exist" }
        }
    }

    companion object {
        const val CURRENT_SCHEMA_VERSION: Int = 1
    }
}

private fun requireSyncIdentifier(value: String, name: String) {
    require(value.isNotBlank()) { "$name must not be blank" }
    require(value.length <= 128) { "$name must be at most 128 characters" }
}
