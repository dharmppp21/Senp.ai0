package ai.senp.core.contracts

import kotlinx.serialization.Serializable
import kotlin.math.abs
import kotlin.math.sqrt

@Serializable
data class Vector3(
    val x: Double,
    val y: Double,
    val z: Double,
) {
    init {
        requireFinite(x, "vector x")
        requireFinite(y, "vector y")
        requireFinite(z, "vector z")
    }
}

@Serializable
data class UnitQuaternion(
    val x: Double,
    val y: Double,
    val z: Double,
    val w: Double,
) {
    init {
        listOf(x, y, z, w).forEachIndexed { index, value -> requireFinite(value, "quaternion component $index") }
        val norm = sqrt(x * x + y * y + z * z + w * w)
        require(abs(norm - 1.0) <= NORMALIZATION_TOLERANCE) { "quaternion must be unit length" }
    }

    companion object {
        private const val NORMALIZATION_TOLERANCE = 1e-6
        val Identity = UnitQuaternion(0.0, 0.0, 0.0, 1.0)
    }
}

@Serializable
data class BodyCentricTransform(
    val fromCoordinateSpace: String,
    val toCoordinateSpace: String,
    val translation: Vector3,
    val rotation: UnitQuaternion,
    val uniformScale: Double,
) {
    init {
        requireSpatialIdentifier(fromCoordinateSpace, "source coordinate space")
        requireSpatialIdentifier(toCoordinateSpace, "target coordinate space")
        requireFinite(uniformScale, "body-centric uniform scale")
        require(uniformScale > 0.0) { "body-centric uniform scale must be positive" }
    }
}

@Serializable
data class BodyCentricTransformEstimate(
    val role: VideoRole,
    val range: TimestampRange,
    val transform: BodyCentricTransform,
    val confidence: Double,
    val stability: Double,
) {
    init {
        requireProbability(confidence, "body-centric transform confidence")
        requireProbability(stability, "body-centric transform stability")
    }
}

@Serializable
enum class MirrorHypothesis {
    NOT_MIRRORED,
    MIRRORED,
    AMBIGUOUS,
    UNKNOWN,
}

@Serializable
enum class BodySideHypothesis {
    LEFT,
    RIGHT,
    BILATERAL,
    UNKNOWN,
}

@Serializable
data class RelativeViewHypothesis(
    val sourceRange: TimestampRange,
    val referenceRange: TimestampRange,
    val relativeYawDegrees: Double? = null,
    val relativeElevationDegrees: Double? = null,
    val mirror: MirrorHypothesis,
    val selectedBodySide: BodySideHypothesis,
    val confidence: Double,
    val sideSelectionStability: Double,
) {
    init {
        relativeYawDegrees?.let {
            requireFinite(it, "relative yaw")
            require(it in -180.0..180.0) { "relative yaw must be in [-180, 180] degrees" }
        }
        relativeElevationDegrees?.let {
            requireFinite(it, "relative elevation")
            require(it in -90.0..90.0) { "relative elevation must be in [-90, 90] degrees" }
        }
        requireProbability(confidence, "relative-view confidence")
        requireProbability(sideSelectionStability, "side-selection stability")
    }
}

@Serializable
enum class SpatialReliabilityStatus {
    COMPATIBLE,
    UNRELIABLE,
    INCOMPATIBLE,
    DISCONTINUITY,
}

@Serializable
enum class SpatialDiagnosticReason {
    POOR_OBSERVATION_COVERAGE,
    INSUFFICIENT_3D,
    VIEW_AMBIGUITY,
    MIRROR_AMBIGUITY,
    SIDE_SELECTION_UNSTABLE,
    TRANSFORM_UNSTABLE,
    CAMERA_MOVEMENT,
    CAMERA_DISCONTINUITY,
    SUBJECT_AMBIGUITY,
    OCCLUSION,
    SPATIAL_INCOMPATIBILITY,
}

@Serializable
data class SpatialReliabilitySegment(
    val role: VideoRole,
    val range: TimestampRange,
    val status: SpatialReliabilityStatus,
    val confidence: Double,
    val reasons: Set<SpatialDiagnosticReason> = emptySet(),
) {
    init {
        requireProbability(confidence, "spatial reliability confidence")
        require(status != SpatialReliabilityStatus.COMPATIBLE || reasons.isEmpty()) {
            "compatible spatial segment cannot contain failure reasons"
        }
        require(status == SpatialReliabilityStatus.COMPATIBLE || reasons.isNotEmpty()) {
            "non-compatible spatial segment requires at least one reason"
        }
    }
}

@Serializable
data class SpatialSynchronizationDiagnostics(
    val sourceTransforms: List<BodyCentricTransformEstimate> = emptyList(),
    val referenceTransforms: List<BodyCentricTransformEstimate> = emptyList(),
    val relativeViewHypotheses: List<RelativeViewHypothesis> = emptyList(),
    val reliabilitySegments: List<SpatialReliabilitySegment> = emptyList(),
    val aggregateConfidence: Double,
) {
    init {
        require(sourceTransforms.all { it.role == VideoRole.SOURCE }) { "source transform must have SOURCE role" }
        require(referenceTransforms.all { it.role == VideoRole.REFERENCE }) { "reference transform must have REFERENCE role" }
        requireOrderedNonOverlapping(sourceTransforms.map(BodyCentricTransformEstimate::range), "source transform")
        requireOrderedNonOverlapping(referenceTransforms.map(BodyCentricTransformEstimate::range), "reference transform")
        requireOrderedNonOverlapping(relativeViewHypotheses.map(RelativeViewHypothesis::sourceRange), "relative-view source")
        VideoRole.entries.forEach { role ->
            requireOrderedNonOverlapping(
                reliabilitySegments.filter { it.role == role }.map(SpatialReliabilitySegment::range),
                "${role.name.lowercase()} spatial reliability",
            )
        }
        requireProbability(aggregateConfidence, "aggregate spatial confidence")
    }
}

internal fun requireOrderedNonOverlapping(ranges: List<TimestampRange>, name: String) {
    require(ranges.zipWithNext().all { (left, right) -> left.endExclusive <= right.start }) {
        "$name ranges must be ordered and non-overlapping"
    }
}

private fun requireSpatialIdentifier(value: String, name: String) {
    require(value.isNotBlank()) { "$name must not be blank" }
    require(value.length <= 128) { "$name must be at most 128 characters" }
}
