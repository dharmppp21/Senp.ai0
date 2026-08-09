package ai.senp.motion

import ai.senp.core.contracts.BodyCentricTransform
import ai.senp.core.contracts.CanonicalObservationSequence
import ai.senp.core.contracts.DurationMs
import ai.senp.core.contracts.ObservationChannel
import ai.senp.core.contracts.ObservationSampling
import ai.senp.core.contracts.SpatialSynchronizationDiagnostics
import ai.senp.core.contracts.TimestampMs
import ai.senp.core.contracts.UnitQuaternion
import ai.senp.core.contracts.Vector3
import ai.senp.core.contracts.VideoRole
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.sqrt

/** Stable identifiers for the synchronization-v2 spatial implementation. */
object SpatialSynchronizationVersions {
    const val ENGINE: String = "sync-v2-spatial/1"
    const val BODY_CENTRIC_3D: String = "body_centric_3d"
    const val BODY_CENTRIC_2D: String = "body_centric_2d"
}

enum class SpatialEvidenceKind {
    THREE_D,
    IMAGE_2D,
    PARTIAL,
    UNAVAILABLE,
}

/**
 * Root/global pose evidence kept separate from intrinsic body articulation.
 * A null rotation means the input did not contain trustworthy 3D orientation evidence.
 */
data class SpatialRootOrientation(
    val timestamp: TimestampMs,
    val coordinateSpace: String?,
    val pelvisInInputSpace: Vector3?,
    val inputToBodyRotation: UnitQuaternion?,
    val lateralAxisInInputSpace: Vector3?,
    val torsoUpAxisInInputSpace: Vector3?,
    val forwardAxisInInputSpace: Vector3?,
    val planarTorsoTiltDegrees: Double?,
    val confidence: Double,
) {
    init {
        requireSpatialProbability(confidence, "root-orientation confidence")
        planarTorsoTiltDegrees?.let {
            require(it.isFinite() && it in -180.0..180.0) {
                "image torso tilt must be finite and in [-180, 180] degrees"
            }
        }
    }
}

/**
 * Rotation/translation/scale-invariant articulation and body-proportion features.
 * Side-labelled features remain side-labelled; [distanceTo] can explicitly test a mirrored hypothesis.
 */
data class SpatialIntrinsicDescriptor(
    val values: Map<String, Double>,
    val confidence: Double,
) {
    init {
        requireSpatialProbability(confidence, "intrinsic-descriptor confidence")
        require(values.keys.all { it.isNotBlank() }) { "intrinsic-descriptor keys must be non-blank" }
        require(values.values.all(Double::isFinite)) { "intrinsic-descriptor values must be finite" }
    }

    fun distanceTo(other: SpatialIntrinsicDescriptor, mirroredOther: Boolean = false): Double? =
        descriptorDistance(values, other.values, mirroredOther)?.distance
}

/** Exact per-timestamp spatial seam for later coarse matching and robust refinement. */
data class SpatialObservationFrame(
    val timestamp: TimestampMs,
    val evidenceKind: SpatialEvidenceKind,
    val canonicalPose: ObservationChannel?,
    val bodyTransform: BodyCentricTransform?,
    val rootOrientation: SpatialRootOrientation?,
    val intrinsicDescriptor: SpatialIntrinsicDescriptor,
    val transformConfidence: Double,
    val selectedSubjectId: String?,
    val spatialSegmentId: Int?,
) {
    init {
        requireSpatialProbability(transformConfidence, "spatial transform confidence")
        require((canonicalPose == null) == (bodyTransform == null)) {
            "canonical pose and body transform must either both be present or both be absent"
        }
        require(spatialSegmentId == null || spatialSegmentId >= 0) { "spatial segment ID must be non-negative" }
    }
}

data class SpatialSequenceAnalysis(
    val role: VideoRole,
    val duration: DurationMs,
    val sampling: ObservationSampling,
    val frames: List<SpatialObservationFrame>,
    val analyzableFraction: Double,
) {
    init {
        requireSpatialProbability(analyzableFraction, "spatial analyzable fraction")
        require(frames.zipWithNext().all { (left, right) -> left.timestamp < right.timestamp }) {
            "spatial observation timestamps must be strictly increasing"
        }
        require(frames.lastOrNull()?.timestamp?.value?.let { it < duration.value } ?: true) {
            "spatial observation timestamp must be before sequence duration"
        }
    }
}

/** Missing generic semantic channels are surfaced for orchestration/refusal policy; they are never guessed. */
data class SpatialRequirementGaps(
    val sourceMissingSemanticTypes: Set<String> = emptySet(),
    val referenceMissingSemanticTypes: Set<String> = emptySet(),
) {
    val anyMissing: Boolean
        get() = sourceMissingSemanticTypes.isNotEmpty() || referenceMissingSemanticTypes.isNotEmpty()
}

data class SpatialSynchronizationOutput(
    val diagnostics: SpatialSynchronizationDiagnostics,
    val source: SpatialSequenceAnalysis,
    val reference: SpatialSequenceAnalysis,
    val requirementGaps: SpatialRequirementGaps,
)

/**
 * Generic anatomical-key schema. The spatial kernel does not require a fixed pose topology; adapters may bind
 * any topology by mapping only these stable torso/limb roles to keys already present in their human-pose channel.
 */
data class SpatialLandmarkSchema(
    val humanPoseSemanticTypes: Set<String> = setOf("human_pose", "human_pose_2d", "human_pose_3d"),
    val leftShoulder: String = "left_shoulder",
    val rightShoulder: String = "right_shoulder",
    val leftElbow: String = "left_elbow",
    val rightElbow: String = "right_elbow",
    val leftWrist: String = "left_wrist",
    val rightWrist: String = "right_wrist",
    val leftHip: String = "left_hip",
    val rightHip: String = "right_hip",
    val leftKnee: String = "left_knee",
    val rightKnee: String = "right_knee",
    val leftAnkle: String = "left_ankle",
    val rightAnkle: String = "right_ankle",
    val leftFoot: String = "left_foot_index",
    val rightFoot: String = "right_foot_index",
) {
    init {
        require(humanPoseSemanticTypes.isNotEmpty() && humanPoseSemanticTypes.all { it.isNotBlank() })
        require(
            listOf(
                leftShoulder,
                rightShoulder,
                leftElbow,
                rightElbow,
                leftWrist,
                rightWrist,
                leftHip,
                rightHip,
                leftKnee,
                rightKnee,
                leftAnkle,
                rightAnkle,
                leftFoot,
                rightFoot,
            ).all { it.isNotBlank() },
        ) { "spatial landmark keys must be non-blank" }
    }
}

data class SpatialSynchronizationConfig(
    val landmarkSchema: SpatialLandmarkSchema = SpatialLandmarkSchema(),
    val minimumChannelConfidence: Double = 0.30,
    val minimumAnchorConfidence: Double = 0.45,
    val minimumDescriptorConfidence: Double = 0.30,
    val minimumTorsoScale: Double = 1e-6,
    val shortOcclusionMs: Long = 220L,
    val discontinuityGapMs: Long = 600L,
    val cameraMotionRotationDegrees: Double = 8.0,
    val cameraDiscontinuityRotationDegrees: Double = 45.0,
    val cameraMotionRootTorsoUnits: Double = 0.12,
    val cameraDiscontinuityRootTorsoUnits: Double = 0.65,
    val cameraDiscontinuityScaleRatio: Double = 1.70,
    val stableIntrinsicDistance: Double = 0.08,
    val mirrorDecisionMargin: Double = 0.045,
    val minimumChiralityMagnitude: Double = 0.012,
    val sideCoverageThreshold: Double = 0.45,
    val sideDominanceMargin: Double = 0.12,
    val minimumSideSelectionStability: Double = 0.70,
) {
    init {
        listOf(minimumChannelConfidence, minimumAnchorConfidence, minimumDescriptorConfidence).forEach {
            requireSpatialProbability(it, "spatial threshold")
        }
        require(minimumTorsoScale.isFinite() && minimumTorsoScale > 0.0)
        require(shortOcclusionMs >= 0L)
        require(discontinuityGapMs > shortOcclusionMs)
        require(cameraMotionRotationDegrees.isFinite() && cameraMotionRotationDegrees >= 0.0)
        require(
            cameraDiscontinuityRotationDegrees.isFinite() &&
                cameraDiscontinuityRotationDegrees > cameraMotionRotationDegrees,
        )
        require(cameraMotionRootTorsoUnits.isFinite() && cameraMotionRootTorsoUnits >= 0.0)
        require(
            cameraDiscontinuityRootTorsoUnits.isFinite() &&
                cameraDiscontinuityRootTorsoUnits > cameraMotionRootTorsoUnits,
        )
        require(cameraDiscontinuityScaleRatio.isFinite() && cameraDiscontinuityScaleRatio > 1.0)
        require(stableIntrinsicDistance.isFinite() && stableIntrinsicDistance >= 0.0)
        require(mirrorDecisionMargin.isFinite() && mirrorDecisionMargin >= 0.0)
        require(minimumChiralityMagnitude.isFinite() && minimumChiralityMagnitude >= 0.0)
        requireSpatialProbability(sideCoverageThreshold, "side coverage threshold")
        requireSpatialProbability(sideDominanceMargin, "side dominance margin")
        requireSpatialProbability(minimumSideSelectionStability, "minimum side-selection stability")
    }
}

internal data class DescriptorDistance(val distance: Double, val commonFeatureCount: Int)

internal fun descriptorDistance(
    left: Map<String, Double>,
    right: Map<String, Double>,
    mirrorRight: Boolean,
): DescriptorDistance? {
    var squared = 0.0
    var count = 0
    for ((key, leftValue) in left) {
        val rightKey = if (mirrorRight) mirroredSpatialKey(key) else key
        val rightValue = right[rightKey] ?: continue
        val normalizedDifference = when {
            key.startsWith("angle.") -> abs(leftValue - rightValue) / 180.0
            key.startsWith("ratio.") -> abs(ln((abs(leftValue) + 1e-9) / (abs(rightValue) + 1e-9))).coerceAtMost(2.0) / 2.0
            else -> abs(leftValue - rightValue)
        }
        squared += normalizedDifference * normalizedDifference
        count += 1
    }
    if (count == 0) return null
    return DescriptorDistance(sqrt(squared / count.toDouble()), count)
}

internal fun mirroredSpatialKey(key: String): String = when {
    ".left_" in key -> key.replace(".left_", ".__swap__").replace(".right_", ".left_").replace(".__swap__", ".right_")
    ".right_" in key -> key.replace(".right_", ".left_")
    key.startsWith("left_") -> "right_" + key.removePrefix("left_")
    key.startsWith("right_") -> "left_" + key.removePrefix("right_")
    else -> key
}

internal fun requireSpatialProbability(value: Double, name: String) {
    require(value.isFinite() && value in 0.0..1.0) { "$name must be finite and in [0, 1]" }
}

internal fun CanonicalObservationSequence.availableSemanticTypes(): Set<String> = observations
    .flatMap { observation -> observation.channels }
    .filter { channel -> channel.confidence > 0.0 && channel.availability != ai.senp.core.contracts.ChannelAvailability.MISSING }
    .mapTo(linkedSetOf()) { channel -> channel.semanticType }
