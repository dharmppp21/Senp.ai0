package ai.senp.sync.v2

import ai.senp.core.contracts.CanonicalObservation
import ai.senp.core.contracts.CanonicalObservationSequence
import ai.senp.core.contracts.ChannelAvailability
import ai.senp.core.contracts.FrameValidityStatus
import ai.senp.core.contracts.ObservationChannel
import ai.senp.core.contracts.ObservationSampling
import ai.senp.core.contracts.ObservationValue
import ai.senp.core.contracts.PoseFrame
import ai.senp.core.contracts.PoseLandmark
import ai.senp.core.contracts.VideoPoseExtraction
import kotlin.math.min

/**
 * Timestamp-preserving bridge from the production MP33 pose envelope into the generic Sync-v2 observation contract.
 * Image and real world coordinates remain separate channels; absent world coordinates stay explicitly masked.
 */
class PoseObservationAdapter {
    fun adapt(
        extraction: VideoPoseExtraction,
        analysisFramesPerSecond: Double,
        inputNominalFramesPerSecond: Double? = null,
    ): CanonicalObservationSequence {
        require(analysisFramesPerSecond.isFinite() && analysisFramesPerSecond > 0.0)
        inputNominalFramesPerSecond?.let { require(it.isFinite() && it > 0.0) }
        return CanonicalObservationSequence(
            role = extraction.role,
            duration = extraction.duration,
            sampling = ObservationSampling(
                inputNominalFramesPerSecond = inputNominalFramesPerSecond,
                analysisFramesPerSecond = analysisFramesPerSecond,
            ),
            observations = extraction.poses.frames.map(::adaptFrame),
        )
    }

    private fun adaptFrame(frame: PoseFrame): CanonicalObservation = CanonicalObservation(
        timestamp = frame.timestamp,
        channels = listOf(imageChannel(frame), worldChannel(frame)),
    )

    private fun imageChannel(frame: PoseFrame): ObservationChannel {
        val usable = frame.validity.status !in HARD_MISSING_STATUSES
        val values = frame.landmarks.map { landmark ->
            val confidence = landmarkConfidence(frame, landmark)
            ObservationValue(
                key = landmark.id.name.lowercase(),
                values = if (usable) {
                    listOf(landmark.image.x, landmark.image.y, landmark.image.z)
                } else {
                    listOf(null, null, null)
                },
                mask = List(3) { usable },
                confidence = if (usable) confidence else 0.0,
            )
        }
        return ObservationChannel(
            channelId = "human-pose-image",
            schemaVersion = 1,
            semanticType = "human_pose_2d",
            coordinateSpace = "image_normalized",
            subjectId = "primary-subject",
            componentAxes = listOf("x", "y", "z"),
            values = values,
            availability = if (usable) ChannelAvailability.OBSERVED else ChannelAvailability.MISSING,
            confidence = if (usable) values.map(ObservationValue::confidence).averageOrZero() else 0.0,
        )
    }

    private fun worldChannel(frame: PoseFrame): ObservationChannel {
        val frameUsable = frame.validity.status !in HARD_MISSING_STATUSES
        val values = frame.landmarks.map { landmark ->
            val world = landmark.world.takeIf { frameUsable }
            val present = world != null
            ObservationValue(
                key = landmark.id.name.lowercase(),
                values = if (present) {
                    listOf(world!!.xMeters, world.yMeters, world.zMeters)
                } else {
                    listOf(null, null, null)
                },
                mask = List(3) { present },
                confidence = if (present) landmarkConfidence(frame, landmark) else 0.0,
            )
        }
        val presentComponents = values.sumOf { value -> value.mask.count { it } }
        val totalComponents = values.sumOf { it.mask.size }
        val availability = when {
            presentComponents == 0 -> ChannelAvailability.MISSING
            presentComponents == totalComponents -> ChannelAvailability.OBSERVED
            else -> ChannelAvailability.PARTIAL
        }
        val confidence = values.filter { it.mask.any { present -> present } }
            .map(ObservationValue::confidence)
            .averageOrZero()
        return ObservationChannel(
            channelId = "human-pose-world",
            schemaVersion = 1,
            semanticType = "human_pose_3d",
            coordinateSpace = "world_metric_3d",
            subjectId = "primary-subject",
            componentAxes = listOf("x", "y", "z"),
            values = values,
            availability = availability,
            confidence = if (availability == ChannelAvailability.MISSING) 0.0 else confidence,
        )
    }

    private fun landmarkConfidence(frame: PoseFrame, landmark: PoseLandmark): Double {
        var confidence = frame.validity.confidence
        landmark.visibility?.let { confidence = min(confidence, it) }
        landmark.presence?.let { confidence = min(confidence, it) }
        return confidence.coerceIn(0.0, 1.0)
    }

    private companion object {
        val HARD_MISSING_STATUSES = setOf(FrameValidityStatus.BLIND, FrameValidityStatus.CONTINUITY_BREAK)
    }
}

private fun List<Double>.averageOrZero(): Double = if (isEmpty()) 0.0 else average()
