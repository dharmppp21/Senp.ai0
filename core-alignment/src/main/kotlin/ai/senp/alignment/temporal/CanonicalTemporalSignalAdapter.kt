package ai.senp.alignment.temporal

import ai.senp.core.contracts.CanonicalObservationSequence
import ai.senp.core.contracts.ChannelAvailability
import ai.senp.core.contracts.SpatialReliabilityStatus
import ai.senp.core.contracts.SpatialSynchronizationDiagnostics
import ai.senp.core.contracts.TimestampMs
import ai.senp.core.contracts.VideoRole
import kotlin.math.min

/**
 * Converts the frozen generic observation envelope into activity-agnostic temporal feature maps.
 * It never assumes a pose topology and never uses frame index/FPS as temporal truth.
 */
class CanonicalTemporalSignalAdapter(
    private val config: TemporalSynchronizationConfig = TemporalSynchronizationConfig(),
) {
    fun adapt(
        sequence: CanonicalObservationSequence,
        spatialDiagnostics: SpatialSynchronizationDiagnostics? = null,
    ): TemporalSignalSequence {
        val featureUniverse = sequence.observations.flatMap { observation ->
            observation.channels.flatMap { channel ->
                channel.values.flatMap { value ->
                    channel.componentAxes.indices.mapNotNull { componentIndex ->
                        if (value.mask[componentIndex] && value.values[componentIndex] != null) {
                            featureKey(channel.semanticType, value.key, channel.componentAxes[componentIndex])
                        } else {
                            null
                        }
                    }
                }
            }
        }.toSet()

        var previousSubjects: Set<String>? = null
        var previousTimestamp: TimestampMs? = null
        val frames = sequence.observations.map { observation ->
            val candidates = linkedMapOf<String, TemporalFeatureSample>()
            val subjects = mutableSetOf<String>()
            observation.channels.forEach { channel ->
                channel.subjectId?.let(subjects::add)
                if (
                    channel.availability == ChannelAvailability.MISSING ||
                    channel.availability == ChannelAvailability.UNRELIABLE ||
                    channel.confidence < config.minimumFeatureConfidence
                ) {
                    return@forEach
                }
                channel.values.forEach { value ->
                    val sampleConfidence = min(channel.confidence, value.confidence)
                    if (sampleConfidence < config.minimumFeatureConfidence) return@forEach
                    channel.componentAxes.indices.forEach { componentIndex ->
                        val component = value.values[componentIndex]
                        if (!value.mask[componentIndex] || component == null || !component.isFinite()) return@forEach
                        val key = featureKey(channel.semanticType, value.key, channel.componentAxes[componentIndex])
                        val current = candidates[key]
                        if (current == null || sampleConfidence > current.confidence) {
                            candidates[key] = TemporalFeatureSample(component, sampleConfidence)
                        }
                    }
                }
            }

            val coverage = if (featureUniverse.isEmpty()) 0.0 else candidates.size.toDouble() / featureUniverse.size.toDouble()
            val evidenceConfidence = candidates.values.map(TemporalFeatureSample::confidence).averageOrZero()
            val spatial = reliabilityAt(sequence.role, observation.timestamp, spatialDiagnostics)
            val spatialFactor = when (spatial) {
                SpatialReliabilityStatus.COMPATIBLE, null -> 1.0
                SpatialReliabilityStatus.UNRELIABLE -> 0.45
                SpatialReliabilityStatus.INCOMPATIBLE, SpatialReliabilityStatus.DISCONTINUITY -> 0.0
            }
            val frameConfidence = (evidenceConfidence * coverage.coerceIn(0.0, 1.0) * spatialFactor).coerceIn(0.0, 1.0)
            val timestampGap = previousTimestamp?.let { observation.timestamp.value - it.value } ?: 0L
            val subjectDiscontinuity = previousSubjects?.let { previous ->
                previous.isNotEmpty() && subjects.isNotEmpty() && previous.intersect(subjects).isEmpty()
            } ?: false
            val discontinuity = spatial == SpatialReliabilityStatus.DISCONTINUITY ||
                timestampGap > config.discontinuityGapMs || subjectDiscontinuity
            previousSubjects = subjects
            previousTimestamp = observation.timestamp

            TemporalSignalFrame(
                timestamp = observation.timestamp,
                features = candidates,
                confidence = frameConfidence,
                discontinuityBefore = discontinuity,
            )
        }
        return TemporalSignalSequence(sequence.role, sequence.duration, frames)
    }

    private fun reliabilityAt(
        role: VideoRole,
        timestamp: TimestampMs,
        diagnostics: SpatialSynchronizationDiagnostics?,
    ): SpatialReliabilityStatus? = diagnostics?.reliabilitySegments
        ?.firstOrNull { segment -> segment.role == role && segment.range.contains(timestamp) }
        ?.status

    private fun featureKey(semanticType: String, valueKey: String, axis: String): String =
        "$semanticType/$valueKey/${axis.lowercase()}"
}

private fun List<Double>.averageOrZero(): Double = if (isEmpty()) 0.0 else average()
