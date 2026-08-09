package ai.senp.alignment.temporal

import ai.senp.core.contracts.CanonicalObservation
import ai.senp.core.contracts.CanonicalObservationSequence
import ai.senp.core.contracts.ChannelAvailability
import ai.senp.core.contracts.DurationMs
import ai.senp.core.contracts.ObservationChannel
import ai.senp.core.contracts.ObservationSampling
import ai.senp.core.contracts.ObservationValue
import ai.senp.core.contracts.SpatialDiagnosticReason
import ai.senp.core.contracts.SpatialReliabilitySegment
import ai.senp.core.contracts.SpatialReliabilityStatus
import ai.senp.core.contracts.SpatialSynchronizationDiagnostics
import ai.senp.core.contracts.TimestampMs
import ai.senp.core.contracts.TimestampRange
import ai.senp.core.contracts.VideoRole
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class CanonicalTemporalSignalAdapterTest {
    private val adapter = CanonicalTemporalSignalAdapter()

    @Test
    fun preservesIrregularAuthoritativeTimestampsAndGenericSemanticKeys() {
        val timestamps = listOf(0L, 83L, 211L, 420L, 677L)
        val sequence = scalarSequence(VideoRole.SOURCE, timestamps, 900L)
        val signal = adapter.adapt(sequence)

        assertEquals(timestamps, signal.frames.map { it.timestamp.value })
        assertEquals(setOf("human_pose/joint/x"), signal.frames.first().features.keys)
        assertTrue(signal.frames.all { it.confidence > 0.9 })
    }

    @Test
    fun spatialDiscontinuityIsPropagatedIntoTemporalEvidence() {
        val sequence = scalarSequence(VideoRole.SOURCE, listOf(0L, 100L, 200L, 300L), 400L)
        val diagnostics = SpatialSynchronizationDiagnostics(
            reliabilitySegments = listOf(
                SpatialReliabilitySegment(
                    role = VideoRole.SOURCE,
                    range = TimestampRange(TimestampMs(100L), TimestampMs(200L)),
                    status = SpatialReliabilityStatus.DISCONTINUITY,
                    confidence = 0.0,
                    reasons = setOf(SpatialDiagnosticReason.CAMERA_DISCONTINUITY),
                ),
            ),
            aggregateConfidence = 0.75,
        )
        val signal = adapter.adapt(sequence, diagnostics)
        val discontinuityFrame = signal.frames.first { it.timestamp.value == 100L }

        assertTrue(discontinuityFrame.discontinuityBefore)
        assertEquals(0.0, discontinuityFrame.confidence)
    }

    private fun scalarSequence(
        role: VideoRole,
        timestamps: List<Long>,
        durationMs: Long,
    ): CanonicalObservationSequence = CanonicalObservationSequence(
        role = role,
        duration = DurationMs(durationMs),
        sampling = ObservationSampling(inputNominalFramesPerSecond = 60.0, analysisFramesPerSecond = 15.0),
        observations = timestamps.map { timestamp ->
            CanonicalObservation(
                timestamp = TimestampMs(timestamp),
                channels = listOf(
                    ObservationChannel(
                        channelId = "pose",
                        schemaVersion = 1,
                        semanticType = "human_pose",
                        componentAxes = listOf("x"),
                        values = listOf(
                            ObservationValue(
                                key = "joint",
                                values = listOf(timestamp.toDouble() / durationMs.toDouble()),
                                mask = listOf(true),
                                confidence = 0.98,
                            ),
                        ),
                        availability = ChannelAvailability.OBSERVED,
                        confidence = 0.98,
                    ),
                ),
            )
        },
    )
}
