package ai.senp.motion

import ai.senp.core.contracts.CanonicalObservation
import ai.senp.core.contracts.CanonicalObservationSequence
import ai.senp.core.contracts.ChannelAvailability
import ai.senp.core.contracts.DurationMs
import ai.senp.core.contracts.ObservationChannel
import ai.senp.core.contracts.ObservationSampling
import ai.senp.core.contracts.ObservationValue
import ai.senp.core.contracts.TimestampMs
import ai.senp.core.contracts.VideoRole
import java.nio.file.Files
import java.nio.file.Path
import kotlin.system.measureNanoTime

object SpatialMicrobenchmark {
    @JvmStatic
    fun main(args: Array<String>) {
        val report = Path.of(args.getOrElse(0) { "build/reports/microbenchmark/spatial-sync-v2-10s.json" })
        val iterations = args.getOrNull(1)?.toIntOrNull() ?: 120
        require(iterations > 0)
        val source = sequence(VideoRole.SOURCE, xOffset = 0.12)
        val reference = sequence(VideoRole.REFERENCE, xOffset = 0.0)
        val engine = SpatialSynchronizationEngine()

        repeat(12) { engine.analyze(source, reference) }
        var checksum = 0.0
        val elapsedNs = measureNanoTime {
            repeat(iterations) {
                checksum += engine.analyze(source, reference).diagnostics.aggregateConfidence
            }
        }
        val totalFrames = iterations * (source.observations.size + reference.observations.size)
        val elapsedMs = elapsedNs / 1_000_000.0
        val msPerFrame = elapsedMs / totalFrames.toDouble()
        val analysesPerSecond = iterations * 1_000.0 / elapsedMs
        Files.createDirectories(report.parent)
        Files.writeString(
            report,
            """{
  "engine": "${SpatialSynchronizationVersions.ENGINE}",
  "clip_seconds": 10,
  "analysis_fps": 15.0,
  "frames_per_pair": ${source.observations.size + reference.observations.size},
  "iterations": $iterations,
  "elapsed_ms": ${"%.3f".format(elapsedMs)},
  "ms_per_input_frame": ${"%.6f".format(msPerFrame)},
  "pair_analyses_per_second": ${"%.3f".format(analysesPerSecond)},
  "checksum": ${"%.6f".format(checksum)}
}
""",
        )
        println(Files.readString(report))
    }

    private fun sequence(role: VideoRole, xOffset: Double): CanonicalObservationSequence {
        val frameCount = 150
        val stepMs = 67L
        val observations = (0 until frameCount).map { index ->
            val phase = index.toDouble() / frameCount.toDouble()
            val values = baseValues(phase, xOffset)
            CanonicalObservation(
                timestamp = TimestampMs(index * stepMs),
                channels = listOf(
                    ObservationChannel(
                        channelId = "pose-${role.name.lowercase()}",
                        schemaVersion = 1,
                        semanticType = "human_pose",
                        coordinateSpace = "camera_metric",
                        subjectId = "primary",
                        componentAxes = listOf("x", "y", "z"),
                        values = values.map { (key, point) ->
                            ObservationValue(
                                key = key,
                                values = listOf(point.x, point.y, point.z),
                                mask = listOf(true, true, true),
                                confidence = 0.94,
                            )
                        },
                        availability = ChannelAvailability.OBSERVED,
                        confidence = 0.94,
                    ),
                ),
            )
        }
        return CanonicalObservationSequence(
            role = role,
            duration = DurationMs(frameCount * stepMs),
            sampling = ObservationSampling(inputNominalFramesPerSecond = 60.0, analysisFramesPerSecond = 15.0),
            observations = observations,
        )
    }

    private fun baseValues(phase: Double, xOffset: Double): LinkedHashMap<String, Vec3> {
        val wristMotion = 0.16 * kotlin.math.sin(phase * kotlin.math.PI * 4.0)
        val values = linkedMapOf(
            "left_shoulder" to Vec3(-0.34 + xOffset, 1.00, 0.04),
            "right_shoulder" to Vec3(0.34 + xOffset, 1.00, -0.04),
            "left_elbow" to Vec3(-0.61 + xOffset, 0.58, 0.26),
            "right_elbow" to Vec3(0.50 + xOffset, 0.69, -0.09),
            "left_wrist" to Vec3(-0.44 + xOffset, 0.20 + wristMotion, 0.48),
            "right_wrist" to Vec3(0.77 + xOffset, 0.47 - wristMotion, -0.16),
            "left_hip" to Vec3(-0.23 + xOffset, 0.00, 0.02),
            "right_hip" to Vec3(0.23 + xOffset, 0.00, -0.02),
            "left_knee" to Vec3(-0.27 + xOffset, -0.88, 0.11),
            "right_knee" to Vec3(0.29 + xOffset, -0.94, -0.08),
            "left_ankle" to Vec3(-0.22 + xOffset, -1.76, 0.05),
            "right_ankle" to Vec3(0.32 + xOffset, -1.82, -0.03),
            "left_foot_index" to Vec3(-0.18 + xOffset, -1.86, 0.34),
            "right_foot_index" to Vec3(0.38 + xOffset, -1.91, 0.29),
        )
        repeat(19) { index ->
            values["aux_$index"] = Vec3(
                xOffset + (index % 5 - 2) * 0.05,
                1.2 - index * 0.06,
                (index % 3 - 1) * 0.03,
            )
        }
        return values
    }
}
