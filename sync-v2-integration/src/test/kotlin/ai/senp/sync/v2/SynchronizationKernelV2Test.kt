package ai.senp.sync.v2

import ai.senp.core.contracts.AnalysisFailure
import ai.senp.core.contracts.CanonicalObservation
import ai.senp.core.contracts.CanonicalObservationSequence
import ai.senp.core.contracts.ChannelAvailability
import ai.senp.core.contracts.DurationMs
import ai.senp.core.contracts.FrameValidity
import ai.senp.core.contracts.ImageLandmark
import ai.senp.core.contracts.ObservationChannel
import ai.senp.core.contracts.ObservationSampling
import ai.senp.core.contracts.ObservationValue
import ai.senp.core.contracts.PoseFrame
import ai.senp.core.contracts.PoseLandmark
import ai.senp.core.contracts.PoseLandmarkId
import ai.senp.core.contracts.PoseModelConfiguration
import ai.senp.core.contracts.PoseSequence
import ai.senp.core.contracts.SamplingConfiguration
import ai.senp.core.contracts.Sha256
import ai.senp.core.contracts.StageResult
import ai.senp.core.contracts.SynchronizationRefusalReason
import ai.senp.core.contracts.SynchronizationRequest
import ai.senp.core.contracts.SynchronizationRequirements
import ai.senp.core.contracts.SynchronizationStatus
import ai.senp.core.contracts.TimestampMs
import ai.senp.core.contracts.VideoPoseDiagnostics
import ai.senp.core.contracts.VideoPoseExtraction
import ai.senp.core.contracts.VideoPoseFailureKind
import ai.senp.core.contracts.VideoRole
import ai.senp.core.contracts.VideoSource
import ai.senp.core.contracts.WorldLandmark
import ai.senp.core.pipeline.VideoPoseExtractor
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking

class SynchronizationKernelV2Test {
    @Test
    fun `identity uses bounded refinement and remains analyzable`() {
        val source = canonicalSequence(VideoRole.SOURCE)
        val reference = canonicalSequence(VideoRole.REFERENCE)

        val run = SynchronizationKernelV2().synchronize(SynchronizationRequest(source = source, reference = reference))

        assertNotEquals(SynchronizationStatus.REFUSED, run.result.status)
        assertTrue(run.result.diagnostics.sourceAnalyzableFraction > 0.8)
        assertTrue(run.result.diagnostics.referenceAnalyzableFraction > 0.8)
        assertEquals(1, run.stats.iterationCount)
        assertTrue(run.stats.maximumFineBandWidth < source.observations.size)
        assertTrue(run.stats.totalFineCellsEvaluated < source.observations.size.toLong() * reference.observations.size * run.stats.iterationCount)
        val mappedDiagnostics = run.mappingDiagnostics.filter { it.referenceTimestamp != null }
        assertTrue(mappedDiagnostics.isNotEmpty())
        assertTrue(mappedDiagnostics.any { it.sourceDirection != null && it.referenceDirection != null })
        assertTrue(mappedDiagnostics.any { it.sourceState != null && it.referenceState != null })
        assertTrue(mappedDiagnostics.none {
            it.sourceDirection != null && it.referenceDirection != null && it.sourceDirection * it.referenceDirection < 0
        })
    }

    @Test
    fun `required missing modality is a typed refusal`() {
        val run = SynchronizationKernelV2().synchronize(
            SynchronizationRequest(
                source = canonicalSequence(VideoRole.SOURCE),
                reference = canonicalSequence(VideoRole.REFERENCE),
                requirements = SynchronizationRequirements(setOf("human_pose", "object_pose")),
            ),
        )

        assertEquals(SynchronizationStatus.REFUSED, run.result.status)
        assertEquals(SynchronizationRefusalReason.REQUIRED_CHANNEL_MISSING, run.result.refusal?.reason)
        assertEquals(setOf("object_pose"), run.result.refusal?.missingRequiredChannelSemanticTypes)
    }

    @Test
    fun `synthetic truth oracle channel cannot affect production features`() {
        val source = canonicalSequence(VideoRole.SOURCE)
        val reference = canonicalSequence(VideoRole.REFERENCE)
        val baseline = SynchronizationKernelV2().synchronize(SynchronizationRequest(source = source, reference = reference))
        val poisoned = source.copy(observations = source.observations.mapIndexed { index, observation ->
            observation.copy(channels = observation.channels + oracleChannel(index))
        })
        val withOracle = SynchronizationKernelV2().synchronize(SynchronizationRequest(source = poisoned, reference = reference))

        assertEquals(baseline.result, withOracle.result)
        assertEquals(baseline.stats.totalFineCellsEvaluated, withOracle.stats.totalFineCellsEvaluated)
        assertEquals(baseline.stats.totalCoarseUnitComparisons, withOracle.stats.totalCoarseUnitComparisons)
    }

    @Test
    fun `video seam reuses pose cache without re-extraction`() = runBlocking {
        val extractor = CountingExtractor()
        val pipeline = VideoSynchronizationPipelineV2(extractor, InMemorySynchronizationPoseCache())
        val request = videoRequest()

        val first = assertIs<VideoSynchronizationOutcome.Success>(pipeline.synchronize(request)).run
        val second = assertIs<VideoSynchronizationOutcome.Success>(pipeline.synchronize(request)).run

        assertEquals(2, extractor.calls)
        assertTrue(!first.timings.sourcePoseCacheHit && !first.timings.referencePoseCacheHit)
        assertTrue(second.timings.sourcePoseCacheHit && second.timings.referencePoseCacheHit)
        assertEquals(first.synchronization.result, second.synchronization.result)
    }

    @Test
    fun `video seam preserves typed extractor failure`() = runBlocking {
        val failure = AnalysisFailure.VideoPose(
            VideoRole.SOURCE,
            VideoPoseFailureKind.CORRUPT_VIDEO,
            "fixture decode failure",
        )
        val extractor = object : VideoPoseExtractor {
            override suspend fun extract(
                role: VideoRole,
                source: VideoSource,
                sampling: SamplingConfiguration,
                model: PoseModelConfiguration,
            ): StageResult<VideoPoseExtraction> = StageResult.Failure(failure)
        }
        val outcome = VideoSynchronizationPipelineV2(extractor).synchronize(videoRequest())

        assertEquals(failure, assertIs<VideoSynchronizationOutcome.Failure>(outcome).failure)
    }

    private fun canonicalSequence(role: VideoRole): CanonicalObservationSequence {
        val observations = (0 until 30).map { index ->
            val phase = index / 10.0
            val motion = -cos(2.0 * PI * phase)
            val secondary = sin(2.0 * PI * phase)
            val points = linkedMapOf(
                "left_hip" to listOf(-0.20, -0.05, 0.0),
                "right_hip" to listOf(0.20, -0.05, 0.0),
                "left_shoulder" to listOf(-0.42, 0.95, 0.0),
                "right_shoulder" to listOf(0.42, 0.95, 0.0),
                "left_elbow" to listOf(-0.56, 0.69 + 0.20 * secondary, 0.10 * motion),
                "right_elbow" to listOf(0.56, 0.69 + 0.20 * secondary, 0.10 * motion),
                "left_wrist" to listOf(-0.66, 0.42 + 0.32 * motion, 0.16 * motion),
                "right_wrist" to listOf(0.66, 0.42 + 0.32 * motion, 0.16 * motion),
                "left_knee" to listOf(-0.20, -0.55 + 0.08 * secondary, 0.05 * motion),
                "right_knee" to listOf(0.20, -0.55 + 0.08 * secondary, 0.05 * motion),
                "left_ankle" to listOf(-0.20, -1.0, 0.0),
                "right_ankle" to listOf(0.20, -1.0, 0.0),
                "left_foot_index" to listOf(-0.20, -1.16, 0.18),
                "right_foot_index" to listOf(0.20, -1.16, 0.18),
            )
            CanonicalObservation(
                timestamp = TimestampMs(index * 100L),
                channels = listOf(
                    ObservationChannel(
                        channelId = "human-primary",
                        schemaVersion = 1,
                        semanticType = "human_pose",
                        coordinateSpace = "camera_3d",
                        subjectId = "subject-a",
                        componentAxes = listOf("x", "y", "z"),
                        values = points.map { (key, values) -> ObservationValue(key, values, listOf(true, true, true), 0.98) },
                        availability = ChannelAvailability.OBSERVED,
                        confidence = 0.98,
                    ),
                ),
            )
        }
        return CanonicalObservationSequence(role, DurationMs(3000), ObservationSampling(30.0, 10.0), observations)
    }

    private fun oracleChannel(index: Int): ObservationChannel = ObservationChannel(
        channelId = "synthetic-oracle",
        schemaVersion = 1,
        semanticType = "synthetic_motion_truth",
        componentAxes = listOf("value"),
        values = listOf(ObservationValue("phase", listOf((index % 2) * 1000.0), listOf(true), 1.0)),
        availability = ChannelAvailability.OBSERVED,
        confidence = 1.0,
    )

    private fun videoRequest(): VideoSynchronizationRequest = VideoSynchronizationRequest(
        source = VideoSource("source.mp4", Sha256("1".repeat(64))),
        reference = VideoSource("reference.mp4", Sha256("2".repeat(64))),
        sampling = SamplingConfiguration(targetFramesPerSecond = 10),
        model = PoseModelConfiguration(Sha256("3".repeat(64))),
    )

    private class CountingExtractor : VideoPoseExtractor {
        var calls: Int = 0

        override suspend fun extract(
            role: VideoRole,
            source: VideoSource,
            sampling: SamplingConfiguration,
            model: PoseModelConfiguration,
        ): StageResult<VideoPoseExtraction> {
            calls += 1
            return StageResult.Success(poseExtraction(role))
        }
    }

    private companion object {
        fun poseExtraction(role: VideoRole): VideoPoseExtraction {
            val frames = (0 until 30).map { index ->
                val motion = -cos(2.0 * PI * (index / 10.0))
                val landmarks = PoseLandmarkId.entries.map { id ->
                    val base = when (id) {
                        PoseLandmarkId.LEFT_HIP -> Triple(-0.20, -0.05, 0.0)
                        PoseLandmarkId.RIGHT_HIP -> Triple(0.20, -0.05, 0.0)
                        PoseLandmarkId.LEFT_SHOULDER -> Triple(-0.42, 0.95, 0.0)
                        PoseLandmarkId.RIGHT_SHOULDER -> Triple(0.42, 0.95, 0.0)
                        PoseLandmarkId.LEFT_ELBOW -> Triple(-0.56, 0.69, 0.10 * motion)
                        PoseLandmarkId.RIGHT_ELBOW -> Triple(0.56, 0.69, 0.10 * motion)
                        PoseLandmarkId.LEFT_WRIST -> Triple(-0.66, 0.42 + 0.32 * motion, 0.16 * motion)
                        PoseLandmarkId.RIGHT_WRIST -> Triple(0.66, 0.42 + 0.32 * motion, 0.16 * motion)
                        PoseLandmarkId.LEFT_KNEE -> Triple(-0.20, -0.55, 0.05 * motion)
                        PoseLandmarkId.RIGHT_KNEE -> Triple(0.20, -0.55, 0.05 * motion)
                        PoseLandmarkId.LEFT_ANKLE -> Triple(-0.20, -1.0, 0.0)
                        PoseLandmarkId.RIGHT_ANKLE -> Triple(0.20, -1.0, 0.0)
                        PoseLandmarkId.LEFT_FOOT_INDEX -> Triple(-0.20, -1.16, 0.18)
                        PoseLandmarkId.RIGHT_FOOT_INDEX -> Triple(0.20, -1.16, 0.18)
                        else -> Triple(0.0, 0.2, 0.0)
                    }
                    PoseLandmark(
                        id = id,
                        image = ImageLandmark(0.5 + base.first * 0.2, 0.5 - base.second * 0.2, base.third),
                        world = WorldLandmark(base.first, base.second, base.third),
                        visibility = 0.98,
                        presence = 0.98,
                    )
                }
                PoseFrame(TimestampMs(index * 100L), index.toLong(), landmarks, FrameValidity.Valid)
            }
            return VideoPoseExtraction(
                role = role,
                duration = DurationMs(3000),
                poses = PoseSequence(role, frames),
                diagnostics = VideoPoseDiagnostics(
                    decodedFrameCount = 30,
                    sampledFrameCount = 30,
                    detectedFrameCount = 30,
                    noPersonFrameCount = 0,
                    unusableTrackingFrameCount = 0,
                    decodeNanos = 1_000_000,
                    inferenceNanos = 2_000_000,
                    maxInFlightFrames = 2,
                    peakInFlightFrames = 1,
                ),
            )
        }
    }
}
