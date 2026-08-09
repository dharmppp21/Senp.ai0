package ai.senp.sync.v2

import ai.senp.core.contracts.AnalysisFailure
import ai.senp.core.contracts.PipelineStageId
import ai.senp.core.contracts.PoseModelConfiguration
import ai.senp.core.contracts.SamplingConfiguration
import ai.senp.core.contracts.StageResult
import ai.senp.core.contracts.SynchronizationRequirements
import ai.senp.core.contracts.SynchronizationRequest
import ai.senp.core.contracts.SynchronizationSemantics
import ai.senp.core.contracts.VideoPoseExtraction
import ai.senp.core.contracts.VideoPoseFailureKind
import ai.senp.core.contracts.VideoRole
import ai.senp.core.contracts.VideoSource
import ai.senp.core.pipeline.VideoPoseExtractor
import java.util.LinkedHashMap

fun interface SynchronizationNanoClock {
    fun nanoTime(): Long
}

data class VideoSynchronizationRequest(
    val source: VideoSource,
    val reference: VideoSource,
    val sampling: SamplingConfiguration = SamplingConfiguration(),
    val model: PoseModelConfiguration,
    val semantics: SynchronizationSemantics = SynchronizationSemantics(),
    val requirements: SynchronizationRequirements = SynchronizationRequirements(),
)

data class VideoSynchronizationTimings(
    val sourcePoseExtractionNanos: Long,
    val referencePoseExtractionNanos: Long,
    val postPoseSynchronizationNanos: Long,
    val totalNanos: Long,
    val sourceDecodeNanos: Long,
    val sourceInferenceNanos: Long,
    val referenceDecodeNanos: Long,
    val referenceInferenceNanos: Long,
    val sourcePoseCacheHit: Boolean,
    val referencePoseCacheHit: Boolean,
) {
    init {
        require(
            listOf(
                sourcePoseExtractionNanos,
                referencePoseExtractionNanos,
                postPoseSynchronizationNanos,
                totalNanos,
                sourceDecodeNanos,
                sourceInferenceNanos,
                referenceDecodeNanos,
                referenceInferenceNanos,
            ).all { it >= 0L },
        )
    }

    val poseAndPreprocessingNanos: Long
        get() = sourcePoseExtractionNanos + referencePoseExtractionNanos

    val postPoseFraction: Double
        get() = if (totalNanos <= 0L) 0.0 else postPoseSynchronizationNanos.toDouble() / totalNanos.toDouble()
}

data class VideoSynchronizationRun(
    val synchronization: SynchronizationKernelRun,
    val sourcePoseExtraction: VideoPoseExtraction,
    val referencePoseExtraction: VideoPoseExtraction,
    val timings: VideoSynchronizationTimings,
)

sealed interface VideoSynchronizationOutcome {
    data class Success(val run: VideoSynchronizationRun) : VideoSynchronizationOutcome
    data class Failure(val failure: AnalysisFailure) : VideoSynchronizationOutcome
}

data class SynchronizationPoseCacheKey(
    val role: VideoRole,
    val videoSha256: String,
    val modelSha256: String,
    val modelVariant: String,
    val targetFramesPerSecond: Int,
    val longEdgeCapPx: Int,
    val minimumDetectionConfidence: Double,
    val minimumPresenceConfidence: Double,
    val minimumTrackingConfidence: Double,
)

interface SynchronizationPoseCache {
    suspend fun lookup(key: SynchronizationPoseCacheKey): VideoPoseExtraction?
    suspend fun store(key: SynchronizationPoseCacheKey, extraction: VideoPoseExtraction)
}

object NoopSynchronizationPoseCache : SynchronizationPoseCache {
    override suspend fun lookup(key: SynchronizationPoseCacheKey): VideoPoseExtraction? = null
    override suspend fun store(key: SynchronizationPoseCacheKey, extraction: VideoPoseExtraction) = Unit
}

class InMemorySynchronizationPoseCache(
    private val maximumEntries: Int = 8,
) : SynchronizationPoseCache {
    init {
        require(maximumEntries > 0)
    }

    private val entries = object : LinkedHashMap<SynchronizationPoseCacheKey, VideoPoseExtraction>(16, 0.75f, true) {
        override fun removeEldestEntry(
            eldest: MutableMap.MutableEntry<SynchronizationPoseCacheKey, VideoPoseExtraction>?,
        ): Boolean = size > maximumEntries
    }

    override suspend fun lookup(key: SynchronizationPoseCacheKey): VideoPoseExtraction? = synchronized(entries) {
        entries[key]
    }

    override suspend fun store(key: SynchronizationPoseCacheKey, extraction: VideoPoseExtraction) {
        synchronized(entries) {
            entries[key] = extraction
        }
    }

    fun size(): Int = synchronized(entries) { entries.size }
}

/**
 * Real-video Sync-v2 entry point. Decode/MediaPipe extraction remains behind the existing production
 * [VideoPoseExtractor] port; post-pose synchronization consumes only timestamped canonical observations.
 */
class VideoSynchronizationPipelineV2(
    private val videoPoseExtractor: VideoPoseExtractor,
    private val poseCache: SynchronizationPoseCache = NoopSynchronizationPoseCache,
    private val kernel: SynchronizationKernelV2 = SynchronizationKernelV2(),
    private val observationAdapter: PoseObservationAdapter = PoseObservationAdapter(),
    private val clock: SynchronizationNanoClock = SynchronizationNanoClock(System::nanoTime),
) {
    suspend fun synchronize(request: VideoSynchronizationRequest): VideoSynchronizationOutcome {
        val totalStart = clock.nanoTime()
        val source = when (val extracted = extract(VideoRole.SOURCE, request.source, request.sampling, request.model)) {
            is ExtractionOutcome.Failure -> return VideoSynchronizationOutcome.Failure(extracted.failure)
            is ExtractionOutcome.Success -> extracted
        }
        val reference = when (val extracted = extract(VideoRole.REFERENCE, request.reference, request.sampling, request.model)) {
            is ExtractionOutcome.Failure -> return VideoSynchronizationOutcome.Failure(extracted.failure)
            is ExtractionOutcome.Success -> extracted
        }

        return try {
            val postPoseStart = clock.nanoTime()
            val sourceObservations = observationAdapter.adapt(
                extraction = source.extraction,
                analysisFramesPerSecond = request.sampling.targetFramesPerSecond.toDouble(),
            )
            val referenceObservations = observationAdapter.adapt(
                extraction = reference.extraction,
                analysisFramesPerSecond = request.sampling.targetFramesPerSecond.toDouble(),
            )
            val synchronization = kernel.synchronize(
                SynchronizationRequest(
                    source = sourceObservations,
                    reference = referenceObservations,
                    semantics = request.semantics,
                    requirements = request.requirements,
                ),
            )
            val postPoseEnd = clock.nanoTime()
            VideoSynchronizationOutcome.Success(
                VideoSynchronizationRun(
                    synchronization = synchronization,
                    sourcePoseExtraction = source.extraction,
                    referencePoseExtraction = reference.extraction,
                    timings = VideoSynchronizationTimings(
                        sourcePoseExtractionNanos = source.elapsedNanos,
                        referencePoseExtractionNanos = reference.elapsedNanos,
                        postPoseSynchronizationNanos = elapsed(postPoseStart, postPoseEnd),
                        totalNanos = elapsed(totalStart, postPoseEnd),
                        sourceDecodeNanos = source.extraction.diagnostics.decodeNanos,
                        sourceInferenceNanos = source.extraction.diagnostics.inferenceNanos,
                        referenceDecodeNanos = reference.extraction.diagnostics.decodeNanos,
                        referenceInferenceNanos = reference.extraction.diagnostics.inferenceNanos,
                        sourcePoseCacheHit = source.cacheHit,
                        referencePoseCacheHit = reference.cacheHit,
                    ),
                ),
            )
        } catch (error: Exception) {
            VideoSynchronizationOutcome.Failure(
                AnalysisFailure.Unexpected(
                    stage = PipelineStageId.ALIGNMENT,
                    exceptionType = error::class.qualifiedName ?: "unknown",
                    message = error.message ?: "Sync-v2 post-pose synchronization failed",
                ),
            )
        }
    }

    private suspend fun extract(
        role: VideoRole,
        source: VideoSource,
        sampling: SamplingConfiguration,
        model: PoseModelConfiguration,
    ): ExtractionOutcome {
        val key = SynchronizationPoseCacheKey(
            role = role,
            videoSha256 = source.sha256.value,
            modelSha256 = model.modelSha256.value,
            modelVariant = model.modelVariant,
            targetFramesPerSecond = sampling.targetFramesPerSecond,
            longEdgeCapPx = sampling.longEdgeCapPx,
            minimumDetectionConfidence = model.thresholds.minimumDetectionConfidence,
            minimumPresenceConfidence = model.thresholds.minimumPresenceConfidence,
            minimumTrackingConfidence = model.thresholds.minimumTrackingConfidence,
        )
        poseCache.lookup(key)?.let { cached ->
            validateExtraction(role, cached)?.let { failure -> return ExtractionOutcome.Failure(failure) }
            return ExtractionOutcome.Success(cached, elapsedNanos = 0L, cacheHit = true)
        }

        val start = clock.nanoTime()
        val result = videoPoseExtractor.extract(role, source, sampling, model)
        val end = clock.nanoTime()
        return when (result) {
            is StageResult.Failure -> ExtractionOutcome.Failure(result.failure)
            is StageResult.Success -> {
                val extraction = result.value
                validateExtraction(role, extraction)?.let { return ExtractionOutcome.Failure(it) }
                poseCache.store(key, extraction)
                ExtractionOutcome.Success(extraction, elapsed(start, end), cacheHit = false)
            }
        }
    }

    private fun validateExtraction(role: VideoRole, extraction: VideoPoseExtraction): AnalysisFailure? = when {
        extraction.role != role -> AnalysisFailure.VideoPose(
            role,
            VideoPoseFailureKind.INFERENCE,
            "extractor returned ${extraction.role} for $role",
        )
        extraction.poses.frames.isEmpty() -> AnalysisFailure.VideoPose(
            role,
            VideoPoseFailureKind.INFERENCE,
            "extractor returned no sampled pose frames",
        )
        else -> null
    }

    private fun elapsed(start: Long, end: Long): Long = (end - start).coerceAtLeast(0L)

    private sealed interface ExtractionOutcome {
        data class Success(
            val extraction: VideoPoseExtraction,
            val elapsedNanos: Long,
            val cacheHit: Boolean,
        ) : ExtractionOutcome

        data class Failure(val failure: AnalysisFailure) : ExtractionOutcome
    }
}
