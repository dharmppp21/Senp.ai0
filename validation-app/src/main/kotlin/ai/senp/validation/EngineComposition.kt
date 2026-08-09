package ai.senp.validation

import ai.senp.core.cache.InMemoryAnalysisCache
import ai.senp.core.contracts.TimestampMs
import ai.senp.core.pipeline.AnalysisPipeline
import ai.senp.core.pipeline.TimestampFirstAlignmentEngine
import ai.senp.core.pipeline.TimestampFirstPhaseDetector
import ai.senp.motion.CoreMotionProcessor
import ai.senp.pose.mediapipe.AndroidVideoPoseExtractor
import ai.senp.sync.v2.InMemorySynchronizationPoseCache
import ai.senp.sync.v2.VideoSynchronizationPipelineV2
import android.content.Context
import android.os.SystemClock

/** Concrete, lifecycle-owned Android composition root used by headless validation. */
internal class EngineComposition(context: Context) {
    val videoPoseExtractor = AndroidVideoPoseExtractor(context)
    val motionProcessor = CoreMotionProcessor()
    val phaseDetector = TimestampFirstPhaseDetector()
    val alignmentEngine = TimestampFirstAlignmentEngine()
    val cache = InMemoryAnalysisCache(maximumEntries = 8)
    val synchronizationPoseCache = InMemorySynchronizationPoseCache(maximumEntries = 8)
    val synchronizationPipeline = VideoSynchronizationPipelineV2(
        videoPoseExtractor = videoPoseExtractor,
        poseCache = synchronizationPoseCache,
    )

    val pipeline = AnalysisPipeline(
        videoPoseExtractor = videoPoseExtractor,
        motionProcessor = motionProcessor,
        phaseDetector = phaseDetector,
        alignmentEngine = alignmentEngine,
        cache = cache,
        monotonicClock = { SystemClock.elapsedRealtime() },
        wallClock = { TimestampMs(System.currentTimeMillis()) },
        engineVersion = ENGINE_VERSION,
    )

    companion object {
        const val ENGINE_VERSION = "senp-android-e2e/1"
        const val PIPELINE_VERSION = "senp-analysis-pipeline/1"
    }
}
