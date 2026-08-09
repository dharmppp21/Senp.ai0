package ai.senp.alignment.temporal

import ai.senp.core.contracts.DurationMs
import ai.senp.core.contracts.SynchronizationResult
import ai.senp.core.contracts.TimestampMs
import ai.senp.core.contracts.VideoRole

/** Stable implementation identifier for Synchronization Kernel v2 temporal/correspondence behavior. */
object TemporalSynchronizationVersions {
    const val ENGINE: String = "sync-v2-temporal/1"
}

data class TemporalFeatureSample(
    val value: Double,
    val confidence: Double,
) {
    init {
        require(value.isFinite()) { "temporal feature value must be finite" }
        requireProbability(confidence, "temporal feature confidence")
    }
}

/**
 * Generic timestamped temporal evidence. Integration may populate [features] from spatially canonicalized pose,
 * intrinsic articulation, learned descriptors, object state, or any future channel without activity-name logic.
 */
data class TemporalSignalFrame(
    val timestamp: TimestampMs,
    val features: Map<String, TemporalFeatureSample>,
    val confidence: Double,
    val discontinuityBefore: Boolean = false,
) {
    init {
        require(features.keys.all { it.isNotBlank() }) { "temporal feature keys must be non-blank" }
        requireProbability(confidence, "temporal frame confidence")
    }
}

data class TemporalSignalSequence(
    val role: VideoRole,
    val duration: DurationMs,
    val frames: List<TemporalSignalFrame>,
) {
    init {
        require(frames.zipWithNext().all { (left, right) -> left.timestamp < right.timestamp }) {
            "temporal signal timestamps must be strictly increasing"
        }
        require(frames.lastOrNull()?.timestamp?.value?.let { it < duration.value } ?: true) {
            "temporal signal timestamp must be before sequence duration"
        }
    }
}

data class TemporalSynchronizationConfig(
    val minimumFeatureConfidence: Double = 0.30,
    val minimumFrameConfidence: Double = 0.28,
    val minimumAnalyzableFraction: Double = 0.20,
    val discontinuityGapMs: Long = 650L,
    val motionSmoothingRadiusMs: Long = 100L,
    val activeVelocityQuantile: Double = 0.65,
    val activeVelocityMultiplier: Double = 0.55,
    val minimumNormalizedActiveSpeedPerSecond: Double = 0.10,
    val minimumActiveRunMs: Long = 220L,
    val bridgeQuietGapMs: Long = 160L,
    val maximumBridgeUnreliableGapMs: Long = 0L,
    val minimumHoldMs: Long = 320L,
    val restGapMs: Long = 900L,
    val minimumMotionUnitMs: Long = 260L,
    val minimumCycleMs: Long = 360L,
    val singleCycleEndpointToleranceFraction: Double = 0.20,
    val derivativeContextMs: Long = 160L,
    val temporalContextMs: Long = 240L,
    val reliableDirectionMagnitude: Double = 0.18,
    val oppositeDirectionFraction: Double = 0.55,
    val oppositeDirectionCellPenalty: Double = 0.75,
    val descriptorMinimumCoverage: Double = 0.35,
    val coarseSamplesPerUnit: Int = 12,
    val coarseShiftSamples: Int = 4,
    val coarseMaximumCost: Double = 1.10,
    val coarseShortlistSize: Int = 5,
    val unitSkipCost: Double = 0.72,
    val fineBandFraction: Double = 0.20,
    val fineStepPenalty: Double = 0.045,
    val fineMaximumCellCost: Double = 1.45,
    val minimumMatchedTimestampFraction: Double = 0.45,
    val minimumMatchedUnitConfidence: Double = 0.32,
    val minimumCrossClassMatchedUnitConfidence: Double = 0.10,
    val minimumSynchronizedConfidence: Double = 0.62,
    val minimumCorrespondenceForPartial: Double = 0.20,
    val minimumWarpSlope: Double = 0.16,
    val maximumWarpSlope: Double = 6.0,
    val slopeWindowMs: Long = 300L,
    val maximumFrozenMappingMs: Long = 550L,
    val maximumAmbiguityForConfidentMatch: Double = 0.92,
    val maximumCrossClassAmbiguityForMatch: Double = 0.90,
) {
    init {
        listOf(
            minimumFeatureConfidence,
            minimumFrameConfidence,
            minimumAnalyzableFraction,
            activeVelocityQuantile,
            activeVelocityMultiplier,
            singleCycleEndpointToleranceFraction,
            reliableDirectionMagnitude,
            oppositeDirectionFraction,
            descriptorMinimumCoverage,
            fineBandFraction,
            minimumMatchedTimestampFraction,
            minimumMatchedUnitConfidence,
            minimumCrossClassMatchedUnitConfidence,
            minimumSynchronizedConfidence,
            minimumCorrespondenceForPartial,
            maximumAmbiguityForConfidentMatch,
            maximumCrossClassAmbiguityForMatch,
        ).forEach { value -> requireProbability(value, "temporal synchronization probability/fraction") }
        require(discontinuityGapMs > 0L)
        require(motionSmoothingRadiusMs >= 0L)
        require(minimumNormalizedActiveSpeedPerSecond > 0.0)
        require(minimumActiveRunMs > 0L)
        require(bridgeQuietGapMs >= 0L)
        require(maximumBridgeUnreliableGapMs >= 0L)
        require(minimumHoldMs > 0L)
        require(restGapMs >= minimumHoldMs)
        require(minimumMotionUnitMs > 0L)
        require(minimumCycleMs >= minimumMotionUnitMs)
        require(derivativeContextMs > 0L)
        require(temporalContextMs >= derivativeContextMs)
        require(oppositeDirectionCellPenalty >= 0.0)
        require(coarseSamplesPerUnit >= 6)
        require(coarseShiftSamples >= 0)
        require(coarseMaximumCost > 0.0)
        require(coarseShortlistSize > 0)
        require(unitSkipCost > 0.0)
        require(fineStepPenalty >= 0.0)
        require(fineMaximumCellCost > 0.0)
        require(minimumWarpSlope > 0.0)
        require(maximumWarpSlope > minimumWarpSlope)
        require(slopeWindowMs > 0L)
        require(maximumFrozenMappingMs > 0L)
    }
}

data class TemporalComputationStats(
    val sourceFrameCount: Int,
    val referenceFrameCount: Int,
    val sourceUnitCount: Int,
    val referenceUnitCount: Int,
    val coarseUnitComparisons: Long,
    val coarseAcceptedCandidateCount: Int,
    val bestCoarseCandidateCost: Double?,
    val fineCellsEvaluated: Long,
    val finePathTimestampCount: Long,
    val fineAcceptedTimestampCount: Long,
    val fineRejectedOppositeDirectionCount: Long,
    val fineRejectedCostCount: Long,
    val fineRejectedCoverageCount: Long,
    val fineRejectedConfidenceCount: Long,
    val fineRejectedWarpCount: Long,
    val bestFineMatchedFraction: Double?,
    val bestFineDecisionConfidence: Double?,
    val maximumFineBandWidth: Int,
    val fineAlignmentCount: Int,
) {
    init {
        require(
            listOf(
                sourceFrameCount,
                referenceFrameCount,
                sourceUnitCount,
                referenceUnitCount,
                coarseAcceptedCandidateCount,
                maximumFineBandWidth,
                fineAlignmentCount,
            )
                .all { it >= 0 },
        )
        require(coarseUnitComparisons >= 0L)
        bestCoarseCandidateCost?.let { require(it.isFinite() && it >= 0.0) }
        require(
            listOf(
                fineCellsEvaluated,
                finePathTimestampCount,
                fineAcceptedTimestampCount,
                fineRejectedOppositeDirectionCount,
                fineRejectedCostCount,
                fineRejectedCoverageCount,
                fineRejectedConfidenceCount,
                fineRejectedWarpCount,
            ).all { it >= 0L },
        )
        bestFineMatchedFraction?.let { requireProbability(it, "best fine matched fraction") }
        bestFineDecisionConfidence?.let { requireProbability(it, "best fine decision confidence") }
    }

    val naiveWholeVideoCellCount: Long
        get() = sourceFrameCount.toLong() * referenceFrameCount.toLong()
}

data class TemporalSynchronizationRun(
    val result: SynchronizationResult,
    val stats: TemporalComputationStats,
)

internal fun requireProbability(value: Double, name: String) {
    require(value.isFinite() && value in 0.0..1.0) { "$name must be finite and in [0, 1]" }
}
