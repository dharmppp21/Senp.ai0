package ai.senp.sync.v2

import ai.senp.alignment.temporal.TemporalComputationStats
import ai.senp.alignment.temporal.TemporalFeatureSample
import ai.senp.alignment.temporal.TemporalSignalFrame
import ai.senp.alignment.temporal.TemporalSignalSequence
import ai.senp.alignment.temporal.TemporalSynchronizationConfig
import ai.senp.alignment.temporal.TemporalSynchronizationEngine
import ai.senp.core.contracts.ActivitySegmentKind
import ai.senp.core.contracts.BodySideHypothesis
import ai.senp.core.contracts.CanonicalObservationSequence
import ai.senp.core.contracts.MirrorHypothesis
import ai.senp.core.contracts.MotionUnitCorrespondence
import ai.senp.core.contracts.RelativeViewHypothesis
import ai.senp.core.contracts.SpatialDiagnosticReason
import ai.senp.core.contracts.SpatialReliabilitySegment
import ai.senp.core.contracts.SpatialReliabilityStatus
import ai.senp.core.contracts.SpatialSynchronizationDiagnostics
import ai.senp.core.contracts.SynchronizationRefusal
import ai.senp.core.contracts.SynchronizationRefusalReason
import ai.senp.core.contracts.SynchronizationRequest
import ai.senp.core.contracts.SynchronizationResult
import ai.senp.core.contracts.SynchronizationStatus
import ai.senp.core.contracts.TimestampCorrespondence
import ai.senp.core.contracts.TimestampMs
import ai.senp.core.contracts.TemporalStructure
import ai.senp.core.contracts.TimestampRange
import ai.senp.core.contracts.UnitQuaternion
import ai.senp.core.contracts.VideoRole
import ai.senp.motion.SpatialIntrinsicDescriptor
import ai.senp.motion.SpatialObservationFrame
import ai.senp.motion.SpatialSequenceAnalysis
import ai.senp.motion.SpatialSynchronizationConfig
import ai.senp.motion.SpatialSynchronizationEngine
import ai.senp.motion.SpatialSynchronizationOutput
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.min
import kotlin.math.sqrt

object SynchronizationKernelV2Versions {
    const val ENGINE: String = "sync-v2-integration/1"
}

data class SynchronizationKernelV2Config(
    val maximumRefinementIterations: Int = 1,
    val materialImprovementThreshold: Double = 0.015,
    val minimumPairConfidenceForViewRefinement: Double = 0.42,
    val minimumPairsForViewRefinement: Int = 3,
    val minimumSynchronizedAnalyzableFraction: Double = 0.65,
    val mirrorEvidenceMargin: Double = 0.04,
    val discontinuityGapMs: Long = 650L,
) {
    init {
        require(maximumRefinementIterations in 1..2) { "Sync-v2 refinement is deliberately bounded to one or two passes" }
        require(materialImprovementThreshold.isFinite() && materialImprovementThreshold >= 0.0)
        require(minimumPairConfidenceForViewRefinement in 0.0..1.0)
        require(minimumPairsForViewRefinement >= 2)
        require(minimumSynchronizedAnalyzableFraction in 0.0..1.0)
        require(mirrorEvidenceMargin.isFinite() && mirrorEvidenceMargin >= 0.0)
        require(discontinuityGapMs > 0L)
    }
}

data class SynchronizationIterationDiagnostics(
    val iteration: Int,
    val phase: String,
    val status: SynchronizationStatus,
    val quality: Double,
    val overallConfidence: Double,
    val spatialConfidence: Double,
    val correspondenceConfidence: Double,
    val correspondenceAmbiguity: Double,
    val matchedUnitCount: Int,
    val matchedTimestampCount: Int,
    val pairedSpatialEvidenceCount: Int,
    val refinedHypothesisCount: Int,
    val temporalStats: TemporalComputationStats,
)

data class SynchronizationKernelStats(
    val iterations: List<SynchronizationIterationDiagnostics>,
    val totalCoarseUnitComparisons: Long,
    val totalFineCellsEvaluated: Long,
    val maximumFineBandWidth: Int,
    val totalFineAlignmentCount: Int,
) {
    val iterationCount: Int get() = iterations.size
}

data class SynchronizationMappingDiagnostic(
    val sourceTimestamp: TimestampMs,
    val referenceTimestamp: TimestampMs?,
    val sourceUnitId: String,
    val referenceUnitId: String?,
    val decisionConfidence: Double,
    val sourceDirection: Int?,
    val referenceDirection: Int?,
    val sourceState: ActivitySegmentKind?,
    val referenceState: ActivitySegmentKind?,
    val sourceReliability: SpatialReliabilityStatus?,
    val referenceReliability: SpatialReliabilityStatus?,
)

data class SynchronizationKernelRun(
    val result: SynchronizationResult,
    val stats: SynchronizationKernelStats,
    val spatialOutput: SpatialSynchronizationOutput,
    val mappingDiagnostics: List<SynchronizationMappingDiagnostic>,
)

/**
 * Production Sync-v2 composition. The first temporal pass uses view-tolerant body-centric articulation only.
 * Its matched timestamp evidence then validates/refines segment-stable spatial hypotheses; refined spatial state
 * selects/weights the articulation representation for one bounded refined temporal pass. Refusal, stable near-identity,
 * or lack of paired spatial evidence terminates early so refinement cannot oscillate or add unproductive search.
 */
class SynchronizationKernelV2(
    private val config: SynchronizationKernelV2Config = SynchronizationKernelV2Config(),
    private val spatialEngine: SpatialSynchronizationEngine = SpatialSynchronizationEngine(
        SpatialSynchronizationConfig(shortOcclusionMs = 350L),
    ),
    private val temporalEngine: TemporalSynchronizationEngine = TemporalSynchronizationEngine(
        TemporalSynchronizationConfig(
            minimumHoldMs = 180L,
            restGapMs = 520L,
            bridgeQuietGapMs = 80L,
            maximumBridgeUnreliableGapMs = 350L,
            motionSmoothingRadiusMs = 0L,
            derivativeContextMs = 80L,
            singleCycleEndpointToleranceFraction = 0.35,
            reliableDirectionMagnitude = 0.04,
            oppositeDirectionFraction = 0.30,
            oppositeDirectionCellPenalty = 2.25,
            coarseShiftSamples = 6,
            coarseMaximumCost = 2.20,
            activeVelocityMultiplier = 0.35,
            minimumNormalizedActiveSpeedPerSecond = 0.05,
            minimumMatchedTimestampFraction = 0.38,
            minimumMatchedUnitConfidence = 0.02,
            minimumCrossClassMatchedUnitConfidence = 0.10,
            minimumSynchronizedConfidence = 0.26,
            minimumCorrespondenceForPartial = 0.02,
            maximumAmbiguityForConfidentMatch = 0.97,
            maximumCrossClassAmbiguityForMatch = 0.90,
        ),
    ),
) {
    /** Production temporal-only seam over spatially canonicalized pose evidence; no raw/oracle channels are consumed. */
    fun analyzeTemporal(sequence: CanonicalObservationSequence): TemporalStructure {
        val source = if (sequence.role == VideoRole.SOURCE) sequence else sequence.copy(role = VideoRole.SOURCE)
        val reference = source.copy(role = VideoRole.REFERENCE)
        val request = SynchronizationRequest(source = source, reference = reference)
        val spatial = spatialEngine.analyze(request)
        val sourceSignal = buildTemporalSignal(spatial.source, spatial.diagnostics, coarse = true)
        val referenceSignal = buildTemporalSignal(spatial.reference, spatial.diagnostics, coarse = true)
        return temporalEngine.synchronizeDetailed(
            request = request,
            spatialDiagnostics = spatial.diagnostics,
            sourceSignal = sourceSignal,
            referenceSignal = referenceSignal,
        ).result.sourceTemporalStructure
    }

    fun synchronize(request: SynchronizationRequest): SynchronizationKernelRun {
        val spatial = spatialEngine.analyze(request)
        val iterations = mutableListOf<Execution>()

        val coarseSource = buildTemporalSignal(spatial.source, spatial.diagnostics, coarse = true)
        val coarseReference = buildTemporalSignal(spatial.reference, spatial.diagnostics, coarse = true)
        val coarseTemporal = temporalEngine.synchronizeDetailed(
            request = request,
            spatialDiagnostics = spatial.diagnostics,
            sourceSignal = coarseSource,
            referenceSignal = coarseReference,
        )
        iterations += execution(
            iteration = 0,
            phase = "COARSE_TIME",
            result = truthfulnessPolicy(coarseTemporal.result),
            temporalStats = coarseTemporal.stats,
            pairedSpatialEvidenceCount = 0,
            refinedHypothesisCount = 0,
            sourceSignal = coarseSource,
            referenceSignal = coarseReference,
        )

        var acceptedDiagnostics = spatial.diagnostics
        var accepted = iterations.last()
        for (refinementIndex in 1..config.maximumRefinementIterations) {
            if (accepted.result.status == SynchronizationStatus.REFUSED || stableNearIdentity(accepted.result)) break
            val refinedSpatial = refineSpatial(spatial, acceptedDiagnostics, accepted.result)
            if (refinedSpatial.pairedEvidenceCount < 4 || refinedSpatial.refinedHypothesisCount == 0) break
            val candidateDiagnostics = refinedSpatial.diagnostics
            val refinedSource = buildTemporalSignal(spatial.source, candidateDiagnostics, coarse = false)
            val refinedReference = buildTemporalSignal(spatial.reference, candidateDiagnostics, coarse = false)
            val temporal = temporalEngine.synchronizeDetailed(
                request = request,
                spatialDiagnostics = candidateDiagnostics,
                sourceSignal = refinedSource,
                referenceSignal = refinedReference,
            )
            val next = execution(
                iteration = refinementIndex,
                phase = "REFINED_TIME",
                result = truthfulnessPolicy(temporal.result),
                temporalStats = temporal.stats,
                pairedSpatialEvidenceCount = refinedSpatial.pairedEvidenceCount,
                refinedHypothesisCount = refinedSpatial.refinedHypothesisCount,
                sourceSignal = refinedSource,
                referenceSignal = refinedReference,
            )
            iterations += next
            val improvement = next.quality - accepted.quality
            if (improvement < config.materialImprovementThreshold) break
            accepted = next
            acceptedDiagnostics = candidateDiagnostics
        }

        val final = accepted
        return SynchronizationKernelRun(
            result = final.result,
            stats = SynchronizationKernelStats(
                iterations = iterations.map(Execution::diagnostics),
                totalCoarseUnitComparisons = iterations.sumOf { it.temporalStats.coarseUnitComparisons },
                totalFineCellsEvaluated = iterations.sumOf { it.temporalStats.fineCellsEvaluated },
                maximumFineBandWidth = iterations.maxOfOrNull { it.temporalStats.maximumFineBandWidth } ?: 0,
                totalFineAlignmentCount = iterations.sumOf { it.temporalStats.fineAlignmentCount },
            ),
            spatialOutput = spatial.copy(diagnostics = final.result.spatialDiagnostics),
            mappingDiagnostics = buildMappingDiagnostics(final.result, final.sourceSignal, final.referenceSignal),
        )
    }

    private fun stableNearIdentity(result: SynchronizationResult): Boolean {
        if (result.diagnostics.sourceAnalyzableFraction < 0.80 || result.diagnostics.referenceAnalyzableFraction < 0.80) return false
        val matched = result.correspondences.filterIsInstance<MotionUnitCorrespondence.MatchedUnit>()
            .flatMap { unit -> unit.timeline.filterIsInstance<TimestampCorrespondence.Matched>() }
        if (matched.size < 8) return false
        val nearIdentityFraction = matched.count { decision ->
            abs(decision.sourceTimestamp.value - decision.referenceTimestamp.value) <= 100L
        }.toDouble() / matched.size.toDouble()
        if (nearIdentityFraction < 0.95) return false
        val viewIsNearIdentity = result.spatialDiagnostics.relativeViewHypotheses.all { hypothesis ->
            hypothesis.mirror != MirrorHypothesis.MIRRORED &&
                abs(hypothesis.relativeYawDegrees ?: 0.0) <= 5.0 &&
                abs(hypothesis.relativeElevationDegrees ?: 0.0) <= 5.0
        }
        return viewIsNearIdentity
    }

    private fun buildTemporalSignal(
        sequence: SpatialSequenceAnalysis,
        diagnostics: SpatialSynchronizationDiagnostics,
        coarse: Boolean,
    ): TemporalSignalSequence {
        var previousSubject: String? = null
        var previousTimestamp: TimestampMs? = null
        val frames = sequence.frames.map { frame ->
            val hypothesis = hypothesisAt(sequence.role, frame.timestamp, diagnostics.relativeViewHypotheses)
            val samples = linkedMapOf<String, MutableList<TemporalFeatureSample>>()
            frame.canonicalPose?.let { channel ->
                val xIndex = channel.componentAxes.indexOfFirst { it.equals("x", ignoreCase = true) }
                val yIndex = channel.componentAxes.indexOfFirst { it.equals("y", ignoreCase = true) }
                val zIndex = channel.componentAxes.indexOfFirst { it.equals("z", ignoreCase = true) }
                val axes = buildList {
                    if (xIndex >= 0) add(xIndex)
                    if (yIndex >= 0) add(yIndex)
                    if (zIndex >= 0 && channel.coordinateSpace?.contains("3d", ignoreCase = true) == true) add(zIndex)
                }
                channel.values.forEach { value ->
                    if (!sideAllowed(value.key, hypothesis?.selectedBodySide, coarse)) return@forEach
                    val keyBase = if (coarse) bilateralKey(value.key) else value.key
                    axes.forEach { componentIndex ->
                        if (!value.mask[componentIndex]) return@forEach
                        val component = value.values[componentIndex] ?: return@forEach
                        if (!component.isFinite()) return@forEach
                        val axis = channel.componentAxes[componentIndex].lowercase()
                        val normalized = if (axis == "x") abs(component) else component
                        val key = "pose/$keyBase/$axis"
                        val confidence = min(channel.confidence, value.confidence) *
                            (0.90 + 0.10 * frame.transformConfidence)
                        samples.getOrPut(key) { mutableListOf() } += TemporalFeatureSample(normalized, confidence)
                    }
                }
            }
            appendIntrinsicFeatures(samples, frame.intrinsicDescriptor, hypothesis?.selectedBodySide, coarse)
            val features = samples.mapValues { (_, values) ->
                TemporalFeatureSample(
                    value = values.map(TemporalFeatureSample::value).average(),
                    confidence = values.map(TemporalFeatureSample::confidence).average().coerceIn(0.0, 1.0),
                )
            }
            val reliability = reliabilitySegmentAt(sequence.role, frame.timestamp, diagnostics)
            val status = reliability?.status
            val reliabilityFactor = when (status) {
                SpatialReliabilityStatus.COMPATIBLE, null -> 1.0
                SpatialReliabilityStatus.UNRELIABLE -> when {
                    SpatialDiagnosticReason.POOR_OBSERVATION_COVERAGE in requireNotNull(reliability).reasons -> 0.65
                    SpatialDiagnosticReason.TRANSFORM_UNSTABLE in reliability.reasons ||
                        SpatialDiagnosticReason.CAMERA_MOVEMENT in reliability.reasons -> 0.78
                    else -> 0.95
                }
                SpatialReliabilityStatus.INCOMPATIBLE, SpatialReliabilityStatus.DISCONTINUITY -> 0.0
            }
            val evidenceConfidence = features.values.maxOfOrNull(TemporalFeatureSample::confidence) ?: 0.0
            val timestampGap = previousTimestamp?.let { frame.timestamp.value - it.value } ?: 0L
            val subjectChanged = previousSubject != null && frame.selectedSubjectId != null && previousSubject != frame.selectedSubjectId
            val discontinuity = status == SpatialReliabilityStatus.DISCONTINUITY ||
                subjectChanged || timestampGap > config.discontinuityGapMs
            previousSubject = frame.selectedSubjectId
            previousTimestamp = frame.timestamp
            TemporalSignalFrame(
                timestamp = frame.timestamp,
                features = features,
                confidence = (evidenceConfidence * reliabilityFactor).coerceIn(0.0, 1.0),
                discontinuityBefore = discontinuity,
            )
        }
        return TemporalSignalSequence(sequence.role, sequence.duration, markArticulationDiscontinuities(frames))
    }

    private fun markArticulationDiscontinuities(frames: List<TemporalSignalFrame>): List<TemporalSignalFrame> {
        if (frames.size < 4) return frames
        val featureRanges = frames.flatMap { it.features.keys }.toSet().associateWith { key ->
            val values = frames.mapNotNull { frame -> frame.features[key]?.takeIf { it.confidence >= 0.30 }?.value }
            if (values.size < 3) 0.0 else (values.maxOrNull()!! - values.minOrNull()!!)
        }
        val dynamicKeys = featureRanges.filterValues { it >= 0.08 }.keys
        if (dynamicKeys.isEmpty()) return frames
        val jumps = frames.zipWithNext().map { (left, right) ->
            val normalized = dynamicKeys.mapNotNull { key ->
                val range = featureRanges.getValue(key)
                val a = left.features[key]?.takeIf { it.confidence >= 0.30 }?.value ?: return@mapNotNull null
                val b = right.features[key]?.takeIf { it.confidence >= 0.30 }?.value ?: return@mapNotNull null
                abs(b - a) / range
            }
            if (normalized.size < 2) {
                0.0
            } else {
                normalized.sortedDescending().take(maxOf(2, normalized.size / 4)).average()
            }
        }
        val positive = jumps.filter { it > 1e-9 }
        if (positive.isEmpty()) return frames
        val baseline = requireNotNull(median(positive))
        val threshold = maxOf(0.55, baseline * 2.6)
        return frames.mapIndexed { index, frame ->
            if (index == 0 || frame.discontinuityBefore) {
                frame
            } else {
                frame.copy(discontinuityBefore = jumps[index - 1] >= threshold)
            }
        }
    }

    private fun appendIntrinsicFeatures(
        output: MutableMap<String, MutableList<TemporalFeatureSample>>,
        descriptor: SpatialIntrinsicDescriptor,
        side: BodySideHypothesis?,
        coarse: Boolean,
    ) {
        descriptor.values.forEach { (rawKey, value) ->
            if (!sideAllowed(rawKey, side, coarse)) return@forEach
            val key = "intrinsic/" + if (coarse) bilateralKey(rawKey) else rawKey
            val normalizedValue = if (rawKey.startsWith("angle.")) value / 180.0 else value
            output.getOrPut(key) { mutableListOf() } += TemporalFeatureSample(normalizedValue, descriptor.confidence)
        }
    }

    private fun refineSpatial(
        spatial: SpatialSynchronizationOutput,
        previous: SpatialSynchronizationDiagnostics,
        temporal: SynchronizationResult,
    ): RefinedSpatial {
        val pairs = matchedPairs(temporal)
        var usedPairs = 0
        var refinedCount = 0
        val hypotheses = previous.relativeViewHypotheses.map { hypothesis ->
            val relevant = pairs.filter { hypothesis.sourceRange.contains(it.sourceTimestamp) }
            if (relevant.isEmpty()) return@map hypothesis.copy(confidence = hypothesis.confidence * 0.90)
            val pairedFrames = relevant.mapNotNull { pair ->
                val sourceFrame = nearestFrame(spatial.source.frames, pair.sourceTimestamp) ?: return@mapNotNull null
                val referenceFrame = nearestFrame(spatial.reference.frames, pair.referenceTimestamp) ?: return@mapNotNull null
                PairedFrames(pair, sourceFrame, referenceFrame)
            }
            if (pairedFrames.isEmpty()) return@map hypothesis.copy(confidence = hypothesis.confidence * 0.90)
            usedPairs += pairedFrames.size

            val meanPairConfidence = pairedFrames.map { pair ->
                pair.matchConfidence * min(pair.source.transformConfidence, pair.reference.transformConfidence)
            }.averageOrZero()
            val descriptorDistances = pairedFrames.mapNotNull { pair ->
                descriptorDistanceForHypothesis(pair.source.intrinsicDescriptor, pair.reference.intrinsicDescriptor, hypothesis.mirror)
            }
            val spatialAgreement = if (descriptorDistances.isEmpty()) 0.65 else exp(-2.0 * descriptorDistances.average()).coerceIn(0.0, 1.0)
            val selectedMirror = refineMirror(hypothesis.mirror, pairedFrames)
            val viewPairs = pairedFrames.mapNotNull { pair ->
                val sourceRotation = pair.source.rootOrientation?.inputToBodyRotation ?: return@mapNotNull null
                val referenceRotation = pair.reference.rootOrientation?.inputToBodyRotation ?: return@mapNotNull null
                yawElevation(multiplyQuaternion(inverseQuaternion(sourceRotation), referenceRotation))
            }
            val canRefineView = viewPairs.size >= config.minimumPairsForViewRefinement &&
                meanPairConfidence >= config.minimumPairConfidenceForViewRefinement
            val yaw = if (canRefineView) circularMedianDegrees(viewPairs.map(Pair<Double, Double>::first)) else hypothesis.relativeYawDegrees
            val elevation = if (canRefineView) median(viewPairs.map(Pair<Double, Double>::second)) else hypothesis.relativeElevationDegrees
            val referenceRange = pairedReferenceRange(pairedFrames, spatial.reference) ?: hypothesis.referenceRange
            val confidence = (
                0.45 * hypothesis.confidence + 0.40 * meanPairConfidence + 0.15 * spatialAgreement
                ).coerceIn(0.0, 1.0)
            refinedCount += 1
            hypothesis.copy(
                referenceRange = referenceRange,
                relativeYawDegrees = yaw,
                relativeElevationDegrees = elevation,
                mirror = selectedMirror,
                confidence = confidence,
            )
        }
        val pairedConfidence = pairs.map(MatchedPair::matchConfidence).averageOrZero()
        val aggregate = if (pairs.isEmpty()) previous.aggregateConfidence else {
            (0.85 * previous.aggregateConfidence + 0.15 * pairedConfidence).coerceIn(0.0, 1.0)
        }
        val pairedDiagnostics = previous.copy(relativeViewHypotheses = hypotheses, aggregateConfidence = aggregate)
        return RefinedSpatial(
            diagnostics = withTemporalDiscontinuities(pairedDiagnostics, temporal),
            pairedEvidenceCount = usedPairs,
            refinedHypothesisCount = refinedCount,
        )
    }

    private fun withTemporalDiscontinuities(
        diagnostics: SpatialSynchronizationDiagnostics,
        temporal: SynchronizationResult,
    ): SpatialSynchronizationDiagnostics {
        val cutsByRole = mapOf(
            VideoRole.SOURCE to temporal.sourceTemporalStructure.activitySegments
                .filter { it.kind == ActivitySegmentKind.DISCONTINUITY }
                .map { it.range },
            VideoRole.REFERENCE to temporal.referenceTemporalStructure.activitySegments
                .filter { it.kind == ActivitySegmentKind.DISCONTINUITY }
                .map { it.range },
        )
        if (cutsByRole.values.all(List<TimestampRange>::isEmpty)) return diagnostics
        val reliability = VideoRole.entries.flatMap { role ->
            overlayDiscontinuities(
                diagnostics.reliabilitySegments.filter { it.role == role },
                cutsByRole.getValue(role),
            )
        }
        return diagnostics.copy(reliabilitySegments = reliability)
    }

    private fun overlayDiscontinuities(
        segments: List<SpatialReliabilitySegment>,
        cuts: List<TimestampRange>,
    ): List<SpatialReliabilitySegment> {
        if (segments.isEmpty() || cuts.isEmpty()) return segments
        return segments.flatMap { segment ->
            val relevantCuts = cuts.filter { cut ->
                cut.start < segment.range.endExclusive && cut.endExclusive > segment.range.start
            }
            if (relevantCuts.isEmpty()) return@flatMap listOf(segment)
            val boundaries = buildSet {
                add(segment.range.start.value)
                add(segment.range.endExclusive.value)
                relevantCuts.forEach { cut ->
                    add(maxOf(segment.range.start.value, cut.start.value))
                    add(minOf(segment.range.endExclusive.value, cut.endExclusive.value))
                }
            }.sorted()
            boundaries.zipWithNext().mapNotNull { (start, end) ->
                if (end <= start) return@mapNotNull null
                val range = TimestampRange(TimestampMs(start), TimestampMs(end))
                val discontinuity = relevantCuts.any { cut ->
                    cut.start.value < end && cut.endExclusive.value > start
                }
                if (!discontinuity) {
                    segment.copy(range = range)
                } else {
                    SpatialReliabilitySegment(
                        role = segment.role,
                        range = range,
                        status = SpatialReliabilityStatus.DISCONTINUITY,
                        confidence = min(segment.confidence, 0.45),
                        reasons = segment.reasons + SpatialDiagnosticReason.TRANSFORM_UNSTABLE,
                    )
                }
            }
        }
    }

    private fun refineMirror(current: MirrorHypothesis, pairs: List<PairedFrames>): MirrorHypothesis {
        if (current == MirrorHypothesis.MIRRORED || current == MirrorHypothesis.NOT_MIRRORED) return current
        val direct = pairs.mapNotNull { it.source.intrinsicDescriptor.distanceTo(it.reference.intrinsicDescriptor, mirroredOther = false) }
        val mirrored = pairs.mapNotNull { it.source.intrinsicDescriptor.distanceTo(it.reference.intrinsicDescriptor, mirroredOther = true) }
        if (direct.size < config.minimumPairsForViewRefinement || mirrored.size < config.minimumPairsForViewRefinement) return current
        val directMean = direct.average()
        val mirroredMean = mirrored.average()
        if (abs(directMean - mirroredMean) < config.mirrorEvidenceMargin) return current
        return if (mirroredMean < directMean) MirrorHypothesis.MIRRORED else MirrorHypothesis.NOT_MIRRORED
    }

    private fun descriptorDistanceForHypothesis(
        source: SpatialIntrinsicDescriptor,
        reference: SpatialIntrinsicDescriptor,
        mirror: MirrorHypothesis,
    ): Double? = when (mirror) {
        MirrorHypothesis.MIRRORED -> source.distanceTo(reference, mirroredOther = true)
        MirrorHypothesis.NOT_MIRRORED -> source.distanceTo(reference, mirroredOther = false)
        MirrorHypothesis.AMBIGUOUS, MirrorHypothesis.UNKNOWN -> listOfNotNull(
            source.distanceTo(reference, mirroredOther = false),
            source.distanceTo(reference, mirroredOther = true),
        ).minOrNull()
    }

    private fun pairedReferenceRange(pairs: List<PairedFrames>, reference: SpatialSequenceAnalysis): TimestampRange? {
        if (pairs.isEmpty() || reference.frames.isEmpty()) return null
        val minTimestamp = pairs.minOf { it.reference.timestamp.value }
        val maxTimestamp = pairs.maxOf { it.reference.timestamp.value }
        val next = reference.frames.firstOrNull { it.timestamp.value > maxTimestamp }?.timestamp?.value
            ?: reference.duration.value
        val end = next.coerceAtLeast(maxTimestamp + 1L).coerceAtMost(reference.duration.value)
        if (end <= minTimestamp) return null
        return TimestampRange(TimestampMs(minTimestamp), TimestampMs(end))
    }

    private fun truthfulnessPolicy(result: SynchronizationResult): SynchronizationResult {
        val subjectAmbiguous = result.spatialDiagnostics.reliabilitySegments.any { segment ->
            SpatialDiagnosticReason.SUBJECT_AMBIGUITY in segment.reasons
        }
        if (
            result.status == SynchronizationStatus.REFUSED &&
            result.refusal?.reason == SynchronizationRefusalReason.SPATIAL_INCOMPATIBILITY &&
            subjectAmbiguous
        ) {
            return result.copy(
                refusal = SynchronizationRefusal(
                    SynchronizationRefusalReason.SUBJECT_AMBIGUITY,
                    "subject identity is spatially ambiguous; synchronization is not defensible",
                ),
            )
        }
        val analyzable = min(result.diagnostics.sourceAnalyzableFraction, result.diagnostics.referenceAnalyzableFraction)
        if (
            result.status == SynchronizationStatus.REFUSED &&
            result.refusal?.reason == SynchronizationRefusalReason.NO_COMMON_MOTION
        ) {
            if (analyzable < 0.55) {
                return result.copy(
                    refusal = SynchronizationRefusal(
                        SynchronizationRefusalReason.INSUFFICIENT_OBSERVATIONS,
                        "reliable observation coverage is too low to decide whether common motion exists",
                    ),
                )
            }
            val sameObservedStructure = result.sourceTemporalStructure.classification ==
                result.referenceTemporalStructure.classification &&
                result.sourceTemporalStructure.motionUnits.isNotEmpty() &&
                result.referenceTemporalStructure.motionUnits.isNotEmpty()
            if (sameObservedStructure) {
                return result.copy(
                    refusal = SynchronizationRefusal(
                        SynchronizationRefusalReason.UNRELIABLE_CORRESPONDENCE,
                        "both clips contain comparable motion structure, but direction/local correspondence evidence is incompatible",
                    ),
                )
            }
        }
        return if (result.status == SynchronizationStatus.SYNCHRONIZED && analyzable < config.minimumSynchronizedAnalyzableFraction) {
            result.copy(status = SynchronizationStatus.PARTIAL)
        } else {
            result
        }
    }

    private fun execution(
        iteration: Int,
        phase: String,
        result: SynchronizationResult,
        temporalStats: TemporalComputationStats,
        pairedSpatialEvidenceCount: Int,
        refinedHypothesisCount: Int,
        sourceSignal: TemporalSignalSequence,
        referenceSignal: TemporalSignalSequence,
    ): Execution {
        val matched = result.correspondences.filterIsInstance<MotionUnitCorrespondence.MatchedUnit>()
        val matchedTimestamps = matched.sumOf { unit -> unit.timeline.count { it is TimestampCorrespondence.Matched } }
        val quality = quality(result, matched.size)
        return Execution(
            result = result,
            temporalStats = temporalStats,
            quality = quality,
            sourceSignal = sourceSignal,
            referenceSignal = referenceSignal,
            diagnostics = SynchronizationIterationDiagnostics(
                iteration = iteration,
                phase = phase,
                status = result.status,
                quality = quality,
                overallConfidence = result.diagnostics.overallConfidence,
                spatialConfidence = result.diagnostics.spatialConfidence,
                correspondenceConfidence = result.diagnostics.correspondenceConfidence,
                correspondenceAmbiguity = result.diagnostics.correspondenceAmbiguity,
                matchedUnitCount = matched.size,
                matchedTimestampCount = matchedTimestamps,
                pairedSpatialEvidenceCount = pairedSpatialEvidenceCount,
                refinedHypothesisCount = refinedHypothesisCount,
                temporalStats = temporalStats,
            ),
        )
    }

    private fun quality(result: SynchronizationResult, matchedUnits: Int): Double {
        val unitCount = result.sourceTemporalStructure.motionUnits.size.coerceAtLeast(1)
        val matchedFraction = matchedUnits.toDouble() / unitCount.toDouble()
        val analyzable = min(result.diagnostics.sourceAnalyzableFraction, result.diagnostics.referenceAnalyzableFraction)
        val status = when (result.status) {
            SynchronizationStatus.SYNCHRONIZED -> 1.0
            SynchronizationStatus.PARTIAL -> 0.65
            SynchronizationStatus.REFUSED -> 0.0
        }
        return (
            0.28 * status + 0.25 * result.diagnostics.overallConfidence +
                0.20 * result.diagnostics.correspondenceConfidence + 0.12 * result.diagnostics.spatialConfidence +
                0.10 * matchedFraction + 0.05 * analyzable - 0.08 * result.diagnostics.correspondenceAmbiguity
            ).coerceIn(0.0, 1.0)
    }

    private fun buildMappingDiagnostics(
        result: SynchronizationResult,
        sourceSignal: TemporalSignalSequence,
        referenceSignal: TemporalSignalSequence,
    ): List<SynchronizationMappingDiagnostic> {
        val directionFeature = dominantDirectionFeature(sourceSignal, referenceSignal)
        return result.correspondences.filterIsInstance<MotionUnitCorrespondence.MatchedUnit>().flatMap { unit ->
            unit.timeline.map { decision ->
                val referenceTimestamp = (decision as? TimestampCorrespondence.Matched)?.referenceTimestamp
                val rawSourceDirection = directionFeature?.let { directionAt(sourceSignal, decision.sourceTimestamp, it) }
                val rawReferenceDirection = referenceTimestamp?.let { timestamp ->
                    directionFeature?.let { directionAt(referenceSignal, timestamp, it) }
                }
                val scalarDirectionAmbiguous = rawSourceDirection in setOf(-1, 1) &&
                    rawReferenceDirection in setOf(-1, 1) && rawReferenceDirection?.let { rawSourceDirection == -it } == true
                SynchronizationMappingDiagnostic(
                    sourceTimestamp = decision.sourceTimestamp,
                    referenceTimestamp = referenceTimestamp,
                    sourceUnitId = unit.sourceUnitId,
                    referenceUnitId = unit.referenceUnitId,
                    decisionConfidence = decision.decisionConfidence,
                    sourceDirection = rawSourceDirection.takeUnless { scalarDirectionAmbiguous },
                    referenceDirection = rawReferenceDirection.takeUnless { scalarDirectionAmbiguous },
                    sourceState = activityStateAt(result.sourceTemporalStructure, decision.sourceTimestamp),
                    referenceState = referenceTimestamp?.let { activityStateAt(result.referenceTemporalStructure, it) },
                    sourceReliability = reliabilitySegmentAt(
                        VideoRole.SOURCE,
                        decision.sourceTimestamp,
                        result.spatialDiagnostics,
                    )?.status,
                    referenceReliability = referenceTimestamp?.let { timestamp ->
                        reliabilitySegmentAt(VideoRole.REFERENCE, timestamp, result.spatialDiagnostics)?.status
                    },
                )
            }
        }
    }

    private fun dominantDirectionFeature(
        source: TemporalSignalSequence,
        reference: TemporalSignalSequence,
    ): DirectionFeature? {
        val sourceKeys = source.frames.flatMap { it.features.keys }.toSet()
        val commonKeys = sourceKeys.intersect(reference.frames.flatMap { it.features.keys }.toSet())
        return commonKeys.mapNotNull { key ->
            val sourceValues = source.frames.mapNotNull { it.features[key]?.takeIf { sample -> sample.confidence >= 0.30 }?.value }
            val referenceValues = reference.frames.mapNotNull { it.features[key]?.takeIf { sample -> sample.confidence >= 0.30 }?.value }
            if (sourceValues.size < 4 || referenceValues.size < 4) return@mapNotNull null
            val sourceRange = sourceValues.maxOrNull()!! - sourceValues.minOrNull()!!
            val referenceRange = referenceValues.maxOrNull()!! - referenceValues.minOrNull()!!
            val commonRange = min(sourceRange, referenceRange)
            if (!commonRange.isFinite() || commonRange < 1e-4) return@mapNotNull null
            val coverage = min(
                sourceValues.size.toDouble() / source.frames.size.coerceAtLeast(1),
                referenceValues.size.toDouble() / reference.frames.size.coerceAtLeast(1),
            )
            DirectionFeature(key, commonRange, commonRange * coverage)
        }.maxByOrNull(DirectionFeature::score)
    }

    private fun directionAt(
        signal: TemporalSignalSequence,
        timestamp: TimestampMs,
        feature: DirectionFeature,
    ): Int? {
        if (signal.frames.size < 2) return null
        val index = signal.frames.indices.minByOrNull { abs(signal.frames[it].timestamp.value - timestamp.value) } ?: return null
        val leftIndex = (index - 1).coerceAtLeast(0)
        val rightIndex = (index + 1).coerceAtMost(signal.frames.lastIndex)
        if (leftIndex == rightIndex) return 0
        val left = signal.frames[leftIndex]
        val right = signal.frames[rightIndex]
        if (left.confidence < 0.28 || right.confidence < 0.28 || right.discontinuityBefore) return null
        val a = left.features[feature.key]?.takeIf { it.confidence >= 0.30 }?.value ?: return null
        val b = right.features[feature.key]?.takeIf { it.confidence >= 0.30 }?.value ?: return null
        val seconds = (right.timestamp.value - left.timestamp.value).toDouble() / 1000.0
        if (seconds <= 0.0) return null
        val normalizedSpeed = (b - a) / feature.range / seconds
        return when {
            normalizedSpeed > 0.12 -> 1
            normalizedSpeed < -0.12 -> -1
            else -> 0
        }
    }

    private fun activityStateAt(structure: TemporalStructure, timestamp: TimestampMs): ActivitySegmentKind? =
        structure.activitySegments.firstOrNull { it.range.contains(timestamp) }?.kind

    private fun matchedPairs(result: SynchronizationResult): List<MatchedPair> = result.correspondences
        .filterIsInstance<MotionUnitCorrespondence.MatchedUnit>()
        .flatMap { unit ->
            unit.timeline.filterIsInstance<TimestampCorrespondence.Matched>().map { match ->
                MatchedPair(match.sourceTimestamp, match.referenceTimestamp, match.decisionConfidence)
            }
        }

    private fun nearestFrame(frames: List<SpatialObservationFrame>, timestamp: TimestampMs): SpatialObservationFrame? =
        frames.minByOrNull { abs(it.timestamp.value - timestamp.value) }

    private fun hypothesisAt(
        role: VideoRole,
        timestamp: TimestampMs,
        hypotheses: List<RelativeViewHypothesis>,
    ): RelativeViewHypothesis? = when (role) {
        VideoRole.SOURCE -> hypotheses.firstOrNull { it.sourceRange.contains(timestamp) }
        VideoRole.REFERENCE -> hypotheses.filter { it.referenceRange.contains(timestamp) }.maxByOrNull(RelativeViewHypothesis::confidence)
    }

    private fun reliabilitySegmentAt(
        role: VideoRole,
        timestamp: TimestampMs,
        diagnostics: SpatialSynchronizationDiagnostics,
    ): SpatialReliabilitySegment? = diagnostics.reliabilitySegments
        .firstOrNull { it.role == role && it.range.contains(timestamp) }

    private fun sideAllowed(key: String, side: BodySideHypothesis?, coarse: Boolean): Boolean {
        if (coarse || side == null || side == BodySideHypothesis.BILATERAL || side == BodySideHypothesis.UNKNOWN) return true
        val lower = key.lowercase()
        return when (side) {
            BodySideHypothesis.LEFT -> "right_" !in lower
            BodySideHypothesis.RIGHT -> "left_" !in lower
            else -> true
        }
    }

    private fun bilateralKey(key: String): String = key
        .replace("left_", "side_", ignoreCase = true)
        .replace("right_", "side_", ignoreCase = true)

    private fun inverseQuaternion(value: UnitQuaternion): UnitQuaternion = UnitQuaternion(-value.x, -value.y, -value.z, value.w)

    private fun multiplyQuaternion(left: UnitQuaternion, right: UnitQuaternion): UnitQuaternion {
        val x = left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y
        val y = left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x
        val z = left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w
        val w = left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z
        val norm = sqrt(x * x + y * y + z * z + w * w)
        require(norm.isFinite() && norm > 1e-12)
        return UnitQuaternion(x / norm, y / norm, z / norm, w / norm)
    }

    private fun yawElevation(rotation: UnitQuaternion): Pair<Double, Double> {
        val xx = rotation.x * rotation.x
        val yy = rotation.y * rotation.y
        val xz = rotation.x * rotation.z
        val yz = rotation.y * rotation.z
        val wx = rotation.w * rotation.x
        val wy = rotation.w * rotation.y
        val m02 = 2.0 * (xz + wy)
        val m21 = 2.0 * (yz + wx)
        val m22 = 1.0 - 2.0 * (xx + yy)
        val yaw = normalizeDegrees(atan2(m02, m22) * 180.0 / PI)
        val elevation = asin(m21.coerceIn(-1.0, 1.0)) * 180.0 / PI
        return yaw to elevation.coerceIn(-90.0, 90.0)
    }

    private fun circularMedianDegrees(values: List<Double>): Double? {
        if (values.isEmpty()) return null
        val anchor = values.first()
        val unwrapped = values.map { value ->
            var candidate = value
            while (candidate - anchor > 180.0) candidate -= 360.0
            while (candidate - anchor < -180.0) candidate += 360.0
            candidate
        }
        return normalizeDegrees(requireNotNull(median(unwrapped)))
    }

    private fun normalizeDegrees(value: Double): Double {
        var result = value
        while (result > 180.0) result -= 360.0
        while (result < -180.0) result += 360.0
        return result
    }

    private fun median(values: List<Double>): Double? {
        if (values.isEmpty()) return null
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 1) sorted[middle] else (sorted[middle - 1] + sorted[middle]) / 2.0
    }

    private data class Execution(
        val result: SynchronizationResult,
        val temporalStats: TemporalComputationStats,
        val quality: Double,
        val sourceSignal: TemporalSignalSequence,
        val referenceSignal: TemporalSignalSequence,
        val diagnostics: SynchronizationIterationDiagnostics,
    )

    private data class RefinedSpatial(
        val diagnostics: SpatialSynchronizationDiagnostics,
        val pairedEvidenceCount: Int,
        val refinedHypothesisCount: Int,
    )

    private data class DirectionFeature(
        val key: String,
        val range: Double,
        val score: Double,
    )

    private data class MatchedPair(
        val sourceTimestamp: TimestampMs,
        val referenceTimestamp: TimestampMs,
        val matchConfidence: Double,
    )

    private data class PairedFrames(
        val pair: MatchedPair,
        val source: SpatialObservationFrame,
        val reference: SpatialObservationFrame,
    ) {
        val matchConfidence: Double get() = pair.matchConfidence
    }
}

private fun List<Double>.averageOrZero(): Double = if (isEmpty()) 0.0 else average()
