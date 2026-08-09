package ai.senp.alignment.temporal

import ai.senp.core.contracts.ActivitySegmentKind
import ai.senp.core.contracts.CanonicalObservationSequence
import ai.senp.core.contracts.ChannelAvailability
import ai.senp.core.contracts.MotionStructureClass
import ai.senp.core.contracts.MotionUnitCorrespondence
import ai.senp.core.contracts.SpatialReliabilitySegment
import ai.senp.core.contracts.SpatialReliabilityStatus
import ai.senp.core.contracts.SpatialSynchronizationDiagnostics
import ai.senp.core.contracts.SynchronizationDiagnostics
import ai.senp.core.contracts.SynchronizationRefusal
import ai.senp.core.contracts.SynchronizationRefusalReason
import ai.senp.core.contracts.SynchronizationRequest
import ai.senp.core.contracts.SynchronizationResult
import ai.senp.core.contracts.SynchronizationSemantics
import ai.senp.core.contracts.SynchronizationStatus
import ai.senp.core.contracts.TemporalStructure
import ai.senp.core.contracts.UnitBoundaryStatus
import ai.senp.core.contracts.UnmatchedReason
import ai.senp.core.contracts.VideoRole
import kotlin.math.min
import kotlin.math.pow

/**
 * Public orchestration seam for the Synchronization Kernel v2 temporal/correspondence lane.
 * Spatial canonicalization remains an upstream responsibility; callers can supply spatially derived generic signals.
 */
class TemporalSynchronizationEngine(
    private val config: TemporalSynchronizationConfig = TemporalSynchronizationConfig(),
) {
    fun synchronize(
        request: SynchronizationRequest,
        spatialDiagnostics: SpatialSynchronizationDiagnostics,
        sourceSignal: TemporalSignalSequence? = null,
        referenceSignal: TemporalSignalSequence? = null,
    ): SynchronizationResult = synchronizeDetailed(
        request,
        spatialDiagnostics,
        sourceSignal,
        referenceSignal,
    ).result

    fun synchronizeDetailed(
        request: SynchronizationRequest,
        spatialDiagnostics: SpatialSynchronizationDiagnostics,
        sourceSignal: TemporalSignalSequence? = null,
        referenceSignal: TemporalSignalSequence? = null,
    ): TemporalSynchronizationRun {
        val adapter = CanonicalTemporalSignalAdapter(config)
        val source = sourceSignal ?: adapter.adapt(request.source, spatialDiagnostics)
        val reference = referenceSignal ?: adapter.adapt(request.reference, spatialDiagnostics)
        validateSignal(source, request.source, VideoRole.SOURCE)
        validateSignal(reference, request.reference, VideoRole.REFERENCE)

        val descriptorSpace = TemporalDescriptorSpace(source, reference, config)
        val preparedSource = descriptorSpace.prepare(source)
        val preparedReference = descriptorSpace.prepare(reference)
        val detector = TemporalStructureDetector(config)
        val sourceDetection = detector.detect(preparedSource)
        val referenceDetection = detector.detect(preparedReference)
        val stats = MutableTemporalStats(source.frames.size, reference.frames.size)

        val missingRequired = missingRequiredChannels(request)
        if (missingRequired.isNotEmpty()) {
            return refused(
                request,
                sourceDetection,
                referenceDetection,
                spatialDiagnostics,
                SynchronizationRefusalReason.REQUIRED_CHANNEL_MISSING,
                "required synchronization evidence is unavailable: ${missingRequired.sorted().joinToString()}",
                stats,
                missingRequired,
            )
        }
        if (spatiallyIncompatible(spatialDiagnostics)) {
            return refused(
                request,
                sourceDetection,
                referenceDetection,
                spatialDiagnostics,
                SynchronizationRefusalReason.SPATIAL_INCOMPATIBILITY,
                "spatial reliability does not support a defensible temporal correspondence",
                stats,
            )
        }
        if (
            source.frames.size < 2 || reference.frames.size < 2 ||
            sourceDetection.analyzableFraction < config.minimumAnalyzableFraction ||
            referenceDetection.analyzableFraction < config.minimumAnalyzableFraction
        ) {
            return refused(
                request,
                sourceDetection,
                referenceDetection,
                spatialDiagnostics,
                SynchronizationRefusalReason.INSUFFICIENT_OBSERVATIONS,
                "insufficient reliable timestamped observations for synchronization",
                stats,
            )
        }
        if (hasDisallowedOpenBoundary(request.semantics, sourceDetection.structure, referenceDetection.structure)) {
            return refused(
                request,
                sourceDetection,
                referenceDetection,
                spatialDiagnostics,
                SynchronizationRefusalReason.UNRELIABLE_CORRESPONDENCE,
                "observed partial motion requires an open boundary disallowed by the requested semantics",
                stats,
            )
        }
        if (sourceDetection.structure.motionUnits.isEmpty() || referenceDetection.structure.motionUnits.isEmpty()) {
            return refused(
                request,
                sourceDetection,
                referenceDetection,
                spatialDiagnostics,
                SynchronizationRefusalReason.NO_COMMON_MOTION,
                "no synchronizable motion or sustained-state units were detected in both clips",
                stats,
            )
        }

        val outcome = TemporalCorrespondenceSolver(
            config,
            descriptorSpace,
            preparedSource,
            preparedReference,
            stats,
        ).solve(request.semantics, sourceDetection.structure, referenceDetection.structure)
        if (outcome.matchedCount == 0) {
            return refused(
                request,
                sourceDetection,
                referenceDetection,
                spatialDiagnostics,
                SynchronizationRefusalReason.NO_COMMON_MOTION,
                "no defensible common motion survived direction, coverage, and local-warp constraints",
                stats,
            )
        }
        if (
            (!request.semantics.allowUnmatchedSource && outcome.correspondences.any {
                it is MotionUnitCorrespondence.SourceUnmatchedUnit ||
                    (it is MotionUnitCorrespondence.MatchedUnit && it.timeline.any { decision ->
                        decision is ai.senp.core.contracts.TimestampCorrespondence.UnmatchedSource
                    })
            }) ||
            (!request.semantics.allowUnmatchedReference && outcome.correspondences.any {
                it is MotionUnitCorrespondence.ReferenceUnmatchedUnit
            })
        ) {
            return refused(
                request,
                sourceDetection,
                referenceDetection,
                spatialDiagnostics,
                SynchronizationRefusalReason.UNRELIABLE_CORRESPONDENCE,
                "requested synchronization semantics disallow unmatched material required by the evidence",
                stats,
            )
        }
        if (outcome.confidence < config.minimumCorrespondenceForPartial) {
            return refused(
                request,
                sourceDetection,
                referenceDetection,
                spatialDiagnostics,
                SynchronizationRefusalReason.UNRELIABLE_CORRESPONDENCE,
                "candidate motion exists but bounded correspondence confidence is too low",
                stats,
            )
        }

        val temporalConfidence = min(sourceDetection.structure.confidence, referenceDetection.structure.confidence)
        val hasDiscontinuity = (sourceDetection.structure.activitySegments + referenceDetection.structure.activitySegments)
            .any { segment -> segment.kind == ActivitySegmentKind.DISCONTINUITY }
        val adjustedCorrespondenceConfidence = outcome.confidence * if (hasDiscontinuity) 0.80 else 1.0
        val status = if (
            !hasDiscontinuity && !outcome.hasUnmatchedUnits && !outcome.hasUnmatchedTimestamps &&
            adjustedCorrespondenceConfidence >= config.minimumSynchronizedConfidence &&
            outcome.ambiguity <= config.maximumAmbiguityForConfidentMatch
        ) {
            SynchronizationStatus.SYNCHRONIZED
        } else {
            SynchronizationStatus.PARTIAL
        }
        val result = SynchronizationResult(
            status = status,
            semantics = request.semantics,
            sourceTemporalStructure = sourceDetection.structure,
            referenceTemporalStructure = referenceDetection.structure,
            spatialDiagnostics = spatialDiagnostics,
            correspondences = outcome.correspondences,
            diagnostics = diagnostics(
                spatialDiagnostics,
                temporalConfidence,
                adjustedCorrespondenceConfidence,
                sourceDetection.analyzableFraction,
                referenceDetection.analyzableFraction,
                outcome.ambiguity,
            ),
        )
        return TemporalSynchronizationRun(result, stats.freeze())
    }

    private fun refused(
        request: SynchronizationRequest,
        sourceDetection: DetectedTemporalStructure,
        referenceDetection: DetectedTemporalStructure,
        spatialDiagnostics: SpatialSynchronizationDiagnostics,
        reason: SynchronizationRefusalReason,
        message: String,
        stats: MutableTemporalStats,
        missingRequired: Set<String> = emptySet(),
    ): TemporalSynchronizationRun {
        val sourceStructure = refusalStructure(sourceDetection.structure, request.semantics.allowUnmatchedSource)
        val referenceStructure = refusalStructure(referenceDetection.structure, request.semantics.allowUnmatchedReference)
        stats.sourceUnits = sourceStructure.motionUnits.size
        stats.referenceUnits = referenceStructure.motionUnits.size
        val coverage = mutableListOf<MotionUnitCorrespondence>()
        if (request.semantics.allowUnmatchedSource) {
            sourceStructure.motionUnits.forEach { unit ->
                coverage += MotionUnitCorrespondence.SourceUnmatchedUnit(
                    unit.unitId,
                    refusalUnmatchedReason(reason),
                    0.9,
                )
            }
        }
        if (request.semantics.allowUnmatchedReference) {
            referenceStructure.motionUnits.forEach { unit ->
                coverage += MotionUnitCorrespondence.ReferenceUnmatchedUnit(
                    unit.unitId,
                    if (unit.structureClass == MotionStructureClass.ACYCLIC) {
                        UnmatchedReason.MISSING_REFERENCE_STEP
                    } else {
                        refusalUnmatchedReason(reason)
                    },
                    0.9,
                )
            }
        }
        val result = SynchronizationResult(
            status = SynchronizationStatus.REFUSED,
            semantics = request.semantics,
            sourceTemporalStructure = sourceStructure,
            referenceTemporalStructure = referenceStructure,
            spatialDiagnostics = spatialDiagnostics,
            correspondences = coverage,
            diagnostics = diagnostics(
                spatialDiagnostics,
                min(sourceDetection.structure.confidence, referenceDetection.structure.confidence),
                0.0,
                sourceDetection.analyzableFraction,
                referenceDetection.analyzableFraction,
                if (reason == SynchronizationRefusalReason.TEMPORAL_AMBIGUITY) 1.0 else 0.0,
            ),
            refusal = SynchronizationRefusal(reason, message, missingRequired),
        )
        return TemporalSynchronizationRun(result, stats.freeze())
    }

    private fun refusalStructure(structure: TemporalStructure, unmatchedAllowed: Boolean): TemporalStructure =
        if (unmatchedAllowed) structure else structure.copy(motionUnits = emptyList())

    private fun refusalUnmatchedReason(reason: SynchronizationRefusalReason): UnmatchedReason = when (reason) {
        SynchronizationRefusalReason.INSUFFICIENT_OBSERVATIONS -> UnmatchedReason.INSUFFICIENT_DATA
        SynchronizationRefusalReason.TEMPORAL_AMBIGUITY -> UnmatchedReason.AMBIGUOUS
        SynchronizationRefusalReason.DISCONTINUITY -> UnmatchedReason.DISCONTINUITY
        else -> UnmatchedReason.NO_COMPATIBLE_COUNTERPART
    }

    private fun diagnostics(
        spatial: SpatialSynchronizationDiagnostics,
        temporalConfidence: Double,
        correspondenceConfidence: Double,
        sourceAnalyzable: Double,
        referenceAnalyzable: Double,
        ambiguity: Double,
    ): SynchronizationDiagnostics {
        val overall = if (correspondenceConfidence <= 0.0) {
            0.0
        } else {
            (spatial.aggregateConfidence * temporalConfidence * correspondenceConfidence)
                .coerceAtLeast(0.0)
                .pow(1.0 / 3.0)
        }
        return SynchronizationDiagnostics(
            overallConfidence = overall.coerceIn(0.0, 1.0),
            spatialConfidence = spatial.aggregateConfidence,
            temporalConfidence = temporalConfidence.coerceIn(0.0, 1.0),
            correspondenceConfidence = correspondenceConfidence.coerceIn(0.0, 1.0),
            sourceAnalyzableFraction = sourceAnalyzable.coerceIn(0.0, 1.0),
            referenceAnalyzableFraction = referenceAnalyzable.coerceIn(0.0, 1.0),
            correspondenceAmbiguity = ambiguity.coerceIn(0.0, 1.0),
        )
    }

    private fun missingRequiredChannels(request: SynchronizationRequest): Set<String> {
        val required = request.requirements.requiredChannelSemanticTypes
        if (required.isEmpty()) return emptySet()
        val sourceAvailable = availableSemanticTypes(request.source)
        val referenceAvailable = availableSemanticTypes(request.reference)
        return required.filterTo(linkedSetOf()) { it !in sourceAvailable || it !in referenceAvailable }
    }

    private fun availableSemanticTypes(sequence: CanonicalObservationSequence): Set<String> = sequence.observations
        .flatMap { observation -> observation.channels }
        .filter { channel -> channel.availability != ChannelAvailability.MISSING && channel.confidence > 0.0 }
        .mapTo(linkedSetOf()) { channel -> channel.semanticType }

    private fun spatiallyIncompatible(spatial: SpatialSynchronizationDiagnostics): Boolean {
        val source = spatial.reliabilitySegments.filter { it.role == VideoRole.SOURCE }
        val reference = spatial.reliabilitySegments.filter { it.role == VideoRole.REFERENCE }
        return allHardIncompatible(source) || allHardIncompatible(reference)
    }

    private fun allHardIncompatible(segments: List<SpatialReliabilitySegment>): Boolean = segments.isNotEmpty() &&
        segments.all { segment ->
            segment.status == SpatialReliabilityStatus.INCOMPATIBLE ||
                segment.status == SpatialReliabilityStatus.DISCONTINUITY
        }

    private fun hasDisallowedOpenBoundary(
        semantics: SynchronizationSemantics,
        source: TemporalStructure,
        reference: TemporalStructure,
    ): Boolean {
        val sourceFirst = source.motionUnits.firstOrNull()
        val sourceLast = source.motionUnits.lastOrNull()
        val referenceFirst = reference.motionUnits.firstOrNull()
        val referenceLast = reference.motionUnits.lastOrNull()
        return (!semantics.allowOpenSourceBegin && sourceFirst?.startBoundary == UnitBoundaryStatus.OPEN) ||
            (!semantics.allowOpenSourceEnd && sourceLast?.endBoundary == UnitBoundaryStatus.OPEN) ||
            (!semantics.allowOpenReferenceBegin && referenceFirst?.startBoundary == UnitBoundaryStatus.OPEN) ||
            (!semantics.allowOpenReferenceEnd && referenceLast?.endBoundary == UnitBoundaryStatus.OPEN)
    }

    private fun validateSignal(
        signal: TemporalSignalSequence,
        sequence: CanonicalObservationSequence,
        role: VideoRole,
    ) {
        require(signal.role == role) { "temporal signal role must be $role" }
        require(signal.duration == sequence.duration) { "temporal signal duration must match canonical sequence duration" }
    }
}
