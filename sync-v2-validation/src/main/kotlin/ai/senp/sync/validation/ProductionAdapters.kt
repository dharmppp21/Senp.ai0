package ai.senp.sync.validation

import ai.senp.core.contracts.CanonicalObservation
import ai.senp.core.contracts.CanonicalObservationSequence
import ai.senp.core.contracts.SynchronizationRequest
import ai.senp.core.contracts.TemporalStructure
import ai.senp.core.contracts.VideoRole
import ai.senp.sync.v2.SynchronizationKernelRun
import ai.senp.sync.v2.SynchronizationKernelV2

/** Concrete validation adapters for the production Sync-v2 composition. */
class ProductionSynchronizationHarnessAdapter(
    private val kernel: SynchronizationKernelV2 = SynchronizationKernelV2(),
) : SynchronizationHarnessAdapter {
    override fun synchronize(request: SynchronizationRequest) = kernel.synchronize(request).result

    fun synchronizeDetailed(request: SynchronizationRequest): SynchronizationKernelRun = kernel.synchronize(request)
}

class ProductionSpatialHarnessAdapter : SpatialHarnessAdapter {
    private val kernel = SynchronizationKernelV2()

    override fun synchronize(
        source: CanonicalObservationSequence,
        reference: CanonicalObservationSequence,
    ): SpatialHarnessOutput {
        val run = kernel.synchronize(SynchronizationRequest(source = source, reference = reference))
        return SpatialHarnessOutput(
            canonicalSource = canonicalSequence(run, VideoRole.SOURCE),
            canonicalReference = canonicalSequence(run, VideoRole.REFERENCE),
            diagnostics = run.result.spatialDiagnostics,
        )
    }

    private fun canonicalSequence(run: SynchronizationKernelRun, role: VideoRole): CanonicalObservationSequence {
        val analysis = if (role == VideoRole.SOURCE) run.spatialOutput.source else run.spatialOutput.reference
        return CanonicalObservationSequence(
            role = role,
            duration = analysis.duration,
            sampling = analysis.sampling,
            observations = analysis.frames.map { frame ->
                CanonicalObservation(frame.timestamp, listOfNotNull(frame.canonicalPose))
            },
        )
    }
}

class ProductionTemporalHarnessAdapter : TemporalHarnessAdapter {
    private val kernel = SynchronizationKernelV2()

    override fun analyze(sequence: CanonicalObservationSequence): TemporalStructure = kernel.analyzeTemporal(sequence)
}
