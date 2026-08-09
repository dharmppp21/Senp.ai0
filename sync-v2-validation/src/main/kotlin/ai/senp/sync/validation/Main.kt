package ai.senp.sync.validation

import ai.senp.core.contracts.CanonicalObservationSequence
import ai.senp.core.contracts.DurationMs
import ai.senp.core.contracts.SynchronizationRequest
import ai.senp.core.contracts.SynchronizationResult
import ai.senp.core.contracts.TimestampMs
import ai.senp.core.contracts.VideoPoseExtraction
import ai.senp.sync.v2.PoseObservationAdapter
import ai.senp.sync.v2.SynchronizationKernelV2
import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.roundToLong
import kotlin.io.path.writeText
import kotlin.math.ceil
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.double
import kotlinx.serialization.json.long
import kotlinx.serialization.json.put

private val json = Json {
    prettyPrint = true
    encodeDefaults = true
    explicitNulls = true
    classDiscriminator = "type"
    ignoreUnknownKeys = false
}

fun main(args: Array<String>) {
    require(args.isNotEmpty()) {
        "usage: generate <output-dir> [seed] | validate <scenario-id> <result-json> <report-json> [seed] | evaluate-production <output-dir> [seed] | benchmark-adapter <request-json>"
    }
    when (args[0]) {
        "generate" -> generate(args)
        "validate" -> validate(args)
        "evaluate-production" -> evaluateProduction(args)
        "benchmark-adapter" -> benchmarkAdapter(args)
        "evaluate-pose-pair" -> evaluatePosePair(args)
        else -> error("unknown validation command: " + args[0])
    }
}

private fun generate(args: Array<String>) {
    require(args.size in 2..3) { "generate requires <output-dir> [seed]" }
    val output = Path.of(args[1]).toAbsolutePath().normalize()
    val seed = args.getOrNull(2)?.toLong() ?: SyntheticScenarioGenerator.DEFAULT_SEED
    val suite = SyntheticScenarioGenerator.generate(seed)
    Files.createDirectories(output)
    output.resolve("synthetic-suite.json").writeText(json.encodeToString(suite))
    val scenarioDir = output.resolve("scenarios")
    Files.createDirectories(scenarioDir)
    suite.scenarios.forEach { scenario ->
        scenarioDir.resolve(scenario.scenarioId + ".json").writeText(json.encodeToString(scenario))
    }
    output.resolve("coverage-matrix.json").writeText(json.encodeToString(suite.coverage))
    println(json.encodeToString(mapOf("output" to output.toString(), "scenario_count" to suite.scenarioCount.toString(), "seed" to seed.toString())))
}

private fun validate(args: Array<String>) {
    require(args.size in 4..5) { "validate requires <scenario-id> <result-json> <report-json> [seed]" }
    val scenarioId = args[1]
    val resultPath = Path.of(args[2])
    val reportPath = Path.of(args[3])
    val seed = args.getOrNull(4)?.toLong() ?: SyntheticScenarioGenerator.DEFAULT_SEED
    val scenario = SyntheticScenarioGenerator.generate(seed).scenarios.single { it.scenarioId == scenarioId }
    val result = json.decodeFromString<SynchronizationResult>(Files.readString(resultPath))
    val report = InvariantValidator.validate(scenario, result)
    reportPath.parent?.let(Files::createDirectories)
    reportPath.writeText(json.encodeToString(report))
    println(json.encodeToString(report))
    check(report.passed) { "Sync-v2 invariant validation failed for " + scenarioId }
}

private fun evaluateProduction(args: Array<String>) {
    require(args.size in 2..3) { "evaluate-production requires <output-dir> [seed]" }
    val output = Path.of(args[1]).toAbsolutePath().normalize()
    val seed = args.getOrNull(2)?.toLong() ?: SyntheticScenarioGenerator.DEFAULT_SEED
    val suite = SyntheticScenarioGenerator.generate(seed)
    Files.createDirectories(output)
    val integrated = ProductionSynchronizationHarnessAdapter()
    val spatial = ProductionSpatialHarnessAdapter()
    val temporal = ProductionTemporalHarnessAdapter()
    val rows = mutableListOf<kotlinx.serialization.json.JsonObject>()
    val failed = mutableListOf<String>()

    suite.scenarios.forEach { scenario ->
        val scenarioDirectory = output.resolve("scenarios").resolve(scenario.scenarioId)
        Files.createDirectories(scenarioDirectory)
        val run = integrated.synchronizeDetailed(scenario.request)
        val integrationReport = InvariantValidator.validate(scenario, run.result)
        val spatialOutput = spatial.synchronize(scenario.request.source, scenario.request.reference)
        val spatialReport = InvariantValidator.validateSpatialOutput(scenario, spatialOutput)
        val temporalReport = InvariantValidator.validateTemporalOutput(scenario, temporal.analyze(scenario.request.source))
        scenarioDirectory.resolve("result.json").writeText(json.encodeToString(run.result))
        scenarioDirectory.resolve("integration-report.json").writeText(json.encodeToString(integrationReport))
        scenarioDirectory.resolve("spatial-report.json").writeText(json.encodeToString(spatialReport))
        scenarioDirectory.resolve("temporal-report.json").writeText(json.encodeToString(temporalReport))
        val passed = integrationReport.passed && spatialReport.passed && temporalReport.passed
        if (!passed) failed += scenario.scenarioId
        rows += buildJsonObject {
            put("scenario_id", scenario.scenarioId)
            put("passed", passed)
            put("status", run.result.status.name)
            put("integration_findings", integrationReport.findings.size)
            put("spatial_findings", spatialReport.findings.size)
            put("temporal_findings", temporalReport.findings.size)
            put("iteration_count", run.stats.iterationCount)
            put("coarse_unit_comparisons", run.stats.totalCoarseUnitComparisons)
            put("fine_cells_evaluated", run.stats.totalFineCellsEvaluated)
            put("maximum_fine_band_width", run.stats.maximumFineBandWidth)
            put("iterations", buildJsonArray {
                run.stats.iterations.forEach { iteration ->
                    add(buildJsonObject {
                        put("iteration", iteration.iteration)
                        put("phase", iteration.phase)
                        put("quality", iteration.quality)
                        put("status", iteration.status.name)
                        put("spatial_confidence", iteration.spatialConfidence)
                        put("correspondence_confidence", iteration.correspondenceConfidence)
                        put("paired_spatial_evidence_count", iteration.pairedSpatialEvidenceCount)
                        put("refined_hypothesis_count", iteration.refinedHypothesisCount)
                        put("fine_cells_evaluated", iteration.temporalStats.fineCellsEvaluated)
                    })
                }
            })
        }
    }

    val expectedLabels = suite.scenarios.flatMap { it.expectedInvariants }.toSet()
    val summary = buildJsonObject {
        put("schema_version", 1)
        put("seed", seed)
        put("scenario_count", suite.scenarioCount)
        put("passed_count", suite.scenarioCount - failed.size)
        put("failed_count", failed.size)
        put("production_integration", CoverageState.EXECUTABLE.name)
        put("frozen_invariant_vocabulary_count", InvariantValidator.supportedFrozenInvariants.size)
        put("expected_invariant_label_count", expectedLabels.size)
        put("all_frozen_invariants_accounted", expectedLabels == InvariantValidator.supportedFrozenInvariants)
        put("failed_scenarios", buildJsonArray { failed.forEach { add(kotlinx.serialization.json.JsonPrimitive(it)) } })
        put("scenarios", buildJsonArray { rows.forEach(::add) })
    }
    output.resolve("production-summary.json").writeText(json.encodeToString(summary))
    output.resolve("coverage-matrix.json").writeText(json.encodeToString(suite.coverage))
    println(json.encodeToString(summary))
    check(failed.isEmpty()) { "production Sync-v2 validation failed scenarios=$failed" }
    check(expectedLabels == InvariantValidator.supportedFrozenInvariants) { "frozen invariant vocabulary is not exhaustively covered" }
}


private fun benchmarkAdapter(args: Array<String>) {
    require(args.size == 2) { "benchmark-adapter requires <request-json>" }
    val descriptorPath = Path.of(args[1]).toAbsolutePath().normalize()
    val descriptor = json.parseToJsonElement(Files.readString(descriptorPath)).jsonObject
    require(descriptor["protocol"]?.jsonPrimitive?.content == "senp-sync-v2-validation-adapter/1")
    require(descriptor["mode"]?.jsonPrimitive?.content == "post_pose_benchmark")
    val plan = requireNotNull(descriptor["case"]).jsonObject
    val scenarioId = requireNotNull(plan["scenario"]).jsonPrimitive.content
    val analysisFps = requireNotNull(plan["analysis_fps"]).jsonPrimitive.int
    val requestedSamples = plan["samples"]?.let { element ->
        if (element is JsonNull) null else element.jsonPrimitive.int
    }
    val outputPath = Path.of(requireNotNull(descriptor["result_output"]).jsonPrimitive.content).toAbsolutePath().normalize()
    val scenario = SyntheticScenarioGenerator.generate().scenarios.single { it.scenarioId == scenarioId }
    val request = scenario.request.copy(
        source = benchmarkSequence(scenario.request.source, requestedSamples, analysisFps),
        reference = benchmarkSequence(scenario.request.reference, requestedSamples, analysisFps),
    )
    val kernel = SynchronizationKernelV2()
    kernel.synchronize(request)
    val start = System.nanoTime()
    val run = kernel.synchronize(request)
    val elapsedMs = (System.nanoTime() - start).toDouble() / 1_000_000.0
    val naiveCells = run.stats.iterations.sumOf { it.temporalStats.naiveWholeVideoCellCount }
    val fineFraction = if (naiveCells == 0L) 0.0 else run.stats.totalFineCellsEvaluated.toDouble() / naiveCells.toDouble()
    val firstStats = run.stats.iterations.first().temporalStats
    val result = buildJsonObject {
        put("schema_version", 1)
        put("engine", ai.senp.sync.v2.SynchronizationKernelV2Versions.ENGINE)
        put("scenario", scenarioId)
        put("analysis_fps", analysisFps)
        put("source_frames", firstStats.sourceFrameCount)
        put("reference_frames", firstStats.referenceFrameCount)
        put("source_units", firstStats.sourceUnitCount)
        put("reference_units", firstStats.referenceUnitCount)
        put("post_pose_sync_ms", elapsedMs)
        put("pose_preprocessing_ms", JsonNull)
        put("total_pipeline_ms", JsonNull)
        put("peak_rss_bytes", readVmHwmBytes())
        put("status", run.result.status.name)
        put("iteration_count", run.stats.iterationCount)
        put("coarse_unit_comparisons", run.stats.totalCoarseUnitComparisons)
        put("fine_cells_evaluated", run.stats.totalFineCellsEvaluated)
        put("maximum_fine_band_width", run.stats.maximumFineBandWidth)
        put("fine_alignment_count", run.stats.totalFineAlignmentCount)
        put("naive_whole_video_cells", naiveCells)
        put("fine_to_naive_fraction", fineFraction)
        put("synthetic_truth_consumed", false)
    }
    outputPath.parent?.let(Files::createDirectories)
    outputPath.writeText(json.encodeToString(JsonObject.serializer(), result))
    println(json.encodeToString(JsonObject.serializer(), result))
}

private fun benchmarkSequence(
    sequence: CanonicalObservationSequence,
    samples: Int?,
    analysisFramesPerSecond: Int,
): CanonicalObservationSequence {
    require(analysisFramesPerSecond > 0)
    require(sequence.observations.isNotEmpty())
    val templates = sequence.observations.map { observation ->
        observation.copy(channels = observation.channels.filterNot { it.semanticType == "synthetic_motion_truth" })
    }
    if (samples == null) {
        return sequence.copy(
            sampling = sequence.sampling.copy(analysisFramesPerSecond = analysisFramesPerSecond.toDouble()),
            observations = templates,
        )
    }
    require(samples > 1)
    val stepMs = 1000.0 / analysisFramesPerSecond.toDouble()
    val observations = List(samples) { index ->
        val timestamp = (index * stepMs).roundToLong()
        val templateTimestamp = timestamp % sequence.duration.value
        val template = templates.minByOrNull { observation ->
            kotlin.math.abs(observation.timestamp.value - templateTimestamp)
        } ?: templates.first()
        template.copy(timestamp = TimestampMs(timestamp))
    }
    val durationMs = maxOf(
        observations.last().timestamp.value + 1L,
        ceil(samples * stepMs).toLong(),
    )
    return sequence.copy(
        duration = DurationMs(durationMs),
        sampling = sequence.sampling.copy(analysisFramesPerSecond = analysisFramesPerSecond.toDouble()),
        observations = observations,
    )
}

private fun readVmHwmBytes(): Long {
    val line = File("/proc/self/status").useLines { lines -> lines.firstOrNull { it.startsWith("VmHWM:") } } ?: return 0L
    val kib = Regex("VmHWM:\\s+(\\d+)\\s+kB").find(line)?.groupValues?.get(1)?.toLongOrNull() ?: return 0L
    return kib * 1024L
}


private fun evaluatePosePair(args: Array<String>) {
    require(args.size in 4..5) { "evaluate-pose-pair requires <source-extraction-json> <reference-extraction-json> <output-dir> [analysis-fps]" }
    val sourceExtraction = json.decodeFromString<VideoPoseExtraction>(Files.readString(Path.of(args[1])))
    val referenceExtraction = json.decodeFromString<VideoPoseExtraction>(Files.readString(Path.of(args[2])))
    val analysisFramesPerSecond = args.getOrNull(4)?.toDouble() ?: 15.0
    val adapter = PoseObservationAdapter()
    val request = SynchronizationRequest(
        source = adapter.adapt(sourceExtraction, analysisFramesPerSecond),
        reference = adapter.adapt(referenceExtraction, analysisFramesPerSecond),
    )
    val start = System.nanoTime()
    val run = SynchronizationKernelV2().synchronize(request)
    val elapsedMs = (System.nanoTime() - start).toDouble() / 1_000_000.0
    val output = Path.of(args[3]).toAbsolutePath().normalize()
    Files.createDirectories(output)
    output.resolve("result.json").writeText(json.encodeToString(run.result))
    output.resolve("summary.json").writeText(
        json.encodeToString(
            JsonObject.serializer(),
            buildJsonObject {
                put("status", run.result.status.name)
                run.result.refusal?.let { put("refusal_reason", it.reason.name) }
                put("overall_confidence", run.result.diagnostics.overallConfidence)
                put("source_analyzable_fraction", run.result.diagnostics.sourceAnalyzableFraction)
                put("reference_analyzable_fraction", run.result.diagnostics.referenceAnalyzableFraction)
                put("source_units", run.result.sourceTemporalStructure.motionUnits.size)
                put("reference_units", run.result.referenceTemporalStructure.motionUnits.size)
                put("matched_units", run.result.correspondences.count { it is ai.senp.core.contracts.MotionUnitCorrespondence.MatchedUnit })
                put("elapsed_ms", elapsedMs)
                put("iteration_count", run.stats.iterationCount)
                put("coarse_unit_comparisons", run.stats.totalCoarseUnitComparisons)
                put("coarse_accepted_candidates", run.stats.iterations.sumOf { it.temporalStats.coarseAcceptedCandidateCount })
                run.stats.iterations.mapNotNull { it.temporalStats.bestCoarseCandidateCost }.minOrNull()?.let { put("best_coarse_candidate_cost", it) }
                put("fine_cells_evaluated", run.stats.totalFineCellsEvaluated)
                put("fine_path_timestamps", run.stats.iterations.sumOf { it.temporalStats.finePathTimestampCount })
                put("fine_accepted_timestamps", run.stats.iterations.sumOf { it.temporalStats.fineAcceptedTimestampCount })
                put("fine_rejected_opposite_direction", run.stats.iterations.sumOf { it.temporalStats.fineRejectedOppositeDirectionCount })
                put("fine_rejected_cost", run.stats.iterations.sumOf { it.temporalStats.fineRejectedCostCount })
                put("fine_rejected_coverage", run.stats.iterations.sumOf { it.temporalStats.fineRejectedCoverageCount })
                put("fine_rejected_confidence", run.stats.iterations.sumOf { it.temporalStats.fineRejectedConfidenceCount })
                put("fine_rejected_warp", run.stats.iterations.sumOf { it.temporalStats.fineRejectedWarpCount })
                run.stats.iterations.mapNotNull { it.temporalStats.bestFineMatchedFraction }.maxOrNull()?.let { put("best_fine_matched_fraction", it) }
                run.stats.iterations.mapNotNull { it.temporalStats.bestFineDecisionConfidence }.maxOrNull()?.let { put("best_fine_decision_confidence", it) }
                put("maximum_fine_band_width", run.stats.maximumFineBandWidth)
                put("mapping_rows", run.mappingDiagnostics.size)
                put("reliable_opposite_direction_rows", run.mappingDiagnostics.count { diagnostic ->
                    diagnostic.referenceTimestamp != null &&
                        diagnostic.sourceReliability in setOf(null, ai.senp.core.contracts.SpatialReliabilityStatus.COMPATIBLE) &&
                        diagnostic.referenceReliability in setOf(null, ai.senp.core.contracts.SpatialReliabilityStatus.COMPATIBLE) &&
                        diagnostic.sourceDirection in setOf(-1, 1) &&
                        diagnostic.referenceDirection in setOf(-1, 1) &&
                        diagnostic.referenceDirection?.let { direction -> diagnostic.sourceDirection == -direction } == true
                })
            },
        ),
    )
    println(Files.readString(output.resolve("summary.json")))
}
