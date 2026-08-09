package ai.senp.core.contracts

import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue

class SyncV2ContractsTest {
    private val json = Json {
        encodeDefaults = true
        explicitNulls = true
        classDiscriminator = "type"
    }

    @Test
    fun `observation contract separates input FPS from analysis FPS and requires explicit masks`() {
        val value = ObservationValue(
            key = "left_wrist",
            values = listOf(0.2, 0.7, null),
            mask = listOf(true, true, false),
            confidence = 0.85,
        )
        val channel = ObservationChannel(
            channelId = "pose-human-image",
            schemaVersion = 1,
            semanticType = "human_pose",
            coordinateSpace = "image_normalized",
            subjectId = "primary",
            componentAxes = listOf("x", "y", "z"),
            values = listOf(value),
            availability = ChannelAvailability.PARTIAL,
            confidence = 0.8,
        )
        val sequence = CanonicalObservationSequence(
            role = VideoRole.SOURCE,
            duration = DurationMs(1000),
            sampling = ObservationSampling(inputNominalFramesPerSecond = 59.94, analysisFramesPerSecond = 15.0),
            observations = listOf(CanonicalObservation(TimestampMs(100), listOf(channel))),
        )

        assertEquals(59.94, sequence.sampling.inputNominalFramesPerSecond)
        assertEquals(15.0, sequence.sampling.analysisFramesPerSecond)
        assertFailsWith<IllegalArgumentException> {
            ObservationValue("bad", listOf(1.0), listOf(false), 1.0)
        }
        assertFailsWith<IllegalArgumentException> {
            channel.copy(availability = ChannelAvailability.MISSING, confidence = 0.5)
        }
    }

    @Test
    fun `canonical channels are semantic and do not require a MediaPipe landmark topology`() {
        val objectChannel = ObservationChannel(
            channelId = "object-tool-0",
            schemaVersion = 3,
            semanticType = "object_pose",
            coordinateSpace = "camera_metric",
            componentAxes = listOf("x", "y", "z", "qx", "qy", "qz", "qw"),
            values = listOf(
                ObservationValue(
                    key = "tool",
                    values = listOf(0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 1.0),
                    mask = List(7) { true },
                    confidence = 0.9,
                ),
            ),
            availability = ChannelAvailability.OBSERVED,
            confidence = 0.9,
        )
        assertEquals("object_pose", objectChannel.semanticType)
        assertEquals(7, objectChannel.componentAxes.size)
    }

    @Test
    fun `spatial diagnostics expose similarity transform mirror side and unreliable intervals`() {
        val estimate = BodyCentricTransformEstimate(
            role = VideoRole.SOURCE,
            range = range(0, 1000),
            transform = BodyCentricTransform(
                fromCoordinateSpace = "camera_metric",
                toCoordinateSpace = "body_centric",
                translation = Vector3(0.1, -0.2, 0.3),
                rotation = UnitQuaternion.Identity,
                uniformScale = 1.2,
            ),
            confidence = 0.8,
            stability = 0.75,
        )
        val diagnostics = SpatialSynchronizationDiagnostics(
            sourceTransforms = listOf(estimate),
            relativeViewHypotheses = listOf(
                RelativeViewHypothesis(
                    sourceRange = range(0, 1000),
                    referenceRange = range(0, 1000),
                    relativeYawDegrees = 70.0,
                    relativeElevationDegrees = 20.0,
                    mirror = MirrorHypothesis.AMBIGUOUS,
                    selectedBodySide = BodySideHypothesis.LEFT,
                    confidence = 0.55,
                    sideSelectionStability = 0.4,
                ),
            ),
            reliabilitySegments = listOf(
                SpatialReliabilitySegment(
                    role = VideoRole.SOURCE,
                    range = range(400, 650),
                    status = SpatialReliabilityStatus.UNRELIABLE,
                    confidence = 0.3,
                    reasons = setOf(SpatialDiagnosticReason.CAMERA_MOVEMENT, SpatialDiagnosticReason.TRANSFORM_UNSTABLE),
                ),
            ),
            aggregateConfidence = 0.6,
        )

        assertEquals(SpatialReliabilityStatus.UNRELIABLE, diagnostics.reliabilitySegments.single().status)
        assertFailsWith<IllegalArgumentException> {
            estimate.copy(transform = estimate.transform.copy(uniformScale = 0.0))
        }
        assertFailsWith<IllegalArgumentException> {
            UnitQuaternion(0.0, 0.0, 0.0, 2.0)
        }
    }

    @Test
    fun `temporal contract supports partial units holds rests discontinuities and isometric motion`() {
        val structure = TemporalStructure(
            role = VideoRole.SOURCE,
            duration = DurationMs(5000),
            classification = MotionStructureClass.MIXED,
            activitySegments = listOf(
                ActivitySegment(range(0, 500), ActivitySegmentKind.SETUP, 0.9),
                ActivitySegment(range(500, 1500), ActivitySegmentKind.ACTIVE, 0.9),
                ActivitySegment(range(1500, 2200), ActivitySegmentKind.HOLD, 0.85),
                ActivitySegment(range(2200, 3000), ActivitySegmentKind.REST, 0.95),
                ActivitySegment(range(3000, 3050), ActivitySegmentKind.DISCONTINUITY, 1.0),
            ),
            motionUnits = listOf(
                MotionUnit(
                    unitId = "unit-0",
                    range = range(500, 1500),
                    structureClass = MotionStructureClass.CYCLIC,
                    completeness = UnitCompleteness.PARTIAL,
                    startBoundary = UnitBoundaryStatus.OPEN,
                    endBoundary = UnitBoundaryStatus.CLOSED,
                    confidence = 0.8,
                ),
                MotionUnit(
                    unitId = "unit-1",
                    range = range(1500, 2200),
                    structureClass = MotionStructureClass.ISOMETRIC,
                    completeness = UnitCompleteness.COMPLETE,
                    startBoundary = UnitBoundaryStatus.CLOSED,
                    endBoundary = UnitBoundaryStatus.CLOSED,
                    confidence = 0.85,
                ),
            ),
            confidence = 0.8,
        )

        assertEquals(UnitBoundaryStatus.OPEN, structure.motionUnits.first().startBoundary)
        assertEquals(ActivitySegmentKind.HOLD, structure.activitySegments[2].kind)
        assertFailsWith<IllegalArgumentException> {
            structure.motionUnits.first().copy(
                completeness = UnitCompleteness.COMPLETE,
                startBoundary = UnitBoundaryStatus.OPEN,
            )
        }
    }

    @Test
    fun `one reference unit can be reused independently by ten source units`() {
        val source = temporal(VideoRole.SOURCE, 10)
        val reference = temporal(VideoRole.REFERENCE, 1)
        val correspondences = source.motionUnits.mapIndexed { index, sourceUnit ->
            MotionUnitCorrespondence.MatchedUnit(
                sourceUnitId = sourceUnit.unitId,
                referenceUnitId = reference.motionUnits.single().unitId,
                timeline = listOf(
                    TimestampCorrespondence.Matched(
                        sourceTimestamp = sourceUnit.range.start,
                        referenceTimestamp = reference.motionUnits.single().range.start,
                        decisionConfidence = 0.95,
                    ),
                ),
                decisionConfidence = 0.95 - index * 0.01,
                ambiguity = 0.05,
            )
        }

        val result = result(source, reference, SynchronizationStatus.SYNCHRONIZED, correspondences)
        assertEquals(10, result.correspondences.size)
        assertEquals(1, result.correspondences.filterIsInstance<MotionUnitCorrespondence.MatchedUnit>()
            .map(MotionUnitCorrespondence.MatchedUnit::referenceUnitId).distinct().size)
    }

    @Test
    fun `source timestamps can be unmatched and missing reference units are explicit`() {
        val source = temporal(VideoRole.SOURCE, 2)
        val reference = temporal(VideoRole.REFERENCE, 2)
        val firstSource = source.motionUnits[0]
        val firstReference = reference.motionUnits[0]
        val correspondences = listOf(
            MotionUnitCorrespondence.MatchedUnit(
                sourceUnitId = firstSource.unitId,
                referenceUnitId = firstReference.unitId,
                timeline = listOf(
                    TimestampCorrespondence.Matched(firstSource.range.start, firstReference.range.start, 0.9),
                    TimestampCorrespondence.UnmatchedSource(
                        TimestampMs(firstSource.range.start.value + 100),
                        UnmatchedReason.OCCLUSION,
                        0.8,
                    ),
                    TimestampCorrespondence.Matched(
                        TimestampMs(firstSource.range.start.value + 200),
                        TimestampMs(firstReference.range.start.value + 250),
                        0.9,
                    ),
                ),
                decisionConfidence = 0.85,
                ambiguity = 0.1,
            ),
            MotionUnitCorrespondence.SourceUnmatchedUnit(source.motionUnits[1].unitId, UnmatchedReason.EXTRA_ACTION, 0.95),
            MotionUnitCorrespondence.ReferenceUnmatchedUnit(reference.motionUnits[1].unitId, UnmatchedReason.MISSING_REFERENCE_STEP, 0.95),
        )

        val result = result(source, reference, SynchronizationStatus.PARTIAL, correspondences)
        assertIs<TimestampCorrespondence.UnmatchedSource>(
            result.correspondences.filterIsInstance<MotionUnitCorrespondence.MatchedUnit>().single().timeline[1],
        )
    }

    @Test
    fun `unit mappings are locally monotonic but may reset reference time across reused units`() {
        val first = MotionUnitCorrespondence.MatchedUnit(
            sourceUnitId = "s0",
            referenceUnitId = "r0",
            timeline = listOf(
                TimestampCorrespondence.Matched(TimestampMs(0), TimestampMs(100), 1.0),
                TimestampCorrespondence.Matched(TimestampMs(100), TimestampMs(200), 1.0),
            ),
            decisionConfidence = 1.0,
            ambiguity = 0.0,
        )
        val second = MotionUnitCorrespondence.MatchedUnit(
            sourceUnitId = "s1",
            referenceUnitId = "r0",
            timeline = listOf(
                TimestampCorrespondence.Matched(TimestampMs(1000), TimestampMs(100), 1.0),
                TimestampCorrespondence.Matched(TimestampMs(1100), TimestampMs(200), 1.0),
            ),
            decisionConfidence = 1.0,
            ambiguity = 0.0,
        )
        assertTrue(second.timeline.first().sourceTimestamp > first.timeline.last().sourceTimestamp)
        assertEquals(
            (first.timeline.first() as TimestampCorrespondence.Matched).referenceTimestamp,
            (second.timeline.first() as TimestampCorrespondence.Matched).referenceTimestamp,
        )
        assertFailsWith<IllegalArgumentException> {
            first.copy(
                timeline = listOf(
                    TimestampCorrespondence.Matched(TimestampMs(0), TimestampMs(200), 1.0),
                    TimestampCorrespondence.Matched(TimestampMs(100), TimestampMs(100), 1.0),
                ),
            )
        }
    }

    @Test
    fun `every motion unit must be explicitly covered while reference reuse remains legal`() {
        val source = temporal(VideoRole.SOURCE, 2)
        val reference = temporal(VideoRole.REFERENCE, 1)
        val onlyFirst = listOf(
            MotionUnitCorrespondence.MatchedUnit(
                source.motionUnits[0].unitId,
                reference.motionUnits[0].unitId,
                listOf(TimestampCorrespondence.Matched(source.motionUnits[0].range.start, reference.motionUnits[0].range.start, 1.0)),
                1.0,
                0.0,
            ),
        )
        assertFailsWith<IllegalArgumentException> {
            result(source, reference, SynchronizationStatus.PARTIAL, onlyFirst)
        }
    }

    @Test
    fun `refused result carries diagnostics instead of fabricated successful comparison`() {
        val source = temporal(VideoRole.SOURCE, 0)
        val reference = temporal(VideoRole.REFERENCE, 0)
        val refused = SynchronizationResult(
            status = SynchronizationStatus.REFUSED,
            sourceTemporalStructure = source,
            referenceTemporalStructure = reference,
            spatialDiagnostics = spatial(0.1),
            correspondences = emptyList(),
            diagnostics = diagnostics(0.1),
            refusal = SynchronizationRefusal(
                reason = SynchronizationRefusalReason.INSUFFICIENT_OBSERVATIONS,
                message = "Pose coverage is insufficient for reliable synchronization",
            ),
        )

        assertEquals(SynchronizationStatus.REFUSED, refused.status)
        assertTrue(refused.correspondences.isEmpty())
        assertFailsWith<IllegalArgumentException> {
            refused.copy(status = SynchronizationStatus.SYNCHRONIZED, refusal = null)
        }
    }

    @Test
    fun `object dependent requirement can be declared without putting task names in kernel contract`() {
        val request = SynchronizationRequest(
            source = observationSequence(VideoRole.SOURCE, 30.0, 15.0),
            reference = observationSequence(VideoRole.REFERENCE, 24.0, 12.0),
            requirements = SynchronizationRequirements(setOf("human_pose", "object_pose")),
        )
        assertEquals(setOf("human_pose", "object_pose"), request.requirements.requiredChannelSemanticTypes)
    }

    @Test
    fun `result semantics can forbid unmatched decisions and reference reuse`() {
        val source = temporal(VideoRole.SOURCE, 2)
        val reference = temporal(VideoRole.REFERENCE, 1)
        val reused = source.motionUnits.map { sourceUnit ->
            MotionUnitCorrespondence.MatchedUnit(
                sourceUnitId = sourceUnit.unitId,
                referenceUnitId = reference.motionUnits.single().unitId,
                timeline = listOf(
                    TimestampCorrespondence.Matched(
                        sourceUnit.range.start,
                        reference.motionUnits.single().range.start,
                        1.0,
                    ),
                ),
                decisionConfidence = 1.0,
                ambiguity = 0.0,
            )
        }
        assertFailsWith<IllegalArgumentException> {
            SynchronizationResult(
                status = SynchronizationStatus.SYNCHRONIZED,
                semantics = SynchronizationSemantics(allowReferenceUnitReuse = false),
                sourceTemporalStructure = source,
                referenceTemporalStructure = reference,
                spatialDiagnostics = spatial(1.0),
                correspondences = reused,
                diagnostics = diagnostics(1.0),
            )
        }

        val oneSource = temporal(VideoRole.SOURCE, 1)
        val oneReference = temporal(VideoRole.REFERENCE, 1)
        val withUnmatchedTimestamp = listOf(
            MotionUnitCorrespondence.MatchedUnit(
                oneSource.motionUnits.single().unitId,
                oneReference.motionUnits.single().unitId,
                listOf(
                    TimestampCorrespondence.Matched(
                        oneSource.motionUnits.single().range.start,
                        oneReference.motionUnits.single().range.start,
                        0.9,
                    ),
                    TimestampCorrespondence.UnmatchedSource(
                        TimestampMs(oneSource.motionUnits.single().range.start.value + 100),
                        UnmatchedReason.OCCLUSION,
                        0.9,
                    ),
                ),
                0.9,
                0.1,
            ),
        )
        assertFailsWith<IllegalArgumentException> {
            SynchronizationResult(
                status = SynchronizationStatus.PARTIAL,
                semantics = SynchronizationSemantics(allowUnmatchedSource = false),
                sourceTemporalStructure = oneSource,
                referenceTemporalStructure = oneReference,
                spatialDiagnostics = spatial(0.9),
                correspondences = withUnmatchedTimestamp,
                diagnostics = diagnostics(0.9),
            )
        }
    }

    @Test
    fun `confidence and refusal constructors reject invalid values`() {
        assertFailsWith<IllegalArgumentException> {
            TimestampCorrespondence.Matched(TimestampMs(0), TimestampMs(0), 1.01)
        }
        assertFailsWith<IllegalArgumentException> {
            SynchronizationDiagnostics(
                overallConfidence = Double.NaN,
                spatialConfidence = 0.5,
                temporalConfidence = 0.5,
                correspondenceConfidence = 0.5,
                sourceAnalyzableFraction = 0.5,
                referenceAnalyzableFraction = 0.5,
                correspondenceAmbiguity = 0.5,
            )
        }
        assertFailsWith<IllegalArgumentException> {
            SynchronizationRefusal(
                reason = SynchronizationRefusalReason.REQUIRED_CHANNEL_MISSING,
                message = "required evidence is missing",
            )
        }
        val refusal = SynchronizationRefusal(
            reason = SynchronizationRefusalReason.REQUIRED_CHANNEL_MISSING,
            message = "object evidence is missing",
            missingRequiredChannelSemanticTypes = setOf("object_pose"),
        )
        assertEquals(setOf("object_pose"), refusal.missingRequiredChannelSemanticTypes)
    }

    @Test
    fun `synchronization result with sealed matched and unmatched decisions round trips JSON`() {
        val source = temporal(VideoRole.SOURCE, 2)
        val reference = temporal(VideoRole.REFERENCE, 2)
        val correspondences = listOf(
            MotionUnitCorrespondence.MatchedUnit(
                source.motionUnits[0].unitId,
                reference.motionUnits[0].unitId,
                listOf(
                    TimestampCorrespondence.Matched(source.motionUnits[0].range.start, reference.motionUnits[0].range.start, 0.9),
                    TimestampCorrespondence.UnmatchedSource(
                        TimestampMs(source.motionUnits[0].range.start.value + 100),
                        UnmatchedReason.OCCLUSION,
                        0.8,
                    ),
                ),
                0.9,
                0.1,
            ),
            MotionUnitCorrespondence.SourceUnmatchedUnit(source.motionUnits[1].unitId, UnmatchedReason.EXTRA_ACTION, 0.9),
            MotionUnitCorrespondence.ReferenceUnmatchedUnit(reference.motionUnits[1].unitId, UnmatchedReason.MISSING_REFERENCE_STEP, 0.9),
        )
        val original = result(source, reference, SynchronizationStatus.PARTIAL, correspondences)
        val encoded = json.encodeToString(original)
        val decoded = json.decodeFromString<SynchronizationResult>(encoded)

        assertEquals(original, decoded)
        assertTrue(encoded.contains("unmatched_source"))
        assertTrue(encoded.contains("reference_unmatched_unit"))
    }

    private fun observationSequence(role: VideoRole, inputFps: Double, analysisFps: Double): CanonicalObservationSequence {
        val channel = ObservationChannel(
            channelId = "human-pose-2d",
            schemaVersion = 1,
            semanticType = "human_pose",
            coordinateSpace = "image_normalized",
            componentAxes = listOf("x", "y"),
            values = listOf(ObservationValue("joint-0", listOf(0.5, 0.5), listOf(true, true), 1.0)),
            availability = ChannelAvailability.OBSERVED,
            confidence = 1.0,
        )
        return CanonicalObservationSequence(
            role,
            DurationMs(1000),
            ObservationSampling(inputFps, analysisFps),
            listOf(CanonicalObservation(TimestampMs(0), listOf(channel))),
        )
    }

    private fun temporal(role: VideoRole, unitCount: Int): TemporalStructure {
        val units = (0 until unitCount).map { index ->
            val start = index * 1000L
            MotionUnit(
                unitId = "${role.name.lowercase()}-$index",
                range = range(start, start + 900),
                structureClass = MotionStructureClass.CYCLIC,
                completeness = UnitCompleteness.COMPLETE,
                startBoundary = UnitBoundaryStatus.CLOSED,
                endBoundary = UnitBoundaryStatus.CLOSED,
                confidence = 0.95,
            )
        }
        return TemporalStructure(
            role = role,
            duration = DurationMs(maxOf(1000L, unitCount * 1000L)),
            classification = if (unitCount == 0) MotionStructureClass.UNKNOWN else MotionStructureClass.CYCLIC,
            activitySegments = emptyList(),
            motionUnits = units,
            confidence = if (unitCount == 0) 0.0 else 0.95,
        )
    }

    private fun result(
        source: TemporalStructure,
        reference: TemporalStructure,
        status: SynchronizationStatus,
        correspondences: List<MotionUnitCorrespondence>,
    ): SynchronizationResult = SynchronizationResult(
        status = status,
        sourceTemporalStructure = source,
        referenceTemporalStructure = reference,
        spatialDiagnostics = spatial(0.9),
        correspondences = correspondences,
        diagnostics = diagnostics(0.9),
    )

    private fun spatial(confidence: Double) = SpatialSynchronizationDiagnostics(aggregateConfidence = confidence)

    private fun diagnostics(confidence: Double) = SynchronizationDiagnostics(
        overallConfidence = confidence,
        spatialConfidence = confidence,
        temporalConfidence = confidence,
        correspondenceConfidence = confidence,
        sourceAnalyzableFraction = confidence,
        referenceAnalyzableFraction = confidence,
        correspondenceAmbiguity = 1.0 - confidence,
    )

    private fun range(start: Long, endExclusive: Long) = TimestampRange(TimestampMs(start), TimestampMs(endExclusive))
}
