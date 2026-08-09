package ai.senp.motion

import ai.senp.core.contracts.BodySideHypothesis
import ai.senp.core.contracts.CanonicalObservation
import ai.senp.core.contracts.CanonicalObservationSequence
import ai.senp.core.contracts.ChannelAvailability
import ai.senp.core.contracts.DurationMs
import ai.senp.core.contracts.MirrorHypothesis
import ai.senp.core.contracts.ObservationChannel
import ai.senp.core.contracts.ObservationSampling
import ai.senp.core.contracts.ObservationValue
import ai.senp.core.contracts.SpatialDiagnosticReason
import ai.senp.core.contracts.SpatialReliabilityStatus
import ai.senp.core.contracts.SynchronizationRequest
import ai.senp.core.contracts.SynchronizationRequirements
import ai.senp.core.contracts.TimestampMs
import ai.senp.core.contracts.VideoRole
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SpatialSynchronizationEngineTest {
    private val engine = SpatialSynchronizationEngine()

    @Test
    fun `3d camera yaw elevation translation and uniform scale canonicalize to the same intrinsic pose`() {
        val reference = sequence(VideoRole.REFERENCE)
        val source = sequence(
            VideoRole.SOURCE,
            yawDegrees = 38.0,
            elevationDegrees = 17.0,
            globalScale = 1.65,
            translation = Vec3(2.2, -0.7, 3.4),
        )

        val output = engine.analyze(source, reference)

        assertEquals(source.sampling, output.source.sampling)
        assertEquals(source.observations.map { it.timestamp }, output.source.frames.map { it.timestamp })
        assertTrue(output.source.frames.all { it.evidenceKind == SpatialEvidenceKind.THREE_D })
        assertTrue(output.source.frames.all { it.rootOrientation?.inputToBodyRotation != null })
        assertTrue(output.diagnostics.reliabilitySegments
            .filter { it.role == VideoRole.SOURCE }
            .all { it.status == SpatialReliabilityStatus.COMPATIBLE })

        val sourcePose = output.source.frames.first().canonicalPose
        val referencePose = output.reference.frames.first().canonicalPose
        assertNotNull(sourcePose)
        assertNotNull(referencePose)
        assertTrue(canonicalRms(sourcePose, referencePose) < 1e-8)

        val view = output.diagnostics.relativeViewHypotheses.single()
        val relativeYaw = assertNotNull(view.relativeYawDegrees)
        val relativeElevation = assertNotNull(view.relativeElevationDegrees)
        assertTrue(abs(relativeYaw) > 20.0)
        assertTrue(abs(relativeElevation) > 5.0)
        assertEquals(MirrorHypothesis.NOT_MIRRORED, view.mirror)
        assertEquals(BodySideHypothesis.BILATERAL, view.selectedBodySide)
        assertTrue(view.sideSelectionStability > 0.95)
        assertTrue(output.diagnostics.aggregateConfidence > 0.65)
    }

    @Test
    fun `segment mirror selection is stable and makes swapped-side evidence comparable`() {
        val reference = sequence(VideoRole.REFERENCE)
        val source = sequence(VideoRole.SOURCE, mirrorGeometryAndLabels = true)

        val output = engine.analyze(source, reference)

        val hypotheses = output.diagnostics.relativeViewHypotheses
        assertEquals(1, hypotheses.size)
        assertEquals(MirrorHypothesis.MIRRORED, hypotheses.single().mirror)
        assertTrue(hypotheses.single().sideSelectionStability > 0.95)
        val direct = output.source.frames.first().intrinsicDescriptor.distanceTo(
            output.reference.frames.first().intrinsicDescriptor,
            mirroredOther = false,
        )
        val mirrored = output.source.frames.first().intrinsicDescriptor.distanceTo(
            output.reference.frames.first().intrinsicDescriptor,
            mirroredOther = true,
        )
        assertNotNull(direct)
        assertNotNull(mirrored)
        assertTrue(mirrored < direct)
    }

    @Test
    fun `uniform body scale is nuisance but nonrigid body proportion changes remain detectable`() {
        val reference = sequence(VideoRole.REFERENCE)
        val uniformlyScaled = sequence(VideoRole.SOURCE, bodyScale = 1.55)
        val longForearm = sequence(
            VideoRole.SOURCE,
            poseMutation = { pose ->
                val elbow = pose.getValue("left_elbow")
                val wrist = pose.getValue("left_wrist")
                pose["left_wrist"] = elbow + (wrist - elbow) * 1.75
            },
        )

        val scaledOutput = engine.analyze(uniformlyScaled, reference)
        val proportionOutput = engine.analyze(longForearm, reference)
        val scaledDistance = scaledOutput.source.frames.first().intrinsicDescriptor.distanceTo(
            scaledOutput.reference.frames.first().intrinsicDescriptor,
        )
        val proportionDistance = proportionOutput.source.frames.first().intrinsicDescriptor.distanceTo(
            proportionOutput.reference.frames.first().intrinsicDescriptor,
        )
        assertNotNull(scaledDistance)
        assertNotNull(proportionDistance)
        assertTrue(scaledDistance < 1e-9)
        assertTrue(proportionDistance > 0.02)
        assertTrue(
            canonicalRms(
                scaledOutput.source.frames.first().canonicalPose!!,
                scaledOutput.reference.frames.first().canonicalPose!!,
            ) < 1e-8,
        )
    }

    @Test
    fun `genuine joint form difference survives viewpoint normalization`() {
        val reference = sequence(VideoRole.REFERENCE)
        val changedForm = sequence(
            VideoRole.SOURCE,
            yawDegrees = 31.0,
            elevationDegrees = -12.0,
            poseMutation = { pose ->
                pose["left_ankle"] = Vec3(0.28, -1.28, 0.58)
            },
        )

        val output = engine.analyze(changedForm, reference)
        val distance = output.source.frames.first().intrinsicDescriptor.distanceTo(
            output.reference.frames.first().intrinsicDescriptor,
        )
        assertNotNull(distance)
        assertTrue(distance > 0.035)
        val sourceKnee = output.source.frames.first().intrinsicDescriptor.values.getValue("angle.left_knee")
        val referenceKnee = output.reference.frames.first().intrinsicDescriptor.values.getValue("angle.left_knee")
        assertTrue(abs(sourceKnee - referenceKnee) > 15.0)
    }

    @Test
    fun `2d evidence remains usable but cannot fabricate yaw elevation or world orientation`() {
        val source = sequence(VideoRole.SOURCE, image2d = true)
        val reference = sequence(VideoRole.REFERENCE, image2d = true)

        val output = engine.analyze(source, reference)

        assertTrue(output.source.frames.all { it.evidenceKind == SpatialEvidenceKind.IMAGE_2D })
        assertTrue(output.source.frames.all { it.rootOrientation?.inputToBodyRotation == null })
        assertTrue(output.source.frames.all { it.rootOrientation?.planarTorsoTiltDegrees != null })
        val view = output.diagnostics.relativeViewHypotheses.single()
        assertNull(view.relativeYawDegrees)
        assertNull(view.relativeElevationDegrees)
        assertTrue(output.diagnostics.reliabilitySegments
            .filter { it.role == VideoRole.SOURCE }
            .all { it.status == SpatialReliabilityStatus.UNRELIABLE && SpatialDiagnosticReason.INSUFFICIENT_3D in it.reasons })
        assertTrue(output.diagnostics.aggregateConfidence in 0.1..0.65)
    }

    @Test
    fun `unqualified xyz evidence remains partial and cannot fabricate calibrated 3d orientation`() {
        val source = sequence(VideoRole.SOURCE, unqualifiedDepth = true)
        val reference = sequence(VideoRole.REFERENCE, unqualifiedDepth = true)

        val output = engine.analyze(source, reference)

        assertTrue(output.source.frames.all { it.evidenceKind == SpatialEvidenceKind.PARTIAL })
        assertTrue(output.source.frames.all { it.rootOrientation?.inputToBodyRotation == null })
        val view = output.diagnostics.relativeViewHypotheses.single()
        assertNull(view.relativeYawDegrees)
        assertNull(view.relativeElevationDegrees)
        assertTrue(output.diagnostics.reliabilitySegments
            .filter { it.role == VideoRole.SOURCE }
            .all {
                it.status == SpatialReliabilityStatus.UNRELIABLE &&
                    SpatialDiagnosticReason.INSUFFICIENT_3D in it.reasons &&
                    SpatialDiagnosticReason.VIEW_AMBIGUITY in it.reasons
            })

        val inputWrist = source.observations.first().channels.single().values.single { it.key == "left_wrist" }
        val canonicalWrist = output.source.frames.first().canonicalPose!!.values.single { it.key == "left_wrist" }
        assertEquals(inputWrist.mask, canonicalWrist.mask)
        assertEquals(inputWrist.confidence, canonicalWrist.confidence)
        assertEquals(inputWrist.values[2], canonicalWrist.values[2])
    }

    @Test
    fun `missing depth never becomes a fabricated canonical 3d coordinate`() {
        val original = sequence(VideoRole.SOURCE)
        val source = original.copy(
            observations = original.observations.map { observation ->
                val channel = observation.channels.single()
                observation.copy(
                    channels = listOf(
                        channel.copy(
                            availability = ChannelAvailability.PARTIAL,
                            values = channel.values.map { value ->
                                if (value.key != "left_wrist") {
                                    value
                                } else {
                                    value.copy(
                                        values = listOf(value.values[0], value.values[1], null),
                                        mask = listOf(true, true, false),
                                    )
                                }
                            },
                        ),
                    ),
                )
            },
        )
        val output = engine.analyze(source, sequence(VideoRole.REFERENCE))

        assertTrue(output.source.frames.all { it.bodyTransform != null })
        output.source.frames.forEach { frame ->
            val wrist = frame.canonicalPose!!.values.single { it.key == "left_wrist" }
            assertEquals(listOf(false, false, false), wrist.mask)
            assertEquals(listOf(null, null, null), wrist.values)
            assertEquals(0.95, wrist.confidence)
        }
    }

    @Test
    fun `gradual camera rotation is diagnosed without framewise hypothesis churn`() {
        val source = sequence(VideoRole.SOURCE, yawByFrame = { index -> index * 12.0 })
        val reference = sequence(VideoRole.REFERENCE)

        val output = engine.analyze(source, reference)

        assertEquals(1, output.source.frames.mapNotNull { it.spatialSegmentId }.distinct().size)
        assertEquals(1, output.diagnostics.relativeViewHypotheses.size)
        assertTrue(output.diagnostics.reliabilitySegments.any { segment ->
            segment.role == VideoRole.SOURCE &&
                segment.status == SpatialReliabilityStatus.UNRELIABLE &&
                SpatialDiagnosticReason.CAMERA_MOVEMENT in segment.reasons
        })
        assertTrue(output.diagnostics.reliabilitySegments.none { segment ->
            segment.role == VideoRole.SOURCE && SpatialDiagnosticReason.CAMERA_DISCONTINUITY in segment.reasons
        })
    }

    @Test
    fun `side visibility switch stays segment stable and surfaces side instability while preserving masks`() {
        val source = sequence(
            VideoRole.SOURCE,
            leftLimbMaskedFrameIndices = (0 until FRAME_COUNT / 2).toSet(),
            rightLimbMaskedFrameIndices = (FRAME_COUNT / 2 until FRAME_COUNT).toSet(),
        )
        val reference = sequence(VideoRole.REFERENCE)

        val output = engine.analyze(source, reference)

        assertEquals(1, output.source.frames.mapNotNull { it.spatialSegmentId }.distinct().size)
        val hypothesis = output.diagnostics.relativeViewHypotheses.single()
        assertEquals(BodySideHypothesis.BILATERAL, hypothesis.selectedBodySide)
        assertTrue(hypothesis.sideSelectionStability < 0.10)
        assertTrue(output.diagnostics.reliabilitySegments.any { segment ->
            segment.role == VideoRole.SOURCE &&
                segment.status == SpatialReliabilityStatus.UNRELIABLE &&
                SpatialDiagnosticReason.SIDE_SELECTION_UNSTABLE in segment.reasons
        })

        val inputWrist = source.observations.first().channels.single().values.single { it.key == "left_wrist" }
        val canonicalWrist = output.source.frames.first().canonicalPose!!.values.single { it.key == "left_wrist" }
        assertEquals(inputWrist.mask, canonicalWrist.mask)
        assertEquals(inputWrist.values, canonicalWrist.values)
        assertEquals(inputWrist.confidence, canonicalWrist.confidence)
    }

    @Test
    fun `short occlusion preserves segment hypothesis while masks and uncertainty remain explicit`() {
        val source = sequence(VideoRole.SOURCE, occludedFrameIndices = setOf(5, 6))
        val reference = sequence(VideoRole.REFERENCE)

        val output = engine.analyze(source, reference)

        assertEquals(1, output.diagnostics.relativeViewHypotheses.size)
        assertEquals(1, output.source.frames.mapNotNull { it.spatialSegmentId }.distinct().size)
        val occluded = output.source.frames.filterIndexed { index, _ -> index in setOf(5, 6) }
        assertTrue(occluded.all { it.canonicalPose == null && it.transformConfidence == 0.0 })
        assertTrue(output.diagnostics.reliabilitySegments.any { segment ->
            segment.role == VideoRole.SOURCE &&
                segment.status == SpatialReliabilityStatus.UNRELIABLE &&
                SpatialDiagnosticReason.OCCLUSION in segment.reasons
        })
        assertTrue(output.source.analyzableFraction < 1.0)
    }

    @Test
    fun `long occlusion creates explicit discontinuity instead of bridging incompatible spans`() {
        val source = sequence(VideoRole.SOURCE, occludedFrameIndices = setOf(4, 5, 6, 7))
        val reference = sequence(VideoRole.REFERENCE)

        val output = engine.analyze(source, reference)

        assertEquals(2, output.source.frames.mapNotNull { it.spatialSegmentId }.distinct().size)
        assertEquals(2, output.diagnostics.relativeViewHypotheses.size)
        assertTrue(output.diagnostics.reliabilitySegments.any { segment ->
            segment.role == VideoRole.SOURCE &&
                segment.status == SpatialReliabilityStatus.DISCONTINUITY &&
                SpatialDiagnosticReason.OCCLUSION in segment.reasons &&
                SpatialDiagnosticReason.TRANSFORM_UNSTABLE in segment.reasons
        })
    }

    @Test
    fun `abrupt stable-articulation view jump creates explicit spatial discontinuity and a new stable segment`() {
        val source = sequence(
            VideoRole.SOURCE,
            yawByFrame = { index -> if (index < FRAME_COUNT / 2) 0.0 else 82.0 },
        )
        val reference = sequence(VideoRole.REFERENCE)

        val output = engine.analyze(source, reference)

        assertEquals(2, output.source.frames.mapNotNull { it.spatialSegmentId }.distinct().size)
        assertEquals(2, output.diagnostics.relativeViewHypotheses.size)
        val discontinuities = output.diagnostics.reliabilitySegments.filter { segment ->
            segment.role == VideoRole.SOURCE && segment.status == SpatialReliabilityStatus.DISCONTINUITY
        }
        assertTrue(discontinuities.isNotEmpty())
        assertTrue(discontinuities.any { SpatialDiagnosticReason.CAMERA_DISCONTINUITY in it.reasons })
        assertTrue(output.diagnostics.relativeViewHypotheses.size < output.source.frames.size)
    }

    @Test
    fun `multi subject ambiguity is incompatible rather than selecting a convenient person`() {
        val source = sequence(VideoRole.SOURCE, ambiguousSubjects = true)
        val reference = sequence(VideoRole.REFERENCE)

        val output = engine.analyze(source, reference)

        assertTrue(output.source.frames.all { it.canonicalPose == null && it.selectedSubjectId == null })
        assertTrue(output.diagnostics.relativeViewHypotheses.isEmpty())
        assertTrue(output.diagnostics.reliabilitySegments
            .filter { it.role == VideoRole.SOURCE }
            .all { it.status == SpatialReliabilityStatus.INCOMPATIBLE && SpatialDiagnosticReason.SUBJECT_AMBIGUITY in it.reasons })
        assertEquals(0.0, output.source.analyzableFraction)
        assertEquals(0.0, output.diagnostics.aggregateConfidence)
    }

    @Test
    fun `required object evidence gap is surfaced for refusal policy without pose-only guessing`() {
        val source = sequence(VideoRole.SOURCE)
        val reference = sequence(VideoRole.REFERENCE)
        val request = SynchronizationRequest(
            source = source,
            reference = reference,
            requirements = SynchronizationRequirements(setOf("human_pose", "object_pose")),
        )

        val output = engine.analyze(request)

        assertEquals(setOf("object_pose"), output.requirementGaps.sourceMissingSemanticTypes)
        assertEquals(setOf("object_pose"), output.requirementGaps.referenceMissingSemanticTypes)
        assertTrue(output.requirementGaps.anyMissing)
        assertTrue(output.diagnostics.aggregateConfidence > 0.0)
    }

    @Test
    fun `torso-only partial visibility keeps a transform but lowers spatial reliability`() {
        val allFrames = (0 until FRAME_COUNT).toSet()
        val source = sequence(
            VideoRole.SOURCE,
            leftLimbMaskedFrameIndices = allFrames,
            rightLimbMaskedFrameIndices = allFrames,
        )
        val reference = sequence(VideoRole.REFERENCE)

        val output = engine.analyze(source, reference)

        assertTrue(output.source.frames.all { it.bodyTransform != null })
        assertTrue(output.source.frames.all { it.intrinsicDescriptor.confidence < 0.30 })
        assertTrue(output.diagnostics.reliabilitySegments
            .filter { it.role == VideoRole.SOURCE }
            .all {
                it.status == SpatialReliabilityStatus.UNRELIABLE &&
                    SpatialDiagnosticReason.POOR_OBSERVATION_COVERAGE in it.reasons
            })
        assertTrue(output.diagnostics.aggregateConfidence < 0.70)
    }

    @Test
    fun `weak pose evidence never becomes fabricated spatial certainty`() {
        val source = sequence(VideoRole.SOURCE, channelConfidence = 0.18)
        val reference = sequence(VideoRole.REFERENCE)

        val output = engine.analyze(source, reference)

        assertTrue(output.source.frames.all { it.evidenceKind == SpatialEvidenceKind.UNAVAILABLE })
        assertTrue(output.source.frames.all { it.canonicalPose == null })
        assertTrue(output.diagnostics.relativeViewHypotheses.isEmpty())
        assertEquals(0.0, output.source.analyzableFraction)
        assertEquals(0.0, output.diagnostics.aggregateConfidence)
        assertTrue(output.diagnostics.reliabilitySegments.any { segment ->
            segment.role == VideoRole.SOURCE && SpatialDiagnosticReason.POOR_OBSERVATION_COVERAGE in segment.reasons
        })
    }

    private fun sequence(
        role: VideoRole,
        yawDegrees: Double = 0.0,
        elevationDegrees: Double = 0.0,
        globalScale: Double = 1.0,
        bodyScale: Double = 1.0,
        translation: Vec3 = Vec3(0.0, 0.0, 0.0),
        image2d: Boolean = false,
        mirrorGeometryAndLabels: Boolean = false,
        occludedFrameIndices: Set<Int> = emptySet(),
        leftLimbMaskedFrameIndices: Set<Int> = emptySet(),
        rightLimbMaskedFrameIndices: Set<Int> = emptySet(),
        ambiguousSubjects: Boolean = false,
        channelConfidence: Double = 0.95,
        unqualifiedDepth: Boolean = false,
        yawByFrame: ((Int) -> Double)? = null,
        poseMutation: (MutableMap<String, Vec3>) -> Unit = {},
    ): CanonicalObservationSequence {
        val observations = (0 until FRAME_COUNT).map { index ->
            val pose = basePose().mapValuesTo(linkedMapOf()) { (_, point) -> point * bodyScale }
            poseMutation(pose)
            val frameYaw = yawByFrame?.invoke(index) ?: yawDegrees
            val channel = poseChannel(
                pose = pose,
                channelId = "human-${role.name.lowercase()}",
                subjectId = "primary",
                yawDegrees = frameYaw,
                elevationDegrees = elevationDegrees,
                globalScale = globalScale,
                translation = translation,
                image2d = image2d,
                mirrorGeometryAndLabels = mirrorGeometryAndLabels,
                occluded = index in occludedFrameIndices,
                maskLeftLimb = index in leftLimbMaskedFrameIndices,
                maskRightLimb = index in rightLimbMaskedFrameIndices,
                confidence = channelConfidence,
                unqualifiedDepth = unqualifiedDepth,
            )
            val channels = if (ambiguousSubjects) {
                listOf(
                    channel.copy(channelId = "human-a", subjectId = "person-a"),
                    channel.copy(channelId = "human-b", subjectId = "person-b"),
                )
            } else {
                listOf(channel)
            }
            CanonicalObservation(TimestampMs(index * FRAME_STEP_MS), channels)
        }
        return CanonicalObservationSequence(
            role = role,
            duration = DurationMs(FRAME_COUNT * FRAME_STEP_MS),
            sampling = ObservationSampling(inputNominalFramesPerSecond = 59.94, analysisFramesPerSecond = 15.0),
            observations = observations,
        )
    }

    private fun poseChannel(
        pose: Map<String, Vec3>,
        channelId: String,
        subjectId: String,
        yawDegrees: Double,
        elevationDegrees: Double,
        globalScale: Double,
        translation: Vec3,
        image2d: Boolean,
        mirrorGeometryAndLabels: Boolean,
        occluded: Boolean,
        maskLeftLimb: Boolean,
        maskRightLimb: Boolean,
        confidence: Double,
        unqualifiedDepth: Boolean,
    ): ObservationChannel {
        val axes = if (image2d) listOf("x", "y") else listOf("x", "y", "z")
        val values = pose.keys.map { outputKey ->
            val sourceKey = if (mirrorGeometryAndLabels) mirrorLandmarkKey(outputKey) else outputKey
            var point = cameraTransform(pose.getValue(sourceKey), yawDegrees, elevationDegrees) * globalScale + translation
            if (mirrorGeometryAndLabels) point = Vec3(-point.x, point.y, point.z)
            val missing = (occluded && outputKey in TORSO_ANCHORS) ||
                (maskLeftLimb && outputKey in LEFT_NON_ANCHORS) ||
                (maskRightLimb && outputKey in RIGHT_NON_ANCHORS)
            if (missing) {
                ObservationValue(
                    key = outputKey,
                    values = List(axes.size) { null },
                    mask = List(axes.size) { false },
                    confidence = 0.0,
                )
            } else if (image2d) {
                ObservationValue(outputKey, listOf(point.x, point.y), listOf(true, true), confidence)
            } else {
                ObservationValue(outputKey, listOf(point.x, point.y, point.z), listOf(true, true, true), confidence)
            }
        }
        val availability = if (values.any { value -> value.mask.any { !it } }) ChannelAvailability.PARTIAL else ChannelAvailability.OBSERVED
        return ObservationChannel(
            channelId = channelId,
            schemaVersion = 1,
            semanticType = "human_pose",
            coordinateSpace = when {
                image2d -> "image_normalized"
                unqualifiedDepth -> "normalized_coordinates"
                else -> "camera_metric"
            },
            subjectId = subjectId,
            componentAxes = axes,
            values = values,
            availability = availability,
            confidence = confidence,
        )
    }

    private fun cameraTransform(point: Vec3, yawDegrees: Double, elevationDegrees: Double): Vec3 {
        val yaw = yawDegrees * PI / 180.0
        val elevation = elevationDegrees * PI / 180.0
        val xYaw = cos(yaw) * point.x + sin(yaw) * point.z
        val yYaw = point.y
        val zYaw = -sin(yaw) * point.x + cos(yaw) * point.z
        return Vec3(
            x = xYaw,
            y = cos(elevation) * yYaw - sin(elevation) * zYaw,
            z = sin(elevation) * yYaw + cos(elevation) * zYaw,
        )
    }

    private fun canonicalRms(left: ObservationChannel, right: ObservationChannel): Double {
        val rightByKey = right.values.associateBy(ObservationValue::key)
        var sum = 0.0
        var count = 0
        for (leftValue in left.values) {
            val rightValue = rightByKey.getValue(leftValue.key)
            for (index in leftValue.values.indices) {
                if (!leftValue.mask[index] || !rightValue.mask[index]) continue
                val delta = leftValue.values[index]!! - rightValue.values[index]!!
                sum += delta * delta
                count += 1
            }
        }
        return sqrt(sum / count.toDouble())
    }

    private fun mirrorLandmarkKey(key: String): String = when {
        key.startsWith("left_") -> "right_" + key.removePrefix("left_")
        key.startsWith("right_") -> "left_" + key.removePrefix("right_")
        else -> key
    }

    private fun basePose(): LinkedHashMap<String, Vec3> = linkedMapOf(
        "left_shoulder" to Vec3(-0.34, 1.00, 0.04),
        "right_shoulder" to Vec3(0.34, 1.00, -0.04),
        "left_elbow" to Vec3(-0.61, 0.58, 0.26),
        "right_elbow" to Vec3(0.50, 0.69, -0.09),
        "left_wrist" to Vec3(-0.44, 0.20, 0.48),
        "right_wrist" to Vec3(0.77, 0.47, -0.16),
        "left_hip" to Vec3(-0.23, 0.00, 0.02),
        "right_hip" to Vec3(0.23, 0.00, -0.02),
        "left_knee" to Vec3(-0.27, -0.88, 0.11),
        "right_knee" to Vec3(0.29, -0.94, -0.08),
        "left_ankle" to Vec3(-0.22, -1.76, 0.05),
        "right_ankle" to Vec3(0.32, -1.82, -0.03),
        "left_foot_index" to Vec3(-0.18, -1.86, 0.34),
        "right_foot_index" to Vec3(0.38, -1.91, 0.29),
    )

    companion object {
        private const val FRAME_COUNT = 12
        private const val FRAME_STEP_MS = 67L
        private val TORSO_ANCHORS = setOf("left_shoulder", "right_shoulder", "left_hip", "right_hip")
        private val LEFT_NON_ANCHORS = setOf("left_elbow", "left_wrist", "left_knee", "left_ankle", "left_foot_index")
        private val RIGHT_NON_ANCHORS = setOf("right_elbow", "right_wrist", "right_knee", "right_ankle", "right_foot_index")
    }
}
