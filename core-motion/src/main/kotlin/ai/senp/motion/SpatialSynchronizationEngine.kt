package ai.senp.motion

import ai.senp.core.contracts.BodyCentricTransform
import ai.senp.core.contracts.BodyCentricTransformEstimate
import ai.senp.core.contracts.BodySideHypothesis
import ai.senp.core.contracts.CanonicalObservation
import ai.senp.core.contracts.CanonicalObservationSequence
import ai.senp.core.contracts.ChannelAvailability
import ai.senp.core.contracts.MirrorHypothesis
import ai.senp.core.contracts.ObservationChannel
import ai.senp.core.contracts.ObservationValue
import ai.senp.core.contracts.RelativeViewHypothesis
import ai.senp.core.contracts.SpatialDiagnosticReason
import ai.senp.core.contracts.SpatialReliabilitySegment
import ai.senp.core.contracts.SpatialReliabilityStatus
import ai.senp.core.contracts.SpatialSynchronizationDiagnostics
import ai.senp.core.contracts.SynchronizationRequest
import ai.senp.core.contracts.TimestampMs
import ai.senp.core.contracts.TimestampRange
import ai.senp.core.contracts.UnitQuaternion
import ai.senp.core.contracts.Vector3
import ai.senp.core.contracts.VideoRole
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Synchronization-v2 spatial lane. It performs only spatial evidence extraction/canonicalization and diagnostics;
 * it deliberately does not segment activity, discover motion units, or establish timestamp correspondence.
 */
class SpatialSynchronizationEngine(
    private val config: SpatialSynchronizationConfig = SpatialSynchronizationConfig(),
) {
    fun analyze(request: SynchronizationRequest): SpatialSynchronizationOutput = analyze(
        source = request.source,
        reference = request.reference,
        requiredChannelSemanticTypes = request.requirements.requiredChannelSemanticTypes,
    )

    fun analyze(
        source: CanonicalObservationSequence,
        reference: CanonicalObservationSequence,
        requiredChannelSemanticTypes: Set<String> = emptySet(),
    ): SpatialSynchronizationOutput {
        require(source.role == VideoRole.SOURCE) { "spatial source sequence must have SOURCE role" }
        require(reference.role == VideoRole.REFERENCE) { "spatial reference sequence must have REFERENCE role" }
        require(requiredChannelSemanticTypes.all { it.isNotBlank() }) { "required semantic types must be non-blank" }

        val sourceWork = processSequence(source)
        val referenceWork = processSequence(reference)
        val relativeViews = buildRelativeViewHypotheses(sourceWork, referenceWork)
        val diagnostics = SpatialSynchronizationDiagnostics(
            sourceTransforms = sourceWork.transformEstimates,
            referenceTransforms = referenceWork.transformEstimates,
            relativeViewHypotheses = relativeViews,
            reliabilitySegments = sourceWork.reliabilitySegments + referenceWork.reliabilitySegments,
            aggregateConfidence = aggregateConfidence(sourceWork, referenceWork, relativeViews),
        )
        val gaps = SpatialRequirementGaps(
            sourceMissingSemanticTypes = requiredChannelSemanticTypes - source.availableSemanticTypes(),
            referenceMissingSemanticTypes = requiredChannelSemanticTypes - reference.availableSemanticTypes(),
        )
        return SpatialSynchronizationOutput(
            diagnostics = diagnostics,
            source = sourceWork.toPublicAnalysis(),
            reference = referenceWork.toPublicAnalysis(),
            requirementGaps = gaps,
        )
    }

    private fun processSequence(sequence: CanonicalObservationSequence): SequenceWork {
        if (sequence.observations.isEmpty()) {
            val reliability = if (sequence.duration.value > 0L) {
                listOf(
                    SpatialReliabilitySegment(
                        role = sequence.role,
                        range = TimestampRange(TimestampMs(0), TimestampMs(sequence.duration.value)),
                        status = SpatialReliabilityStatus.UNRELIABLE,
                        confidence = 0.0,
                        reasons = setOf(SpatialDiagnosticReason.POOR_OBSERVATION_COVERAGE),
                    ),
                )
            } else {
                emptyList()
            }
            return SequenceWork(
                sequence = sequence,
                frames = emptyList(),
                segments = emptyList(),
                transformEstimates = emptyList(),
                reliabilitySegments = reliability,
                analyzableFraction = 0.0,
                reliabilityScore = 0.0,
            )
        }

        val works = sequence.observations.mapIndexed { index, observation ->
            val intervalEnd = if (index + 1 < sequence.observations.size) {
                sequence.observations[index + 1].timestamp.value
            } else {
                sequence.duration.value
            }
            createFrameWork(observation, intervalEnd)
        }.toMutableList()

        detectContinuityBoundaries(works)
        preserveShortOcclusionsAndSplitLongGaps(works)
        assignSpatialSegments(works)

        val segments = summarizeSegments(works)
        markSideInstability(works, segments)
        val transforms = summarizeTransformRuns(sequence.role, works)
        val reliability = buildReliabilitySegments(sequence, works)
        val analyzableDuration = works.sumOf { work ->
            if (work.transform != null && !work.subjectAmbiguous) work.intervalDurationMs else 0L
        }
        val analyzableFraction = if (sequence.duration.value > 0L) {
            (analyzableDuration.toDouble() / sequence.duration.value.toDouble()).coerceIn(0.0, 1.0)
        } else {
            0.0
        }
        val reliabilityScore = reliabilityScore(sequence.duration.value, reliability)
        return SequenceWork(
            sequence = sequence,
            frames = works,
            segments = segments,
            transformEstimates = transforms,
            reliabilitySegments = reliability,
            analyzableFraction = analyzableFraction,
            reliabilityScore = reliabilityScore,
        )
    }

    private fun createFrameWork(observation: CanonicalObservation, intervalEndMs: Long): FrameWork {
        val selection = selectPoseEvidence(observation)
        val evidence = selection.evidence
        if (evidence == null) {
            return FrameWork(
                observation = observation,
                intervalEndMs = intervalEndMs,
                evidence = null,
                subjectAmbiguous = selection.subjectAmbiguous,
                transform = null,
                canonicalPose = null,
                rootOrientation = null,
                descriptor = SpatialIntrinsicDescriptor(emptyMap(), 0.0),
                transformConfidence = 0.0,
                degenerateGeometry = false,
                leftCoverage = 0.0,
                rightCoverage = 0.0,
            )
        }

        val descriptor = buildIntrinsicDescriptor(evidence)
        val transformResult = if (!selection.subjectAmbiguous) buildBodyTransform(observation.timestamp, evidence) else null
        return FrameWork(
            observation = observation,
            intervalEndMs = intervalEndMs,
            evidence = evidence,
            subjectAmbiguous = selection.subjectAmbiguous,
            transform = transformResult?.transform,
            canonicalPose = transformResult?.let { canonicalizeChannel(evidence.channel, it) },
            rootOrientation = transformResult?.rootOrientation,
            descriptor = descriptor,
            transformConfidence = transformResult?.confidence ?: 0.0,
            degenerateGeometry = transformResult == null && evidence.anchorCoverage >= 0.99 && evidence.anchorConfidence >= config.minimumAnchorConfidence,
            leftCoverage = sideCoverage(evidence, left = true),
            rightCoverage = sideCoverage(evidence, left = false),
        )
    }

    private fun selectPoseEvidence(observation: CanonicalObservation): EvidenceSelection {
        val semanticTypes = config.landmarkSchema.humanPoseSemanticTypes.mapTo(hashSetOf()) { it.lowercase() }
        val candidates = observation.channels
            .filter { channel ->
                channel.semanticType.lowercase() in semanticTypes &&
                    channel.availability != ChannelAvailability.MISSING &&
                    channel.confidence >= config.minimumChannelConfidence
            }
            .mapNotNull(::toPoseEvidence)
        if (candidates.isEmpty()) return EvidenceSelection(null, false)

        val subjectIds = candidates.map { it.channel.subjectId }
        val distinctNamedSubjects = subjectIds.filterNotNull().distinct()
        val subjectAmbiguous = candidates.size > 1 &&
            (distinctNamedSubjects.size != 1 || subjectIds.any { it == null })
        val best = candidates.maxWithOrNull(
            compareBy<PoseEvidence> { if (it.isReliable3d) 1 else 0 }
                .thenBy { it.anchorCoverage }
                .thenBy { it.anchorConfidence }
                .thenBy { it.channel.confidence },
        )
        return EvidenceSelection(best, subjectAmbiguous)
    }

    private fun toPoseEvidence(channel: ObservationChannel): PoseEvidence? {
        val xIndex = channel.componentAxes.indexOfFirst { it.equals("x", ignoreCase = true) }
        val yIndex = channel.componentAxes.indexOfFirst { it.equals("y", ignoreCase = true) }
        if (xIndex < 0 || yIndex < 0) return null
        val zIndex = channel.componentAxes.indexOfFirst { it.equals("z", ignoreCase = true) }
        val coordinateSpace = channel.coordinateSpace?.lowercase()
        val semanticType = channel.semanticType.lowercase()
        val imageLike = coordinateSpace?.let { space ->
            "image" in space || "pixel" in space || "screen" in space
        } == true
        val explicitThreeDimensionalSpace = coordinateSpace?.let { space ->
            "world" in space || "metric" in space || "3d" in space
        } == true || "3d" in semanticType
        val threeDimensionalCandidate = zIndex >= 0 && explicitThreeDimensionalSpace && !imageLike
        val points = channel.values.associate { value ->
            value.key to extractPoint(value, channel.confidence, xIndex, yIndex, zIndex)
        }
        val schema = config.landmarkSchema
        val anchors = listOf(schema.leftShoulder, schema.rightShoulder, schema.leftHip, schema.rightHip)
        val anchorCoverage = anchors.count { key ->
            val point = points[key]
            point != null && point.xyPresent && (!threeDimensionalCandidate || point.zPresent)
        }.toDouble() / anchors.size.toDouble()
        val anchorConfidence = anchors.mapNotNull { key ->
            val point = points[key]
            if (point != null && point.xyPresent && (!threeDimensionalCandidate || point.zPresent)) point.confidence else null
        }.averageOrZero()
        val reliable3d = threeDimensionalCandidate && anchorCoverage >= 0.999 &&
            anchorConfidence >= config.minimumAnchorConfidence
        return PoseEvidence(
            channel = channel,
            points = points,
            isImageLike = imageLike,
            isReliable3d = reliable3d,
            anchorCoverage = anchorCoverage,
            anchorConfidence = anchorConfidence,
        )
    }

    private fun extractPoint(
        value: ObservationValue,
        channelConfidence: Double,
        xIndex: Int,
        yIndex: Int,
        zIndex: Int,
    ): PointEvidence {
        val x = value.values[xIndex].takeIf { value.mask[xIndex] }
        val y = value.values[yIndex].takeIf { value.mask[yIndex] }
        val z = if (zIndex >= 0) value.values[zIndex].takeIf { value.mask[zIndex] } else null
        val xyPresent = x != null && y != null
        return PointEvidence(
            position = if (xyPresent) Vec3(x!!, y!!, z ?: 0.0) else null,
            xyPresent = xyPresent,
            zPresent = z != null,
            confidence = min(value.confidence, channelConfidence),
        )
    }

    private fun buildBodyTransform(timestamp: TimestampMs, evidence: PoseEvidence): FrameTransformResult? {
        val schema = config.landmarkSchema
        val leftHip = evidence.requiredPoint(schema.leftHip) ?: return null
        val rightHip = evidence.requiredPoint(schema.rightHip) ?: return null
        val leftShoulder = evidence.requiredPoint(schema.leftShoulder) ?: return null
        val rightShoulder = evidence.requiredPoint(schema.rightShoulder) ?: return null
        val required = listOf(leftHip, rightHip, leftShoulder, rightShoulder)
        if (required.any { it.confidence < config.minimumAnchorConfidence }) return null
        if (evidence.isReliable3d && required.any { !it.zPresent }) return null

        fun comparablePosition(point: PointEvidence): Vec3 {
            val position = point.position!!
            return if (evidence.isReliable3d) position else Vec3(position.x, position.y, 0.0)
        }
        val leftHipPosition = comparablePosition(leftHip)
        val rightHipPosition = comparablePosition(rightHip)
        val leftShoulderPosition = comparablePosition(leftShoulder)
        val rightShoulderPosition = comparablePosition(rightShoulder)
        val root = (leftHipPosition + rightHipPosition) / 2.0
        val shoulderCenter = (leftShoulderPosition + rightShoulderPosition) / 2.0
        val torso = shoulderCenter - root
        val torsoScale = torso.norm()
        if (!torsoScale.isFinite() || torsoScale < config.minimumTorsoScale) return null
        val confidence = (
            required.minOf(PointEvidence::confidence) *
                geometryConfidence(
                    leftHipPosition,
                    rightHipPosition,
                    leftShoulderPosition,
                    rightShoulderPosition,
                    torsoScale,
                )
            ).coerceIn(0.0, 1.0)
        if (confidence <= 0.0) return null

        return if (evidence.isReliable3d) {
            val lateral = (((rightShoulderPosition - leftShoulderPosition) +
                (rightHipPosition - leftHipPosition)) / 2.0).unitOrNull(config.minimumTorsoScale) ?: return null
            val projectedUp = torso - lateral * torso.dot(lateral)
            val up = projectedUp.unitOrNull(config.minimumTorsoScale) ?: return null
            val forward = lateral.cross(up).unitOrNull(config.minimumTorsoScale) ?: return null
            val correctedUp = forward.cross(lateral).unitOrNull(config.minimumTorsoScale) ?: return null
            val rotation = quaternionFromRows(lateral, correctedUp, forward)
            val scale = 1.0 / torsoScale
            val rotatedRoot = rotateByRows(root, lateral, correctedUp, forward)
            val translation = rotatedRoot * -scale
            val transform = BodyCentricTransform(
                fromCoordinateSpace = inputCoordinateSpace(evidence.channel),
                toCoordinateSpace = SpatialSynchronizationVersions.BODY_CENTRIC_3D,
                translation = translation.toContractVector(),
                rotation = rotation,
                uniformScale = scale,
            )
            FrameTransformResult(
                transform = transform,
                root = root,
                torsoScale = torsoScale,
                confidence = confidence,
                basis = Basis3(lateral, correctedUp, forward),
                rootOrientation = SpatialRootOrientation(
                    timestamp = timestamp,
                    coordinateSpace = evidence.channel.coordinateSpace,
                    pelvisInInputSpace = root.toContractVector(),
                    inputToBodyRotation = rotation,
                    lateralAxisInInputSpace = lateral.toContractVector(),
                    torsoUpAxisInInputSpace = correctedUp.toContractVector(),
                    forwardAxisInInputSpace = forward.toContractVector(),
                    planarTorsoTiltDegrees = null,
                    confidence = confidence,
                ),
            )
        } else {
            val tilt = atan2(torso.x, -torso.y) * 180.0 / PI
            val scale = 1.0 / torsoScale
            val transform = BodyCentricTransform(
                fromCoordinateSpace = inputCoordinateSpace(evidence.channel),
                toCoordinateSpace = SpatialSynchronizationVersions.BODY_CENTRIC_2D,
                translation = (root * -scale).toContractVector(),
                rotation = UnitQuaternion.Identity,
                uniformScale = scale,
            )
            val lateral2d = (((rightShoulderPosition - leftShoulderPosition) +
                (rightHipPosition - leftHipPosition)) / 2.0).unitOrNull(config.minimumTorsoScale)
            val up2d = torso.unitOrNull(config.minimumTorsoScale)
            FrameTransformResult(
                transform = transform,
                root = root,
                torsoScale = torsoScale,
                confidence = confidence * 0.72,
                basis = null,
                rootOrientation = SpatialRootOrientation(
                    timestamp = timestamp,
                    coordinateSpace = evidence.channel.coordinateSpace,
                    pelvisInInputSpace = root.toContractVector(),
                    inputToBodyRotation = null,
                    lateralAxisInInputSpace = lateral2d?.toContractVector(),
                    torsoUpAxisInInputSpace = up2d?.toContractVector(),
                    forwardAxisInInputSpace = null,
                    planarTorsoTiltDegrees = normalizeDegrees(tilt),
                    confidence = (confidence * 0.72).coerceIn(0.0, 1.0),
                ),
            )
        }
    }

    private fun geometryConfidence(
        leftHip: Vec3?,
        rightHip: Vec3?,
        leftShoulder: Vec3?,
        rightShoulder: Vec3?,
        torsoScale: Double,
    ): Double {
        if (listOf(leftHip, rightHip, leftShoulder, rightShoulder).any { it == null }) return 0.0
        val shoulderWidth = (rightShoulder!! - leftShoulder!!).norm() / torsoScale
        val hipWidth = (rightHip!! - leftHip!!).norm() / torsoScale
        if (!shoulderWidth.isFinite() || !hipWidth.isFinite()) return 0.0
        if (shoulderWidth < 0.05 || hipWidth < 0.03 || shoulderWidth > 3.5 || hipWidth > 3.5) return 0.0
        val widthBalance = min(shoulderWidth, hipWidth) / max(shoulderWidth, hipWidth)
        return (0.70 + 0.30 * widthBalance).coerceIn(0.0, 1.0)
    }

    private fun canonicalizeChannel(channel: ObservationChannel, result: FrameTransformResult): ObservationChannel {
        val xIndex = channel.componentAxes.indexOfFirst { it.equals("x", ignoreCase = true) }
        val yIndex = channel.componentAxes.indexOfFirst { it.equals("y", ignoreCase = true) }
        val zIndex = channel.componentAxes.indexOfFirst { it.equals("z", ignoreCase = true) }
        fun maskUncanonicalizableCoordinates(value: ObservationValue, requireZ: Boolean): ObservationValue {
            val values = value.values.toMutableList()
            val mask = value.mask.toMutableList()
            listOf(xIndex, yIndex).filter { it >= 0 }.forEach { index ->
                values[index] = null
                mask[index] = false
            }
            if (requireZ && zIndex >= 0) {
                values[zIndex] = null
                mask[zIndex] = false
            }
            return value.copy(values = values, mask = mask)
        }
        val transformedValues = channel.values.map { value ->
            if (xIndex < 0 || yIndex < 0 || !value.mask[xIndex] || !value.mask[yIndex]) {
                return@map maskUncanonicalizableCoordinates(value, requireZ = result.basis != null)
            }
            val x = value.values[xIndex] ?: return@map maskUncanonicalizableCoordinates(value, requireZ = result.basis != null)
            val y = value.values[yIndex] ?: return@map maskUncanonicalizableCoordinates(value, requireZ = result.basis != null)
            val zPresent = zIndex >= 0 && value.mask[zIndex] && value.values[zIndex] != null
            if (result.basis != null && !zPresent) {
                return@map maskUncanonicalizableCoordinates(value, requireZ = true)
            }
            val raw = Vec3(x, y, if (zPresent) value.values[zIndex]!! else 0.0)
            val canonical = if (result.basis != null) {
                rotateByRows(raw - result.root, result.basis.x, result.basis.y, result.basis.z) / result.torsoScale
            } else {
                (Vec3(raw.x, raw.y, 0.0) - result.root) / result.torsoScale
            }
            val components = value.values.toMutableList()
            components[xIndex] = canonical.x
            components[yIndex] = canonical.y
            if (result.basis != null && zIndex >= 0 && zPresent) components[zIndex] = canonical.z
            value.copy(values = components)
        }
        return channel.copy(
            channelId = canonicalChannelId(channel.channelId),
            coordinateSpace = result.transform.toCoordinateSpace,
            values = transformedValues,
        )
    }

    private fun canonicalChannelId(channelId: String): String = if (channelId.length <= 123) {
        "$channelId-body"
    } else {
        channelId.take(123) + "-body"
    }

    private fun buildIntrinsicDescriptor(evidence: PoseEvidence): SpatialIntrinsicDescriptor {
        val schema = config.landmarkSchema
        val values = linkedMapOf<String, Double>()
        val confidences = mutableListOf<Double>()

        fun comparablePosition(point: PointEvidence): Vec3 {
            val position = point.position!!
            return if (evidence.isReliable3d) position else Vec3(position.x, position.y, 0.0)
        }

        fun angle(name: String, a: String, b: String, c: String) {
            val pa = evidence.usablePoint(a, config.minimumDescriptorConfidence) ?: return
            val pb = evidence.usablePoint(b, config.minimumDescriptorConfidence) ?: return
            val pc = evidence.usablePoint(c, config.minimumDescriptorConfidence) ?: return
            val angle = vectorAngleDegrees(
                comparablePosition(pa) - comparablePosition(pb),
                comparablePosition(pc) - comparablePosition(pb),
            ) ?: return
            values["angle.$name"] = angle
            confidences += minOf(pa.confidence, pb.confidence, pc.confidence)
        }

        angle("left_shoulder", schema.leftElbow, schema.leftShoulder, schema.leftHip)
        angle("right_shoulder", schema.rightElbow, schema.rightShoulder, schema.rightHip)
        angle("left_elbow", schema.leftShoulder, schema.leftElbow, schema.leftWrist)
        angle("right_elbow", schema.rightShoulder, schema.rightElbow, schema.rightWrist)
        angle("left_hip", schema.leftShoulder, schema.leftHip, schema.leftKnee)
        angle("right_hip", schema.rightShoulder, schema.rightHip, schema.rightKnee)
        angle("left_knee", schema.leftHip, schema.leftKnee, schema.leftAnkle)
        angle("right_knee", schema.rightHip, schema.rightKnee, schema.rightAnkle)
        angle("left_ankle", schema.leftKnee, schema.leftAnkle, schema.leftFoot)
        angle("right_ankle", schema.rightKnee, schema.rightAnkle, schema.rightFoot)

        val leftHip = evidence.usablePoint(schema.leftHip, config.minimumDescriptorConfidence)
        val rightHip = evidence.usablePoint(schema.rightHip, config.minimumDescriptorConfidence)
        val leftShoulder = evidence.usablePoint(schema.leftShoulder, config.minimumDescriptorConfidence)
        val rightShoulder = evidence.usablePoint(schema.rightShoulder, config.minimumDescriptorConfidence)
        val root = if (leftHip != null && rightHip != null) {
            (comparablePosition(leftHip) + comparablePosition(rightHip)) / 2.0
        } else {
            null
        }
        val shoulderCenter = if (leftShoulder != null && rightShoulder != null) {
            (comparablePosition(leftShoulder) + comparablePosition(rightShoulder)) / 2.0
        } else {
            null
        }
        val torso = if (root != null && shoulderCenter != null) (shoulderCenter - root).norm() else null
        if (torso != null && torso >= config.minimumTorsoScale) {
            fun ratio(name: String, a: String, b: String) {
                val pa = evidence.usablePoint(a, config.minimumDescriptorConfidence) ?: return
                val pb = evidence.usablePoint(b, config.minimumDescriptorConfidence) ?: return
                val length = (comparablePosition(pa) - comparablePosition(pb)).norm()
                if (!length.isFinite()) return
                values["ratio.$name"] = length / torso
                confidences += min(pa.confidence, pb.confidence)
            }
            ratio("shoulder_width_over_torso", schema.leftShoulder, schema.rightShoulder)
            ratio("hip_width_over_torso", schema.leftHip, schema.rightHip)
            ratio("left_upper_arm_over_torso", schema.leftShoulder, schema.leftElbow)
            ratio("right_upper_arm_over_torso", schema.rightShoulder, schema.rightElbow)
            ratio("left_forearm_over_torso", schema.leftElbow, schema.leftWrist)
            ratio("right_forearm_over_torso", schema.rightElbow, schema.rightWrist)
            ratio("left_thigh_over_torso", schema.leftHip, schema.leftKnee)
            ratio("right_thigh_over_torso", schema.rightHip, schema.rightKnee)
            ratio("left_shin_over_torso", schema.leftKnee, schema.leftAnkle)
            ratio("right_shin_over_torso", schema.rightKnee, schema.rightAnkle)

            if (evidence.isReliable3d) {
                val leftWrist = evidence.usablePoint(schema.leftWrist, config.minimumDescriptorConfidence)
                val rightWrist = evidence.usablePoint(schema.rightWrist, config.minimumDescriptorConfidence)
                if (leftWrist != null && rightWrist != null && leftShoulder != null && rightShoulder != null) {
                    val stableRoot = requireNotNull(root)
                    val stableShoulderCenter = requireNotNull(shoulderCenter)
                    val lateral = comparablePosition(rightShoulder) - comparablePosition(leftShoulder)
                    val up = stableShoulderCenter - stableRoot
                    val handCenter = (comparablePosition(leftWrist) + comparablePosition(rightWrist)) / 2.0
                    val handOffset = handCenter - stableShoulderCenter
                    val normalizedSignedVolume = lateral.dot(up.cross(handOffset)) / (torso * torso * torso)
                    if (normalizedSignedVolume.isFinite()) {
                        // Proper rotations preserve scalar-triple-product parity; reflections invert it.
                        values["signed.chirality"] = normalizedSignedVolume
                        confidences += minOf(
                            leftShoulder.confidence,
                            rightShoulder.confidence,
                            leftWrist.confidence,
                            rightWrist.confidence,
                        )
                    }
                }
            }
        }
        val confidence = if (values.isEmpty()) 0.0 else (
            confidences.averageOrZero() * min(1.0, values.size / 8.0)
            ).coerceIn(0.0, 1.0)
        return SpatialIntrinsicDescriptor(values, confidence)
    }

    private fun sideCoverage(evidence: PoseEvidence, left: Boolean): Double {
        val schema = config.landmarkSchema
        val keys = if (left) {
            listOf(schema.leftShoulder, schema.leftElbow, schema.leftWrist, schema.leftHip, schema.leftKnee, schema.leftAnkle)
        } else {
            listOf(schema.rightShoulder, schema.rightElbow, schema.rightWrist, schema.rightHip, schema.rightKnee, schema.rightAnkle)
        }
        return keys.map { key ->
            val point = evidence.points[key]
            if (point?.xyPresent == true) point.confidence else 0.0
        }.averageOrZero().coerceIn(0.0, 1.0)
    }

    private fun detectContinuityBoundaries(works: MutableList<FrameWork>) {
        for (index in 1 until works.size) {
            val previous = works[index - 1]
            val current = works[index]
            val elapsedMs = current.timestampMs - previous.timestampMs
            if (elapsedMs > config.discontinuityGapMs) {
                current.boundaryBefore = true
                current.discontinuityReasons += SpatialDiagnosticReason.TRANSFORM_UNSTABLE
                current.discontinuityReasons += SpatialDiagnosticReason.OCCLUSION
            }
            if (current.subjectAmbiguous) {
                current.boundaryBefore = true
                current.discontinuityReasons += SpatialDiagnosticReason.SUBJECT_AMBIGUITY
            } else if (previous.subjectAmbiguous) {
                current.boundaryBefore = true
                current.discontinuityReasons += SpatialDiagnosticReason.SUBJECT_AMBIGUITY
                current.discontinuityReasons += SpatialDiagnosticReason.TRANSFORM_UNSTABLE
            }
            val previousSubject = previous.evidence?.channel?.subjectId
            val currentSubject = current.evidence?.channel?.subjectId
            if (previousSubject != null && currentSubject != null && previousSubject != currentSubject) {
                current.boundaryBefore = true
                current.discontinuityReasons += SpatialDiagnosticReason.SUBJECT_AMBIGUITY
            }
            val previousTransform = previous.transform
            val currentTransform = current.transform
            if (previousTransform == null || currentTransform == null || elapsedMs <= 0L || elapsedMs > config.discontinuityGapMs) {
                continue
            }
            if (previousTransform.fromCoordinateSpace != currentTransform.fromCoordinateSpace ||
                previousTransform.toCoordinateSpace != currentTransform.toCoordinateSpace
            ) {
                current.boundaryBefore = true
                current.discontinuityReasons += SpatialDiagnosticReason.TRANSFORM_UNSTABLE
                continue
            }

            val descriptorDistance = previous.descriptor.distanceTo(current.descriptor)
            val intrinsicStable = descriptorDistance != null && descriptorDistance <= config.stableIntrinsicDistance
            if (!intrinsicStable) continue

            val rotationDelta = quaternionAngleDegrees(previousTransform.rotation, currentTransform.rotation)
            val rootDelta = normalizedRootDelta(previous, current)
            val scaleRatio = max(
                previousTransform.uniformScale / currentTransform.uniformScale,
                currentTransform.uniformScale / previousTransform.uniformScale,
            )
            val discontinuity = rotationDelta >= config.cameraDiscontinuityRotationDegrees ||
                rootDelta >= config.cameraDiscontinuityRootTorsoUnits ||
                scaleRatio >= config.cameraDiscontinuityScaleRatio
            if (discontinuity) {
                current.boundaryBefore = true
                current.discontinuityReasons += SpatialDiagnosticReason.CAMERA_DISCONTINUITY
                current.discontinuityReasons += SpatialDiagnosticReason.TRANSFORM_UNSTABLE
            } else if (rotationDelta >= config.cameraMotionRotationDegrees ||
                rootDelta >= config.cameraMotionRootTorsoUnits ||
                abs(ln(scaleRatio)) >= 0.08
            ) {
                current.cameraMovement = true
            }
        }
    }

    private fun normalizedRootDelta(previous: FrameWork, current: FrameWork): Double {
        val previousRoot = previous.rootOrientation?.pelvisInInputSpace?.toInternalVec() ?: return 0.0
        val currentRoot = current.rootOrientation?.pelvisInInputSpace?.toInternalVec() ?: return 0.0
        val previousScale = previous.transform?.uniformScale?.let { 1.0 / it } ?: return 0.0
        val currentScale = current.transform?.uniformScale?.let { 1.0 / it } ?: return 0.0
        val torso = (previousScale + currentScale) / 2.0
        return if (torso <= config.minimumTorsoScale) 0.0 else (currentRoot - previousRoot).norm() / torso
    }

    private fun preserveShortOcclusionsAndSplitLongGaps(works: MutableList<FrameWork>) {
        var index = 0
        while (index < works.size) {
            if (works[index].transform != null || works[index].subjectAmbiguous) {
                index += 1
                continue
            }
            val start = index
            var end = index
            while (end + 1 < works.size && works[end + 1].transform == null && !works[end + 1].subjectAmbiguous) end += 1
            val runStart = works[start].timestampMs
            val runEnd = works[end].intervalEndMs
            val duration = max(0L, runEnd - runStart)
            if (duration > config.shortOcclusionMs) {
                if (start > 0 && works[start - 1].transform != null) {
                    works[start].boundaryBefore = true
                }
                if (end + 1 < works.size && works[end + 1].transform != null) {
                    works[end + 1].boundaryBefore = true
                    works[end + 1].discontinuityReasons += SpatialDiagnosticReason.OCCLUSION
                    works[end + 1].discontinuityReasons += SpatialDiagnosticReason.TRANSFORM_UNSTABLE
                }
            }
            index = end + 1
        }
    }

    private fun assignSpatialSegments(works: MutableList<FrameWork>) {
        var segment = -1
        var haveGoodFrame = false
        var previousSegment: Int? = null
        for (work in works) {
            if (work.transform != null && !work.subjectAmbiguous) {
                if (!haveGoodFrame || work.boundaryBefore) segment += 1
                haveGoodFrame = true
                work.segmentId = segment
                previousSegment = segment
            } else if (work.boundaryBefore) {
                work.segmentId = null
                previousSegment = null
                haveGoodFrame = false
            } else if (previousSegment != null) {
                work.segmentId = previousSegment
            } else {
                work.segmentId = null
            }
        }
    }

    private fun summarizeSegments(works: List<FrameWork>): List<SegmentSummary> {
        return works.filter { it.transform != null && it.segmentId != null && !it.subjectAmbiguous }
            .groupBy { it.segmentId!! }
            .toSortedMap()
            .values
            .mapNotNull { frames -> summarizeSegment(frames, works) }
    }

    private fun summarizeSegment(frames: List<FrameWork>, allWorks: List<FrameWork>): SegmentSummary? {
        if (frames.isEmpty()) return null
        val segmentId = frames.first().segmentId ?: return null
        val indices = allWorks.indices.filter { allWorks[it].segmentId == segmentId }
        val rangeStart = allWorks[indices.first()].timestampMs
        val rangeEnd = allWorks[indices.last()].intervalEndMs
        if (rangeEnd <= rangeStart) return null
        val descriptor = medianDescriptor(frames.map(FrameWork::descriptor))
        val threeDFrames = frames.filter { it.evidenceKind == SpatialEvidenceKind.THREE_D }
        val rotation = if (threeDFrames.isNotEmpty()) {
            averageQuaternion(threeDFrames.mapNotNull { it.transform?.rotation })
        } else {
            null
        }
        val stability = transformStability(frames)
        val left = frames.map(FrameWork::leftCoverage).averageOrZero()
        val right = frames.map(FrameWork::rightCoverage).averageOrZero()
        val preferred = preferredSide(left, right)
        val sideStability = sideStability(frames, preferred)
        return SegmentSummary(
            segmentId = segmentId,
            range = TimestampRange(TimestampMs(rangeStart), TimestampMs(rangeEnd)),
            descriptor = descriptor,
            representativeInputToBody = rotation,
            threeDFraction = threeDFrames.size.toDouble() / frames.size.toDouble(),
            confidence = frames.map(FrameWork::transformConfidence).averageOrZero().coerceIn(0.0, 1.0),
            stability = stability,
            leftCoverage = left,
            rightCoverage = right,
            preferredSide = preferred,
            sideStability = sideStability,
        )
    }

    private fun medianDescriptor(descriptors: List<SpatialIntrinsicDescriptor>): SpatialIntrinsicDescriptor {
        val keys = descriptors.flatMap { it.values.keys }.toSet()
        val values = linkedMapOf<String, Double>()
        for (key in keys.sorted()) {
            val samples = descriptors.mapNotNull { it.values[key] }
            if (samples.isNotEmpty()) values[key] = median(samples)
        }
        val confidence = descriptors.map(SpatialIntrinsicDescriptor::confidence).averageOrZero().coerceIn(0.0, 1.0)
        return SpatialIntrinsicDescriptor(values, confidence)
    }

    private fun preferredSide(left: Double, right: Double): BodySideHypothesis = when {
        max(left, right) < config.sideCoverageThreshold -> BodySideHypothesis.UNKNOWN
        left >= config.sideCoverageThreshold && right >= config.sideCoverageThreshold &&
            abs(left - right) <= config.sideDominanceMargin -> BodySideHypothesis.BILATERAL
        left > right + config.sideDominanceMargin -> BodySideHypothesis.LEFT
        right > left + config.sideDominanceMargin -> BodySideHypothesis.RIGHT
        else -> BodySideHypothesis.BILATERAL
    }

    private fun sideStability(frames: List<FrameWork>, side: BodySideHypothesis): Double {
        if (frames.isEmpty()) return 0.0
        val stable = frames.count { frame ->
            when (side) {
                BodySideHypothesis.LEFT -> frame.leftCoverage + config.sideDominanceMargin >= frame.rightCoverage
                BodySideHypothesis.RIGHT -> frame.rightCoverage + config.sideDominanceMargin >= frame.leftCoverage
                BodySideHypothesis.BILATERAL ->
                    frame.leftCoverage >= config.sideCoverageThreshold && frame.rightCoverage >= config.sideCoverageThreshold
                BodySideHypothesis.UNKNOWN -> max(frame.leftCoverage, frame.rightCoverage) < config.sideCoverageThreshold
            }
        }
        return stable.toDouble() / frames.size.toDouble()
    }

    private fun markSideInstability(works: List<FrameWork>, segments: List<SegmentSummary>) {
        val unstableSegmentIds = segments
            .filter { segment ->
                segment.preferredSide != BodySideHypothesis.UNKNOWN &&
                    segment.sideStability < config.minimumSideSelectionStability
            }
            .mapTo(hashSetOf(), SegmentSummary::segmentId)
        works.forEach { work ->
            work.sideSelectionUnstable = work.transform != null && work.segmentId in unstableSegmentIds
        }
    }

    private fun summarizeTransformRuns(role: VideoRole, works: List<FrameWork>): List<BodyCentricTransformEstimate> {
        val estimates = mutableListOf<BodyCentricTransformEstimate>()
        var index = 0
        while (index < works.size) {
            val startWork = works[index]
            if (startWork.transform == null || startWork.subjectAmbiguous || startWork.intervalDurationMs <= 0L) {
                index += 1
                continue
            }
            val segmentId = startWork.segmentId
            val from = startWork.transform.fromCoordinateSpace
            val to = startWork.transform.toCoordinateSpace
            var end = index
            while (end + 1 < works.size) {
                val next = works[end + 1]
                val transform = next.transform ?: break
                if (next.subjectAmbiguous || next.segmentId != segmentId ||
                    transform.fromCoordinateSpace != from || transform.toCoordinateSpace != to ||
                    next.timestampMs != works[end].intervalEndMs
                ) break
                end += 1
            }
            val run = works.subList(index, end + 1)
            val representative = representativeTransform(run) ?: run.first().transform!!
            val rangeEnd = run.last().intervalEndMs
            if (rangeEnd > run.first().timestampMs) {
                estimates += BodyCentricTransformEstimate(
                    role = role,
                    range = TimestampRange(TimestampMs(run.first().timestampMs), TimestampMs(rangeEnd)),
                    transform = representative,
                    confidence = run.map(FrameWork::transformConfidence).averageOrZero().coerceIn(0.0, 1.0),
                    stability = transformStability(run),
                )
            }
            index = end + 1
        }
        return estimates
    }

    private fun representativeTransform(frames: List<FrameWork>): BodyCentricTransform? {
        val transforms = frames.mapNotNull(FrameWork::transform)
        if (transforms.isEmpty()) return null
        val first = transforms.first()
        val rotation = averageQuaternion(transforms.map(BodyCentricTransform::rotation))
        val translation = Vector3(
            transforms.map { it.translation.x }.average(),
            transforms.map { it.translation.y }.average(),
            transforms.map { it.translation.z }.average(),
        )
        val uniformScale = exp(transforms.map { ln(it.uniformScale) }.average())
        return BodyCentricTransform(
            fromCoordinateSpace = first.fromCoordinateSpace,
            toCoordinateSpace = first.toCoordinateSpace,
            translation = translation,
            rotation = rotation,
            uniformScale = uniformScale,
        )
    }

    private fun transformStability(frames: List<FrameWork>): Double {
        if (frames.size <= 1) return frames.firstOrNull()?.transformConfidence?.times(0.85) ?: 0.0
        val deltas = frames.zipWithNext().mapNotNull { (a, b) ->
            val at = a.transform ?: return@mapNotNull null
            val bt = b.transform ?: return@mapNotNull null
            if (at.fromCoordinateSpace != bt.fromCoordinateSpace || at.toCoordinateSpace != bt.toCoordinateSpace) {
                return@mapNotNull 1.0
            }
            val rotationPenalty = quaternionAngleDegrees(at.rotation, bt.rotation) / 30.0
            val scalePenalty = abs(ln(at.uniformScale / bt.uniformScale)) / 0.20
            val translationDelta = sqrt(
                (at.translation.x - bt.translation.x) * (at.translation.x - bt.translation.x) +
                    (at.translation.y - bt.translation.y) * (at.translation.y - bt.translation.y) +
                    (at.translation.z - bt.translation.z) * (at.translation.z - bt.translation.z),
            )
            val translationPenalty = translationDelta / max(0.25, config.cameraMotionRootTorsoUnits * 2.0)
            min(1.0, 0.50 * rotationPenalty + 0.25 * scalePenalty + 0.25 * translationPenalty)
        }
        if (deltas.isEmpty()) return 0.5
        val smoothness = 1.0 - median(deltas).coerceIn(0.0, 1.0)
        val confidence = frames.map(FrameWork::transformConfidence).averageOrZero()
        return (0.55 * smoothness + 0.45 * confidence).coerceIn(0.0, 1.0)
    }

    private fun buildReliabilitySegments(
        sequence: CanonicalObservationSequence,
        works: List<FrameWork>,
    ): List<SpatialReliabilitySegment> {
        if (sequence.duration.value <= 0L) return emptyList()
        val spans = mutableListOf<ReliabilitySpan>()
        val firstTimestamp = works.first().timestampMs
        if (firstTimestamp > 0L) {
            spans += ReliabilitySpan(
                start = 0L,
                end = firstTimestamp,
                status = SpatialReliabilityStatus.UNRELIABLE,
                confidence = 0.0,
                reasons = setOf(SpatialDiagnosticReason.POOR_OBSERVATION_COVERAGE),
            )
        }
        for (work in works) {
            if (work.intervalEndMs <= work.timestampMs) continue
            val assessment = assessReliability(work)
            spans += ReliabilitySpan(
                start = work.timestampMs,
                end = work.intervalEndMs,
                status = assessment.status,
                confidence = assessment.confidence,
                reasons = assessment.reasons,
            )
        }
        val merged = mutableListOf<ReliabilitySpan>()
        for (span in spans) {
            val previous = merged.lastOrNull()
            if (previous != null && previous.end == span.start && previous.status == span.status && previous.reasons == span.reasons) {
                val previousDuration = previous.end - previous.start
                val currentDuration = span.end - span.start
                val total = previousDuration + currentDuration
                val weightedConfidence = if (total > 0L) {
                    (previous.confidence * previousDuration + span.confidence * currentDuration) / total.toDouble()
                } else {
                    min(previous.confidence, span.confidence)
                }
                merged[merged.lastIndex] = previous.copy(end = span.end, confidence = weightedConfidence.coerceIn(0.0, 1.0))
            } else {
                merged += span
            }
        }
        return merged.filter { it.end > it.start }.map { span ->
            SpatialReliabilitySegment(
                role = sequence.role,
                range = TimestampRange(TimestampMs(span.start), TimestampMs(span.end)),
                status = span.status,
                confidence = span.confidence.coerceIn(0.0, 1.0),
                reasons = span.reasons,
            )
        }
    }

    private fun assessReliability(work: FrameWork): ReliabilityAssessment {
        if (work.subjectAmbiguous) {
            return ReliabilityAssessment(
                SpatialReliabilityStatus.INCOMPATIBLE,
                0.0,
                setOf(SpatialDiagnosticReason.SUBJECT_AMBIGUITY),
            )
        }
        if (work.discontinuityReasons.isNotEmpty()) {
            return ReliabilityAssessment(
                SpatialReliabilityStatus.DISCONTINUITY,
                min(0.25, work.transformConfidence),
                work.discontinuityReasons.toSet(),
            )
        }
        if (work.transform == null) {
            val reasons = linkedSetOf(SpatialDiagnosticReason.POOR_OBSERVATION_COVERAGE)
            if (work.evidence != null) reasons += SpatialDiagnosticReason.OCCLUSION
            if (work.degenerateGeometry) reasons += SpatialDiagnosticReason.SPATIAL_INCOMPATIBILITY
            return ReliabilityAssessment(
                if (work.degenerateGeometry) SpatialReliabilityStatus.INCOMPATIBLE else SpatialReliabilityStatus.UNRELIABLE,
                0.0,
                reasons,
            )
        }
        val poorObservationCoverage = work.descriptor.confidence < config.minimumDescriptorConfidence
        if (work.evidenceKind != SpatialEvidenceKind.THREE_D) {
            val reasons = linkedSetOf(
                SpatialDiagnosticReason.INSUFFICIENT_3D,
                SpatialDiagnosticReason.VIEW_AMBIGUITY,
            )
            var confidenceFactor = 0.72
            if (work.cameraMovement) {
                reasons += SpatialDiagnosticReason.CAMERA_MOVEMENT
                reasons += SpatialDiagnosticReason.TRANSFORM_UNSTABLE
                confidenceFactor = min(confidenceFactor, 0.50)
            }
            if (work.sideSelectionUnstable) {
                reasons += SpatialDiagnosticReason.SIDE_SELECTION_UNSTABLE
                confidenceFactor = min(confidenceFactor, 0.62)
            }
            if (poorObservationCoverage) {
                reasons += SpatialDiagnosticReason.POOR_OBSERVATION_COVERAGE
                confidenceFactor = min(confidenceFactor, 0.55)
            }
            return ReliabilityAssessment(
                SpatialReliabilityStatus.UNRELIABLE,
                (work.transformConfidence * confidenceFactor).coerceIn(0.0, 1.0),
                reasons,
            )
        }
        val reasons = linkedSetOf<SpatialDiagnosticReason>()
        var confidenceFactor = 1.0
        if (work.cameraMovement) {
            reasons += SpatialDiagnosticReason.CAMERA_MOVEMENT
            reasons += SpatialDiagnosticReason.TRANSFORM_UNSTABLE
            confidenceFactor = min(confidenceFactor, 0.62)
        }
        if (work.sideSelectionUnstable) {
            reasons += SpatialDiagnosticReason.SIDE_SELECTION_UNSTABLE
            confidenceFactor = min(confidenceFactor, 0.75)
        }
        if (poorObservationCoverage) {
            reasons += SpatialDiagnosticReason.POOR_OBSERVATION_COVERAGE
            confidenceFactor = min(confidenceFactor, 0.55)
        }
        if (reasons.isNotEmpty()) {
            return ReliabilityAssessment(
                SpatialReliabilityStatus.UNRELIABLE,
                (work.transformConfidence * confidenceFactor).coerceIn(0.0, 1.0),
                reasons,
            )
        }
        return ReliabilityAssessment(SpatialReliabilityStatus.COMPATIBLE, work.transformConfidence, emptySet())
    }

    private fun reliabilityScore(durationMs: Long, segments: List<SpatialReliabilitySegment>): Double {
        if (durationMs <= 0L) return 0.0
        var weighted = 0.0
        for (segment in segments) {
            val duration = segment.range.endExclusive.value - segment.range.start.value
            val statusWeight = when (segment.status) {
                SpatialReliabilityStatus.COMPATIBLE -> 1.0
                SpatialReliabilityStatus.UNRELIABLE -> 0.60
                SpatialReliabilityStatus.INCOMPATIBLE -> 0.10
                SpatialReliabilityStatus.DISCONTINUITY -> 0.05
            }
            weighted += duration.toDouble() * segment.confidence * statusWeight
        }
        return (weighted / durationMs.toDouble()).coerceIn(0.0, 1.0)
    }

    private fun buildRelativeViewHypotheses(
        source: SequenceWork,
        reference: SequenceWork,
    ): List<RelativeViewHypothesis> {
        if (source.segments.isEmpty() || reference.segments.isEmpty()) return emptyList()
        val referenceAnchor = reference.segments.maxByOrNull { segment ->
            segment.range.duration().value.toDouble() * segment.confidence * (0.5 + 0.5 * segment.stability)
        } ?: return emptyList()
        return source.segments.map { sourceSegment ->
            relativeViewHypothesis(sourceSegment, referenceAnchor)
        }
    }

    private fun relativeViewHypothesis(
        source: SegmentSummary,
        reference: SegmentSummary,
    ): RelativeViewHypothesis {
        val direct = descriptorDistance(source.descriptor.values, reference.descriptor.values, mirrorRight = false)
        val mirrored = descriptorDistance(source.descriptor.values, reference.descriptor.values, mirrorRight = true)
        val mirrorDecision = selectMirror(direct, mirrored, source, reference)
        val selectedSide = selectCommonSide(source, reference, mirrorDecision.hypothesis)
        val sideStability = min(source.sideStability, reference.sideStability)
        val relativeRotation = if (
            source.representativeInputToBody != null &&
            reference.representativeInputToBody != null &&
            source.threeDFraction >= 0.5 && reference.threeDFraction >= 0.5
        ) {
            multiplyQuaternion(
                inverseQuaternion(source.representativeInputToBody),
                reference.representativeInputToBody,
            )
        } else {
            null
        }
        val viewAngles = relativeRotation?.let(::yawElevationDegrees)
        val threeDConfidence = min(source.threeDFraction, reference.threeDFraction)
        val base = min(source.confidence, reference.confidence) * min(source.stability, reference.stability)
        val viewConfidence = if (relativeRotation != null) {
            (0.55 + 0.45 * threeDConfidence)
        } else {
            0.45
        }
        val confidence = (
            base * viewConfidence * (0.55 + 0.45 * mirrorDecision.confidence) * (0.65 + 0.35 * sideStability)
            ).coerceIn(0.0, 1.0)
        return RelativeViewHypothesis(
            sourceRange = source.range,
            referenceRange = reference.range,
            relativeYawDegrees = viewAngles?.first,
            relativeElevationDegrees = viewAngles?.second,
            mirror = mirrorDecision.hypothesis,
            selectedBodySide = selectedSide,
            confidence = confidence,
            sideSelectionStability = sideStability.coerceIn(0.0, 1.0),
        )
    }

    private fun selectMirror(
        direct: DescriptorDistance?,
        mirrored: DescriptorDistance?,
        source: SegmentSummary,
        reference: SegmentSummary,
    ): MirrorDecision {
        val chiralityDecision = chiralityMirrorDecision(source, reference)
        if (chiralityDecision?.hypothesis == MirrorHypothesis.MIRRORED) return chiralityDecision
        if (direct == null || mirrored == null || direct.commonFeatureCount < 4 || mirrored.commonFeatureCount < 4) {
            if (chiralityDecision != null) return chiralityDecision
            return MirrorDecision(MirrorHypothesis.UNKNOWN, 0.0)
        }
        val base = min(source.descriptor.confidence, reference.descriptor.confidence)
        val difference = direct.distance - mirrored.distance
        if (abs(difference) <= config.mirrorDecisionMargin) {
            val ambiguityConfidence = (base * (1.0 - abs(difference) / max(config.mirrorDecisionMargin, 1e-9)) * 0.55)
                .coerceIn(0.0, 1.0)
            return MirrorDecision(MirrorHypothesis.AMBIGUOUS, ambiguityConfidence)
        }
        val selected = if (difference > 0.0) MirrorHypothesis.MIRRORED else MirrorHypothesis.NOT_MIRRORED
        val separation = abs(difference) / max(0.05, min(direct.distance, mirrored.distance) + abs(difference))
        return MirrorDecision(selected, (base * (0.60 + 0.40 * separation.coerceIn(0.0, 1.0))).coerceIn(0.0, 1.0))
    }

    private fun chiralityMirrorDecision(source: SegmentSummary, reference: SegmentSummary): MirrorDecision? {
        val sourceChirality = source.descriptor.values["signed.chirality"] ?: return null
        val referenceChirality = reference.descriptor.values["signed.chirality"] ?: return null
        val strength = min(abs(sourceChirality), abs(referenceChirality))
        if (strength < config.minimumChiralityMagnitude) return null
        val base = min(source.descriptor.confidence, reference.descriptor.confidence)
        val hypothesis = if (sourceChirality * referenceChirality < 0.0) {
            MirrorHypothesis.MIRRORED
        } else {
            MirrorHypothesis.NOT_MIRRORED
        }
        val confidence = (
            base * (0.70 + 0.30 * (strength / (config.minimumChiralityMagnitude * 4.0)).coerceIn(0.0, 1.0))
            ).coerceIn(0.0, 1.0)
        return MirrorDecision(hypothesis, confidence)
    }

    private fun selectCommonSide(
        source: SegmentSummary,
        reference: SegmentSummary,
        mirror: MirrorHypothesis,
    ): BodySideHypothesis {
        if (mirror == MirrorHypothesis.AMBIGUOUS || mirror == MirrorHypothesis.UNKNOWN) {
            return if (
                min(source.leftCoverage, source.rightCoverage) >= config.sideCoverageThreshold &&
                min(reference.leftCoverage, reference.rightCoverage) >= config.sideCoverageThreshold
            ) {
                BodySideHypothesis.BILATERAL
            } else {
                BodySideHypothesis.UNKNOWN
            }
        }
        val refLeft = if (mirror == MirrorHypothesis.MIRRORED) reference.rightCoverage else reference.leftCoverage
        val refRight = if (mirror == MirrorHypothesis.MIRRORED) reference.leftCoverage else reference.rightCoverage
        val left = min(source.leftCoverage, refLeft)
        val right = min(source.rightCoverage, refRight)
        return when {
            left >= config.sideCoverageThreshold && right >= config.sideCoverageThreshold &&
                abs(left - right) <= config.sideDominanceMargin -> BodySideHypothesis.BILATERAL
            left > right + config.sideDominanceMargin && left >= config.sideCoverageThreshold -> BodySideHypothesis.LEFT
            right > left + config.sideDominanceMargin && right >= config.sideCoverageThreshold -> BodySideHypothesis.RIGHT
            max(left, right) >= config.sideCoverageThreshold -> if (left >= right) BodySideHypothesis.LEFT else BodySideHypothesis.RIGHT
            else -> BodySideHypothesis.UNKNOWN
        }
    }

    private fun aggregateConfidence(
        source: SequenceWork,
        reference: SequenceWork,
        views: List<RelativeViewHypothesis>,
    ): Double {
        val sequenceConfidence = sqrt(source.reliabilityScore * reference.reliabilityScore)
        if (sequenceConfidence <= 0.0) return 0.0
        val viewConfidence = if (views.isEmpty()) 0.35 else views.map(RelativeViewHypothesis::confidence).average()
        return (sequenceConfidence * (0.80 + 0.20 * viewConfidence)).coerceIn(0.0, 1.0)
    }

    private data class EvidenceSelection(val evidence: PoseEvidence?, val subjectAmbiguous: Boolean)

    private data class PointEvidence(
        val position: Vec3?,
        val xyPresent: Boolean,
        val zPresent: Boolean,
        val confidence: Double,
    )

    private data class PoseEvidence(
        val channel: ObservationChannel,
        val points: Map<String, PointEvidence>,
        val isImageLike: Boolean,
        val isReliable3d: Boolean,
        val anchorCoverage: Double,
        val anchorConfidence: Double,
    ) {
        fun requiredPoint(key: String): PointEvidence? = points[key]?.takeIf { point ->
            point.xyPresent && (!isReliable3d || point.zPresent)
        }

        fun usablePoint(key: String, minimumConfidence: Double): PointEvidence? = points[key]?.takeIf { point ->
            point.xyPresent && point.confidence >= minimumConfidence && (!isReliable3d || point.zPresent)
        }
    }

    private data class Basis3(val x: Vec3, val y: Vec3, val z: Vec3)

    private data class FrameTransformResult(
        val transform: BodyCentricTransform,
        val root: Vec3,
        val torsoScale: Double,
        val confidence: Double,
        val basis: Basis3?,
        val rootOrientation: SpatialRootOrientation,
    )

    private data class FrameWork(
        val observation: CanonicalObservation,
        val intervalEndMs: Long,
        val evidence: PoseEvidence?,
        val subjectAmbiguous: Boolean,
        val transform: BodyCentricTransform?,
        val canonicalPose: ObservationChannel?,
        val rootOrientation: SpatialRootOrientation?,
        val descriptor: SpatialIntrinsicDescriptor,
        val transformConfidence: Double,
        val degenerateGeometry: Boolean,
        val leftCoverage: Double,
        val rightCoverage: Double,
        var boundaryBefore: Boolean = false,
        var cameraMovement: Boolean = false,
        var sideSelectionUnstable: Boolean = false,
        val discontinuityReasons: MutableSet<SpatialDiagnosticReason> = linkedSetOf(),
        var segmentId: Int? = null,
    ) {
        val timestampMs: Long get() = observation.timestamp.value
        val intervalDurationMs: Long get() = max(0L, intervalEndMs - timestampMs)
        val evidenceKind: SpatialEvidenceKind
            get() = when {
                transform != null && evidence?.isReliable3d == true -> SpatialEvidenceKind.THREE_D
                transform != null && evidence?.isImageLike == true -> SpatialEvidenceKind.IMAGE_2D
                transform != null -> SpatialEvidenceKind.PARTIAL
                evidence != null -> SpatialEvidenceKind.PARTIAL
                else -> SpatialEvidenceKind.UNAVAILABLE
            }
    }

    private data class SegmentSummary(
        val segmentId: Int,
        val range: TimestampRange,
        val descriptor: SpatialIntrinsicDescriptor,
        val representativeInputToBody: UnitQuaternion?,
        val threeDFraction: Double,
        val confidence: Double,
        val stability: Double,
        val leftCoverage: Double,
        val rightCoverage: Double,
        val preferredSide: BodySideHypothesis,
        val sideStability: Double,
    )

    private data class ReliabilityAssessment(
        val status: SpatialReliabilityStatus,
        val confidence: Double,
        val reasons: Set<SpatialDiagnosticReason>,
    )

    private data class ReliabilitySpan(
        val start: Long,
        val end: Long,
        val status: SpatialReliabilityStatus,
        val confidence: Double,
        val reasons: Set<SpatialDiagnosticReason>,
    )

    private data class MirrorDecision(val hypothesis: MirrorHypothesis, val confidence: Double)

    private data class SequenceWork(
        val sequence: CanonicalObservationSequence,
        val frames: List<FrameWork>,
        val segments: List<SegmentSummary>,
        val transformEstimates: List<BodyCentricTransformEstimate>,
        val reliabilitySegments: List<SpatialReliabilitySegment>,
        val analyzableFraction: Double,
        val reliabilityScore: Double,
    ) {
        fun toPublicAnalysis(): SpatialSequenceAnalysis = SpatialSequenceAnalysis(
            role = sequence.role,
            duration = sequence.duration,
            sampling = sequence.sampling,
            frames = frames.map { frame ->
                SpatialObservationFrame(
                    timestamp = frame.observation.timestamp,
                    evidenceKind = frame.evidenceKind,
                    canonicalPose = frame.canonicalPose,
                    bodyTransform = frame.transform,
                    rootOrientation = frame.rootOrientation,
                    intrinsicDescriptor = frame.descriptor,
                    transformConfidence = frame.transformConfidence,
                    selectedSubjectId = if (frame.subjectAmbiguous) null else frame.evidence?.channel?.subjectId,
                    spatialSegmentId = frame.segmentId,
                )
            },
            analyzableFraction = analyzableFraction,
        )
    }

    private fun inputCoordinateSpace(channel: ObservationChannel): String =
        channel.coordinateSpace ?: "unspecified:${channel.channelId}".take(128)

    private fun Vec3.toContractVector(): Vector3 = Vector3(x, y, z)
    private fun Vector3.toInternalVec(): Vec3 = Vec3(x, y, z)

    private fun Vec3.unitOrNull(minNorm: Double): Vec3? {
        val norm = norm()
        return if (!norm.isFinite() || norm < minNorm) null else this / norm
    }

    private fun rotateByRows(point: Vec3, xAxis: Vec3, yAxis: Vec3, zAxis: Vec3): Vec3 =
        Vec3(point.dot(xAxis), point.dot(yAxis), point.dot(zAxis))

    private fun vectorAngleDegrees(first: Vec3, second: Vec3): Double? {
        val denominator = first.norm() * second.norm()
        if (!denominator.isFinite() || denominator < config.minimumTorsoScale) return null
        return acos((first.dot(second) / denominator).coerceIn(-1.0, 1.0)) * 180.0 / PI
    }

    private fun quaternionFromRows(row0: Vec3, row1: Vec3, row2: Vec3): UnitQuaternion {
        val m00 = row0.x
        val m01 = row0.y
        val m02 = row0.z
        val m10 = row1.x
        val m11 = row1.y
        val m12 = row1.z
        val m20 = row2.x
        val m21 = row2.y
        val m22 = row2.z
        val trace = m00 + m11 + m22
        val raw = if (trace > 0.0) {
            val s = sqrt(trace + 1.0) * 2.0
            QuaternionValues((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
        } else if (m00 > m11 && m00 > m22) {
            val s = sqrt(1.0 + m00 - m11 - m22) * 2.0
            QuaternionValues(0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
        } else if (m11 > m22) {
            val s = sqrt(1.0 + m11 - m00 - m22) * 2.0
            QuaternionValues((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
        } else {
            val s = sqrt(1.0 + m22 - m00 - m11) * 2.0
            QuaternionValues((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)
        }
        return raw.normalized().toContract()
    }

    private fun quaternionAngleDegrees(left: UnitQuaternion, right: UnitQuaternion): Double {
        val dot = abs(left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w).coerceIn(0.0, 1.0)
        return 2.0 * acos(dot) * 180.0 / PI
    }

    private fun inverseQuaternion(value: UnitQuaternion): UnitQuaternion =
        UnitQuaternion(-value.x, -value.y, -value.z, value.w)

    private fun multiplyQuaternion(left: UnitQuaternion, right: UnitQuaternion): UnitQuaternion {
        val raw = QuaternionValues(
            x = left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
            y = left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
            z = left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
            w = left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z,
        )
        return raw.normalized().toContract()
    }

    private fun averageQuaternion(values: List<UnitQuaternion>): UnitQuaternion {
        if (values.isEmpty()) return UnitQuaternion.Identity
        val reference = values.first()
        var x = 0.0
        var y = 0.0
        var z = 0.0
        var w = 0.0
        for (value in values) {
            val dot = reference.x * value.x + reference.y * value.y + reference.z * value.z + reference.w * value.w
            val sign = if (dot < 0.0) -1.0 else 1.0
            x += sign * value.x
            y += sign * value.y
            z += sign * value.z
            w += sign * value.w
        }
        return QuaternionValues(x, y, z, w).normalized().toContract()
    }

    private fun yawElevationDegrees(rotation: UnitQuaternion): Pair<Double, Double> {
        val q = QuaternionValues(rotation.x, rotation.y, rotation.z, rotation.w)
        val xx = q.x * q.x
        val yy = q.y * q.y
        val zz = q.z * q.z
        val xy = q.x * q.y
        val xz = q.x * q.z
        val yz = q.y * q.z
        val wx = q.w * q.x
        val wy = q.w * q.y
        val wz = q.w * q.z
        val m02 = 2.0 * (xz + wy)
        val m21 = 2.0 * (yz + wx)
        val m22 = 1.0 - 2.0 * (xx + yy)
        val yaw = atan2(m02, m22) * 180.0 / PI
        val elevation = asin(m21.coerceIn(-1.0, 1.0)) * 180.0 / PI
        return normalizeDegrees(yaw) to elevation.coerceIn(-90.0, 90.0)
    }

    private fun normalizeDegrees(value: Double): Double {
        var normalized = value
        while (normalized > 180.0) normalized -= 360.0
        while (normalized < -180.0) normalized += 360.0
        return normalized
    }

    private data class QuaternionValues(val x: Double, val y: Double, val z: Double, val w: Double) {
        fun normalized(): QuaternionValues {
            val norm = sqrt(x * x + y * y + z * z + w * w)
            require(norm.isFinite() && norm > 1e-12) { "cannot normalize degenerate quaternion" }
            return QuaternionValues(x / norm, y / norm, z / norm, w / norm)
        }

        fun toContract(): UnitQuaternion = UnitQuaternion(x, y, z, w)
    }

    private fun median(values: List<Double>): Double {
        if (values.isEmpty()) return 0.0
        val sorted = values.sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 1) sorted[middle] else (sorted[middle - 1] + sorted[middle]) / 2.0
    }

    private fun Iterable<Double>.averageOrZero(): Double {
        var total = 0.0
        var count = 0
        for (value in this) {
            total += value
            count += 1
        }
        return if (count == 0) 0.0 else total / count.toDouble()
    }
}
