package ai.senp.alignment.temporal

import ai.senp.core.contracts.TimestampMs
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

internal data class PreparedTemporalFrame(
    val timestamp: TimestampMs,
    val values: Map<String, TemporalFeatureSample>,
    val derivatives: Map<String, Double>,
    val historyDelta: Map<String, Double>,
    val lookaheadDelta: Map<String, Double>,
    val motionMagnitude: Double,
    val confidence: Double,
    val discontinuityBefore: Boolean,
)

internal data class PreparedTemporalSequence(
    val signal: TemporalSignalSequence,
    val frames: List<PreparedTemporalFrame>,
    val featureScales: Map<String, Double>,
) {
    fun framesInside(startMs: Long, endExclusiveMs: Long): List<PreparedTemporalFrame> = frames.filter {
        it.timestamp.value >= startMs && it.timestamp.value < endExclusiveMs
    }
}

internal data class TemporalDescriptorDistance(
    val cost: Double,
    val coverage: Double,
    val oppositeReliable: Boolean,
    val oppositeFraction: Double,
)

internal class TemporalDescriptorSpace(
    source: TemporalSignalSequence,
    reference: TemporalSignalSequence,
    private val config: TemporalSynchronizationConfig,
) {
    private val scales: Map<String, Double> = buildFeatureScales(source, reference)

    fun prepare(sequence: TemporalSignalSequence): PreparedTemporalSequence {
        val frames = sequence.frames.mapIndexed { index, frame ->
            val before = contextualIndex(sequence.frames, index, -config.derivativeContextMs)
            val after = contextualIndex(sequence.frames, index, config.derivativeContextMs)
            val history = contextualIndex(sequence.frames, index, -config.temporalContextMs)
            val lookahead = contextualIndex(sequence.frames, index, config.temporalContextMs)
            val derivatives = derivative(frame, before?.let(sequence.frames::get), after?.let(sequence.frames::get))
            val historyDelta = contextualDelta(frame, history?.let(sequence.frames::get), reverse = true)
            val lookaheadDelta = contextualDelta(frame, lookahead?.let(sequence.frames::get), reverse = false)
            val reliableDerivatives = derivatives.values.filter { it.isFinite() }
            PreparedTemporalFrame(
                timestamp = frame.timestamp,
                values = frame.features,
                derivatives = derivatives,
                historyDelta = historyDelta,
                lookaheadDelta = lookaheadDelta,
                motionMagnitude = if (reliableDerivatives.isEmpty()) 0.0 else {
                    sqrt(reliableDerivatives.sumOf { it * it } / reliableDerivatives.size.toDouble())
                },
                confidence = frame.confidence,
                discontinuityBefore = frame.discontinuityBefore,
            )
        }
        return PreparedTemporalSequence(sequence, frames, scales)
    }

    fun distance(left: PreparedTemporalFrame, right: PreparedTemporalFrame): TemporalDescriptorDistance {
        val common = left.values.keys intersect right.values.keys
        val unionCount = (left.values.keys + right.values.keys).size
        val usable = common.filter { key ->
            left.values.getValue(key).confidence >= config.minimumFeatureConfidence &&
                right.values.getValue(key).confidence >= config.minimumFeatureConfidence
        }
        val coverage = if (unionCount == 0) 0.0 else usable.size.toDouble() / unionCount.toDouble()
        if (usable.isEmpty()) {
            return TemporalDescriptorDistance(2.5, 0.0, false, 0.0)
        }

        val staticCost = usable.map { key ->
            abs(left.values.getValue(key).value - right.values.getValue(key).value) / scale(key)
        }.average().coerceAtMost(3.0)
        val derivativeKeys = usable.filter { key -> key in left.derivatives && key in right.derivatives }
        val derivativeCost = if (derivativeKeys.isEmpty()) 0.8 else derivativeKeys.map { key ->
            abs(left.derivatives.getValue(key) - right.derivatives.getValue(key)).coerceAtMost(3.0)
        }.average()
        val reliableDirectionKeys = derivativeKeys.filter { key ->
            abs(left.derivatives.getValue(key)) >= config.reliableDirectionMagnitude &&
                abs(right.derivatives.getValue(key)) >= config.reliableDirectionMagnitude
        }
        val oppositeCount = reliableDirectionKeys.count { key ->
            left.derivatives.getValue(key) * right.derivatives.getValue(key) < 0.0
        }
        val oppositeFraction = if (reliableDirectionKeys.isEmpty()) 0.0 else {
            oppositeCount.toDouble() / reliableDirectionKeys.size.toDouble()
        }
        val oppositeReliable = reliableDirectionKeys.isNotEmpty() &&
            oppositeFraction >= config.oppositeDirectionFraction
        val magnitudeCost = abs(left.motionMagnitude - right.motionMagnitude).coerceAtMost(3.0)
        val historyCost = contextDistance(left.historyDelta, right.historyDelta, usable)
        val lookaheadCost = contextDistance(left.lookaheadDelta, right.lookaheadDelta, usable)
        val contextCost = (historyCost + lookaheadCost) / 2.0
        val confidence = min(left.confidence, right.confidence).coerceIn(0.0, 1.0)
        val missingPenalty = (1.0 - coverage) * 0.75
        val oppositePenalty = if (oppositeReliable) 1.35 else oppositeFraction * 0.55
        val confidencePenalty = (1.0 - confidence) * 0.45
        val cost = 0.38 * staticCost + 0.28 * derivativeCost + 0.11 * magnitudeCost +
            0.23 * contextCost + missingPenalty + oppositePenalty + confidencePenalty
        return TemporalDescriptorDistance(cost, coverage, oppositeReliable, oppositeFraction)
    }

    private fun derivative(
        center: TemporalSignalFrame,
        before: TemporalSignalFrame?,
        after: TemporalSignalFrame?,
    ): Map<String, Double> {
        val left = before ?: center
        val right = after ?: center
        val deltaMs = right.timestamp.value - left.timestamp.value
        if (deltaMs <= 0L || left.discontinuityBefore || right.discontinuityBefore) return emptyMap()
        return (left.features.keys intersect right.features.keys).mapNotNull { key ->
            val a = left.features.getValue(key)
            val b = right.features.getValue(key)
            if (a.confidence < config.minimumFeatureConfidence || b.confidence < config.minimumFeatureConfidence) {
                null
            } else {
                key to ((b.value - a.value) / scale(key) * 1000.0 / deltaMs.toDouble())
            }
        }.toMap()
    }

    private fun contextualDelta(
        center: TemporalSignalFrame,
        context: TemporalSignalFrame?,
        reverse: Boolean,
    ): Map<String, Double> {
        if (context == null || context.discontinuityBefore || center.discontinuityBefore) return emptyMap()
        val common = center.features.keys intersect context.features.keys
        return common.mapNotNull { key ->
            val a = if (reverse) context.features.getValue(key) else center.features.getValue(key)
            val b = if (reverse) center.features.getValue(key) else context.features.getValue(key)
            if (a.confidence < config.minimumFeatureConfidence || b.confidence < config.minimumFeatureConfidence) null
            else key to ((b.value - a.value) / scale(key))
        }.toMap()
    }

    private fun contextDistance(
        left: Map<String, Double>,
        right: Map<String, Double>,
        preferredKeys: List<String>,
    ): Double {
        val common = preferredKeys.filter { it in left && it in right }
        return if (common.isEmpty()) 0.7 else common.map { key ->
            abs(left.getValue(key) - right.getValue(key)).coerceAtMost(3.0)
        }.average()
    }

    private fun scale(key: String): Double = scales[key] ?: 1.0

    private fun buildFeatureScales(
        source: TemporalSignalSequence,
        reference: TemporalSignalSequence,
    ): Map<String, Double> {
        val all = (source.frames + reference.frames).flatMap { frame ->
            frame.features.map { (key, sample) -> key to sample }
        }.groupBy({ it.first }, { it.second })
        return all.mapValues { (_, samples) ->
            val values = samples.filter { it.confidence >= config.minimumFeatureConfidence }.map { it.value }
            if (values.size < 2) 1.0 else {
                val spread = quantile(values, 0.90) - quantile(values, 0.10)
                val absolute = quantile(values.map(::abs), 0.50)
                max(spread, max(absolute * 0.10, 1e-6))
            }
        }
    }
}

internal fun quantile(values: List<Double>, probability: Double): Double {
    if (values.isEmpty()) return 0.0
    val sorted = values.sorted()
    val position = probability.coerceIn(0.0, 1.0) * (sorted.size - 1)
    val lower = position.toInt()
    val upper = min(sorted.lastIndex, lower + 1)
    val fraction = position - lower
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

private fun contextualIndex(
    frames: List<TemporalSignalFrame>,
    centerIndex: Int,
    offsetMs: Long,
): Int? {
    if (frames.isEmpty()) return null
    val target = frames[centerIndex].timestamp.value + offsetMs
    if (offsetMs < 0L) {
        var index = centerIndex
        while (index > 0 && frames[index].timestamp.value > target) index -= 1
        return index.takeIf { it != centerIndex && !crossesDiscontinuity(frames, index, centerIndex) }
    }
    var index = centerIndex
    while (index < frames.lastIndex && frames[index].timestamp.value < target) index += 1
    return index.takeIf { it != centerIndex && !crossesDiscontinuity(frames, centerIndex, index) }
}

private fun crossesDiscontinuity(frames: List<TemporalSignalFrame>, leftIndex: Int, rightIndex: Int): Boolean {
    val start = min(leftIndex, rightIndex) + 1
    val end = max(leftIndex, rightIndex)
    return start <= end && (start..end).any { index -> frames[index].discontinuityBefore }
}
