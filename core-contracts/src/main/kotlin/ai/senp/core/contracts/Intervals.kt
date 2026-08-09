package ai.senp.core.contracts

import kotlinx.serialization.Serializable

@Serializable
data class TimestampRange(
    val start: TimestampMs,
    val endExclusive: TimestampMs,
) {
    init {
        require(endExclusive > start) { "timestamp-range end must be after start" }
    }

    fun contains(timestamp: TimestampMs): Boolean = timestamp >= start && timestamp < endExclusive

    fun duration(): DurationMs = DurationMs(endExclusive.value - start.value)
}
