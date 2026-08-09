package ai.senp.core.contracts

import kotlinx.serialization.Serializable

@Serializable
data class ObservationSampling(
    val inputNominalFramesPerSecond: Double? = null,
    val analysisFramesPerSecond: Double? = null,
) {
    init {
        inputNominalFramesPerSecond?.let { requirePositiveFinite(it, "input nominal FPS") }
        analysisFramesPerSecond?.let { requirePositiveFinite(it, "analysis FPS") }
    }
}

@Serializable
enum class ChannelAvailability {
    OBSERVED,
    PARTIAL,
    MISSING,
    UNRELIABLE,
}

@Serializable
data class ObservationValue(
    val key: String,
    val values: List<Double?>,
    val mask: List<Boolean>,
    val confidence: Double,
) {
    init {
        requireIdentifier(key, "observation value key")
        require(values.isNotEmpty()) { "observation value must contain at least one component" }
        require(values.size == mask.size) { "observation values and mask must have equal length" }
        requireProbability(confidence, "observation value confidence")
        values.zip(mask).forEachIndexed { index, (value, present) ->
            if (present) {
                require(value != null) { "masked-present component $index must contain a value" }
                requireFinite(value, "observation component $index")
            } else {
                require(value == null) { "masked-missing component $index must be null" }
            }
        }
    }
}

@Serializable
data class ObservationChannel(
    val channelId: String,
    val schemaVersion: Int,
    val semanticType: String,
    val coordinateSpace: String? = null,
    val subjectId: String? = null,
    val componentAxes: List<String>,
    val values: List<ObservationValue>,
    val availability: ChannelAvailability,
    val confidence: Double,
) {
    init {
        requireIdentifier(channelId, "channel ID")
        require(schemaVersion > 0) { "channel schema version must be positive" }
        requireIdentifier(semanticType, "channel semantic type")
        coordinateSpace?.let { requireIdentifier(it, "channel coordinate space") }
        subjectId?.let { requireIdentifier(it, "channel subject ID") }
        require(componentAxes.isNotEmpty()) { "channel component axes must not be empty" }
        require(componentAxes.all { it.isNotBlank() }) { "channel component axes must not contain blanks" }
        require(componentAxes.distinct().size == componentAxes.size) { "channel component axes must be unique" }
        require(values.isNotEmpty()) { "channel must contain at least one observation value" }
        require(values.map(ObservationValue::key).distinct().size == values.size) { "channel observation value keys must be unique" }
        require(values.all { it.values.size == componentAxes.size }) {
            "all channel values must match component-axis dimensionality"
        }
        requireProbability(confidence, "channel confidence")

        val masks = values.flatMap(ObservationValue::mask)
        val presentCount = masks.count { it }
        when (availability) {
            ChannelAvailability.OBSERVED -> require(presentCount == masks.size) {
                "observed channel must have all components present"
            }
            ChannelAvailability.PARTIAL -> require(presentCount in 1 until masks.size) {
                "partial channel must contain both present and missing components"
            }
            ChannelAvailability.MISSING -> {
                require(presentCount == 0) { "missing channel cannot contain present components" }
                require(confidence == 0.0) { "missing channel confidence must be zero" }
            }
            ChannelAvailability.UNRELIABLE -> Unit
        }
    }
}

@Serializable
data class CanonicalObservation(
    val timestamp: TimestampMs,
    val channels: List<ObservationChannel>,
) {
    init {
        require(channels.map(ObservationChannel::channelId).distinct().size == channels.size) {
            "observation channel IDs must be unique at a timestamp"
        }
    }
}

@Serializable
data class CanonicalObservationSequence(
    val role: VideoRole,
    val duration: DurationMs,
    val sampling: ObservationSampling,
    val observations: List<CanonicalObservation>,
) {
    init {
        require(observations.zipWithNext().all { (left, right) -> left.timestamp < right.timestamp }) {
            "canonical observation timestamps must be strictly increasing"
        }
        require(observations.lastOrNull()?.timestamp?.value?.let { it < duration.value } ?: true) {
            "canonical observation timestamp must be before sequence duration"
        }
    }
}

private fun requirePositiveFinite(value: Double, name: String) {
    requireFinite(value, name)
    require(value > 0.0) { "$name must be positive" }
}

private fun requireIdentifier(value: String, name: String) {
    require(value.isNotBlank()) { "$name must not be blank" }
    require(value.length <= 128) { "$name must be at most 128 characters" }
}
