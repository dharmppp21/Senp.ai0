package ai.senp.core.contracts

import kotlinx.serialization.Serializable

@Serializable
enum class MotionStructureClass {
    CYCLIC,
    ACYCLIC,
    ISOMETRIC,
    MIXED,
    UNKNOWN,
}

@Serializable
enum class ActivitySegmentKind {
    ACTIVE,
    HOLD,
    IDLE,
    REST,
    SETUP,
    DISCONTINUITY,
    UNRELIABLE,
}

@Serializable
data class ActivitySegment(
    val range: TimestampRange,
    val kind: ActivitySegmentKind,
    val confidence: Double,
) {
    init {
        requireProbability(confidence, "activity-segment confidence")
    }
}

@Serializable
enum class UnitCompleteness {
    COMPLETE,
    PARTIAL,
}

@Serializable
enum class UnitBoundaryStatus {
    CLOSED,
    OPEN,
}

@Serializable
data class MotionUnit(
    val unitId: String,
    val range: TimestampRange,
    val structureClass: MotionStructureClass,
    val completeness: UnitCompleteness,
    val startBoundary: UnitBoundaryStatus,
    val endBoundary: UnitBoundaryStatus,
    val confidence: Double,
) {
    init {
        requireTemporalIdentifier(unitId, "motion-unit ID")
        requireProbability(confidence, "motion-unit confidence")
        when (completeness) {
            UnitCompleteness.COMPLETE -> require(
                startBoundary == UnitBoundaryStatus.CLOSED && endBoundary == UnitBoundaryStatus.CLOSED,
            ) { "complete motion unit must have closed begin and end boundaries" }
            UnitCompleteness.PARTIAL -> require(
                startBoundary == UnitBoundaryStatus.OPEN || endBoundary == UnitBoundaryStatus.OPEN,
            ) { "partial motion unit must have at least one open boundary" }
        }
    }
}

@Serializable
data class TemporalStructure(
    val role: VideoRole,
    val duration: DurationMs,
    val classification: MotionStructureClass,
    val activitySegments: List<ActivitySegment>,
    val motionUnits: List<MotionUnit>,
    val confidence: Double,
) {
    init {
        requireOrderedNonOverlapping(activitySegments.map(ActivitySegment::range), "activity-segment")
        requireOrderedNonOverlapping(motionUnits.map(MotionUnit::range), "motion-unit")
        require(motionUnits.map(MotionUnit::unitId).distinct().size == motionUnits.size) {
            "motion-unit IDs must be unique within a temporal structure"
        }
        require(activitySegments.all { it.range.endExclusive.value <= duration.value }) {
            "activity segment must lie within temporal-structure duration"
        }
        require(motionUnits.all { it.range.endExclusive.value <= duration.value }) {
            "motion unit must lie within temporal-structure duration"
        }
        requireProbability(confidence, "temporal-structure confidence")
    }
}

private fun requireTemporalIdentifier(value: String, name: String) {
    require(value.isNotBlank()) { "$name must not be blank" }
    require(value.length <= 128) { "$name must be at most 128 characters" }
}
