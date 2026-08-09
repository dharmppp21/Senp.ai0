# Synchronization Kernel v2 contract freeze

## Status and scope

This document freezes Stage 1 of Synchronization Kernel v2. The serialized schema version is `1`; "v2" names the synchronization architecture generation, not the JSON schema revision.

The freeze is additive to Wave5. Existing `PoseSequence`, `MotionSeries`, `PhaseSeries`, `AlignmentResult`, `AnalysisResult`, pipeline ports, and cache contracts are unchanged so the stable Wave5 path can continue while spatial, temporal, correspondence, and validation lanes implement v2 independently.

No new Gradle module is justified. `core-contracts` is already the canonical pure Kotlin/JVM boundary for immutable, serializable, timestamp-first values, so Sync v2 contracts live there with no Android, MediaPipe, video, rendering, or algorithm dependency.

## Frozen public boundary

### Observation and timing

`CanonicalObservationSequence` is the generic observation input. It owns a `VideoRole`, duration, `ObservationSampling`, and strictly increasing timestamped `CanonicalObservation` values.

`ObservationSampling.inputNominalFramesPerSecond` and `analysisFramesPerSecond` are deliberately separate nullable diagnostics. Neither is temporal truth. Correspondence is defined only in `TimestampMs`; variable-rate media and irregular analysis sampling remain legal.

Each `ObservationChannel` has an opaque stable channel ID, independent schema version, semantic type, optional coordinate space/subject ID, named component axes, explicit per-component masks, per-value confidence, channel availability, and channel confidence. The kernel does not require a fixed pose topology. Human 2D and 3D pose can be separate channels now; future hand, object, appearance, or learned channels use the same envelope without changing synchronization semantics.

Mask invariant: a present component has a finite value; a missing component is explicitly masked and serialized as `null`. A `MISSING` channel has no present components and zero confidence. Missing data is never encoded as numeric zero.

### Spatial synchronization

`BodyCentricTransform` is a 3D similarity transform: translation, unit-quaternion rotation, and one positive uniform scale. This is intentionally less permissive than an arbitrary affine/non-rigid transform so nuisance coordinate changes can be removed without shearing or independently resizing body parts and erasing genuine form differences.

`BodyCentricTransformEstimate` adds role, timestamp range, confidence, and stability. `RelativeViewHypothesis` exposes optional relative yaw/elevation, mirror hypothesis, selected body side, confidence, and side-selection stability. `SpatialReliabilitySegment` explicitly marks `COMPATIBLE`, `UNRELIABLE`, `INCOMPATIBLE`, or `DISCONTINUITY` intervals and gives generic diagnostic reasons such as camera movement, view ambiguity, subject ambiguity, insufficient 3D, occlusion, or unstable transforms.

The public contract does not name or require a transform estimator, landmark topology, camera model, pose framework, or fitting algorithm.

### Temporal structure

`TemporalStructure` classifies a clip as `CYCLIC`, `ACYCLIC`, `ISOMETRIC`, `MIXED`, or `UNKNOWN`. Ordered `ActivitySegment` values represent active motion, holds, idle, rest, setup, discontinuities, and unreliable spans.

`MotionUnit` is the reusable temporal unit. Units have stable IDs, timestamp ranges, structure class, confidence, and explicit complete/partial boundary semantics. A partial unit must have an open begin and/or open end. An isometric hold that is synchronizable should be represented as an isometric motion unit rather than requiring a cycle.

Units are ordered and non-overlapping within one clip. There is no invariant that source and reference unit counts, durations, sample counts, or frame rates match.

### Correspondence

`TimestampCorrespondence` is source-driven. Every emitted source timestamp decision is explicit within a matched unit:

- `Matched`: SOURCE timestamp -> REFERENCE timestamp with decision confidence.
- `UnmatchedSource`: SOURCE timestamp -> UNMATCHED with a typed generic reason and decision confidence.

`MotionUnitCorrespondence` is the unit-level decision:

- `MatchedUnit`: one source unit -> one reference unit plus a per-unit timestamp correspondence timeline and ambiguity.
- `SourceUnmatchedUnit`: an extra or indefensible source unit.
- `ReferenceUnmatchedUnit`: a missing source step/reference unit with no counterpart.

Within one `MatchedUnit`, source timestamps are strictly increasing and matched reference timestamps are monotonic, allowing nonlinear local time warp and many-to-one samples. Monotonicity deliberately does **not** extend across source units. The same reference unit ID may be reused independently by any number of source units, so one reference repetition can synchronize ten source repetitions without manufacturing ten reference copies.

Every source motion unit must have exactly one unit-level decision. Every reference unit must be matched at least once or explicitly unmatched. Reference reuse is legal; declaring the same reference unit both matched and unmatched is not.

### Open/local/subsequence semantics

`SynchronizationSemantics` freezes support flags for full-sequence, subsequence, and local synchronization, open source/reference begin/end, unmatched source/reference material, and reference-unit reuse. These are semantic capabilities, not an algorithm prescription.

Setup, rest, idle, extra actions, missing actions, partial clips, and discontinuities are therefore representable without forcing a correspondence.

### Confidence, refusal, and required channels

`SynchronizationDiagnostics` exposes overall, spatial, temporal, and correspondence confidence; source/reference analyzable fractions; and correspondence ambiguity. `SynchronizationStatus` is `SYNCHRONIZED`, `PARTIAL`, or `REFUSED`.

A `REFUSED` result requires a typed `SynchronizationRefusal`. Non-refused results require at least one matched motion unit. A refused result may have no correspondences and still carry high-confidence diagnostics explaining *why* synchronization was refused. Confidence in a refusal diagnosis is distinct from confidence in a correspondence.

The v2 result intentionally contains no score, coaching label, error count, or deviation metric. Unanalyzable data therefore cannot become a fabricated "0 errors" success through this contract.

`SynchronizationRequirements.requiredChannelSemanticTypes` lets an optional domain/task adapter declare generic evidence requirements. For example, a task that genuinely depends on an object may require an `object_pose` semantic channel; pose-only input can then be refused with `REQUIRED_CHANNEL_MISSING`. Task or exercise names do not belong in the generic kernel contract.

## Constructor invariants

The contract constructors enforce the following independent of implementation:

1. Timestamps are non-negative; ranges are non-empty and end-exclusive; observation timestamps are strictly increasing.
2. Input FPS and analysis FPS are separate, optional, positive finite metadata.
3. Observation values and masks have equal dimensions; present values are finite and missing values are null.
4. Confidence, stability, analyzable fractions, and ambiguity are finite in `[0, 1]`.
5. Body transforms are similarity transforms with a unit quaternion and positive uniform scale.
6. Spatial transform/reliability intervals are ordered and non-overlapping per role.
7. Activity segments and motion units are ordered/non-overlapping and lie within clip duration.
8. Complete units have closed boundaries; partial units have at least one open boundary.
9. Per-unit source correspondence timestamps are strictly increasing; matched reference timestamps are monotonic.
10. Unit IDs referenced by correspondence must exist; each source unit has exactly one decision.
11. Every reference unit is matched at least once or explicitly unmatched; one reference unit may be matched by many source units.
12. Refused results require refusal diagnostics; non-refused results require a real matched unit.
13. Result correspondence decisions must honor the serialized unmatched/reference-reuse semantics; `SYNCHRONIZED` results cannot contain unmatched motion units.

These invariants reject internally contradictory DTOs but do not impose confidence thresholds, segmentation heuristics, transform fitting thresholds, or a temporal alignment algorithm.

## Adversarial acceptance matrix

The executable fixture is `core-contracts/src/test/resources/fixtures/sync-v2-adversarial-acceptance-v1.json`. Its test verifies complete scenario coverage, invariant names, and absence of activity/algorithm-specific golden paths.

| Scenario | Invariant-oriented expected outcome |
|---|---|
| Same video vs itself | If observations are analyzable, identity must not be refused and no synthetic unmatched material is introduced. |
| Different FPS | Timestamp-first mapping; no frame-count/FPS equality assumption; local warp allowed. |
| Different resolution | Resolution is spatial nuisance; real form differences remain. |
| Different codec | Codec/container choice is not correspondence evidence when decoded timestamps/observations are valid. |
| Rotation | Spatial diagnostics/canonicalization absorb display/camera orientation nuisance. |
| Camera yaw/elevation/viewpoint | Explicit view hypothesis and reliability; incompatible geometry may be partial/refused rather than forced. |
| Mirror | Mirror state/ambiguity is explicit; no silent side swap assumption. |
| Side-selection changes | Side-selection stability is diagnostic and ambiguity lowers trust or causes refusal. |
| Camera movement/discontinuity | Mark unreliable/discontinuous intervals; no confident mapping across an unsupported gap. |
| Start mid-motion | Open-begin partial unit/subsequence semantics. |
| End mid-motion | Open-end partial unit/subsequence semantics. |
| 1 reference unit vs 10 source units | Reuse the same reference unit independently; each source unit has its own local warp. |
| 10 reference units vs 1 source unit | Unused reference units are explicitly unmatched; no forced repetition-count equality. |
| 2 reference units vs 7 source units | Reference reuse is legal and local; no global reference-time monotonicity across source units. |
| Multiple sets/rests | Rest/setup/idle are temporal structure, not forced action correspondence. |
| Extra source action | Source unit/timestamps may be unmatched. |
| Missing source action | Reference unit may be explicitly unmatched. |
| Repeated identical-looking phase | Ambiguity/confidence is surfaced; do not force a convenient-looking pair. |
| Variable speed | Per-unit nonlinear time warp is legal; form difference is not explained away as timing. |
| Pause/hold | Holds are explicit temporal structure; timing may locally warp around them. |
| Very slow / very fast | Same timestamp-first/local-warp rules; no hidden FPS or duration equality. |
| Static/isometric | Isometric structure is first-class; cyclic detection is not required. |
| No common motion | Refuse rather than produce an arbitrary mapping or successful zero-error result. |
| Poor pose/observation coverage | Analyzable coverage/confidence is explicit; partial/refused, never fabricated success. |
| Short occlusion | Masks and source-unmatched timestamp decisions preserve the gap explicitly. |
| Long occlusion | Mark unreliable interval and avoid confident mapping across it; partial/refused as needed. |
| Person leaves/re-enters | Subject/temporal discontinuity is explicit; identity continuity is not assumed. |
| Multiple people/subject ambiguity | Subject ambiguity is explicit and may force partial/refused output. |
| Different body proportions | Coordinate nuisance may normalize globally; non-rigid proportion differences are preserved. |
| True form difference | Spatial synchronization cannot shear/non-rigidly normalize the difference away. |
| Reversed video | Do not force high-confidence monotonic correspondence; partial/refused is valid. |
| Edited/spliced video | Cuts are discontinuities; local alignment may resume on defensible segments. |
| Slow-motion edit | Timestamp/local-warp semantics handle duration change without treating encoding metadata as correspondence. |
| Non-cyclic activity | Acyclic structure and missing steps are supported without inventing cycles. |
| Object-dependent task with pose-only input | Adapter-declared required channel is missing -> explicit refusal, not pose-only guesswork. |

## Ownership boundaries for implementation lanes

**Observation adapters** translate upstream evidence into `CanonicalObservationSequence`. They own channel schemas and subject/channel extraction but must preserve timestamps, masks, confidence, and input-vs-analysis FPS distinction.

**Spatial lane** owns estimation and produces the frozen spatial diagnostic types. It must not mutate temporal timestamps, invent missing observation values, or apply non-rigid body deformation that erases form differences.

**Temporal lane** owns activity/hold/rest/discontinuity segmentation, global structure classification, and motion-unit discovery. It produces `TemporalStructure` and does not need activity-specific names.

**Correspondence/synchronization lane** consumes frozen observation/spatial/temporal evidence and produces the frozen unit/timestamp correspondence and `SynchronizationResult`. It owns confidence/refusal policy but must honor explicit unmatched and reference-reuse semantics.

**Validation lane** treats the adversarial fixture invariants as acceptance requirements. It should add concrete media cases and tolerances without converting the matrix into an algorithm-specific golden path.

**Domain adapters/profiles** are optional and outside the generic kernel. They may translate task knowledge into channel requirements or external metadata, but generic contracts and algorithms must not branch on names such as an exercise/activity label.

## Migration from Wave5

1. Keep Wave5 interfaces and execution path intact during v2 development.
2. Add an adapter from legacy video/pose extraction to `CanonicalObservationSequence`. Map human pose 2D and real 3D evidence into separate semantic channels, preserve confidence/missing masks, and populate input/analysis FPS separately. Do not require MP33 in the v2 channel contract.
3. Spatial and temporal v2 branches compile against the additive contracts and return their frozen DTOs. They do not need to edit legacy `MotionSeries`, `PhaseSeries`, or `AlignmentResult`.
4. Do **not** mechanically translate legacy `AlignmentResult` into v2 success: the legacy global path cannot losslessly express reference reuse, missing reference units, extra source units, or explicit unmatched timestamps.
5. When v2 orchestration/cache integration is introduced, assign independent behavior/cache versions. Do not reuse a Wave5 cache entry as a v2 synchronization result.
6. Only after validation passes the adversarial matrix should product scoring/coaching consume synchronization output. Scoring is intentionally not part of this freeze.

## Intentionally implementation-private

The freeze does not prescribe transform fitting, mirror/view estimation, subject tracking, motion segmentation, cycle discovery, feature extraction, confidence thresholds, dynamic-programming formulation, local-warp estimator, candidate search, pruning, or optimization strategy. Specific techniques such as Kabsch/Procrustes fitting, RANSAC, DTW/Drop-DTW variants, learned embeddings, or MediaPipe-specific topology remain implementation details and may change without changing these contracts as long as the frozen invariants are preserved.

## Versioning rule

`SynchronizationRequest.CURRENT_SCHEMA_VERSION` and `SynchronizationResult.CURRENT_SCHEMA_VERSION` are both `1` for this freeze. A serialized field meaning/removal/reinterpretation requires a schema bump. Adding an optional extension still requires coordinated consumer review if it changes behavioral expectations. Channel producers version each `ObservationChannel.schemaVersion` independently of the synchronization envelope.

No unresolved contract decision blocks the spatial, temporal, correspondence, or validation implementation lanes at this stage.
