# Synchronization Kernel v2 validation lane

This module is the independent validation/harness lane for the frozen Sync-v2 contracts. It owns **no synchronization algorithm**; the executable production adapters delegate to `:sync-v2-integration`, `:core-motion`, and `:core-alignment`. Its job is to make adversarial inputs, inspect production outputs, drive the immutable real-video corpus, render human-review evidence, and measure post-pose overhead.

## Boundary

`sync-v2-validation` keeps the frozen contracts read-only while depending on the integrated production modules only through explicit harness adapters. Production logic remains in `:sync-v2-integration`, `:core-motion`, and `:core-alignment`; validation owns fixtures, invariant checks, executable protocol glue, and reports.

Three local seams keep future integration narrow:

- `SynchronizationHarnessAdapter` wraps a concrete end-to-end Sync-v2 implementation.
- `SpatialHarnessAdapter` exposes canonicalized observations plus frozen spatial diagnostics for form-preservation inspection without changing the public contract.
- `TemporalHarnessAdapter` exposes the frozen `TemporalStructure` for independent segmentation/unit validation.

The CLI/media layer uses a separate executable adapter protocol (`senp-sync-v2-validation-adapter/1`) so real-video and performance validation can be wired after spatial/temporal integration without importing their production modules here.

## Synthetic suite

Generate deterministic timestamp-first machine-readable fixtures:

```bash
./gradlew :sync-v2-validation:run --args="generate test-artifacts/sync-v2/synthetic"
```

Outputs:

- `synthetic-suite.json` — all 36 frozen scenarios with canonical observations, truth tracks, motion-unit truth, spatial truth, and allowed synchronization outcomes.
- `scenarios/<id>.json` — one reproducible fixture per scenario.
- `coverage-matrix.json` — executable fixture/invariant coverage and production-integration state.

Default seed: `20260808`. A different seed may be passed as the final argument. Timestamps, not frame indices or nominal FPS, are the source of temporal truth. Each observation also carries concrete `human_pose` 3D landmarks (`x/y/z`) for spatial-lane injection; temporal oracle values live in a separate `synthetic_motion_truth` channel so a spatial adapter is not handed its expected answer as pose geometry. Viewpoint cases rotate those landmarks, mirror cases reflect them, camera-discontinuity cases change the view after an explicit unreliable gap, and body-proportion cases combine similarity-scale nuisance with non-rigid geometry changes.

The result validator is deliberately invariant-oriented. It checks identity, explicit unmatched material, reference-unit reuse, reliable interior direction consistency, open boundaries, rest/isometric/acyclic semantics, discontinuities, ambiguity, coverage, mirror/view diagnostics, required-channel refusals, and true-form preservation through the local spatial inspection seam. It does not prescribe a DTW path, phase-count golden, or coaching outcome.

To validate a frozen `SynchronizationResult` JSON after integration:

```bash
./gradlew :sync-v2-validation:run --args="validate same_video_self result.json report.json"
```

## Frozen 36-scenario matrix

Every row has executable deterministic fixture generation, invariant validation, and a concrete production adapter. `evaluate-production` executes the integrated Sync-v2 kernel for all 36 scenarios and exhaustively accounts for the frozen invariant vocabulary.

| Scenario | Primary lane(s) | Fixture | Invariant evaluator | Production adapter |
|---|---|---:|---:|---:|
| same_video_self | correspondence | executable | executable | executable |
| different_fps | temporal/correspondence | executable | executable | executable |
| different_resolution | spatial/correspondence | executable | executable | executable |
| different_codec | correspondence | executable | executable | executable |
| rotation_metadata | spatial/correspondence | executable | executable | executable |
| yaw_elevation_viewpoint | spatial | executable | executable | executable |
| mirror | spatial | executable | executable | executable |
| side_selection_stability | spatial | executable | executable | executable |
| camera_movement_discontinuity | spatial/temporal | executable | executable | executable |
| start_mid_motion | temporal | executable | executable | executable |
| end_mid_motion | temporal | executable | executable | executable |
| one_reference_ten_source | temporal/correspondence | executable | executable | executable |
| ten_reference_one_source | temporal/correspondence | executable | executable | executable |
| two_reference_seven_source | temporal/correspondence | executable | executable | executable |
| multiple_sets_rests | temporal | executable | executable | executable |
| extra_source_action | temporal/correspondence | executable | executable | executable |
| missing_source_action | temporal/correspondence | executable | executable | executable |
| repeated_identical_phase | temporal/correspondence | executable | executable | executable |
| variable_speed | temporal/correspondence | executable | executable | executable |
| pause_hold | temporal | executable | executable | executable |
| very_slow | temporal | executable | executable | executable |
| very_fast | temporal | executable | executable | executable |
| static_isometric | temporal | executable | executable | executable |
| no_common_motion | truthfulness | executable | executable | executable |
| poor_pose_coverage | truthfulness/spatial | executable | executable | executable |
| short_occlusion | temporal/truthfulness | executable | executable | executable |
| long_occlusion | temporal/spatial/truthfulness | executable | executable | executable |
| person_leaves_reenters | temporal/spatial | executable | executable | executable |
| multiple_people_subject_ambiguity | spatial/truthfulness | executable | executable | executable |
| different_body_proportions | spatial | executable | executable | executable |
| true_form_difference | spatial | executable | executable | executable |
| reversed_video | temporal/truthfulness | executable | executable | executable |
| edited_spliced_video | temporal/spatial | executable | executable | executable |
| slow_motion_edit | temporal | executable | executable | executable |
| non_cyclic_activity | temporal/correspondence | executable | executable | executable |
| object_required_pose_only | truthfulness/channels | executable | executable | executable |

## Real-video corpus runner

The immutable corpus is resolved from `fixtures/real-video-cases.json` and its external manifest. The runner verifies manifest membership and SHA-256 by default; it never modifies media.

Regression targets:

- `biceps-wrong-right` — old alignment had visually verified opposite-phase correspondence errors.
- `legraise-wrong-right` — full-pose-coverage generic temporal regression, including raised-vs-flat leg mismatches.
- `pushup-wrong-right` — low-observation-coverage truthfulness/refusal target. Its corpus descriptor requires at least 0.65 analyzable fraction before a `SYNCHRONIZED` result is accepted; lower coverage must remain partial/refused.
- `biceps-right-right-control` and `legraise-right-right-control` — identity/same-video controls.

Prepare a case descriptor without executing Android:

```bash
python3 sync-v2-validation/tools/sync_v2_validation.py prepare-real \
  --case biceps-wrong-right \
  --output-dir test-artifacts/sync-v2/real/biceps-wrong-right
```

This writes `adapter-request.json` and reports `integration_status: STAGED` because no executable was requested. To run the production Android/MediaPipe path, pass the checked-in executable wrapper:

```bash
python3 sync-v2-validation/tools/sync_v2_validation.py prepare-real \
  --case biceps-wrong-right \
  --output-dir test-artifacts/sync-v2/real/biceps-wrong-right \
  --adapter-executable scripts/sync-v2-validation-adapter
```

The executable receives the request JSON path as its only argument and must write the requested normalized result. Normalized results contain synchronization status/confidence, source/reference analyzable fractions, mapped or unmatched timestamps, motion-unit IDs, direction/phase/state labels when available, unmatched units, refusal reason when applicable, and spatial diagnostics. The validator rejects scoring/coaching/problem-count fields, reliable opposite-direction mappings, non-monotonic per-unit timestamp mappings, and confident forced matches through explicit rest/unreliable holes.

## Human-verification artifacts

Render a deterministic mobile-first HTML report and PNG contact sheet from normalized mapping rows:

```bash
python3 sync-v2-validation/tools/sync_v2_validation.py artifact \
  --case legraise-wrong-right \
  --plan sync-v2-validation/fixtures/legraise-renderer-adversarial-review.json \
  --output-dir test-artifacts/sync-v2/artifacts/legraise-renderer-smoke
```

The checked-in review plan is **not a synchronization result**. It deliberately contains bad raised-vs-flat, opposite-state, opposite-direction, and unmatched rows so reviewers can verify the renderer makes these failure modes obvious instead of demonstrating fake algorithm success.

The artifact includes:

- a deterministic source-to-reference `timeline.svg`, including explicit `UNMATCHED` marks;
- raw source/reference frames selected by decoded presentation timestamp;
- source timestamp and reference timestamp or `UNMATCHED`;
- confidence, source/reference motion-unit IDs, direction, phase, state and reliability labels;
- mirror/side/view/global-scale diagnostics supplied by the adapter;
- a mobile-first HTML page plus a deterministic vertically stacked PNG contact sheet.

Non-uniform scaling or shear is outside the frozen spatial transform family and is rejected by the normalized-result validator; the page keeps spatial diagnostics visible near the top so global scale and form-preservation concerns can be reviewed beside the correspondence timeline and raw frames.

## Performance harness

Run the seven-shape Sync-v2 post-pose scaling plan (and optionally summarize legacy stage evidence separately):

```bash
python3 sync-v2-validation/tools/sync_v2_validation.py performance \
  --output test-artifacts/sync-v2/performance/report.json \
  --adapter-executable scripts/sync-v2-validation-adapter \
  --repetitions 3
```

The benchmark plan covers:

- 150/300/600/1200 post-pose sequence samples;
- coarse 10 FPS, typical 15 FPS, and denser 30 FPS analysis cadences;
- 1:10 reference-unit reuse and repeated 2:7 units;
- long idle/rest clips;
- peak RSS reporting.

`input_nominal_fps` is never treated as `analysis_fps`. Existing stage reports are labeled `legacy_wave5_evidence_only_not_sync_v2` and separate pose/preprocessing time from post-pose motion/phase/alignment time.

With `--adapter-executable`, the wrapper receives `post_pose_benchmark` requests and returns production-kernel post-pose time, peak RSS, and bounded-search statistics after stripping `synthetic_motion_truth`. The JVM benchmark intentionally leaves `total_pipeline_ms` null because it does not run video decode or pose inference; therefore it cannot manufacture a pipeline-fraction denominator. Actual Android/MediaPipe runs record pose/preprocessing, post-pose Sync-v2, total time, and the post-pose fraction. The ordinary target is the 15–20% band of real total-pipeline time.

## Validation commands

```bash
python3 -m unittest discover -s sync-v2-validation/tools/tests -v
./gradlew :sync-v2-validation:test
./gradlew :core-contracts:test
./gradlew checkCoreBoundaries
./gradlew check
```

Generated media, reports, and benchmark outputs live under ignored `test-artifacts/` / `benchmark-results/` paths. Keep the external corpus read-only and keep publish/PR steps outside this lane.

## Integration status and limits

The production spatial, temporal, and iterative integration paths are wired through `ProductionAdapters.kt` and `scripts/sync-v2-validation-adapter`. Synthetic acceptance, API35 Android/MediaPipe real-video execution, and post-pose performance measurements are executable. The checked-in adversarial renderer plan remains only a renderer test and must never be cited as synchronization accuracy evidence; real acceptance artifacts must be derived from actual normalized production results.
