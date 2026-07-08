# Production Build

Spec 4.7 adds the one-command production GPU Tome build path:
`radjax-tome production-build`.

```bash
radjax-tome production-build \
  --teacher-model /models/MODEL \
  --tokenizer-id /models/MODEL \
  --dataset ./corpus_out/corpus.jsonl \
  --corpus-manifest ./corpus_out/corpus_manifest.json \
  --teacher-model-provenance ./teacher_model_provenance.json \
  --output ./tome_out
```

The default production route is intentionally strict:

- `teacher_backend=gpu_torch`
- `runtime_mode=cpu_gpu`
- `target_policy=corridor_exemplar_v1`
- streaming build enabled
- resume supported with `--resume`
- `local_files_only=true`
- `allow_downloads=false`
- fallback policy `error`
- GPU batch policy `auto`, bounded by `--gpu-batch-size-auto-min 1` and
  `--gpu-batch-size-auto-max 64`

## Workflow

The command validates required input paths, validates corpus and teacher-model
provenance, runs runtime doctor diagnostics, writes `run_plan.json`, stops on a
failed plan, passes the planner-selected effective batch size into the
streaming builder, validates the completed artifact, writes `cover_page.json`,
and writes `production_build_report.json`.

If `--parity-left` is supplied, the command compares the production artifact
against that baseline and records `parity_status` plus an optional parity
report path. No parity claim is made when `--parity-left` is omitted.

## Selected-Only Exemplar Delivery

Spec P4.8B adds an opt-in selected-only delivery harness for
`corridor_exemplar_v1`. The default production-build behavior is unchanged
unless exemplar selection is explicitly enabled.

Recommended selected-only Path B invocation:

```bash
radjax-tome production-build \
  --teacher-model /models/MODEL \
  --tokenizer-id /models/MODEL \
  --dataset ./corpus_out/corpus.jsonl \
  --corpus-manifest ./corpus_out/corpus_manifest.json \
  --teacher-model-provenance ./teacher_model_provenance.json \
  --output ./tome_out \
  --target-policy corridor_exemplar_v1 \
  --exemplar-selection-enabled \
  --exemplar-delivery-path two_pass_rerun_selected \
  --selected-exemplar-budget 1024 \
  --no-retain-unselected-exemplar-payloads
```

Path A uses `--exemplar-delivery-path one_pass_pruned_candidate`; Path B uses
`--exemplar-delivery-path two_pass_rerun_selected`. Both delivery modes write
the same selected-only layout:

- `corridors/corridor_summary.json`
- `leaderboards/leaderboard_report.json`
- `leaderboards/selected_exemplars.json`
- `selected_exemplars/selected-exemplars-00000.json`
- `delivery_report.json`

The delivery report records `num_examples_scored`, `num_positions_scored`,
`num_selected_exemplars`, `selected_exemplar_payload_retained`,
`non_selected_exemplar_payload_retained`, `teacher_rerun_count`, and
`delivery_path`. Selected payload shards include compressed dynamic top-k and
bucket fields only for selected winners.

Compare two selected-only artifacts with:

```bash
radjax-tome exemplar-delivery-parity \
  --path-a ./path_a_artifact \
  --path-b ./path_b_artifact \
  --output parity_report.json
```

The parity report compares selected IDs, positions, rank order, score values,
mode keys, selected payload shape compatibility, retained bytes, rerun counts,
and non-selected payload retention status.

Selected-only parity uses the same compact `score_*` surface for Path A and
Path B leaderboard selection. Path A still slices selected compressed payloads
from one-pass candidate shard captures before pruning broad `exemplar_source_*`
arrays from final shards when non-selected retention is disabled.

Use `--track-delivery-timing` to add informational timing fields to production,
delivery, and exemplar-delivery parity reports. Timing fields are environment
specific and do not affect parity pass/fail status.

## Reports

`production_build_report.json` uses `production_build_report_v1`. It records:

- status, blockers, warnings, timestamps, command, and output directory
- input paths and effective local-files/download policy
- doctor summary
- run plan path, run plan status, and effective batch size
- streaming/resume state, run manifest path, and progress log path
- validation report path and validation status
- cover page path
- selected exemplar delivery report path and selected-only delivery truth when
  exemplar selection is enabled
- optional parity report path and parity status
- artifact summary with planned and actual size fields
- explicit non-claims: no downloads, no network verification, no silent CPU
  fallback, no multidevice scheduling, no TPU/JAX, and no unverified parity
  claim

## Guardrails

Use `--fail-on-plan-warnings` to make planner warnings fail the run during
planning. Use `--no-build-if-plan-warn` to stop before emission whenever the
plan status is `warn`. Use `--max-artifact-bytes` to fail if rough artifact
estimates exceed a run-specific byte budget.

Custom `--run-manifest` and `--progress-log` paths are recorded truthfully in
streaming metadata, teacher manifests, emission config, cover pages, and
production reports.

Spec 4.7.1 fixes already-complete production resume truth: when `--resume`
finds a complete, valid run manifest and the artifact validates, the command
writes `production_build_report.json` and returns `pass` without rerunning
doctor, planner, or the streaming builder. If that completed artifact is
invalid, it fails from validation blockers without rerunning planner or build.

The command does not expose an `--allow-downloads` flag. Teacher model setup is
a separate local provenance step; see `docs/TEACHER_MODEL_PROVENANCE.md`.
The low-level build CLI also does not expose `--fail-fast`; only fail-fast
streaming behavior exists today.

Spec 4.7.a adds a separate experimental `multi-gpu-path-b` command for Path B
candidate scheduling. It is not part of the default production-build happy
path. Single-GPU `production-build` remains the recommended path.
