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

## Reports

`production_build_report.json` uses `production_build_report_v1`. It records:

- status, blockers, warnings, timestamps, command, and output directory
- input paths and effective local-files/download policy
- doctor summary
- run plan path, run plan status, and effective batch size
- streaming/resume state, run manifest path, and progress log path
- validation report path and validation status
- cover page path
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
