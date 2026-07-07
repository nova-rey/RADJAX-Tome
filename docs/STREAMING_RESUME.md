# Streaming Resume

Spec 4.6 adds a streaming backend build path for larger Tome artifacts. It is
enabled on the existing build command with `--streaming`.

```bash
radjax-tome build \
  --streaming \
  --teacher-backend gpu_torch \
  --runtime-mode cpu_gpu \
  --target-policy corridor_exemplar_v1 \
  --teacher-model /models/MODEL \
  --teacher-model-provenance ./teacher_model_provenance.json \
  --dataset ./corpus_out/corpus.jsonl \
  --corpus-manifest ./corpus_out/corpus_manifest.json \
  --output ./tome_out \
  --shard-size-examples 1024
```

The streaming path reads `corpus.jsonl` one line at a time, preserves corpus
order, batches examples for backend emission, and writes target shards in
bounded shard groups. It does not change backend reducer math, selector
behavior, model download policy, network policy, multidevice scheduling, or
TPU/JAX support.

## Files

Streaming builds add:

- `run_manifest.json`: source of truth for resume state.
- `progress_log.jsonl`: append-only human/debug progress events.
- `failure_report.json`: written only when a run fails.

Normal Tome sidecars are still written after successful emission:
`metadata.json`, `teacher_manifest.json`, `emission_config.json`,
`vocab_contract.json`, `validation_report.json`, and `cover_page.json`.

## Atomic Shards

Each shard is written to a temporary file under `shards/`, fsynced, and then
renamed into its final `shard-00000.npz` path. A shard is only marked complete
in `run_manifest.json` after the final file exists and its `sha256:<hex>` hash
has been recorded.

On resume, stale `*.tmp` files are removed. Only final shard files whose hashes
match `completed_shards` are trusted.

## Resume

Resume with the same command plus `--resume`:

```bash
radjax-tome build \
  --streaming \
  --resume \
  --teacher-backend gpu_torch \
  --runtime-mode cpu_gpu \
  --target-policy corridor_exemplar_v1 \
  --teacher-model /models/MODEL \
  --teacher-model-provenance ./teacher_model_provenance.json \
  --dataset ./corpus_out/corpus.jsonl \
  --corpus-manifest ./corpus_out/corpus_manifest.json \
  --output ./tome_out \
  --shard-size-examples 1024
```

Resume recomputes `resume_config_hash` and refuses to continue if durable
inputs or settings drifted. The hash covers backend/runtime/target settings,
teacher model identity, dataset hash, corpus hash when supplied, reducer
parameters, batch/shard sizing, exemplar policy fields, GPU batch policy, and
local-files/download policy.

If a run is already complete and the artifact validates, `--resume` returns the
existing validation status.

## Failure Report

On failure, `failure_report.json` records the failure stage, exception type,
completed example/shard counts, whether resume is available, and the
recommended action. Completed shards are preserved.

## Current Limits

The streaming path supports backend-routed target emission and refuses
corpus-global exemplar selection for now. This is intentional: it does not
pretend a global selector ran when the current streaming path cannot finalize
that policy honestly.
