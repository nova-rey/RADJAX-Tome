# GPU Run Planner

Spec 4.5 adds an explicit preflight planner for GPU Tome builds. It writes
`gpu_run_plan_v1` JSON to `run_plan.json` and does not run a production build.

```bash
radjax-tome plan \
  --teacher-backend gpu_torch \
  --runtime-mode cpu_gpu \
  --target-policy corridor_exemplar_v1 \
  --teacher-model /models/MODEL \
  --tokenizer-id /models/MODEL \
  --dataset ./corpus_out/corpus.jsonl \
  --corpus-manifest ./corpus_out/corpus_manifest.json \
  --teacher-model-provenance ./teacher_model_provenance.json \
  --gpu-batch-size-mode auto \
  --gpu-batch-size-auto-min 1 \
  --gpu-batch-size-auto-max 64 \
  --output run_plan.json
```

The CLI prints concise summary lines including status, selected batch size,
probe status, first failing batch size, estimated artifact bytes, and warning
count. Exit code is `0` for `pass` and `warn`, and `1` for `fail`.

## What The Plan Contains

`run_plan.json` records environment diagnostics from `radjax-tome doctor`,
dataset and provenance checks, requested and resolved batch policy, auto batch
probe results, rough memory estimates, rough artifact estimates, capture-mode
implications, a recommended build command, and explicit claims not made.

The planner validates supplied corpus manifests and teacher model provenance. If
they are missing, the plan warns by default because provenance confidence is
weaker. If supplied provenance is invalid, the plan fails.

## Auto Batch Probe

When `--gpu-batch-size-mode auto` is used with `gpu_torch`, the planner tries a
bounded sequence of candidate batch sizes from min to max. Each candidate runs
one tiny local probe through the same model/tokenizer load path and target
reducer path. The probe stops after the first failure and records per-candidate
status, failure stage/reason, duration, and observed memory fields when the
runtime exposes them.

Selection is intentionally simple: the largest passing candidate is selected.
If no candidate passes, the plan fails. If only the minimum passes or every
candidate passes, the plan warns because the result is useful but not optimality
proof.

## Estimate Boundaries

Memory and artifact estimates are rough planning estimates, not contractual
sizes. Dense logits use:

```text
batch_size * sequence_length * vocab_size * dtype_bytes
```

Compact reducers use conservative approximations for top-k arrays, tail mass,
bucket masses, entropy, and corridor/exemplar side data. Large dense-logit
plans warn because they can create large artifacts.

## Non-Claims

The planner does not download models, does not perform network verification,
does not run the full corpus, does not write production artifacts, does not add
streaming/resume, does not schedule multiple devices, and does not add TPU/JAX
support. Backend emission capability statuses are unchanged by Spec 4.5.

After planning, use `radjax-tome build --streaming` for durable large-run
emission and `--resume` after interruption. See `docs/STREAMING_RESUME.md`.
