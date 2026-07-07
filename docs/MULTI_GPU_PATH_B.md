# Experimental Multi-GPU Path B

Spec 4.7.a adds an opt-in experimental harness for Path B candidate scheduling:

```bash
radjax-tome multi-gpu-path-b \
  --teacher-model /models/MODEL \
  --tokenizer-id /models/MODEL \
  --dataset ./corpus_out/corpus.jsonl \
  --corpus-manifest ./corpus_out/corpus_manifest.json \
  --teacher-model-provenance ./teacher_model_provenance.json \
  --output ./multi_gpu_path_b_out \
  --devices cuda:0,cuda:1 \
  --target-policy corridor_exemplar_v1 \
  --sequence-length 128 \
  --batch-size-per-device 4 \
  --shard-size-examples 1024 \
  --fake-workers
```

Single-GPU `production-build` remains the recommended happy path. This command
is experimental/beta, opt-in, and requires an explicit `--devices` list. It
does not auto-use all GPUs.

## What It Does

The harness treats Path B candidate emission as map/reduce-shaped work:

- candidate assignments are split by shard range
- shard ownership is assigned round-robin across selected devices
- worker-local files are written under `workers/worker-*/`
- the CPU coordinator owns `multi_gpu_worker_manifest.json`
- the CPU coordinator writes `multi_gpu_path_b_report.json`
- candidate records are merged deterministically on CPU

The 4.7.a accepted path is the fake-worker scheduler harness. It validates
assignment, output ownership, deterministic merge, and report semantics without
requiring real GPUs or network access. Reports and manifests record
`candidate_execution_mode=fake_for_scheduler_test`.

## Non-Claims

This harness does not use DDP, does not use model parallelism, does not combine
GPU VRAM, does not download models, does not perform network verification, does
not add TPU/JAX support, and does not claim full multi-GPU burn validation.

Future real execution may load one teacher copy per selected device. Even then,
the devices remain independent candidate workers; they are not combined into
one larger model or memory pool.

## Files

The coordinator writes:

- `multi_gpu_worker_manifest.json`
- `multi_gpu_path_b_report.json`
- `merged_candidates.jsonl`

Workers write only worker-local files:

- `workers/worker-000/device.json`
- `workers/worker-000/assignments.jsonl`
- `workers/worker-000/assignment-00000-candidates.jsonl`

Arrival order must not affect merge results. Candidate records are sorted by
score descending, then stable tie-break fields.
