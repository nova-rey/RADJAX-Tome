# Tome Builder

RADJAX-Tome owns the migrated legacy Tome Builder / TeacherTextbook builder.
This phase moves the producer-side code out of the historical `qrwkv-xla` repo
without redesigning the artifact format.

## Current Output

The builder still writes the legacy TeacherTextbook format backed by a
TeacherTargetStore:

```text
metadata.json
vocab_contract.json
teacher_manifest.json
emission_config.json
validation_report.json
shards/shard-00000.npz
```

Supported target types are `dense_logits`, `topk_with_tail_v0`, and
`cascaded_soft_labels_v1`. The new Contract-valid Tome shape and
`cover_page.json` are intentionally deferred.

## Fake/Offline Smoke

```bash
python scripts/build_teacher_textbook.py \
  --output artifacts/fake_teacher_textbook \
  --teacher-mode fake \
  --max-examples 2 \
  --sequence-length 8 \
  --vocab-size 16 \
  --overwrite
```

The fake mode is deterministic, CPU-only, and does not require JAX, torch,
Transformers, CUDA, TPU, internet, or model downloads.

## Validation

```bash
python scripts/validate_teacher_textbook.py \
  --path artifacts/fake_teacher_textbook \
  --write-report
```

The validator checks the legacy sidecars and shard shapes/dtypes. It does not
call the new RADJAX-Contract Tome validator yet because this migration preserves
the older TeacherTextbook format.

## Optional HF Teacher Mode

HF causal-LM teacher emission is preserved behind lazy optional imports:

```bash
python scripts/build_teacher_textbook.py \
  --output artifacts/hf_teacher_textbook \
  --teacher-mode hf \
  --teacher-model sshleifer/tiny-gpt2 \
  --local-files-only \
  --overwrite
```

Install `.[teacher-hf]` and provide cached model files for local-files-only runs.
Use `--allow-downloads` only when downloads are explicitly intended.

## Deferred Work

A later phase will adapt the builder to emit the new Contract-valid Tome format,
including `cover_page.json`, explicit compression metadata, and eventually
dynamic top-k cascading bucket payloads.
