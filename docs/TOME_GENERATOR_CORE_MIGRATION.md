# Tome Generator Core Migration

Spec 2.7 migrates producer-core target tooling from the archived `qrwkv-xla`
repo into `RADJAX-Tome`. This phase keeps the boundary deliberately narrow:
RADJAX-Tome can create, inspect, tokenize, and smoke-load legacy producer target
stores without adding Student runtime paths, JAX defaults, or Contract-valid Tome
emission.

## Migrated Producer Surface

- Target batch loaders for dense logits, top-k/tail, and cascaded soft labels:
  `radjax_tome.targets.consumption`.
- Multi-shard producer smoke helpers and shard iteration:
  `radjax_tome.targets.multishard`.
- Target store inspection:
  `radjax_tome.targets.inspection` and `scripts/inspect_targets.py`.
- Synthetic producer target emission:
  `radjax_tome.backends.synthetic`, `radjax_tome.backends.emission`,
  `radjax_tome.targets.export`, and `scripts/export_teacher_targets.py`.
- Smoke/HF tokenizer registry with lazy optional HF imports:
  `radjax_tome.corpora.tokenizer`.
- JSONL corpus tokenization that preserves `example_id`, source text hash, token
  IDs, and token counts:
  `radjax_tome.corpora.tokenization` and `scripts/tokenize_corpus.py`.
- Tome-side validation wrapper:
  `scripts/validate_producer_pipeline.py`.

## Archived Paths Inspected

The following archived paths were inspected for this migration:

- `scripts/export_teacher_targets.py`
- `scripts/inspect_targets.py`
- `scripts/tokenize_corpus.py`
- `scripts/validate_pipeline.py`
- `src/qrwkv_xla/distill/target_dispatch.py`
- `src/qrwkv_xla/generation/tokenizer.py`
- `src/qrwkv_xla/targets/consumption.py`
- `src/qrwkv_xla/targets/multishard.py`
- `src/qrwkv_xla/teacher_export/hf.py`
- `src/qrwkv_xla/teachers/backend.py`
- `src/qrwkv_xla/teachers/emission.py`
- `src/qrwkv_xla/teachers/hf_specimen_smoke.py`
- `src/qrwkv_xla/teachers/synthetic.py`

## Split, Deferred, Or Waived Paths

- `src/qrwkv_xla/distill/target_dispatch.py` remains deferred because it is
  Student/loss dispatch, not a producer artifact writer.
- `src/qrwkv_xla/targets/consumption.py` was split. NumPy loader and target batch
  structures moved to Tome; JAX loss helpers remain out of scope.
- `src/qrwkv_xla/targets/multishard.py` was split. Shard iteration and layout
  smoke moved to Tome; JAX loss aggregation remains out of scope.
- `scripts/export_teacher_targets.py` was recreated as a narrow synthetic
  producer export CLI. Bulk HF/Qwen teacher export remains deferred to Spec 2.8.
- `scripts/validate_pipeline.py` was not copied. Tome now has
  `scripts/validate_producer_pipeline.py`, which validates only local
  TeacherTextbook and target-store producer artifacts.
- `src/qrwkv_xla/teacher_export/hf.py` and
  `src/qrwkv_xla/teachers/hf_specimen_smoke.py` remain deferred to Spec 2.8
  except for the already existing optional HF causal-LM backend boundary.

## Tests Restored

- `tests/test_target_core_migration.py` covers synthetic emission, target batch
  loading, target store inspection, multi-shard smoke helpers, and target
  export/inspect CLIs.
- `tests/test_corpus_tokenization.py` covers smoke tokenizer behavior, HF
  tokenizer optional dependency boundaries, mocked HF tokenizer loading, source
  identity preservation, and the tokenization CLI.
- Existing builder, A/B parity, audit, import-boundary, and validation tests
  continue to run with the migrated producer-core surface.

## Remaining Blockers

Spec 3 remains blocked. This migration does not emit `cover_page.json`, does not
write the new RADJAX-Contract Tome artifact shape, does not migrate Student
training/runtime logic, and does not make JAX, torch, Transformers, CUDA, TPU, or
network access mandatory.

## Spec 2.8 Bulk Follow-Up

Spec 2.8 adds an explicit bulk manifest and quarantine layer on top of this core
migration. Safe prompt-corpus, Qwen policy, and lightweight fingerprint artifact
inspection paths moved into active code. Mixed burn/fingerprint/student-eval
paths were copied as non-importable quarantine references for Spec 2.9 surgery.
