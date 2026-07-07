# Tome Generation Capability Matrix

## Executive Verdict

Spec 3 may proceed: no teacher-side Tome generation capability in this matrix blocks Contract-valid Tome emission.

| Capability | Status | Proof | Limitation | Blocks Spec 3 |
|---|---|---|---|---|
| dense_teacher_targets | active_generation_supported_synthetic_only | artifacts/tome_generation_capabilities/dense_teacher_targets/targets | Default proof uses SyntheticTeacherBackend; real HF teacher export is classified separately. | no |
| topk_tail_targets | active_generation_supported_synthetic_only | artifacts/tome_generation_capabilities/topk_tail_targets/targets | Compression proof starts from deterministic synthetic dense logits. | no |
| cascaded_bucket_targets | active_generation_supported_synthetic_only | artifacts/tome_generation_capabilities/cascaded_bucket_targets/targets | Bucket proof uses deterministic synthetic logits and probability bucket edges 1.0,0.1,0.0. | no |
| fingerprint_artifact_generation | active_generation_supported_synthetic_only | artifacts/tome_generation_capabilities/fingerprint_artifact_generation/artifact | Validator status: pass; generated from synthetic target store. | no |
| corridor_subset_generation | active_generation_supported | artifacts/tome_generation_capabilities/corridor_subset_generation/budget_subset_receipt.json | Producer artifact receipt role: corridor_subset; Student runtime corridor scoring is out of scope. | no |
| exemplar_reservoir_generation | active_generation_supported_synthetic_only | artifacts/tome_generation_capabilities/fingerprint_artifact_generation/artifact/exemplars | Generated 1 deterministic exemplar record; summary records=1. | no |
| hf_local_teacher_export | schema_validate_inspect_only | artifacts/tome_generation_capabilities/hf_local_teacher_export/hf_export_metadata.json | Default proof validates local-files-only HF metadata only; no optional torch/transformers run or model files are required. | no |
| prompt_corpus_tokenization | active_generation_supported | artifacts/tome_generation_capabilities/prompt_corpus_tokenization/prompts.jsonl | Smoke tokenizer proof generated 2 tokenized sequences. | no |

## Active Generation

dense_teacher_targets, topk_tail_targets, cascaded_bucket_targets, fingerprint_artifact_generation, corridor_subset_generation, exemplar_reservoir_generation, prompt_corpus_tokenization

## Schema Or Metadata Only

hf_local_teacher_export

Spec 3.3F11 adds runtime doctor/preflight reporting and artifact metadata
sanity reporting around existing backend-routed artifacts. It does not change
generation capability statuses, backend emission semantics, reducer math,
selector policy, real auto batch probing, production global selection,
multidevice scheduling, or TPU/JAX support.

Spec 4.1 adds local corpus-builder/provenance utilities. Corpus artifacts can
be hashed, validated, inspected, and linked into generated Tome metadata, but
backend emission capability statuses are unchanged.

## Synthetic Vs Real Teacher Proof

Dense, top-k/tail, cascaded, fingerprint, and exemplar proofs use
deterministic synthetic data by default. Real HF teacher generation
is not claimed by the default proof; only local-files-only metadata
is validated.

## Exact Next Recommendation

Proceed to Spec 3 Contract-valid Tome emission, keeping real HF generation optional/local-files-only until an explicit optional proof is run.
