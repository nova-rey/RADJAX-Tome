# Tome Cover Page

`cover_page.json` is the front door for an unpacked RADJAX-Tome artifact. It
binds existing TeacherTextbook sidecars, target shards, hashes, validation
status, and claims into one inspectable JSON file.

## Unpacked Layout

Spec 3.1 uses a plain directory layout:

```text
artifact/
  cover_page.json
  metadata.json
  vocab_contract.json
  teacher_manifest.json
  emission_config.json
  validation_report.json
  shards/
    shard-00000.npz
```

Spec 3.1 introduced this unpacked layout. Spec 3.2 adds `.rtome` as a
deterministic tar packaging layer around the same cover-page-described files.

## Minimal Example

```json
{
  "artifact_kind": "radjax_tome",
  "cover_page_version": 2,
  "tome_version": 1,
  "layout": "unpacked_directory",
  "created_by": "radjax-tome.radjax_tome.tome.cover_page",
  "created_at": "2026-07-01T00:00:00+00:00",
  "source_artifact_type": "teacher_textbook",
  "teacher": {
    "model_id": "fake-deterministic-teacher",
    "backend_type": "fake",
    "model_family": "fake",
    "local_files_only": true,
    "allow_downloads": false
  },
  "tokenizer": {
    "tokenizer_id": "fake-deterministic-tokenizer",
    "vocab_size": 32,
    "vocab_contract_path": "vocab_contract.json",
    "tokenizer_hash": null
  },
  "targets": {
    "target_type": "dense_logits",
    "dtype": "float32",
    "sequence_length": 8,
    "num_examples": 2,
    "shard_count": 1,
    "target_params": {}
  },
  "contents": [
    {
      "path": "metadata.json",
      "role": "target_store_metadata",
      "sha256": "...",
      "size_bytes": 123,
      "required": true,
      "classification": "integrity_or_provenance"
    }
  ],
  "validation": {
    "status": "pass",
    "validation_report_path": "validation_report.json",
    "validated_by": "radjax_tome.builder.validate_teacher_textbook"
  },
  "claims_not_made": ["no_training_claim"]
}
```

## Contents And Hashes

Every `contents` entry names a file relative to the artifact root, assigns a
semantic role and classification, states whether the entry is required, records
its SHA-256 hash, and records its byte size. Cover-page v2 includes core
sidecars, target shards, packed corridor assignment files, and selected
exemplar files. The cover page does not hash itself.

Cover-page v2 also declares generic `behavioral_surfaces` and a declarative
`recommended_training_plan`. Current selected production Tomes declare a
corridor pass followed by a checkpoint and an exemplar pass followed by a
checkpoint. These are semantic references, not executable schedule classes.

## Validation Scope

RADJAX-Tome validates that cover-page v2 uses the `unpacked_directory` layout,
indexes every required durable role, keeps paths unique and inside the artifact,
matches content hashes and sizes, resolves surface prerequisites, and emits a
valid surface-referenced pass plan.

RADJAX-Contract owns the independent production consumer schema and semantic
validation. The canonical producer/consumer meaning is recorded in the
[Tome/Student consumer handoff](https://github.com/nova-rey/RADJAX-Contract/blob/main/docs/reference/RADJAX_TOME_STUDENT_CONSUMER_HANDOFF.md).
