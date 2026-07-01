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
  "cover_page_version": 1,
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
      "size_bytes": 123
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
role, records its SHA-256 hash, and records its byte size. Spec 3.1 includes the
existing sidecars and every file under `shards/`. The cover page does not hash
itself, avoiding self-referential hashing in v1.

## Validation Scope

RADJAX-Tome validates that the cover page has required v1 fields, uses the
`unpacked_directory` layout, lists required files, does not reference paths
outside the artifact root, and has matching SHA-256 hashes for listed content.

RADJAX-Contract formal validation comes later. The cover page is the local Tome
surface that later Contract and bundle work can target.
