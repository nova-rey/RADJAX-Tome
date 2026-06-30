# Tome Generator Quarantine

Spec 2.8 introduces a non-importable quarantine for archived `qrwkv-xla`
producer-relevant files that should not enter active RADJAX-Tome code yet.

## Layout

```text
quarantine/qrwkv_xla/
  <old archived path>.txt
```

Each file has a short header with the original path and the quarantine reason.
The original source is stored as text so Python import discovery, package
exports, and normal tests do not execute it.

## Rules

- Quarantine is reference material only.
- Do not import quarantine files from `radjax_tome`.
- Do not add quarantine to public APIs.
- Do not treat a quarantined path as migrated runtime behavior.
- Split producer logic from Student/runtime behavior during Spec 2.9 before
  moving any quarantined code into active modules.

## Current Categories

The detailed list lives in
`docs/TOME_GENERATOR_BULK_MIGRATION_MANIFEST.json`.

- `quarantine_for_surgery`: mixed or dependency-heavy producer material retained
  for Spec 2.9 extraction.
- `defer_with_reason`: producer-related support material retained or tracked but
  not required before Spec 3.
- `belongs_contract`: schema/compatibility ownership belongs in RADJAX-Contract.
- `belongs_student`: Student runtime/training/eval ownership belongs in
  RADJAX-Student.
- `waive_with_reason`: explicitly inspected and not a Tome producer migration
  candidate.

## Guardrails

`tests/test_quarantine_guardrails.py` verifies that normal `radjax_tome` imports
do not import quarantine modules, the manifest has reasons, and quarantined
references are `.txt` files outside the package tree.
