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

Quarantine files are tracked reference material even when the archived path
contains a directory named `artifacts`. The repository `.gitignore` keeps runtime
`artifacts/` outputs ignored while explicitly unignoring `quarantine/**`.

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
references are `.txt` files outside the package tree. It also verifies manifest
quarantine references are tracked by git, not merely present in a local working
tree.

## Spec 2.9 Surgery Ledger

Spec 2.9 adds `docs/TOME_GENERATOR_QUARANTINE_SURGERY_LEDGER.json` as the final
disposition ledger for every quarantine-backed manifest path. The ledger covers
273 retained `.txt` references:

- `promoted`: 3
- `split_promoted`: 78
- `kept_quarantined`: 5
- `belongs_student`: 32
- `belongs_contract`: 2
- `deprecated`: 1
- `deferred`: 151
- `waived`: 1

Promoted and split-promoted entries point to active Tome modules under
`src/radjax_tome/fingerprint/`, `src/radjax_tome/reports/`, and
`src/radjax_tome/backends/`. The quarantine files remain retained source
evidence and must still not be imported.
