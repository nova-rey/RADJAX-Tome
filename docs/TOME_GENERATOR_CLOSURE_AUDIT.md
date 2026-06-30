# Tome Generator Closure Audit

## Executive Verdict

Spec 3 may proceed to Contract-valid Tome emission and cover_page.json.

Spec 3 allowed: `True`

This is an adversarial closure report. Quarantine references, path-name
matches, and passing tests are not counted as active migrated behavior
unless the closure record has active paths and explicit evidence.

The committed `docs/TOME_GENERATOR_CLOSURE_AUDIT.json` file is a
compact summary. Full detailed JSON is generated under
`artifacts/tome_generator_closure_audit/` and is intentionally not
committed.

## Closure Metrics

### Files

- `active_behavior_equivalent`: 126
- `contract_bound`: 37
- `deferred`: 151
- `deprecated`: 1
- `missing`: 16
- `quarantine_only`: 5
- `student_bound`: 35
- `waived`: 1

### Symbols

- `active_behavior_equivalent`: 12
- `active_function_equivalent`: 120
- `contract_bound`: 302
- `missing`: 519
- `student_bound`: 183
- `unknown`: 469

### Tests

- `active_equivalent`: 33
- `contract_bound`: 5
- `missing`: 9
- `quarantine_only`: 34
- `student_bound`: 10
- `unknown`: 1

## A/B Parity

- status: `pass`
- cases: `9`
- source: `/Users/Cooper/Documents/Codex/2026-06-29/https-github-com-nova-rey-radjax-5/artifacts/ab_teacher_textbook/ab_summary.json`

## Function And Symbol Parity Summary

No missing/unknown producer symbols were found.

## Fingerprint Closure

Archived `src/qrwkv_xla/artifacts/fingerprint.py` is closed by exact
symbol mappings to active `RADJAX-Tome` fingerprint artifact code.
- `PROBABILITY_LIKE_STATS` -> `src/radjax_tome/fingerprint/artifacts.py:PROBABILITY_LIKE_STATS` status=`active_function_equivalent`; evidence=active Tome artifact schema preserves the probability-like fingerprint stats set
- `TARGET_PAYLOAD_LEGACY_JSONL` -> `src/radjax_tome/fingerprint/artifacts.py:TARGET_PAYLOAD_LEGACY_JSONL` status=`active_function_equivalent`; evidence=active Tome manifest keeps the legacy JSONL payload kind
- `TARGET_PAYLOAD_PACKED_CORRIDOR_V1` -> `src/radjax_tome/fingerprint/artifacts.py:TARGET_PAYLOAD_PACKED_CORRIDOR_V1` status=`active_function_equivalent`; evidence=active Tome manifest keeps the packed corridor payload kind
- `PACKED_TARGET_ARRAYS` -> `src/radjax_tome/fingerprint/artifacts.py:PACKED_TARGET_ARRAYS` status=`active_function_equivalent`; evidence=active Tome artifact schema preserves packed target array ranks
- `ValidationResult` -> `src/radjax_tome/fingerprint/artifacts.py:FingerprintValidationResult` status=`active_function_equivalent`; evidence=ValidationResult is intentionally renamed to FingerprintValidationResult with ok/blockers/warnings/status/to_dict

## CLI Parity Summary

- `active_behavior_equivalent`: 24
- `contract_bound`: 1
- `deferred`: 5
- `quarantine_only`: 5
- `student_bound`: 10
- `unknown`: 3
- blocking CLI records: 0

## Test Parity Summary

Active test coverage is counted separately from quarantine test references.
- `active_equivalent`: 33
- `contract_bound`: 5
- `missing`: 9
- `quarantine_only`: 34
- `student_bound`: 10
- `unknown`: 1

## Quarantine Ledger Verification

- ok: `True`
- entries: `273`
- blockers: `0`

## Active Vs Quarantine Accounting

A quarantine `.txt` file is evidence only. It is classified as
`quarantine_only`, `deferred`, `waived`, `student_bound`,
`contract_bound`, or `deprecated` unless the surgery ledger or direct
active-path evidence maps it to tracked active RADJAX-Tome code.

## Remaining Blockers Or Waivers

- None.

## Exact Next Recommendation

Proceed to Spec 3 Contract-valid Tome emission.

Spec 2.11 adds a command-level generation capability matrix at
`docs/TOME_GENERATION_CAPABILITY_MATRIX.md` and
`docs/TOME_GENERATION_CAPABILITY_MATRIX.json`. That matrix verifies active
teacher-side generation separately from this closure accounting audit.
