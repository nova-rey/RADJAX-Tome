# Tome Generator Closure Audit

## Executive Verdict

Spec 3 blocked.

Spec 3 allowed: `False`

This is an adversarial closure report. Quarantine references, path-name
matches, and passing tests are not counted as active migrated behavior
unless the closure record has active paths and explicit evidence.

## Closure Metrics

### Files

- `active_behavior_equivalent`: 125
- `contract_bound`: 37
- `deferred`: 151
- `deprecated`: 1
- `missing`: 16
- `quarantine_only`: 5
- `student_bound`: 35
- `unknown`: 1
- `waived`: 1

### Symbols

- `active_behavior_equivalent`: 17
- `active_function_equivalent`: 106
- `contract_bound`: 302
- `missing`: 519
- `student_bound`: 183
- `unknown`: 478

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

Missing or unknown producer symbols remain; see blockers below.

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

- file: `src/qrwkv_xla/artifacts/fingerprint.py` status=`unknown` reason=Only weak basename or keyword overlap found.
- symbol: `src/qrwkv_xla/artifacts/fingerprint.py` symbol=`PROBABILITY_LIKE_STATS` status=`unknown` reason=file classified as unknown
- symbol: `src/qrwkv_xla/artifacts/fingerprint.py` symbol=`TARGET_PAYLOAD_LEGACY_JSONL` status=`unknown` reason=file classified as unknown
- symbol: `src/qrwkv_xla/artifacts/fingerprint.py` symbol=`TARGET_PAYLOAD_PACKED_CORRIDOR_V1` status=`unknown` reason=file classified as unknown
- symbol: `src/qrwkv_xla/artifacts/fingerprint.py` symbol=`PACKED_TARGET_ARRAYS` status=`unknown` reason=file classified as unknown
- symbol: `src/qrwkv_xla/artifacts/fingerprint.py` symbol=`ValidationResult` status=`unknown` reason=file classified as unknown

## Exact Next Recommendation

Do not start Spec 3. Resolve or explicitly waive every listed blocking file, symbol, CLI, test, A/B, and validation item first.
