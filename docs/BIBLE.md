# RADJAX-Tome Bible

## 2026-06-29 — Tome Builder migration scaffold

Moved the Tome Builder / TeacherTextbook builder from the historical `qrwkv-xla`
repo into `RADJAX-Tome` with only the minimum required producer-side support code.
The historical repo remains read-only. This phase preserves existing builder
behavior and does not yet implement the new `cover_page.json` Tome contract.

## 2026-06-29 — Toy Tome Contract CI compatibility

Updated the pre-existing toy TeacherTome smoke emitter to validate against the
current RADJAX-Contract dense Tome checks in CI. The migrated legacy
TeacherTextbook builder still preserves its existing output format.

## 2026-06-29 — Legacy Tome Builder A/B parity harness

Added an A/B parity harness comparing the live `RADJAX-Tome` legacy
TeacherTextbook builder against the archived `qrwkv-xla` builder. The harness
covers deterministic fake/offline dense logits, top-k/tail, and cascaded soft
label outputs. This phase verifies migration parity only; `cover_page.json` and
new Contract-valid Tome emission remain deferred.

## 2026-06-30 — Tome Generator extraction completeness audit

Added an extraction audit for the archived `qrwkv-xla` producer-side/Tome-generator
surface against the live `RADJAX-Tome` repo. The audit classifies relevant old
files, symbols, tests, docs, and fixtures as migrated, partial, missing, or
intentionally omitted before any Contract-valid `cover_page.json` emission work
continues.

## 2026-06-30 — Tome Generator migration map

Locked the short-term roadmap after the extraction audit showed major producer-side
migration gaps. Added triage tooling and a migration map that buckets missing
archived `qrwkv-xla` producer files into RADJAX-Tome, RADJAX-Contract,
RADJAX-Student, historical/deprecated, deferred, and human-review categories.
Spec 3 remains blocked until high-risk producer gaps are migrated or explicitly
waived.

## 2026-06-30 — Producer-core target migration

Migrated the highest-priority producer-core target loading, inspection,
multi-shard smoke, synthetic target export, tokenizer, and corpus tokenization
surface into RADJAX-Tome. The migration keeps JAX, Student runtime logic,
Contract-valid Tome emission, and `cover_page.json` out of scope; Spec 3 remains
blocked until the remaining producer/HF/fingerprint gaps are closed or waived.

## 2026-06-30 — Bulk producer migration with quarantine

Switched from micro-migration to a bulk producer migration strategy. Migrated
safe archived `qrwkv-xla` producer-side Tome Generator paths into `RADJAX-Tome`
and quarantined mixed producer/student files for later surgery. This phase moves
or accounts for the remaining producer-relevant gaps in bulk while keeping
Student runtime, Contract ownership, and `cover_page.json` emission out of scope.

## 2026-06-30 — Quarantine surgery pass

Processed the Spec 2.8 quarantine manifest and promoted producer-side artifact
schemas, readers, writers, validators, inspection helpers, and report structures
into active `RADJAX-Tome` modules while leaving Student/runtime/training logic
quarantined or out-of-scope. Added a surgery ledger so every quarantined old path
has an explicit disposition before Spec 3 cover-page work resumes.

## 2026-06-30 — Adversarial Tome Generator closure audit

Added an adversarial closure audit comparing archived `qrwkv-xla` producer-side
Tome Generator files, functions, tests, docs, and artifacts against active
`RADJAX-Tome`, quarantine references, and explicit Student/Contract/waiver
dispositions. The audit distinguishes active migrated behavior from quarantine
evidence before deciding whether Spec 3 cover-page work may resume.
