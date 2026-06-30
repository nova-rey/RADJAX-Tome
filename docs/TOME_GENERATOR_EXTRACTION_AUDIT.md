# Tome Generator Extraction Completeness Audit

## Purpose

This audit measures whether the producer-side Tome Generator surface from the
archived `qrwkv-xla` monorepo has actually been extracted into `RADJAX-Tome`.
It is an inventory and risk report only. It does not implement `cover_page.json`,
change the legacy TeacherTextbook format, or migrate missing producer features.

Spec 2.6 has now triaged these findings into an ordered migration map. See
`docs/TOME_GENERATOR_MIGRATION_MAP.md` for the active short-term roadmap and
Spec 3 gate.

The archived `qrwkv-xla` repository is read-only. Use it only for inspection,
grep, parsing, and local temporary audit outputs. All committed audit tooling and
documentation lives in `RADJAX-Tome`.

## How To Run

```bash
python scripts/audit_tome_generator_extraction.py \
  --old-repo /path/to/qrwkv-xla \
  --new-repo . \
  --output-dir artifacts/tome_generator_extraction_audit \
  --overwrite
```

The script writes:

```text
artifacts/tome_generator_extraction_audit/
  extraction_audit.json
  extraction_audit.md
```

Use `--fail-on-blockers` only when you want CI to fail on high-risk missing or
partial producer-side items.

## Status Labels

- `migrated`: clear same-path, known path-map, filename, or specific-symbol match
  exists in `RADJAX-Tome`.
- `merged_into_other_file`: old public symbols appear in one or more new files.
- `partial`: weak overlap exists, but the audit cannot prove migration.
- `missing`: no path, filename, or specific-symbol equivalent was found.
- `intentionally_omitted_student`: classified as student-side and outside Tome.
- `intentionally_omitted_contract`: classified as contract/schema-side and
  expected to live in `RADJAX-Contract` unless locally wrapped.
- `intentionally_omitted_deprecated`: path appears historical or deprecated.
- `needs_human_review`: the heuristic could not classify the file confidently.

## Current Inventory Summary

Generated against:

```text
old repo: /tmp/qrwkv-xla-tome-generator-audit
new repo: /Users/Cooper/Documents/Codex/2026-06-29/https-github-com-nova-rey-radjax-5
```

Summary:

| Metric | Count |
| --- | ---: |
| total old candidate files | 384 |
| producer-relevant old files | 372 |
| migrated or merged | 22 |
| partial | 1 |
| missing | 317 |
| intentionally omitted as contract | 32 |
| needs human review | 0 |
| producer-relevant old tests | 89 |
| new equivalent tests | 2 |
| missing tests | 87 |
| producer CLI files | 43 |
| producer core files | 18 |
| mixed producer/consumer files | 23 |
| high-risk missing or partial blockers | 82 |

The audit confirms that the narrow legacy TeacherTextbook builder and validator
path has been extracted, but the broader archived producer surface has not.

## High-Risk Missing Or Partial Areas

Spec 3 should not proceed until these areas are migrated or explicitly waived:

- Producer CLIs: `export_teacher_targets.py`, `inspect_targets.py`,
  fingerprint artifact builders/inspectors, HF teacher specimen smoke scripts,
  mini-eval/export smoke scripts, prompt corpus splitting/tokenization scripts,
  and multiple validation scripts are missing.
- Fingerprint and real-teacher capture: `src/qrwkv_xla/fingerprint/*`,
  `src/qrwkv_xla/artifacts/fingerprint*.py`, `capture_summary`-style outputs,
  exemplar artifacts, stat bands, provenance, and quality-per-byte reporting are
  missing or mixed with student-side code.
- Target-store support beyond the migrated narrow store path:
  `targets/multishard.py`, target consumption/dispatch, sparse target loss tests,
  offline target consumption tests, and related target bundle tests are missing.
- HF/local-files-only teacher export and smoke coverage:
  `teacher_export/hf.py`, `teachers/hf_specimen_smoke.py`, backend emission
  tests, and HF export CLI tests are missing.
- Corpus and prompt handling: prompt corpus scripts/tests, prompt manifest
  generation, corpus hashing, split/tokenization utilities, and fixtures are
  missing.
- Producer tests: 87 of 89 producer-relevant old tests/fixtures have no detected
  new equivalent.

## Recommendation

Spec 3 should not proceed until the high-risk missing or partial producer-side
gaps are migrated or explicitly waived.

The current evidence does not support a claim that the Tome Generator was fully
extracted. It supports only a narrower claim: selected legacy TeacherTextbook
fake/offline behavior, validation, and A/B parity coverage exist in
`RADJAX-Tome`.

The Spec 2.6 migration map keeps this recommendation in force and provides the
next migration chunks:

- Spec 2.7 - producer schemas, stores, validators, and core tests
- Spec 2.8 - corpus, source identity, real-teacher/HF producer paths
- Spec 2.9 - behavioral fingerprint, corridor, exemplar producer artifacts
- Spec 2.10 - audit closure, A/B expansion, and explicit waivers

## Follow-Up Tasks

1. Review each high-risk blocker in the generated `extraction_audit.json`.
2. Decide which mixed producer/consumer fingerprint files belong in Tome versus
   Student, and record explicit waivers for the rest.
3. Migrate or waive missing producer CLIs before adding new Tome emission.
4. Restore producer-side HF/local-files-only teacher export smoke coverage or
   explicitly defer it.
5. Restore target inspection/export and multishard target-store tests needed by
   Tome generation.
6. Port the highest-value missing tests first:
   `test_topk_tail_textbook.py`, `test_cascaded_soft_labels_textbook.py`,
   `test_teacher_target_store.py`, `test_export_teacher_targets_cli.py`,
   `test_hf_teacher_backend.py`, and fingerprint artifact validation tests.
7. Rerun the audit after each migration wave and track the blocker count trend.

## Notes On Heuristics

The audit intentionally uses simple, transparent keyword and symbol heuristics.
It ignores generic public symbols such as `main`, `run`, and `to_dict` when
matching files, because those names otherwise inflate migration confidence.
Findings are audit evidence, not automatic migration decisions.
