# Tome Generator Bulk Migration

Spec 2.8 switches the extraction plan from one small producer slice at a time to
a bulk accounting pass. The goal is to move safe producer behavior into active
RADJAX-Tome code and quarantine producer-relevant files that are too tangled
with Student/runtime concerns for safe direct migration.

## Active Migrations

The following producer-safe surfaces moved into active RADJAX-Tome code:

- Prompt corpus loading, validation, deterministic splitting, canonical hashing,
  manifest generation, and prompt inspection/split CLIs.
- Qwen policy loading and resolution with no model downloads and no mandatory HF
  dependencies.
- Lightweight behavioral fingerprint artifact validation and inspection for
  producer artifact metadata, mode files, target shard counts, and exemplar shard
  references.

These are implemented under:

- `src/radjax_tome/corpora/prompts.py`
- `src/radjax_tome/backends/qwen_policy.py`
- `src/radjax_tome/fingerprint/artifacts.py`
- `scripts/inspect_prompt_corpus.py`
- `scripts/split_prompt_corpus.py`
- `scripts/resolve_qwen_policy.py`
- `scripts/validate_fingerprint_artifact.py`
- `scripts/inspect_fingerprint_artifact.py`

## Manifest Counts

`docs/TOME_GENERATOR_BULK_MIGRATION_MANIFEST.json` is the source of truth for
path-level handling in this phase.

| Classification | Count |
| --- | ---: |
| `migrate_now` | 8 |
| `quarantine_for_surgery` | 154 |
| `defer_with_reason` | 118 |
| `belongs_contract` | 8 |
| `belongs_student` | 14 |
| `waive_with_reason` | 1 |

Total handled paths: 303.

## Audit Impact

After rerunning extraction audit and triage against the archived repo:

| Metric | Before Spec 2.8 | After Spec 2.8 |
| --- | ---: | ---: |
| `missing` | 301 | 36 |
| `migrated` | 37 | 44 |
| `new_equivalent_tests` | 7 | 80 |
| high-risk `must_migrate_tome_before_spec3` | 3 | 0 |

The generated Spec 3 gate remains blocked because 47 high-risk producer items
are still quarantined for Spec 2.9 surgery.

## Quarantine Strategy

Producer-relevant files that contain Student training/eval/runtime coupling,
heavy optional-dependency paths, or burn/fingerprint orchestration were copied
as non-importable `.txt` references under `quarantine/qrwkv_xla/`.

Quarantine is intentionally not under `src/radjax_tome`, is not exported from any
public API, and is covered by guardrail tests. These files are evidence and raw
material for Spec 2.9 surgery, not runtime code.

## Ownership Boundaries

The bulk pass does not move Student runtime/training code into Tome. It also
does not take ownership of shared Contract schema and compatibility logic.
Student-owned and Contract-owned paths are recorded in the manifest instead of
being copied into active code.

## Spec 3 Status

Spec 3 remains blocked. This phase does not implement Contract-valid Tome
emission, does not write `cover_page.json`, and does not claim the quarantined
files are production-ready. Spec 2.9 must split quarantined mixed files, and Spec
2.10 must rerun audit closure and waivers before the Spec 3 gate can pass.
