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
- Spec 2.9 quarantine surgery promoted producer-side fingerprint schemas,
  exemplar/corridor metadata, real-teacher capture summaries, provenance,
  producer reports, and HF specimen/export boundaries from mixed archived files.

These are implemented under:

- `src/radjax_tome/corpora/prompts.py`
- `src/radjax_tome/backends/qwen_policy.py`
- `src/radjax_tome/fingerprint/artifacts.py`
- `src/radjax_tome/fingerprint/exemplars.py`
- `src/radjax_tome/fingerprint/corridor.py`
- `src/radjax_tome/fingerprint/capture_summary.py`
- `src/radjax_tome/fingerprint/provenance.py`
- `src/radjax_tome/reports/`
- `src/radjax_tome/backends/hf_export.py`
- `src/radjax_tome/backends/hf_specimen.py`
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

These counts include quarantine references as accounting evidence. They do not
mean quarantined code is active migrated behavior, and `new_equivalent_tests`
includes quarantined test references until the audit distinguishes active tests
from quarantine inputs.

Spec 2.9 adds a separate surgery ledger with active-promotion interpretation:

| Decision | Count |
| --- | ---: |
| `promoted` | 3 |
| `split_promoted` | 78 |
| `kept_quarantined` | 5 |
| `belongs_student` | 32 |
| `belongs_contract` | 2 |
| `deprecated` | 1 |
| `deferred` | 151 |
| `waived` | 1 |

The ledger covers 273 quarantine-backed paths. It is the source of truth for
whether a quarantined reference has active producer behavior, was split, or
remains outside Tome ownership.
The bulk manifest remains a list for existing tooling compatibility and now
annotates quarantine-backed entries with `surgery_decision`,
`surgery_active_new_paths`, `surgery_ledger_path`, and
`surgery_blocks_spec3_after_this_phase`.

## Quarantine Strategy

Producer-relevant files that contain Student training/eval/runtime coupling,
heavy optional-dependency paths, or burn/fingerprint orchestration were copied
as non-importable `.txt` references under `quarantine/qrwkv_xla/`.

Quarantine is intentionally not under `src/radjax_tome`, is not exported from any
public API, and is covered by guardrail tests. These files are evidence and raw
material for Spec 2.9 surgery, not runtime code.

Every manifest-referenced quarantine file must be tracked by git. The guardrail
tests check both local existence and `git ls-files` tracking so clean CI clones
match local migration accounting.

## Ownership Boundaries

The bulk pass does not move Student runtime/training code into Tome. It also
does not take ownership of shared Contract schema and compatibility logic.
Student-owned and Contract-owned paths are recorded in the manifest instead of
being copied into active code.

## Spec 3 Status

Spec 3 remains blocked. Spec 2.9 split the quarantined mixed files, but this
repository still does not implement Contract-valid Tome emission or
`cover_page.json`. Spec 2.10 must rerun audit closure and waivers before the Spec
3 gate can pass.
