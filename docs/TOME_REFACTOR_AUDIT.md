# RADJAX-Tome Refactor Audit

## Executive Summary

The repo is functional, but migration left clear cleanup debt: audit tooling is swollen, several scripts are no longer thin, and report/JSON rendering patterns are repeated.

- status: `complete`
- Spec 3 blocked: `False`
- checklist items: `6`

## Spec 3 Readiness

No refactor finding blocks Spec 3. The issues below should shape cleanup specs, not silently expand the next feature phase.

## Modularity Scorecard

| Category | Score | Evidence |
|---|---:|---|
| core package organization | 3 | Package layout is coherent, but audit and generation helpers need pruning. |
| teacher backend boundary | 4 | HF optional code is isolated behind backend modules. |
| target store boundary | 4 | Store/schema/inspection/compression are separated cleanly. |
| compression boundary | 4 | Compression is now a focused NumPy-only module. |
| fingerprint boundary | 3 | Artifact schemas and generation are separate, but package exports are broad. |
| script thinness | 3 | 1 scripts are large enough to review for business logic. |
| test maintainability | 3 | Coverage is strong, but fixture-heavy tests repeat setup. |
| optional dependency isolation | 5 | 0 suspicious heavy dependency imports found. |
| Contract separation | 5 | No active Contract repo edits or qrwkv_xla runtime imports are required. |
| documentation hygiene | 2 | 2 generated or JSON docs need compactness review. |
| future extensibility | 2 | 5 high/medium cleanup items should shape next specs. |

## Top Refactor Checklist

- `RTA-001` **high**: Shrink swollen audit modules without deleting audit coverage (src/radjax_tome/audit/closure.py, src/radjax_tome/audit/extraction_inventory.py, src/radjax_tome/audit/refactor_surface.py)
- `RTA-002` **medium**: Standardize JSON and Markdown report rendering (docs/TOME_GENERATION_CAPABILITY_MATRIX.md, scripts/prove_tome_generation_capabilities.py, src/radjax_tome/audit/closure.py)
- `RTA-003` **medium**: Move reusable script logic into src modules (scripts/prove_tome_generation_capabilities.py)
- `RTA-004` **low**: Trim broad package export surfaces (src/radjax_tome/fingerprint/__init__.py)
- `RTA-005` **medium**: Consolidate heavy test fixture construction (tests/test_ab_parity_compare.py, tests/test_tome_generation_capabilities.py, tests/test_tome_generator_audit_triage.py)
- `RTA-006` **medium**: Keep generated docs compact by default (docs/TOME_GENERATOR_BULK_MIGRATION_MANIFEST.json, docs/TOME_GENERATOR_QUARANTINE_SURGERY_LEDGER.json)

## Must Fix Before Spec 3

- None.

## Should Fix Before Production Burns

- `RTA-001` **high**: Shrink swollen audit modules without deleting audit coverage (src/radjax_tome/audit/closure.py, src/radjax_tome/audit/extraction_inventory.py, src/radjax_tome/audit/refactor_surface.py)
- `RTA-002` **medium**: Standardize JSON and Markdown report rendering (docs/TOME_GENERATION_CAPABILITY_MATRIX.md, scripts/prove_tome_generation_capabilities.py, src/radjax_tome/audit/closure.py)
- `RTA-003` **medium**: Move reusable script logic into src modules (scripts/prove_tome_generation_capabilities.py)
- `RTA-005` **medium**: Consolidate heavy test fixture construction (tests/test_ab_parity_compare.py, tests/test_tome_generation_capabilities.py, tests/test_tome_generator_audit_triage.py)
- `RTA-006` **medium**: Keep generated docs compact by default (docs/TOME_GENERATOR_BULK_MIGRATION_MANIFEST.json, docs/TOME_GENERATOR_QUARANTINE_SURGERY_LEDGER.json)

## Can Wait

- `RTA-004` **low**: Trim broad package export surfaces (src/radjax_tome/fingerprint/__init__.py)

## File Size and Complexity Hotspots

- `src/radjax_tome/audit/closure.py`: 1262 lines; largest function `run_closure_audit` (112 lines).
- `src/radjax_tome/builder/teacher_textbook.py`: 1200 lines; largest function `validate_teacher_textbook` (157 lines).
- `src/radjax_tome/audit/refactor_surface.py`: 1023 lines; largest function `render_markdown` (101 lines).
- `src/radjax_tome/audit/triage.py`: 933 lines; largest function `render_doc_summary` (106 lines).
- `src/radjax_tome/audit/extraction_inventory.py`: 763 lines; largest function `_status_for_match` (67 lines).
- `src/radjax_tome/fingerprint/artifacts.py`: 558 lines; largest function `validate_fingerprint_artifact` (80 lines).
- `scripts/prove_tome_generation_capabilities.py`: 504 lines; largest function `prove_capabilities` (269 lines).
- `src/radjax_tome/parity/runner.py`: 470 lines; largest function `_run_case` (90 lines).
- `src/radjax_tome/targets/store.py`: 400 lines; largest function `_validate_topk_tail_arrays` (94 lines).
- `src/radjax_tome/parity/artifact_compare.py`: 386 lines; largest function `_compare_array` (76 lines).
- `src/radjax_tome/fingerprint/exemplars.py`: 361 lines; largest function `validate_fingerprint_exemplar_records` (49 lines).
- `src/radjax_tome/corpora/prompts.py`: 334 lines; largest function `assign_prompt_splits` (47 lines).
- `src/radjax_tome/backends/hf_specimen.py`: 311 lines; largest function `run_hf_teacher_specimen_smoke` (51 lines).
- `tests/test_ab_parity_compare.py`: 288 lines; largest function `_artifact` (109 lines).
- `tests/test_tome_generator_closure_audit.py`: 281 lines; largest function `_write_fingerprint_repo_pair` (83 lines).
- `src/radjax_tome/fingerprint/generation.py`: 270 lines; largest function `build_minimal_fingerprint_artifact_from_target_store` (99 lines).
- `src/radjax_tome/targets/consumption.py`: 222 lines; largest function `_load_compressed_target_batch` (78 lines).
- `scripts/build_teacher_textbook.py`: 106 lines; largest function `main` (93 lines).

## Boundary Findings

- **medium** `mixed_responsibility`: src/radjax_tome/audit/closure.py spans CLI/scripts, I/O helpers, audit tooling, fingerprint artifacts, reporting, teacher backend. Recommendation: Split only if the secondary responsibility is real behavior, not glue.
- **high** `audit_module_swollen`: Audit modules mix collection, policy, markdown, and JSON shaping. Recommendation: Extract reusable report rendering and policy tables.
- **medium** `mixed_responsibility`: src/radjax_tome/audit/extraction_inventory.py spans I/O helpers, audit tooling, fingerprint artifacts, reporting, target compression, target store/schema. Recommendation: Split only if the secondary responsibility is real behavior, not glue.
- **high** `audit_module_swollen`: Audit modules mix collection, policy, markdown, and JSON shaping. Recommendation: Extract reusable report rendering and policy tables.
- **medium** `mixed_responsibility`: src/radjax_tome/audit/refactor_surface.py spans CLI/scripts, I/O helpers, audit tooling, fingerprint artifacts, reporting, target compression, target store/schema, teacher backend. Recommendation: Split only if the secondary responsibility is real behavior, not glue.
- **high** `audit_module_swollen`: Audit modules mix collection, policy, markdown, and JSON shaping. Recommendation: Extract reusable report rendering and policy tables.
- **medium** `mixed_responsibility`: src/radjax_tome/audit/triage.py spans I/O helpers, audit tooling, fingerprint artifacts, reporting, target compression. Recommendation: Split only if the secondary responsibility is real behavior, not glue.
- **high** `audit_module_swollen`: Audit modules mix collection, policy, markdown, and JSON shaping. Recommendation: Extract reusable report rendering and policy tables.
- **medium** `mixed_responsibility`: src/radjax_tome/backends/base.py spans target store/schema, teacher backend. Recommendation: Split only if the secondary responsibility is real behavior, not glue.
- **medium** `mixed_responsibility`: src/radjax_tome/backends/emission.py spans target store/schema, teacher backend. Recommendation: Split only if the secondary responsibility is real behavior, not glue.
- **medium** `mixed_responsibility`: src/radjax_tome/backends/synthetic.py spans target store/schema, teacher backend. Recommendation: Split only if the secondary responsibility is real behavior, not glue.
- **medium** `mixed_responsibility`: src/radjax_tome/builder/teacher_textbook.py spans other, target compression, target store/schema, teacher backend. Recommendation: Split only if the secondary responsibility is real behavior, not glue.
- **medium** `mixed_responsibility`: src/radjax_tome/cli/build_teacher_tome.py spans CLI/scripts, other. Recommendation: Split only if the secondary responsibility is real behavior, not glue.
- **medium** `mixed_responsibility`: src/radjax_tome/corpora/prompts.py spans I/O helpers, corpus/tokenization. Recommendation: Split only if the secondary responsibility is real behavior, not glue.
- **medium** `mixed_responsibility`: src/radjax_tome/corpora/tokenization.py spans I/O helpers, corpus/tokenization. Recommendation: Split only if the secondary responsibility is real behavior, not glue.
- **medium** `mixed_responsibility`: src/radjax_tome/corpora/tokenizer.py spans corpus/tokenization, teacher backend. Recommendation: Split only if the secondary responsibility is real behavior, not glue.
- **medium** `mixed_responsibility`: src/radjax_tome/emit/teacher_tome.py spans fingerprint artifacts, other. Recommendation: Split only if the secondary responsibility is real behavior, not glue.
- **medium** `mixed_responsibility`: src/radjax_tome/fingerprint/generation.py spans I/O helpers, fingerprint artifacts, target store/schema. Recommendation: Split only if the secondary responsibility is real behavior, not glue.
- **medium** `mixed_responsibility`: src/radjax_tome/fingerprint/provenance.py spans I/O helpers, fingerprint artifacts. Recommendation: Split only if the secondary responsibility is real behavior, not glue.
- **medium** `mixed_responsibility`: src/radjax_tome/parity/runner.py spans I/O helpers, other, target compression. Recommendation: Split only if the secondary responsibility is real behavior, not glue.

## Duplication Findings

- **medium** `duplicate_json_report_writers`: JSON report writing is repeated across scripts and audit modules. Recommendation: Extract a tiny JSON report writer or standardize on io/json.py.
- **low** `subprocess_test_duplication`: CLI smoke tests repeat subprocess environment setup. Recommendation: Use a shared test helper for PYTHONPATH and subprocess assertions.
- **medium** `markdown_renderer_duplication`: Markdown tables are hand-rendered in multiple report generators. Recommendation: Extract boring table rendering helpers under audit/report utilities.

## API Surface Findings

- **low** `api_surface`: src/radjax_tome/__init__.py exports 4 names. Recommendation: Keep explicit exports but group by domain.
- **low** `api_surface`: src/radjax_tome/audit/__init__.py exports 12 names. Recommendation: Keep explicit exports but group by domain.
- **low** `api_surface`: src/radjax_tome/backends/__init__.py exports 27 names. Recommendation: Keep explicit exports but group by domain.
- **low** `api_surface`: src/radjax_tome/builder/__init__.py exports 10 names. Recommendation: Keep explicit exports but group by domain.
- **low** `api_surface`: src/radjax_tome/cli/__init__.py exports 0 names. Recommendation: Keep explicit exports but group by domain.
- **low** `api_surface`: src/radjax_tome/corpora/__init__.py exports 23 names. Recommendation: Keep explicit exports but group by domain.
- **low** `api_surface`: src/radjax_tome/emit/__init__.py exports 1 names. Recommendation: Keep explicit exports but group by domain.
- **medium** `api_surface`: src/radjax_tome/fingerprint/__init__.py exports 63 names. Recommendation: Split or document the package import surface by role.
- **low** `api_surface`: src/radjax_tome/io/__init__.py exports 4 names. Recommendation: Keep explicit exports but group by domain.
- **low** `api_surface`: src/radjax_tome/parity/__init__.py exports 5 names. Recommendation: Keep explicit exports but group by domain.
- **low** `api_surface`: src/radjax_tome/provenance/__init__.py exports 1 names. Recommendation: Keep explicit exports but group by domain.
- **low** `api_surface`: src/radjax_tome/reports/__init__.py exports 18 names. Recommendation: Keep explicit exports but group by domain.
- **low** `api_surface`: src/radjax_tome/targets/__init__.py exports 19 names. Recommendation: Keep explicit exports but group by domain.

## Script Thinness Findings

- **low** `script_thinness`: scripts/ab_compare_teacher_textbook.py has 52 lines and 1 functions. Recommendation: Keep as thin CLI.
- **low** `script_thinness`: scripts/audit_tome_generator_closure.py has 7 lines and 0 functions. Recommendation: Keep as thin CLI.
- **low** `script_thinness`: scripts/audit_tome_generator_extraction.py has 51 lines and 1 functions. Recommendation: Keep as thin CLI.
- **low** `script_thinness`: scripts/audit_tome_refactor_surface.py has 40 lines and 1 functions. Recommendation: Keep as thin CLI.
- **low** `script_thinness`: scripts/build_teacher_textbook.py has 106 lines and 1 functions. Recommendation: Keep as thin CLI.
- **low** `script_thinness`: scripts/build_teacher_tome.py has 5 lines and 0 functions. Recommendation: Keep as thin CLI.
- **low** `script_thinness`: scripts/export_teacher_targets.py has 52 lines and 1 functions. Recommendation: Keep as thin CLI.
- **low** `script_thinness`: scripts/inspect_fingerprint_artifact.py has 35 lines and 1 functions. Recommendation: Keep as thin CLI.
- **low** `script_thinness`: scripts/inspect_prompt_corpus.py has 67 lines and 1 functions. Recommendation: Keep as thin CLI.
- **low** `script_thinness`: scripts/inspect_targets.py has 44 lines and 1 functions. Recommendation: Keep as thin CLI.
- **medium** `script_thinness`: scripts/prove_tome_generation_capabilities.py has 504 lines and 6 functions. Recommendation: Move reusable business logic into src and keep CLI thin.
- **low** `script_thinness`: scripts/resolve_qwen_policy.py has 53 lines and 1 functions. Recommendation: Keep as thin CLI.
- **low** `script_thinness`: scripts/split_prompt_corpus.py has 58 lines and 1 functions. Recommendation: Keep as thin CLI.
- **low** `script_thinness`: scripts/tokenize_corpus.py has 75 lines and 1 functions. Recommendation: Keep as thin CLI.
- **low** `script_thinness`: scripts/triage_tome_generator_audit.py has 47 lines and 1 functions. Recommendation: Keep as thin CLI.
- **low** `script_thinness`: scripts/validate_fingerprint_artifact.py has 31 lines and 1 functions. Recommendation: Keep as thin CLI.
- **low** `script_thinness`: scripts/validate_producer_pipeline.py has 54 lines and 1 functions. Recommendation: Keep as thin CLI.
- **low** `script_thinness`: scripts/validate_teacher_textbook.py has 37 lines and 1 functions. Recommendation: Keep as thin CLI.

## Test Suite Findings

- **medium** `test_quality`: tests/test_ab_parity_compare.py is fixture-heavy: 288 lines, 1 write_text calls. Recommendation: Extract fixture builders and assert less implementation detail.
- **low** `test_quality`: tests/test_ab_parity_runner.py has focused assertions. Recommendation: Keep.
- **low** `test_quality`: tests/test_bulk_migration.py exercises CLI behavior through subprocess. Recommendation: Keep, but share subprocess helpers.
- **low** `test_quality`: tests/test_corpus_and_emit.py has focused assertions. Recommendation: Keep.
- **low** `test_quality`: tests/test_corpus_tokenization.py exercises CLI behavior through subprocess. Recommendation: Keep, but share subprocess helpers.
- **low** `test_quality`: tests/test_fake_backend.py has focused assertions. Recommendation: Keep.
- **low** `test_quality`: tests/test_fingerprint_artifact_surgery.py has focused assertions. Recommendation: Keep.
- **low** `test_quality`: tests/test_fingerprint_exemplars_surgery.py has focused assertions. Recommendation: Keep.
- **low** `test_quality`: tests/test_fingerprint_reports_surgery.py has focused assertions. Recommendation: Keep.
- **low** `test_quality`: tests/test_hf_specimen_surgery.py has focused assertions. Recommendation: Keep optional/local marker; do not add network CI.
- **low** `test_quality`: tests/test_import.py has focused assertions. Recommendation: Keep.
- **low** `test_quality`: tests/test_import_boundaries.py has focused assertions. Recommendation: Keep.
- **low** `test_quality`: tests/test_quarantine_guardrails.py exercises CLI behavior through subprocess. Recommendation: Keep, but share subprocess helpers.
- **low** `test_quality`: tests/test_quarantine_surgery_ledger.py exercises CLI behavior through subprocess. Recommendation: Keep, but share subprocess helpers.
- **low** `test_quality`: tests/test_target_core_migration.py exercises CLI behavior through subprocess. Recommendation: Keep, but share subprocess helpers.
- **low** `test_quality`: tests/test_teacher_textbook_builder.py exercises CLI behavior through subprocess. Recommendation: Keep, but share subprocess helpers.
- **medium** `test_quality`: tests/test_tome_generation_capabilities.py is fixture-heavy: 216 lines, 0 write_text calls. Recommendation: Extract fixture builders and assert less implementation detail.
- **medium** `test_quality`: tests/test_tome_generator_audit_triage.py is fixture-heavy: 241 lines, 1 write_text calls. Recommendation: Extract fixture builders and assert less implementation detail.
- **medium** `test_quality`: tests/test_tome_generator_closure_audit.py is fixture-heavy: 281 lines, 12 write_text calls. Recommendation: Extract fixture builders and assert less implementation detail.
- **medium** `test_quality`: tests/test_tome_generator_extraction_audit.py is fixture-heavy: 226 lines, 1 write_text calls. Recommendation: Extract fixture builders and assert less implementation detail.

## Documentation Hygiene

- **medium** `large_docs_json`: docs/TOME_GENERATOR_BULK_MIGRATION_MANIFEST.json has 4499 lines. Recommendation: Move full generated payload to artifacts or compact it.
- **low** `long_historical_doc`: docs/TOME_GENERATOR_MIGRATION_MAP.md has 271 lines. Recommendation: Preserve if it is project history; otherwise add a summary top section.
- **medium** `large_docs_json`: docs/TOME_GENERATOR_QUARANTINE_SURGERY_LEDGER.json has 4508 lines. Recommendation: Keep only if the ledger is required as compact history.

## Recommended Follow-up Specs

- Spec 2.14: shrink audit/reporting modules and extract shared renderers.
- Spec 2.15: thin capability/audit scripts and move reusable logic into src.
- Later: test fixture consolidation and optional real-HF local smoke polish.
