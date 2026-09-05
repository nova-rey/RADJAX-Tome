# M11 — TUI Production and Corpus Wizards

Status: finalized plan for approval and implementation. This document is planning-only; it does not close M11.

## Authority and base

Implement from `6b1e3c8563af748b872f68b8f4efd604fcbd4288` on a fresh branch named `m11/tui-production-corpus-wizards`. Contract remains pinned to `373e3d17060d4ce1c4a0db6065c9289da714bde7`. Accepted M8 (`88d6f418e0f39ab2ec61d9047f974f04657e5214`) and M9 (`7550ea9bcae23917fdaaee3d7506efaef849c6bf`, integrated through `ac23f0ea8a8a474d841e80f787003ac6492409ab`) are ancestors of the accepted M10 main tip. Package version is `0.1.0`, Python `>=3.11`, setuptools, and argparse; CI covers Python 3.11 and 3.12.

The current checkout contains preserved dirty historical edits in `bible.md` and `src/radjax_tome/backends/hf_torch.py`. Do not use it for implementation. Create a clean worktree from the accepted main tip and leave the dirty checkout untouched. Every implementation commit appends `bible.md`; integration into `main` requires separate authorization.

## Existing authorities and gaps

Production configuration is owned by `src/radjax_tome/builder/config_io.py::load_tome_build_intent` and `src/radjax_tome/builder/config.py` (`TomeBuildIntent`, `canonical_production_build_intent`, `resolve_tome_build_intent`, `apply_production_preset`, `apply_production_advanced_overrides`, `production_build_config_from_resolved`, `normalize_production_build_request`). The complete v2 loader adapts the corpus reference into the canonical internal shape. `load_tome_build_intent_v2` is only a minimal corpus projection and must not power the production wizard.

Corpus configuration is owned by `src/radjax_tome/corpora/config.py::load_corpus_build_intent`, schema `radjax_tome_corpus_build_intent_v1`, with artifact schema `radjax_tome_corpus_artifact_v2`. Supported adapters are exactly `local_text_tree_v1` and `local_jsonl_text_v1`; accepted policy identifiers and the smoke tokenizer string must be displayed as such. The current corpus schema cannot express an arbitrary HF tokenizer ID/revision, so M11 must not invent those fields. It must report an unusable binding during preflight.

Production execution is `cli.main.main → cli.mainline.main/run → load_tome_build_intent → resolve_tome_build_intent → production_build_config_from_resolved → assess_production_preflight → builder.production.build_production_gpu_tome → native_path_b.api/run_canonical_path_b`. Corpus execution is `load_corpus_build_intent → corpora.builder.build_corpus_artifact_v2 → corpora.lifecycle.preflight_corpus_build → journaled staging → tokenizer/source/shard construction → validate_corpus_artifact_v2 → publish_staging`.

Reuse `cli.models.CLIResult/CLIError/CLIWarning`, `cli.rendering.emit`, `tome.artifact_dispatch.validate_artifact/inspect_artifact`, `corpora.validation.validate_corpus_artifact_v2/inspect_corpus_artifact_v2`, `tome.packaging.plan_package_destination/package_tome_artifact`, `builder.production._ProductionProgressReporter`, and `builder.native_path_b.resume.resolve_native_path_b_resume`. Do not create a second validator, identity calculator, state machine, archive writer, or production route.

Two narrow owner-side gaps must be filled before UI polish:

1. Add complete, reloadable JSON/YAML document parsers and serializers to `builder/config_io.py` and `corpora/config.py`. A production v2 serializer must emit `corpus.artifact_path`, `expected_semantic_identity`, and `max_examples`, rather than dumping internal `dataset_path`/`corpus_manifest_path` fields. Preserve v1 documents as v1.
2. Add a corpus feasibility check and call it before `build_corpus_artifact_v2` creates staging. It checks supported tokenizer construction, source readability, and declared policy without transforming the corpus. Reuse it from a new `corpus build --preflight-only` route. Add read-only status projections for production progress snapshots and corpus journals; do not infer state from filenames.

## Framework and invocation

Use optional `Textual==8.2.8` under `[project.optional-dependencies].tui`. Textual supports Python 3.9+, scrolling/forms/focus, asynchronous workers, and headless `run_test()`/`Pilot` interaction; the base install remains dependency-free from Textual. Cite the versioned Textual documentation in M11 docs and test the exact pin. Do not install Textual devtools, web serving, or syntax extras.

Add `radjax-tome tui [corpus|production] [--config CONFIG]`. No workflow opens a start screen; a supplied config is dispatched by canonical schema. `tui --help`, `--version`, and existing commands work without importing Textual. Non-TTY or `TERM=dumb` execution prints an actionable CLI command and exits cleanly. `--json tui` is rejected because the interactive screen is not a JSON stream; headless CLI commands remain the automation interface.

## UX contract

Use shared screens: Start, Inputs, Behavior, Resources/Destination, Review, Preflight, Save/Confirmation, and Run Status/Results. Tab and Shift-Tab have deterministic focus order; Enter activates only the focused action; Escape backs out or asks before discarding; Ctrl-S saves; Ctrl-Q confirms exit; Ctrl-C requests cancellation. Use words and symbols in addition to color. Keep a single-column scrolling layout usable at 80×24, preserve drafts on resize, and show a resize/help notice below 60×18. Do not claim screen-reader support; prove keyboard-only operation.

Corpus screens select local sources, adapter, source ID/path, include/exclude, JSONL field and record-ID field, accepted normalization/order/chunking policies, minimum/max characters, exact deduplication, smoke tokenizer, resource limits, and destination. Production screens select teacher/model, tokenizer, provenance, validated corpus artifact, canonical preset/behavior/resource choices, destination, resume/overwrite, and package settings. Basic controls are grouped first; an Advanced view and canonical JSON editor preserve every supported field. No hidden defaults: values originate in canonical constructors/presets or are explicit visible choices.

Opening, editing, previewing, and saving never load models, allocate accelerators, create staging, delete output, or start a build. A run requires explicit confirmation after a fresh preflight and a byte check of the saved config. Config drafts contain only canonical document data and UI provenance, never execution state or derived semantic identity.

## Configuration and save behavior

Add `parse_tome_build_intent_document`, `tome_build_intent_document`, and JSON/YAML serialization to `builder/config_io.py`; make `load_tome_build_intent` delegate to them while retaining duplicate-key and unknown-field rejection. Add equivalent corpus functions to `corpora/config.py` plus `apply_corpus_operational_overrides`, extracted from current `cli.mainline.run` behavior.

Add `src/radjax_tome/io/config_export.py` for exclusive, fsynced, no-clobber publication. Reparse serialized text through the canonical parser before writing; write a temporary sibling and publish atomically without replacing an existing target. Resolve relative filesystem paths against the source config directory on load and export absolute paths by default. Preserve model/tokenizer identity strings as strings and do not expand environment variables or `~`.

## Preflight, destination, resume, and overwrite

Production preflight runs canonical config validation, `validate_required_inputs`, tokenizer/corpus compatibility, `assess_production_preflight` with the actual resume/overwrite values, and read-only `resolve_native_path_b_resume` when requested. Corpus preflight runs `preflight_corpus_build` plus the new owner-side feasibility check. Neither performs model loading or corpus transformation.

The matrix is:

| State | Production | Corpus |
|---|---|---|
| Missing destination | create after confirmation | create after confirmation |
| Empty directory | use unless resume requested | ordinary existing destination is refused by current owner rules |
| Unrelated file/nonempty directory/symlink | refuse | refuse |
| Owned incomplete state | explicit canonical resume only | only matching complete artifact or publication-ready owned staging |
| Incompatible state | refuse with canonical reason | refuse identity/validation mismatch |
| Complete valid output | default conflict; explicit resume/overwrite | explicit matching resume or owner-approved overwrite |
| Resume + overwrite | reject | reject |

The TUI never deletes files, guesses completion, or cleans broad paths. Packaging delegates to `plan_package_destination` and `package_tome_artifact`; profile and transport remain explicit. Corpus resume is documented as publication retry, not arbitrary mid-ingestion recovery.

## Execution, progress, cancellation, and output

Run the existing public CLI in an owned asynchronous subprocess using `asyncio.create_subprocess_exec`:

```text
sys.executable -m radjax_tome --json corpus build --config SAVED_CONFIG
sys.executable -m radjax_tome --json build --config SAVED_CONFIG
```

Validation, inspection, and packaging use their existing public commands. Child stdout is one final `radjax_tome_cli_result_v1` JSON document; stderr is drained continuously into a bounded log view. Never parse human stderr into authority. Poll production `production_progress_v1` atomically and corpus journal milestones through owner-side readers. Do not manufacture corpus percentages.

Cancellation confirms, sends SIGINT once to the owned child/process group, drains streams, waits, and preserves all workspace/staging state. No automatic SIGKILL or cleanup. An unresponsive child requires a second force-stop confirmation and is reported as resumability-unverified. Existing result codes remain authoritative: 0 success, 2 invocation/configuration, 3 unsupported projection/artifact, 4 validation, 5 destination conflict, 7 failed production, 130 interrupted, 141 broken pipe. A missing/invalid JSON result is a controller transport error with bounded stderr, not a fabricated canonical receipt.

## File-level implementation plan

Modify `builder/config_io.py`, `corpora/config.py`, `corpora/lifecycle.py`, `corpora/builder.py`, `cli/main.py`, and `cli/mainline.py` only for the adapters and routing described above. Add `builder/status.py`, `io/config_export.py`, and the lazy `tui/` package (`launcher.py`, `draft.py`, `fields.py`, `app.py`, `screens.py`, `controller.py`, `process.py`, and `app.tcss`). Update `pyproject.toml`, CI, `docs/hydra_disposition.json`, and minimal README/help documentation. Add `docs/M11_TUI_WIZARDS.md`, synchronized help fixtures, and focused `tests/test_m11_*.py` files. Do not alter Contract, M10 schemas, Golden fixtures, research scripts, or lower-layer semantics. Append `bible.md` in every commit.

## Test matrix and sequence

First implement and test config parse/serialize/reload, save-as path meaning, advanced-field preservation, corpus feasibility-before-mutation, and the real CPU corpus→production→validate→package flow using the tiny smoke fixtures from `tests/test_m10_corpus_builder.py`, `tests/test_m10_audit_reproductions.py`, `tests/test_m4_live_canonical_execution.py`, and `tests/test_production_build.py`. Compare canonical normalized configuration and applicable identities against direct invocation.

Then add tests for invalid config and backend failures before model/staging calls; all destination states; resume/overwrite exclusivity; progress and stale-state handling; subprocess JSON/stderr/oversize behavior; SIGINT and no-cleanup cancellation; both Textual workflows with headless Pilot; keyboard focus; 80×24 and small-terminal resizing; non-TTY fallback; and optional dependency isolation. Verify existing M4–M10, M9 CLI, Contract, selection-authority, archive, and package-profile suites remain green.

Run the repository gates:

```text
python -m pytest -q -p no:cacheprovider
python -m ruff check .
python -m ruff format --check .
python -m compileall src scripts tests
git diff --check
python -m build
```

Install the wheel in disposable Python 3.11 and 3.12 environments, test base and `[tui]` installations from outside the repository, and record wheel hash. No GPU, Golden regeneration, 50K benchmark, or M13/M14 run is required.

Commit boundaries:

1. Canonical config serializers, safe export, and early real CPU proof.
2. Shared corpus feasibility/status/preflight and safety tests.
3. Textual optional launcher, draft/controller, forms, and subprocess execution.
4. UX/accessibility-by-keyboard, documentation, CI, and compatibility containment.
5. One bounded correction pass after independent review, then closure evidence.

## Compatibility, review, and closure

Keep all old argparse and research commands working as compatibility/research routes. Root help presents only `build`, `corpus`, `validate`, `inspect`, `package`, `doctor`, `research`, and `tui` as public workflow concepts. Research commands remain behind `research`; production modules never import them. Do not delete historical evidence.

Use one independent read-only reviewer to check canonical config/state-machine usage, one public production path, preflight/safety, validate/inspect/package correctness, machine results, optional dependency isolation, honest cancellation/resume claims, and M11 scope. Apply at most one bounded correction and one targeted recheck.

Record `docs/M11_CLOSURE_REPORT.md` and `evidence/m11_tui_wizards/` with authority commits, Contract pin, command/help snapshots, config and identity round trips, CPU artifact/receipt/package identities, destination and cancellation evidence, keyboard/resize evidence, test and quality-gate results, wheel hashes, review verdict, branch push, clean-worktree proof, limitations, and explicit nonclaims. Acceptance is tied to the roadmap: both wizards exist, configs reproduce headlessly, live status/resume/validation/packaging delegate to canonical owners, expert options remain available, and no TUI-only behavior exists.

## Risks and recommendation

The highest risks are invalid v2 serialization, accidental loss of advanced fields, overstated corpus preflight/resume, blocking the UI event loop, and optional Textual leaking into ordinary CLI startup. The owner-side adapters, subprocess transport, round-trip tests, and explicit limitations address each. The accepted repository state is readable and M11 is one-shot implementable after approval; no product decision is required if arbitrary HF tokenizer selection remains explicitly outside the accepted corpus schema.

The next action after approval is Step 0: create the clean worktree/branch from `6b1e3c8563af748b872f68b8f4efd604fcbd4288`, then implement the configuration adapters and early CPU proof before visual polish. Integration into `main` remains separately authorized.

M11_PLAN_READY_FOR_APPROVAL
