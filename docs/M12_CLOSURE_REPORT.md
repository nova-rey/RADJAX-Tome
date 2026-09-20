# M12 closure report

## Authority

- Accepted M11 integration base: 53b19674a724a0999aa1c80f8b40d7e839a17c75
- Final audited content/test commit: c6934f4
- Branch: m12/user-docs-research-map
- Contract: 373e3d17060d4ce1c4a0db6065c9289da714bde7 (unchanged)
- Student revision: no Student checkout was available; this report makes no
  Student training or readiness claim.

M12 changes documentation, examples, Hydra navigation, and a small
documentation-test tranche. No runtime, Contract, Student, Golden, M8, or M11
semantics changed.

## Reader journey

The installed wheel was tested outside the checkout in a disposable Python
3.11 environment with the exact Contract wheel. A local text-tree corpus intent
ran through corpus build, validate, and inspect with one source and one record;
all JSON statuses were pass and the artifact carried a path-independent
sha256 semantic identity. A complete v2 production template loads through the
canonical M5 loader and is marked canonical-loader-validated by the docs checks.

The accepted CPU-reference v2 production smoke (inherited from the M11
authority and unchanged by M12) completed score, selection, selected rerun,
publication, validation, and inspection for one example. This is explicitly
not real teacher inference. The real teacher walkthrough is a prerequisite
bound template, not an executed GPU run.

A copied corridor-selected producer workspace was packaged with the installed
CLI using full_debug_provenance and directory transport; package and unpacked
package validation both passed. The synthetic dynamic CPU artifact is
documented as ineligible for Student profiles because it lacks corridor
artifacts. Direct validation of a .tgz path is documented as unsupported by the
current mainline; extract and validate the directory.

## Documentation inventory

- START_HERE.md: purpose, installation, CPU journey, real-teacher boundary,
  safety and recovery.
- CLI_GUIDE.md: current command grammar, JSON behavior, safety, TUI, and
  compatibility containment.
- CONFIGURATION_REFERENCE.md: v1/v2 M5 fields, normalization, identity, and
  operational overrides.
- CORPUS_BUILDER.md: corpus-v2 sources, policy, shards/indexes, tokenizer
  binding, semantic identity, resume, and migration.
- ARTIFACTS_AND_PACKAGES.md: corpus/producer/archive/Student boundaries,
  integrity, authority, profiles, and package validation.
- TROUBLESHOOTING.md: actionable safe diagnostics and stop conditions.
- RESEARCH_STATUS_MAP.md: human view derived from hydra_disposition.json.
- examples/m12_tome_build_intent_v2.example.json: complete JSON v2 template.
- tests/test_m12_documentation.py: link, loader, CLI grammar, inventory, and
  classification checks.
- PRODUCTION_BUILD.md: historical flag-based material explicitly marked
  compatibility/reference.

The Hydra ledger now records the current public command family and all new
reader pages. The human map is not a competing status registry.

## Verification

- Focused M12/CLI/M9-M11/M10 tests: 44 passed, 2 skipped before the final
  package wording correction; targeted correction checks: 5 passed.
- Full pytest after final correction: 1243 passed, 27 skipped, 0 failed.
- Ruff check: pass.
- Ruff format check: pass.
- compileall: pass.
- git diff --check: pass.
- Wheel: radjax_tome-0.1.0-py3-none-any.whl,
  SHA-256 002675e47469a05f7f5b55422b822a1103863da3445db1b4420190906abb99f3.
- Clean install: Python 3.11.2 disposable environment, pip check pass,
  help/version/doctor pass, Contract commit reported exactly.
- One focused reader review found the direct-tgz validation wording defect;
  correction 4a32260 changed the journey to directory transport and stated the
  limitation. Targeted recheck passed with no remaining blocker.

## Explicit nonclaims and limitations

No GPU, teacher download, training, M13, M14, Golden regeneration, or
historical evidence repair was performed. The documentation does not claim
arbitrary HF tokenizer support, arbitrary mid-shard resume, universal
filesystem atomicity, Student training, or model-quality proof. Optional TUI
installation is separate; headless CLI remains canonical. Historical M8
optimization experiments and legacy research commands remain preserved and are
not presented as current production alternatives.

Evidence seal commit is reported in the final handoff after it exists.
