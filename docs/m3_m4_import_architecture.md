# M3/M4 Import Architecture Characterization

This is the M3A dependency characterization for the approved canonical Path B
refactor. It records the current import graph and public compatibility contract;
it does not move runtime code or change initializer behavior.

## Probe contract

`tests/test_m3a_import_isolation.py` runs every probe in a fresh Python
subprocess with the repository `src/` on `PYTHONPATH`. Each snapshot records the
loaded `radjax_tome.*` module names, any `torch` or `transformers` module names,
the parser command names, help text, and requested compatibility symbol. This
avoids test-order contamination from `sys.modules`.

The 2026-07-19 M3A baseline is:

| Entry point | Current classification | Observed behavior |
| --- | --- | --- |
| `import radjax_tome` | root facade, too eager | Loads 78 project modules, including all classified research/compatibility leaves below. It exposes `FakeTeacherBackend`, `TeacherTextbookBuildConfig`, `build_teacher_textbook`, and `emit_toy_teacher_tome`. |
| Canonical direct leaves (`builder.production`, `builder.corridor_artifacts`, `backends.gpu_torch`, `reports.run_plan`, `audit.selected_linkage`, `tome.packaging`) | canonical runtime | Each currently first executes the root facade and therefore has the same eager research/compatibility leakage. |
| CLI parser and `--help` | canonical public boundary | Preserve all 23 command names and do not import Torch or Transformers; they currently inherit the root facade's eager project-module leakage. |
| Explicit compatibility/research leaf import | compatibility contract | Every leaf and named public symbol listed below imports successfully without importing Torch or Transformers. |

Torch and Transformers are not imported by any of these import-only probes.
They remain execution-time optional dependencies: `hf_torch` and `gpu_torch`
defer importing them through `importlib.import_module`, and tokenizer loading
defers Transformers until it is needed. M3B must preserve this property.

## Current initializer edges and M3B disposition

| Eager facade today | Direct leaves reached from the initializer | Classification | M3B disposition |
| --- | --- | --- | --- |
| `radjax_tome` | `backends.fake`, `builder`, `emit.teacher_tome` | root public facade | Retain the four documented root names. Make the builder-backed compatibility names lazy so root import does not load the builder aggregate. |
| `builder` | `backend_textbook`, `c6_integration`, `exemplar_delivery`, `exemplar_selection`, `multi_gpu_path_b`, `production`, `teacher_textbook` | Path B canonical plus legacy aggregate | Eagerly expose only the canonical Path B surface. Keep legacy aggregate names/direct submodule imports through lazy compatibility resolution. `teacher_textbook` remains a leaf compatibility dependency until M4 characterization authorizes extraction. |
| `backends` | base, CPU/fake/emission, GPU/HF backends, HF export/specimen, Qwen policy, orchestration, registry, synthetic | base/native canonical plus HF research/compatibility | Eagerly expose base contracts and native GPU only. Make `hf_export`, `hf_specimen`, and `qwen_policy` explicit/lazy. Registry imports currently couple the aggregate to HF/GPU backends and need a circular-import review. |
| `reports` | run plan, doctor, parity, writers, arc, baseline, fingerprint quality, metadata sanity | canonical operations plus frozen reports | Keep run plan, doctor, parity, and writers eager. Make arc, baseline, and fingerprint-quality reports explicit/lazy. |
| `audit` | selected linkage, refactor surface | canonical audit plus frozen refactor audit | Keep selected linkage eager; make refactor audit explicit/lazy. |
| `fingerprint` | artifacts, exemplars, generation, inspection | historical generalized facade | Keep C2-C5 leaf modules as canonical direct imports. Make historical facade exports lazy while preserving direct submodules. |
| `tome` | bundle, cover page, packaging | artifact facade | Keep cover/package APIs eager. Its apparent audit reachability is transitive through packaging's local C6 integration import, so M3B audit isolation must be verified from a fresh subprocess. |

## Classified explicit compatibility and research leaves

The following direct submodules are public compatibility edges and must remain
importable after M3B. Their package-level re-exports become lazy; their direct
submodule paths do not change.

| Leaf module | Named compatibility symbol | Owner/disposition |
| --- | --- | --- |
| `backends.hf_export` | `HFTeacherExportConfig` | frozen HF export metadata; lazy backend facade export |
| `backends.hf_specimen` | `HFTeacherSpecimenConfig` | frozen HF specimen tooling; lazy backend facade export |
| `backends.qwen_policy` | `QwenPolicyMap` | research policy map; lazy backend facade export |
| `builder.multi_gpu_path_b` | `MultiGPUPathBConfig` | experimental multi-GPU harness; lazy builder facade export |
| `reports.arc` | `FingerprintArcReport` | frozen arc report; lazy reports facade export |
| `reports.baseline` | `FingerprintBaselineComparisonReport` | frozen baseline report; lazy reports facade export |
| `reports.fingerprint_quality` | `FingerprintQualityPerByteReport` | frozen quality report; lazy reports facade export |
| `audit.refactor_surface` | `RefactorAudit` | frozen refactor audit; lazy audit facade export |
| `fingerprint.artifacts` | `FingerprintManifest` | historical generalized fingerprint API; direct leaf retained, facade export lazy |
| `fingerprint.exemplars` | `FingerprintExemplarRecord` | historical generalized fingerprint API; direct leaf retained, facade export lazy |
| `fingerprint.generation` | `generate_exemplar_reservoir` | historical generalized fingerprint API; direct leaf retained, facade export lazy |
| `fingerprint.inspection` | `inspect_fingerprint_artifact` | historical generalized fingerprint API; direct leaf retained, facade export lazy |

## M3B isolation acceptance targets

M3B changes only initializers and lazy compatibility resolution. It must retain
the parser's exact command inventory and the explicit imports above while
making fresh-process imports of the root, canonical direct leaves, parser, and
help exclude every research/compatibility leaf in the preceding table and all
optional ML modules. Explicit research commands may then load only their own
handler. Actual canonical GPU dispatch may load its required Torch/Transformers
backend, but must not load unrelated research handlers, reports, or policies.

M4 can only change this map after typed stage-contract characterization proves a
need for a canonical `teacher_textbook` type or validator extraction. No M3A
runtime movement is authorized.
