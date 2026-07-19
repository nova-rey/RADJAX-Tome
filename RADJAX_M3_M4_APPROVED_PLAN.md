# M3 + M4 Canonical Boundary and Staged Production Refactor

## 1. Verified Baseline

- `HEAD` and `origin/main` are `700a4bbf53f14eac9078d0fb9ae40da630e6db89`; `371a605` and `fde0051` are ancestors. The capture branch also contains this head.
- `radjax-tome golden validate --fixture tests/fixtures/golden_t4_1k` passes with count `256` and semantic root `sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
- Focused baseline: `111 passed` across import, ledger, golden, production, C6, delivery, and adversarial-linkage tests.
- Only discrepancy is untracked `.DS_Store`; it is unrelated and must remain untouched.

## 2. Canonical Dependency Map

| Surface | Current canonical role | Boundary finding | M3 correction |
|---|---|---|---|
| `radjax_tome.__init__` | Root public import | Eagerly imports `builder`, which exposes legacy and research modules | Minimal root facade; preserve legacy names through lazy compatibility resolution |
| `cli.main` | CLI parser and dispatch | Registers canonical, compatibility, C2–C5 expert, and research commands; handlers import lazily | Retain parser inventory; test canonical imports independently |
| `builder.__init__` | Builder facade | Eagerly imports `multi_gpu_path_b` and all builder APIs | Export canonical Path B eagerly only; lazy compatibility exports |
| `backends.__init__` | Backend facade | Eagerly imports frozen HF specimen/export/Qwen policy modules | Eagerly expose base contracts and native GPU only; lazy explicit compatibility/research access |
| `reports.__init__` | Report facade | Eagerly imports frozen arc/baseline/fingerprint-quality reports | Eagerly expose run-plan, doctor, report writer, and parity utilities only |
| `audit.__init__` | Audit facade | Eagerly imports frozen `refactor_surface` | Export selected-linkage APIs eagerly; make refactor audit explicit/lazy |
| `fingerprint.__init__` | Historical public facade | Eagerly imports frozen generalized fingerprint modules | Keep C2–C5 leaf modules canonical; historical exports become lazy compatibility access |
| `tome.__init__` | Artifact facade | Packaging reaches `audit.__init__` | Keep cover/package APIs; audit isolation removes frozen transitively imported code |

Canonical runtime modules include `builder/production.py`, `backend_textbook.py`, `corridor_artifacts.py`, `c6_integration.py`, `exemplar_delivery.py`, C2–C5 fingerprint modules, `backends/gpu_torch.py`, selected-linkage audit, target-store/provenance/I/O, run-plan/doctor, and cover/package helpers.

Canonical score, corridor, delivery, and production code directly use `builder.teacher_textbook` for `TinyTextExample`, loading, and validation. M3 isolates this as a leaf compatibility dependency; M4 may extract only required canonical types/validators after characterization.

Import-time effects are principally symbol loading. Torch is already lazy, but canonical package imports still load frozen modules. Broad package facades are the primary circular-import risk.

## 3. Corrected Runtime and Phase Map

| Execution point | Exact caller and function | Inputs | Outputs and authority | Progress behavior |
|---|---|---|---|---|
| Early score-surface fingerprint-corridor materialization | `build_production_gpu_tome` calls `build_corridor_artifacts` immediately after score pass | Completed `TeacherTargetStore`, corpus examples, empty selected records/payloads, no callback | Writes initial `corridors/corridor_summary.json`, fingerprints, modes, packed assignments and summary. Supplies full-token-position evidence for C6 feature/passport export; not selected-linked | `phase=fingerprint_corridor_export` is a lightweight checkpoint; no detailed `corridor_export` events |
| Selection authority export | `_export_c6_selection_authorities` | Early corridor files and score store | Owns C6 candidate features/manifest, source passports, global selector/supply, authority manifest, all bound to score-pass authority | `phase=selection_authority_export` |
| C2–C5 integrated selection | `_prepare_c6_selection` | Feature stream, early modes/assignments, global supply, passports | Owns C2 boards, C3 plan, C4 claims/collisions/backfill, C5 frozen records, diagnostics | `phase=corridor_global_selection` |
| Late selected-artifact corridor finalization | `materialize_selected_exemplar_delivery` calls `build_corridor_artifacts` after selected rerun summaries exist | Same score store/examples, C5 records, payload summaries, delivery callback | Rebuilds/overwrites public `corridors/*`, then links selected records and payloads to fingerprint/mode IDs and assignment status; owns final public corridor surface | Emits detailed `phase=corridor_export` events through completion |

The two calls use the same deterministic builder but have different contracts. Preserve this order and never collapse them:

`preflight → score pass → early corridor materialization → selection/global authority export → C2–C5 selection → selected rerun → late corridor finalization → payload/index assembly → validation/linkage → reconciliation/cover → reporting`.

## 4. Artifact Authority Map

| Artifact family | Owner and consumers | Resume/invalidation rule |
|---|---|---|
| Input provenance, corpus/model identity, run plan | CLI/preflight → score pass | Any corpus, teacher, target-policy, resolved config, or batch-policy drift invalidates reuse |
| Score shards, `metadata.json`, `run_manifest.json`, emission/teacher manifests | Streaming score pass → corridor export, delivery, validation | Reuse only with verified streaming manifest and matching resume hash |
| Modes, packed assignments, fingerprint population | Early materialization writes provisional files; late finalization overwrites final public files | Rebuild from verified score store; invalidate on score, mode policy, assignment storage, or stat support changes |
| C6 features and passports | Selection authority export → C2/C5/delivery/audit/golden | Valid only with matching early-corridor and score-pass authority hashes |
| Global board supply | Global authority export → C4 | Must bind the same score-pass authority |
| C2/C3/C4/C5 | Integrated selection → delivery/reconciliation/golden | Invalidate on features, allocation, global supply, dedupe/backfill, passports, or selection-order changes |
| Selected payloads/index/records/routes | Delivery and assembly → validation/audit/package/golden | Reuse only after C5 coordinate, passport, authority-hash, and index reconciliation |
| Final corridor selected-link fields | Late finalization → validation/report/cover/package | Invalidate if C5 selection or payload linkage changes; never reuse as a post-delivery selection input |
| Delivery, validation, audit, reconciliation, coverage | Verification stages → final report/cover/golden | Recompute from matching complete upstream evidence |
| Cover, production report, progress | Final stages → users/package | Derived only; never selection/payload authority |

M4 preserves paths and schemas and derives completion from existing evidence and hashes. No new persistent stage schema is authorized.

## 5. Proposed Target Architecture

```text
builder/
  production.py                 # compatibility facade
  native_path_b/
    api.py                      # canonical entry/config adapter
    contracts.py                # request/evidence/result/failure types
    orchestrator.py             # ordered calls; no workflow engine
    preflight.py
    score_pass.py
    authorities.py
    selection.py
    delivery.py
    assembly.py
    verification.py
    finalization.py
```

`production.py` remains public and delegates to `native_path_b.api` only for explicit native C6 Path B configuration. Global-only and legacy behavior remain behind compatibility adapters.

Direct dataclasses:

- `CanonicalPathBConfig`: resolved native request requiring `corridor_exemplar_v1`, `corridor_first_global_backfill_v1`, selection enabled, and `two_pass_rerun_selected`.
- `StageEvidence`: immutable paths, hashes, counts, and prior-stage proof.
- `StageResult[T]`: status, typed value, evidence, warnings, and failure.
- `StageFailure`: stage, reason, blockers, diagnostics, resumability, remediation.
- `NativePathBRunResult`: final status and existing report references.

No generic workflow engine, plugin framework, dependency-injection container, or replacement artifact schema.

### Target stages

| Stage | Contract/ownership | Evidence, resume, and tests |
|---|---|---|
| `preflight` | Config → preflight evidence; run-plan/doctor | Recompute on config/input drift |
| `score_pass` | Preflight → score evidence; shards/metadata/manifests | Delegate streaming resume; test shard reuse/drift |
| `score_surface_corridor_materialization` | Score → early corridor evidence; provisional `corridors/*` | Test no selected linkage is claimed |
| `fingerprint_corridor_selection_authority_export` | Early corridor → features/passports authority | Fail closed on authority mismatch |
| `global_authority_export` | Score → selector/global supply | Bind same score authority |
| `integrated_selection` | Fingerprint/global authority → frozen C2–C5 evidence | Test allocation, FIFO collisions, dedupe, backfill, indices |
| `selected_delivery_rerun` | C5 + score → staged payload evidence | Reuse only verified staging; test batch/retry/linkage |
| `selected_artifact_corridor_finalization` | C5 + summaries + score → final corridor evidence | Must follow rerun and precede promotion; test final linkage |
| `artifact_assembly` | Delivery + final corridor → assembled artifact | Atomic promotion and coordinate/hash reconciliation |
| `validation_linkage` | Artifact → validation/audit | CPU-only, strict failures |
| `reconciliation_cover` | C2–C5 + artifact + proofs → reconciliation/coverage/cover | CPU-only finalization tests |
| `final_reporting` | Terminal evidence → production result/progress | Idempotent rendering and named terminal failures |

### Compatibility and import policy

- CLI continues mapping into `ProductionBuildConfig`; one resolver creates canonical config.
- Existing public names remain importable through lazy compatibility maps; direct submodule imports remain supported.
- Every command name, argument, parser registration, and dispatch remains unchanged.
- Importing `radjax_tome`, importing canonical modules, constructing the parser, and rendering help must not import research-frozen handlers or eagerly load optional ML dependencies.
- Actual canonical GPU execution may lazily load required Torch/Transformers. It must not load unrelated research backends, handlers, reports, or policies.
- Research commands remain explicitly invocable and may lazily import their own handlers.
- Old artifacts remain inspectable/finalizable through compatibility migration and readers.

## 6. Checkpoints

### M3A — Dependency and behavior proof

Add import snapshots, runtime/artifact/resume characterization, and explicit early-versus-late corridor tests. No runtime movement.

Gate: baseline suite and golden fixture pass; all canonical/research/compatibility edges classified; both corridor phases and their ordering proven.

### M3B — Import isolation

Slim root, builder, backends, reports, audit, and fingerprint initializers; add lazy compatibility exports. Preserve CLI inventory.

Gate: lightweight imports/parser/help load neither frozen handlers nor optional ML stacks; explicit research imports work; actual GPU dispatch loads only its required canonical backend; golden passes.

### M3C — Canonical production boundary

Add `native_path_b/api.py` and typed config resolver; route only exact native C6 configuration through it.

Gate: CLI mapping/defaults remain compatible, global-only regression passes, research policies cannot silently enter canonical runtime, golden passes.

### M4A — Stage contracts

Add typed contracts and evidence readers, including separate early/late corridor contracts. Do not move algorithms.

Gate: unit tests describe current ownership/resume/failure behavior; no schema/output changes.

### M4B — Orchestrator extraction

Extract in preserved order: preflight; score; early corridor; fingerprint/global authority; C2–C5; rerun; late corridor; assembly; validation/linkage; reconciliation/cover/reporting.

Gate after every commit: focused stage and compatibility tests, import isolation, golden validation, no unexplained output change.

### M4C — Resume and failure normalization

Derive earliest incomplete/invalid stage from existing evidence. Preserve compatibility migration and fresh, partial, delivery-pending, finalization-only, already-complete, stale, corrupt, and mismatched cases.

Gate: full resume matrix passes; provisional early corridor evidence cannot be mistaken for final selected-linked evidence.

### M4D — Integration proof

Run appropriate complete non-GPU suite, static checks, import isolation, and golden validation. After review, run T4 Golden 1K comparison.

Gate: unchanged root/count, and passing delivery, validation, linkage, reconciliation, and zero unexplained semantic differences.

## 7. Commit Sequence and Delivery

Create `m3-m4-canonical-path-b-refactor` from `700a4bb`. Push checkpoint commits only there. Merge to `main` only after M4D local and reviewed T4 proof.

1. M3A import/runtime/artifact characterization and docs.
2. M3A early-versus-late corridor characterization.
3. M3B root/backend/audit initializer isolation.
4. M3B builder/reports/fingerprint isolation and lazy exports.
5. M3C canonical config adapter/facade delegation.
6. M4A typed contracts/evidence readers.
7. M4B preflight/score extraction.
8. M4B early corridor plus selection/global authority extraction.
9. M4B integrated C2–C5 extraction.
10. M4B rerun, late corridor, and assembly extraction.
11. M4B verification/finalization/report extraction.
12. M4C resume/failure normalization.
13. M4D local evidence/docs, followed after review by separate T4 evidence.

Every commit appends concise evidence to `bible.md` and is independently reviewable and reversible.

## 8. Subagent Ownership

| Owner | Exclusive scope |
|---|---|
| Architecture/import | Initializers, import graph/tests, research handler isolation, boundary docs |
| Runtime/stage | `builder/native_path_b/*`, contracts and extraction wrappers |
| Test/resume | Characterization, corridor-phase tests, evidence/resume/failure matrix, golden integration |
| Documentation/disposition | Canon docs, Hydra updates, milestone evidence, Bible entries |
| Root integration | `cli/main.py`, `builder/production.py` facade, public API, stage ordering, gates, branch promotion |

Use any and all available subagents. Parallelize independent work, but never allow concurrent edits to a shared integration file. Root alone integrates shared surfaces and resolves architecture conflicts.

## 9. Test Matrix

- Import isolation: root, canonical modules, parser/help, explicit research import, actual canonical backend dispatch.
- Configuration: CLI → production config → canonical config; canonical gate fields; global-only compatibility.
- Stage units: success, structured failure, evidence, progress, filesystem ownership.
- Orchestration: exact order, evidence-based skips, terminal reporting.
- Fingerprint corridors/C2–C5: scoring, allocation, FIFO collisions, dedupe, backfill, indices.
- Delivery: selected rerun, staging/promotion, dynamic top-k/buckets, passport linkage.
- Resume/failure: fresh through complete, partial/stale/corrupt/hash/config mismatches, finalization-only.
- Validation/linkage/reconciliation: indexes, multiplicity, C5 parity, source/passport/hash, cover/report consistency.
- Golden: immutable fixture, roots, portability, comparison policy.
- Rental: unchanged root/count and all terminal proofs pass.

## 10. Risk Register

| Risk | Detection and mitigation |
|---|---|
| Semantic drift | Golden/C2–C5/delivery tests; wrappers only; stop on difference |
| Corridor-phase collapse/reorder | Dedicated characterization and progress/artifact assertions |
| Circular imports | Subprocess import matrix; minimal facades, leaf imports, lazy maps |
| Research leakage | Hydra-boundary allowlist and import snapshots |
| Schema/filesystem drift | Artifact path/schema/ownership assertions; no new schemas |
| Resume regression | Full matrix and hash mutation tests; characterize before normalize |
| Duplicated authority | Evidence consistency and reconciliation; consume authoritative files only |
| Progress/report breakage | Preserve schemas and test terminal events/failures |
| Public API/CLI breakage | Lazy compatibility and parser/dispatch snapshots |
| Agent conflicts | Exclusive ownership and root integration |
| Excess abstraction | Direct functions/dataclasses only; no framework |

## 11. Approved Decisions

1. Retain current package-level names lazily through M3/M4; defer removal.
2. Derive stage state from existing evidence; no new persistent state without review.
3. Preserve all research commands and parser inventory; defer CLI reorganization.

## 12. Stop Conditions

Stop for review if the baseline/fixture differs; corridor phases cannot remain distinct and ordered; research runtime logic lacks an owner; compatibility requires public API or parser changes; an artifact schema/path or semantic algorithm must change; early evidence can be confused with final selected-linked evidence; resume compatibility cannot be preserved; golden validation/comparison or count changes; checkpoint-invalid baseline failures appear; agents produce incompatible architectures; or scope expands into performance, CLI/TUI, corpus, model, multi-GPU, 100K, deletion, or fixture regeneration.

Do not regenerate the fixture to normalize a difference.
