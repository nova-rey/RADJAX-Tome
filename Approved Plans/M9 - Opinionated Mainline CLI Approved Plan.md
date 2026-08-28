# M9 — Opinionated Mainline CLI Implementation Plan

## 1. Authority and repository-state findings

### Authoritative base

The exact M9 code base is:

`88d6f418e0f39ab2ec61d9047f974f04657e5214`

This is the accepted `M8_CLOSED` commit on `origin/m8/sparse-selected-logits-final-closure`.

Remote state verified on 2026-08-28:

| Ref | Commit | Finding |
|---|---|---|
| `origin/main` | `6a6c65378cfd86a190e44e861ed9323927c2acc8` | Does not contain M8 |
| `origin/m8/sparse-selected-logits-final-closure` | `88d6f418e0f39ab2ec61d9047f974f04657e5214` | Accepted M8 closure |
| `origin/m9/opinionated-mainline-cli` | `8a2dc2794ec32227f6187fa22837ba1f8dae0ccb` | Existing but based on pre-M8 main |
| Contract accepted M8 authority | `373e3d17060d4ce1c4a0db6065c9289da714bde7` | Contract 0.9.0 buffer-native codec commit |
| Contract tagged release base | `1fa43e1aea2e198511db86dafb0aeefa525d48c7`, tag `v0.9.0` | Ancestor of `373e3d1` |

`origin/main`, M8, and the existing M9 branch share merge base `6a6c653`. M8 is not an ancestor of either main or the existing M9 tip. The existing M9 branch therefore cannot close M9 regardless of its local tests.

### Required bounded prerequisite

Before creating the M9 implementation branch:

1. Re-fetch `origin`.
2. Require `origin/main == 6a6c65378cfd86a190e44e861ed9323927c2acc8`.
3. Require `origin/m8/sparse-selected-logits-final-closure == 88d6f418e0f39ab2ec61d9047f974f04657e5214`.
4. Fast-forward `main` to `88d6f41`; no synthetic merge commit is needed because M8 directly descends from main.
5. Push that fast-forward and verify the remote.
6. Create `m9/opinionated-mainline-cli-reconciled` at exactly `88d6f41`.

Do not force-push or reuse `m9/opinionated-mainline-cli`. Retain it as unaccepted archaeology.

### Two accepted-authority defects requiring a prerequisite commit

The M8 tree at `88d6f41` contains two concrete inconsistencies:

- `pyproject.toml` pins Contract `46fbf040…`, but current M8 code imports `compact_body_from_buffers` and `encode_compact_body_packed_from_buffers`, which first exist at accepted Contract commit `373e3d1`.
- `apply_production_preset()` still requests selected-rerun batch size 2 for `smoke` and 8 for T4/production presets, although final M8 authority says batch 1 remains canonical.

The first M9-branch commit must therefore:

- Pin Contract exactly to `373e3d17060d4ce1c4a0db6065c9289da714bde7`.
- Change only the canonical preset `selected_rerun_batch_size` values to 1.
- Add pin/import and preset regression tests.
- Append the reconciliation to `bible.md`.

This is an M8-authority reconciliation prerequisite, not a reopened optimization trial. Contract source remains unchanged.

### Package and toolchain state

- Tome package: `radjax-tome 0.1.0`.
- Python: `>=3.11`; CI covers 3.11 and 3.12.
- Build backend: setuptools.
- CLI framework: standard-library `argparse`.
- Console script: `radjax-tome = radjax_tome.cli.main:main`.
- Dependencies on M8: NumPy, PyYAML, DuckDB 1.4.5, exact Git Contract pin.
- Quality gates: pytest, Ruff lint, Ruff format, compileall, and `git diff --check`.
- There is no `radjax_tome.__main__`, so `python -m radjax_tome` currently fails.
- No Tome tags are present in the inspected clone.

### Worktree constraints

The supplied working directory `/Users/Cooper/Documents/radjax-tome` is an unrelated, commitless scratch repository containing untracked M8 files. It must not be used for implementation.

Existing user worktrees contain `.DS_Store`, `uv.lock`, and unrelated dirty P6.U1 work. All must remain untouched. Create a fresh worktree for the reconciled M9 branch.

No GPU, test artifact, or tracked repository file was changed during this planning pass.

## 2. Current CLI inventory

The current public parser is the 2,100-line `src/radjax_tome/cli/main.py`. Its top-level help exposes 27 research, production, compatibility, corpus, Golden, and diagnostic commands as co-equal choices.

| Existing interface | Current role and dependency | M9 treatment |
|---|---|---|
| `build` | Legacy/fake/backend TeacherTextbook builders; creates config directly from many flags | Preserve as hidden compatibility dispatch and `research build`; public `build` becomes canonical M5 config path |
| `production-build` | Current canonical M5 normalization → `build_production_gpu_tome` path, but exposes the entire 60-plus-field Hydra | Preserve as deprecated hidden compatibility dispatch and `research production-build` |
| `validate` | Legacy target-store/bundle validation with optional cover and metadata sanity | Replace public behavior with unified artifact dispatch; preserve `validate --path` as compatibility form |
| `verify-artifact` | Contract v3 standard/governed/external verification | Fold into public `validate --mode`; preserve hidden compatibility dispatch |
| `inspect` | Direct target-store or bundle summary, partly reconstructing from local files | Replace with validated, bounded artifact inspection; preserve `inspect --path` compatibility form |
| `package-artifact` | Canonical `package_tome_artifact` profile writer | Expose through public `package`; preserve old name as compatibility |
| `validate-package` | Canonical package validator | Fold into public `validate` |
| `pack`, `unpack` | Historical `.rtome` transport tools | Keep under `research`; public validation/inspection still accepts `.rtome` |
| `doctor` | Lightweight dependency/runtime report, but configured by many CLI-private flags | Keep public, accepting optional canonical config; old flag-heavy form moves to compatibility/research |
| `status` from existing M9 branch | Scrapes `production_build_report.json` | Reject; incomplete-state inspection must use canonical resume/state evidence |
| `plan` | GPU planning utility that may perform bounded probes | Research-only; M9 build preflight uses the production preflight API |
| C2–C5 corridor commands | Accepted offline research/evidence utilities | `research` plus hidden compatibility dispatch |
| `parity`, `exemplar-delivery-parity`, `audit-selected-linkage` | Engineering comparison/audit tools | Research-only; do not use as public semantic validation substitutes |
| `multi-gpu-path-b` | Explicit experimental harness | Research-only |
| `corpus` | Current local corpus tools; M10 owns product redesign | Research-only in M9 |
| `model` | Model provenance utilities | Research-only; production preflight still consumes their validated output |
| `golden` | Golden engineering commands | Research-only; no Golden regeneration |
| `prove-capabilities`, `finalize-replay-workload` | Diagnostics and M8 evidence utilities | Research-only |
| `src/radjax_tome/cli/build_teacher_tome.py` | Toy compatibility module | Leave unchanged |
| `scripts/build_teacher_textbook.py`, `build_teacher_tome.py`, validation/inspection wrappers | Legacy-compatible evidence workflows | Leave executable and classify explicitly |
| Corpus, tokenizer, model, fingerprint, benchmark, migration, M8 and audit scripts | Research/internal/evidence | Leave unchanged and outside the public paved road |
| `tools/validate_radjax_tome_contract*.py` | Historical portable validator wrappers | Leave unchanged; public CLI uses installed Contract APIs |

Legacy direct invocations remain temporarily usable through a non-advertised compatibility router that emits a deprecation warning and delegates to the existing underlying parser. They will not appear in normal top-level help.

## 3. Canonical dependency path

### Public production path

```text
radjax-tome build --config CONFIG
  → radjax_tome.cli.main
  → radjax_tome.cli.mainline
  → builder.config_io.load_tome_build_intent()
  → builder.config.resolve_tome_build_intent()
  → builder.config.normalize_production_build_request()
  → production_stages.preflight.assess_production_preflight()
  → builder.production.build_production_gpu_tome()
  → native_path_b.api.resolve_canonical_path_b_config()
  → native_path_b.api.run_canonical_path_b()
  → production_stages.path_b_integration.run_post_score_path_b()
  → M4 slices 1–5
  → production_stages.reporting.native_final_reporting_operation()
  → publish_native_path_b_v4() or publish_v3_from_handoff()
  → canonical validation
  → production_build_report_v1
  → m9_cli_receipt_v1
```

The CLI supplies a `TomeBuildIntent` or `ResolvedTomeBuildConfig`. It does not construct `ProductionBuildConfig`, invoke stages, derive identities, or inspect output directories to infer completed work.

### Canonical configuration

Existing authority remains:

- `TomeBuildIntent`
- `ResolvedTomeBuildConfig`
- `NormalizedProductionBuildRequest`
- `resolve_tome_build_intent`
- `normalize_production_build_request`
- `production_build_config_from_resolved`
- `selection_authority_payload_v1`
- `selection_authority_hash_v1`

The missing seam is strict JSON/YAML deserialization into the existing closed dataclasses. Implement it in `builder.config_io`, not in the CLI.

Rules:

- Require the complete `radjax_tome_build_intent_v1` object and all canonical sections.
- Reject duplicate keys, unknown fields, missing fields, wrong section types, and unsupported schema versions.
- Resolve relative paths against the config file’s parent, never the process CWD.
- Do not expand `~`, environment variables, presets, or hidden defaults.
- JSON and YAML expressing the same values must normalize identically.
- `--resume` and `--overwrite` map through `apply_production_advanced_overrides`; they remain operational and do not alter the fixed selection-authority projection.

### Missing preflight seam

Current production initializes progress and can write files before all failures are known. It can also recursively remove a recognized output directory before runtime planning has passed.

Refactor `production_stages/preflight.py` to expose one pure `assess_production_preflight()` result consumed by both the CLI and `run_preflight()`. It must:

- Normalize and validate configuration.
- Validate inputs, provenance, Contract compatibility, backend capability, package constraints, and destination state.
- Use `resolve_native_path_b_resume()` for canonical resume decisions.
- Construct the existing doctor and run-plan data in memory.
- Load no model, construct no backend, allocate no accelerator, execute no score/selected stage, and write nothing.
- Return a typed destination action and structured failures/remediation.
- Permit mutation only after the assessment passes.

### Artifact validation and inspection path

Add a bounded resolver in `tome/artifact_dispatch.py`:

- Contract artifact v3 directory/`.tgz`/`.rtome` → Contract v3 APIs.
- M7 v4 directory/archive → `validate_streaming_tome` and streaming reader.
- Canonical student/full-debug package directories/archives → existing Tome/Contract package validators.
- Historical canonical v1–v5 forms → their frozen native validators/adapters.
- Incomplete producer workspace → native resume resolver plus available validated reports; never filename-only inference.

Inspection validates the applicable artifact contract before reading descriptive fields. It never loads selected payload bodies merely to produce counts or metadata.

### Packaging

Normal `build` already reaches terminal M4 reporting and publishes the canonical M7 v4 pair or explicit Contract-v3 pair. It must not call a second publisher.

Public `package` remains a separate projection of a valid producer workspace into `student` or `full_debug_provenance` through `package_tome_artifact`.

`PackageIntent.profile` and `.transport` are not currently carried through `production_build_config_from_resolved`. Public `build` must therefore fail preflight unless they retain the currently implemented `unpacked`/`directory` values. It must direct users to `radjax-tome package` rather than silently ignoring a requested profile or transport.

## 4. Proposed public command surface

Global options:

```text
radjax-tome [--json] [--quiet] [--debug] [--no-color] [--version] COMMAND
```

### Commands

| Invocation | Purpose and required inputs | API | Result and exit |
|---|---|---|---|
| `radjax-tome build --config CONFIG [--resume\|--overwrite] [--preflight-only]` | Normal production path from complete M5 config | Config loader → pure preflight → `build_production_gpu_tome` | Final directory/archive identities and receipt; 0 pass/warn, categorized failure otherwise |
| `radjax-tome validate ARTIFACT [--mode standard\|governed\|external-attestation] [--expected FILE] [--attestation FILE] [--attestation-policy optional\|required] [--evaluation-time RFC3339]` | Contract-correct validation of supported forms | Artifact dispatcher and Contract-owned validators | Human or `radjax_tome_cli_result_v1`; validation mismatch exit 4 |
| `radjax-tome inspect ARTIFACT` | Bounded self-description after canonical validation | Artifact resolver, indexes, covers, Contract reports | Identity, versions, teacher/tokenizer, corpus, corridor, selected, profile, inventory, reports, limits |
| `radjax-tome package WORKSPACE --output OUTPUT --profile student\|full_debug_provenance [--transport tgz\|directory] [--student-contract-profile v5\|v6] [--overwrite]` | Canonical profile projection and archive creation | `ValidatedProducerArtifact` → `package_tome_artifact` → validation | Package path, semantic identity, profile, size, raw digest, receipt |
| `radjax-tome doctor [--config CONFIG]` | Read-only installation/runtime/config preflight | Runtime doctor plus pure production preflight | Capability/config report; never model load or artifact mutation |
| `radjax-tome research …` | Explicit access to retained engineering commands | Existing parser and APIs | Existing behavior, clearly marked non-mainline |

Defaults:

- `build` requires `--config`; there is no flag-bag fallback.
- `package --transport` defaults to `tgz`.
- Student packaging defaults to current v5.
- Artifact Contract v2 remains the canonical config default; v3 remains explicit.
- `validate` defaults to standard integrity.
- Governed comparison requires `--expected` supplied outside the artifact.
- External attestation requires external `--attestation` and evaluation time.
- `.rtome` remains accepted transport for validation/inspection but is not newly produced by public `package`.

### Representative examples

```bash
radjax-tome build --config tome-build.yaml
radjax-tome build --config tome-build.yaml --resume
radjax-tome build --config tome-build.yaml --preflight-only

radjax-tome validate ./out.v4.tgz
radjax-tome validate ./out.v3 \
  --mode governed \
  --expected ./expected-governed-identity.json

radjax-tome inspect ./out.v3.tgz

radjax-tome package ./producer-workspace \
  --output ./student.tgz \
  --profile student

radjax-tome doctor --config tome-build.yaml
```

Machine mode:

```bash
radjax-tome --json validate ./out.v3.tgz
```

Representative configuration failure:

```text
ERROR M5_CONFIG_INVALID
selection.selected_rerun_batch_size must be positive
config: /work/tome-build.yaml
repair: correct the canonical field and rerun preflight
```

Representative output conflict:

```text
ERROR OUTPUT_UNRELATED_NONEMPTY_DIRECTORY
destination: /work/out
repair: choose another destination; M9 will not overwrite an unowned directory
```

## 5. Safety semantics

### Destination-state matrix

| State | Plain build | `--resume` | `--overwrite` |
|---|---|---|---|
| Destination absent | Proceed after preflight | Reject: nothing to resume | Proceed; no deletion needed |
| Empty directory | Proceed | Reject: no canonical state | Proceed without recursive deletion |
| Compatible incomplete canonical workspace | Refuse with resume guidance | Resume at resolver-selected stage | Allowed only after ownership and closed-member checks |
| Incompatible incomplete canonical workspace | Refuse | Reject with mismatched binding and restart guidance | Allowed only if it is positively identified as Tome-owned and contains no undeclared entries |
| Complete compatible workspace | Refuse as an explicit lifecycle choice | Validate and return `already_complete` without teacher work | Allow exact replacement after complete preflight |
| Complete invalid workspace | Refuse | Reject | Allow only when Tome ownership is proven and undeclared entries are absent |
| Existing canonical `.v4`, `.v4.tgz`, `.v3`, or `.v3.tgz` sibling | Refuse | Invoke the established publication recovery path where applicable | Remove only positively validated siblings included in the exact destination plan |
| Existing unrelated file | Refuse | Refuse | Refuse |
| Unrelated nonempty directory | Refuse | Refuse | Refuse |
| Symlink, root, home, repository root, special file, or unresolved path | Refuse | Refuse | Refuse |
| `--resume` and `--overwrite` together | Invocation/config error | — | — |

The destination plan is frozen during preflight and rechecked immediately before mutation. A changed inode/type/member inventory causes a fail-closed conflict.

Cleanup is limited to exact, resolved, positively owned targets. Unknown files inside a nominal workspace make overwrite fail rather than deleting them.

### Interruption and retries

- Catch `KeyboardInterrupt` at the CLI boundary, emit exit 130, and leave canonical failure/resume evidence untouched.
- Do not catch and conceal canonical transactional exceptions.
- Resume only through `resolve_native_path_b_resume`, M7 publication recovery, or v3 journal recovery.
- Never infer stage completion from directory names or isolated output files.
- A v3 directory may remain promoted when archive publication fails. Report directory success and archive failure separately; do not claim pair atomicity.
- Packaging retry is idempotent only when the existing output validates as the requested exact package. Otherwise it is a conflict.
- Broken pipe exits 141 without traceback.

### Preflight proof

Tests must prove that every preflight rejection happens before:

- progress/report writes;
- output deletion or directory creation;
- `create_backend`;
- teacher/model/tokenizer loading;
- accelerator allocation;
- score pass;
- selected pass;
- M7/v3 publication.

## 6. Human/machine output contract

### Stream ownership

- Human final summaries: stdout.
- Progress, warnings, deprecations, and actionable errors: stderr.
- `--json`: exactly one JSON document plus final newline on stdout.
- Production progress is redirected to stderr and cannot corrupt JSON.
- `--quiet`: suppresses progress and nonessential warnings, never errors or final destinations.
- Non-TTY and `NO_COLOR` disable color. `--no-color` forces plain text.
- `--debug` adds exception type, chained cause, and traceback on stderr.
- Default user errors never print tracebacks.

### `radjax_tome_cli_result_v1`

Every public command returns this stable top-level shape:

```json
{
  "schema_version": "radjax_tome_cli_result_v1",
  "command": "build",
  "status": "pass",
  "exit_code": 0,
  "versions": {
    "radjax_tome": "0.1.0",
    "radjax_contract": "0.9.0",
    "radjax_contract_commit": "373e3d17060d4ce1c4a0db6065c9289da714bde7"
  },
  "config": {
    "schema_version": "radjax_tome_build_intent_v1",
    "selection_authority_hash": "sha256:..."
  },
  "artifact": {
    "input": null,
    "workspace": "/work/out",
    "directory": "/work/out.v4",
    "archive": "/work/out.v4.tgz",
    "format": "m7_v4",
    "profile": "full_debug_provenance",
    "semantic_identity": "sha256:...",
    "authority_identity": null,
    "policy_identity": null,
    "raw_integrity_status": "pass"
  },
  "stages": [],
  "reports": {},
  "timing": {},
  "warnings": [],
  "error": null,
  "receipt_path": "/work/out/m9_cli_receipt.json"
}
```

Unavailable historical fields are `null` and accompanied by limitation codes. They are never inferred.

Warnings and errors use closed records:

```json
{
  "code": "OUTPUT_CONFLICT",
  "message": "destination contains unrelated entries",
  "phase": "preflight",
  "location": "/work/out",
  "repair": "choose an empty destination"
}
```

### Exit codes

| Code | Meaning |
|---:|---|
| 0 | Success, including valid result with nonfatal warnings |
| 2 | Invalid invocation or canonical configuration |
| 3 | Unsupported/incompatible Contract, schema, profile, or artifact |
| 4 | Validation, governed-comparison, or attestation failure |
| 5 | Output conflict or unsafe overwrite/resume |
| 6 | Required runtime/backend/dependency unavailable |
| 7 | Production or packaging execution failed after preflight |
| 70 | Unexpected internal failure |
| 130 | User interruption |
| 141 | Broken pipe |

Contract issue codes, validation phases, stage failures, blockers, and remediation remain visible inside the structured error rather than being flattened into generic strings.

## 7. Exact file-level change plan

| Path | Change |
|---|---|
| `pyproject.toml` | Reconcile Contract pin to `373e3d1`; retain argparse and existing console script; no new CLI dependency |
| `bible.md` | Append one entry in every commit; never rewrite prior entries |
| `src/radjax_tome/__main__.py` | Add `python -m radjax_tome` delegation |
| `src/radjax_tome/cli/main.py` | Replace Hydra root with thin public parser, global output policy, and hidden compatibility router |
| `src/radjax_tome/cli/mainline.py` | Add public command composition and handlers |
| `src/radjax_tome/cli/models.py` | Add CLI result, warning, error, and exit-code models |
| `src/radjax_tome/cli/rendering.py` | Add human/JSON rendering, stream ownership, color, quiet/debug, signal and broken-pipe handling |
| `src/radjax_tome/cli/research.py` | Move the existing parser/handlers intact except program/help labelling; production modules do not import it |
| `src/radjax_tome/builder/config.py` | Set accepted presets to batch 1; retain all M5 types and normalization |
| `src/radjax_tome/builder/config_io.py` | Add strict closed JSON/YAML loading and config-relative path resolution |
| `src/radjax_tome/builder/production_stages/preflight.py` | Extract pure assessment and exact destination plan; defer all mutation until pass |
| `src/radjax_tome/builder/production.py` | Consume the shared assessment before progress/mutation; expose canonical report facts to the CLI without stage duplication |
| `src/radjax_tome/builder/__init__.py` | Export only the deliberate config/preflight interfaces |
| `src/radjax_tome/tome/artifact_dispatch.py` | Add closed artifact-kind dispatch, unified validation, bounded inspection, and incomplete-workspace status |
| `src/radjax_tome/tome/packaging.py` | Add read-only package destination planning and prevent overwrite of unrelated/unowned paths |
| `src/radjax_tome/tome/__init__.py` | Export the new validated artifact dispatch interfaces |
| `tests/test_m9_authority_prerequisite.py` | Contract pin, buffer-native import, batch-1 preset tests |
| `tests/test_m9_config_io.py` | JSON/YAML equivalence, duplicate/unknown/missing fields, path resolution |
| `tests/test_m9_preflight.py` | Full destination matrix and zero-side-effect spies |
| `tests/test_m9_mainline_cli.py` | Help, routing, build, legacy shims, version and module invocation |
| `tests/test_m9_artifact_dispatch.py` | v3, M7 v4, package, historical, corrupt, incompatible, incomplete and streaming cases |
| `tests/test_m9_output_contract.py` | JSON parsing, stdout/stderr, errors, exits, cancellation, TTY/color and broken pipe |
| `tests/test_public_cli_happy_path.py` | Replace the research-era advertised path with the canonical-config CPU smoke |
| `tests/test_m3c_canonical_config_boundary.py` | Separate mainline command assertions from retained research parser assertions |
| `tests/test_tome_v3_verification_cli.py` | Route all three modes through public `validate` |
| `tests/fixtures/m9_cli_help/*.txt` | Durable root/build/validate/inspect/package/doctor help snapshots |
| `tests/fixtures/m9_build_intent/*.json` and `*.yaml` | Equivalent valid and adversarial canonical configuration fixtures |
| `README.md` | Minimal install and happy path |
| `docs/CLI_GUIDE.md` | Mainline reference plus complete compatibility/research classification |
| `docs/examples/m9_tome_build_intent.yaml` | Complete canonical M5 example with explicit placeholder paths |
| `docs/hydra_disposition.json` | Reclassify every old top-level command without deleting history |
| `docs/M9_CLOSURE_REPORT.md` | Final exit-criterion evidence and nonclaims |
| `evidence/m9_mainline_cli/closure.json` | Machine-readable closure receipt |
| `evidence/m9_mainline_cli/SHA256SUMS` | Digest closure evidence and help snapshots |

`builder/delivery/simple_compact_body.py` must not be replaced with the truncated compatibility shim from the obsolete M9 branch. The accepted M8 implementation remains intact.

No Contract, Student, Golden fixture, M8 evidence, corpus policy, model policy, or selected-pass implementation file is expected to change.

## 8. Compatibility plan

### One public path

Top-level help lists only:

```text
build
validate
inspect
package
doctor
research
```

`build` always means canonical M5 configuration driving the M4 state machine.

### Hidden compatibility routing

Before parsing the public surface, `main()` recognizes retained old command names:

- `production-build`
- corridor C2–C5 commands
- `verify-artifact`
- `package-artifact`
- `validate-package`
- `pack`, `unpack`, `parity`, `plan`
- multi-GPU, Golden, corpus, model, capability and M8 utilities

It emits a deprecation message and calls `cli.research.main(argv)`.

For name collisions:

- `build --config …` is mainline.
- Old `build` flag forms route to research with a warning.
- `validate --path …` and `inspect --path …` route to their compatibility handlers.
- Old doctor-specific backend flags route to research doctor.

These compatibility shims are documented as temporary and are absent from normal help. They invoke existing production APIs; they do not create a third implementation.

Scripts remain unchanged and retain their current evidence-reproduction behavior. No research history is deleted.

The existing remote M9 branch and its closure receipt are not accepted evidence because they lack M8 ancestry and required safety/output behavior. No commit from it is cherry-picked wholesale.

## 9. Test matrix

| Risk/criterion | New or reused proof |
|---|---|
| Happy path visible from help | Help snapshots and `test_m9_mainline_cli.py` |
| Help imports no Torch/Transformers/JAX and initializes no accelerator | Subprocess import audit |
| Console and `python -m radjax_tome` agree | Clean-install/module tests |
| Complete M5 JSON and YAML normalize identically | `test_m9_config_io.py` |
| Duplicate keys and ambiguous YAML rejected | Config adversarial fixtures |
| CLI and direct state-machine invocation share normalized config and selection hash | M3C plus M9 routing test |
| Equivalent config with different output destinations produces equal semantic identity | Two small deterministic CPU/smoke-tokenizer builds |
| Batch 1 remains canonical | Preset tests and fixed projection comparison |
| Invalid input fails before any mutation or expensive call | Preflight spies on model/backend/stages/publisher and filesystem snapshot |
| Resume uses canonical recorded state | M4 resume tests plus M9 destination matrix |
| Unsafe overwrite preserves unrelated content | Symlink/file/nonempty/unknown-member adversarial tests |
| Interrupted run retains resume/failure evidence | Existing streaming tests plus CLI interruption test |
| v3 standard/governed/external remain distinct | Existing Contract verification CLI fixtures routed through `validate` |
| Coherent replacement passes standard but fails governed comparison | Existing v3 fixture mutation case |
| Internal evidence cannot satisfy external attestation | Existing Contract case through public CLI |
| M7 v4 directory/archive validation streams | `test_m7_corrective_streaming_reader.py`, memory spies |
| Corrupt shard is rejected before row yield | Existing M7/Contract tests plus dispatcher assertion |
| Student/full-debug and v5/v6 package profiles validate | Packaging profile and B4 fixture tests |
| `.rtome` accepted for validation/inspection but not emitted by public package | Bundle/v3 fixture tests |
| Inspect is bounded and avoids payload-body enumeration | Large fake index plus patched body opener |
| JSON output remains one valid document | Output contract tests |
| Progress never enters JSON stdout | Captured production progress test |
| Exit codes and remediation stable | Parameterized CLI failure matrix |
| Legacy commands still route without appearing in help | Compatibility routing tests |
| No alternate archive writer | Monkeypatch established packagers and assert exact calls |
| Golden projection unchanged | Existing cheap Golden validation/characterization; no regeneration |
| M4–M8 regressions | Existing M4, M5, M6, M7 and M8 focused suites |

### Focused validation commands

```bash
python -m pytest -q \
  tests/test_m9_authority_prerequisite.py \
  tests/test_m9_config_io.py \
  tests/test_m9_preflight.py \
  tests/test_m9_mainline_cli.py \
  tests/test_m9_artifact_dispatch.py \
  tests/test_m9_output_contract.py \
  tests/test_m3c_canonical_config_boundary.py \
  tests/test_m4_live_canonical_execution.py \
  tests/test_native_path_b_resume.py \
  tests/test_tome_v3_verification_cli.py
```

Regression tranche:

```bash
python -m pytest -q \
  tests/test_m4b_production_stage_integration.py \
  tests/test_m4c_resume_assembly.py \
  tests/test_m5c_configuration_normalization.py \
  tests/test_m6a_production_boundary_policy.py \
  tests/test_m7_corrective_streaming_reader.py \
  tests/test_m7_memory_streaming.py \
  tests/test_m7_native_v4_publication_transaction.py \
  tests/test_m7_v4_staging_resume.py \
  tests/test_m8a_selected_pass_measurement.py \
  tests/test_m8g_canonical_modes.py \
  tests/test_production_build.py \
  tests/test_tome_packaging_profiles.py \
  tests/test_tome_artifact_v3_publication.py
```

Final gates:

```bash
python -m pytest -q -p no:cacheprovider
python -m ruff check .
python -m ruff format --check .
python -m compileall src scripts tests
git diff --check
python -m build
```

Clean-install smoke must use a disposable Python 3.12 environment, install the built wheel with its declared dependencies, prove Contract resolves to exact commit `373e3d1`, and run:

```bash
radjax-tome --version
radjax-tome --help
python -m radjax_tome --help
radjax-tome doctor
radjax-tome build --config SMALL_CONFIG
radjax-tome validate OUTPUT_ARCHIVE
radjax-tome inspect OUTPUT_ARCHIVE
radjax-tome package WORKSPACE --output STUDENT.tgz --profile student
radjax-tome validate STUDENT.tgz
```

The bounded smoke uses the existing deterministic four/five-example CPU `smoke_tokenizer` path. No GPU, Golden 1K, 10K, or 100K run is required.

## 10. Implementation sequence

### Prerequisite integration gate

1. Verify remote heads.
2. Fast-forward `main` from `6a6c653` to `88d6f41`.
3. Push and verify.
4. Create a fresh worktree and branch `m9/opinionated-mainline-cli-reconciled`.
5. Confirm only that worktree will be modified.

Stop if main has advanced, M8 moved, or fast-forward is impossible.

### Commit 1 — Authority reconciliation

- Pin Contract `373e3d1`.
- Set all production presets’ selected rerun to batch 1.
- Add pin/API/preset tests.
- Append `bible.md`.

Proposed commit: `M9 prerequisite: align accepted M8 authorities`

Checkpoint: clean install can import the M8 buffer-native boundary and focused M5/M8 tests pass.

### Commit 2 — Canonical config and pure preflight

- Add strict config I/O.
- Extract shared pure production preflight.
- Add safe destination planning.
- Integrate the assessment before production mutation.
- Add full preflight/config tests.
- Append `bible.md`.

Proposed commit: `M9: add canonical config loading and safe preflight`

Checkpoint: all negative cases prove zero filesystem/model/stage side effects.

### Commit 3 — Public CLI root and output contract

- Move existing Hydra parser to `cli.research`.
- Add the six-command root.
- Add result/error/rendering models.
- Add `python -m radjax_tome`.
- Add legacy compatibility routing.
- Add help/output/exit tests.
- Append `bible.md`.

Proposed commit: `M9: establish the opinionated mainline CLI`

Checkpoint: help shows the happy path and imports no heavy runtime.

### Commit 4 — Validation, inspection, and packaging lifecycle

- Add artifact dispatch and bounded inspection.
- Fold verification modes into `validate`.
- Add safe packaging destination planning.
- Emit build/package receipts.
- Add artifact/package/streaming tests.
- Append `bible.md`.

Proposed commit: `M9: unify Tome lifecycle commands`

Checkpoint: CPU smoke builds, validates, inspects, packages, and round-trips through only canonical APIs.

### Commit 5 — Compatibility and documentation

- Update README, CLI guide, examples, help snapshots, and Hydra classification.
- Verify every old entry point’s treatment.
- Append `bible.md`.

Proposed commit: `M9: document the public CLI and contain the Hydra`

### Independent review

Push the implementation commits, then assign one fresh reviewer a read-only scope limited to:

1. Canonical M5/M4 routing.
2. One obvious public path.
3. Preflight/overwrite/resume safety.
4. Contract-correct validate/inspect/package behavior.
5. Machine result and exit-code usability.
6. No second engine or M10–M12 expansion.

Read-only review forbids repository changes but permits disposable builds, temporary environments, and independent calculations. The reviewer returns one finite disposition and concrete findings.

If blocking findings exist, make one bounded correction commit, rerun focused and full gates, and request one targeted recheck of only those findings. No open-ended audit loop.

### Commit 6 — Closure evidence

After review approval:

- Write the closure report and machine receipt.
- Generate checksums.
- Record exact commands/results and review disposition.
- Append `bible.md`.
- Commit and push.
- Require a clean M9 worktree.

Proposed commit: `M9: close the opinionated mainline CLI`

Do not merge the M9 branch into main without the normal integration authorization.

## 11. Closure and evidence plan

`docs/M9_CLOSURE_REPORT.md` and `evidence/m9_mainline_cli/closure.json` must bind:

- Original main: `6a6c653`.
- Accepted M8 base: `88d6f41`.
- Reconciled main commit after fast-forward.
- M9 branch and final commit.
- Contract package 0.9.0 and exact commit `373e3d1`.
- Tome package version.
- Public executable and six-command surface.
- Exact config → state-machine → publication call graph.
- Batch-1 and fixed-projection preservation.
- Help snapshot digests.
- Destination/resume/overwrite policy.
- Exit-code and result-schema versions.
- Human and JSON examples.
- Legacy-command classification.
- Focused/full test results.
- Ruff, format, compileall, diff and build results.
- Clean-install environment and wheel hash.
- CPU smoke input/config digest.
- Producer workspace, v4/v3 directory/archive, and student package identities.
- Validation, inspection, packaging, and round-trip results.
- Independent review disposition and any correction commit.
- Known limitations and nonclaims.
- Branch push and clean-worktree proof.

Exit-criterion mapping:

| Roadmap criterion | Required evidence |
|---|---|
| Fresh user identifies happy path from help | Root and build help snapshots plus clean-install invocation |
| CLI drives M4 through M5 config | Routing spies and M3C/M4 integration tests |
| CLI does not reconstruct domain state | Dependency-boundary inspection and reviewer finding |
| Errors identify invariant and repair | Error fixture matrix |
| Equivalent config produces equivalent semantic identity | JSON/YAML/direct-API dual-build CPU evidence |

Closure nonclaims must explicitly state:

- M9 does not change semantic identity, Contract behavior, corpus/model/selection policy, M8 performance, Golden evidence, Student, or M10–M14.
- No GPU/TPU inference or performance claim was made.
- Research utilities remain available but are not the public paved road.
- M9 closure does not close the Tome project.

## 12. Risks and blockers

| Rank | Risk | Detection | Mitigation/stop |
|---:|---|---|---|
| 1 | Main does not contain accepted M8 | Ancestry check | Mandatory fast-forward prerequisite; stop if remote changed |
| 2 | M8 pin cannot provide imported buffer APIs | Clean install/import | Pin exact accepted `373e3d1` before CLI work |
| 3 | Presets contradict batch-1 M8 closure | Config tests | Correct in prerequisite commit; no new benchmark |
| 4 | Existing M9 branch is accidentally treated as accepted | Ancestry and closure receipt | New branch from `88d6f41`; no wholesale cherry-pick |
| 5 | Preflight mutates before failure | Filesystem snapshots and spies | Shared pure assessment before progress/output action |
| 6 | Overwrite deletes unrelated data | Adversarial destination matrix | Positive ownership, closed members, exact target, recheck before mutation |
| 7 | Config loader becomes a second defaults system | JSON/YAML/direct normalization comparison | Complete closed M5 document; no omitted-field defaults |
| 8 | Package profile/transport is silently ignored | Config preflight test | Reject unsupported build values and direct users to `package` |
| 9 | Artifact resolver misclassifies corruption | Cover/version mutation corpus | Closed dispatch and unsupported-version failures |
| 10 | Validation/inspection defeats M7 streaming | Body-open/read-count instrumentation | Contract streaming APIs and metadata/index-only inspection |
| 11 | Progress corrupts machine output | Captured stdout test | JSON only on stdout; progress to stderr |
| 12 | Legacy shims remain co-equal | Help snapshot and import graph | Hide from help, warn, delegate to research parser |
| 13 | Receipt changes artifact semantics | Before/after semantic roots | Write CLI receipt only after publication in nonsemantic producer workspace |
| 14 | Stale installed Contract contaminates tests | `direct_url.json` and import-path receipt | Disposable exact-pin environment |
| 15 | Work overlaps dirty P6/audit worktrees | Worktree inventory | Fresh isolated worktree; preserve all existing files |
| 16 | Scope expands into M10/M11/M12 | Focused review | Reject corpus redesign, TUI, and documentation overhaul |

True blockers:

- Remote main or M8 authority differs at execution time.
- Accepted M8 cannot be integrated by fast-forward.
- Contract `373e3d1` does not pass clean-install API proof.
- A safe output action cannot be expressed through canonical production ownership without redesigning M4 state.
- Canonical validators cannot distinguish a promised artifact form without Contract changes.
- Correctness requires Golden regeneration, Contract modification, or selected-pass changes.

Everything else above is bounded M9 implementation work.

## 13. Final recommendation

M9 is one-shot implementable after one mandatory, bounded repository prerequisite: fast-forward authoritative main to accepted M8 closure `88d6f41`.

After Rey approves:

1. Verify remote heads.
2. Fast-forward and push main to `88d6f41`.
3. Create `m9/opinionated-mainline-cli-reconciled`.
4. Execute commits 1–5 continuously.
5. Run the complete local and clean-install gates.
6. Obtain the one focused independent review.
7. Correct only demonstrated findings.
8. Commit and push closure evidence.
9. Stop with a clean branch for normal integration review.

Do not use the existing `8a2dc279` closure as accepted M9 evidence, do not modify Contract or Student, and do not begin M10.

**M9_PLAN_READY_WITH_BOUNDED_PREREQUISITE**

