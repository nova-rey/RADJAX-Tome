# M9 Mandatory CI Reconciliation

This report is the durable disposition of the 20 failures observed at the
accepted M8 baseline and at the original M9 tip. The accepted baseline was
`88d6f418e0f39ab2ec61d9047f974f04657e5214`; the original M9 comparison tip was
`1d4910a8530284df7518160b22a63320f217097b`. The shared Python 3.12.14
environment was used for both runs. Baseline: 20 failed, 449 passed, 9
skipped. Original M9: 20 failed, 570 passed, 9 skipped. After reconciliation:
1,212 passed, 9 skipped.

| # | Failure | First differing behavior | M8 | M9 cause/correction | Disposition | Final test |
|---:|---|---|---|---|---|---|
| 1 | HF unsupported dynamic policy | HF admission listed a policy without implementation support | fail | capability table and admission now agree | PREEXISTING_ACCEPTED_BASE_FAILURE | `test_hf_torch_backend_contract.py` |
| 2 | backend builder dynamic payload | compatibility builder requested compact dynamic shape | fail | legacy builder explicitly requests padded mode | PREEXISTING_ACCEPTED_BASE_FAILURE | `test_backend_builder_integration.py` |
| 3 | backend builder mask | padded compatibility payload lacked expected mask shape | fail | compatibility mode retains its established payload contract | PREEXISTING_ACCEPTED_BASE_FAILURE | `test_backend_builder_integration.py` |
| 4 | C6 v4 Contract pin | test asserted retired historical commit | fail | test follows exact current pin `373e3d17060d4ce1c4a0db6065c9289da714bde7` | ACCEPTED_AUTHORITY_ALIGNMENT_REQUIRES_TEST_UPDATE | `test_c6_student_consumption_v4_contract_pin.py` |
| 5 | historical v4 validation | v4 selected payload omitted `top_selection_mask` | fail | materializer supplies the deterministic mask required by validation | M9_REGRESSION | `test_c6_student_consumption_v4_contract_pin.py` |
| 6 | CPU compact payload | legacy helper emitted compact shape | fail | characterization helper explicitly requests legacy padded mode | PREEXISTING_ACCEPTED_BASE_FAILURE | `test_cpu_reference_backend.py` |
| 7 | CPU mask compatibility | mask expectation followed compact shape | fail | legacy helper preserves padded compatibility semantics | PREEXISTING_ACCEPTED_BASE_FAILURE | `test_cpu_reference_backend.py` |
| 8 | CPU dynamic selection | selected rows were compared as compact output | fail | test remains explicit about legacy representation | PREEXISTING_ACCEPTED_BASE_FAILURE | `test_cpu_reference_backend.py` |
| 9 | GPU corridor source shape | corridor source was compact at an internal reducer boundary | fail | corridor source uses padded internal reduction; public result remains compact | M9_REGRESSION | `test_gpu_torch_corridor_exemplar_reducer.py` |
| 10 | GPU corridor mask | internal mask was incompatible with the reducer | fail | same internal padded compatibility boundary | M9_REGRESSION | `test_gpu_torch_corridor_exemplar_reducer.py` |
| 11 | HF local-model error text | Transformers runtime changed the load error wording | fail | stable diagnostic includes the declared local-model guidance | ENVIRONMENT_OR_DEPENDENCY_DRIFT | `test_hf_torch_backend_contract.py` |
| 12 | long-tail default board | test expected the side board as primary | fail | current authority selects the primary board | ACCEPTED_AUTHORITY_ALIGNMENT_REQUIRES_TEST_UPDATE | `test_long_tail_diagnostics.py` |
| 13 | long-tail board identity | default board naming was stale | fail | characterization follows current primary-board authority | ACCEPTED_AUTHORITY_ALIGNMENT_REQUIRES_TEST_UPDATE | `test_long_tail_diagnostics.py` |
| 14 | M3A public parser inventory | public help expected retained engineering commands | fail | six-command public surface is tested separately from compatibility routing | OBSOLETE_LEGACY_CLI_EXPECTATION | `test_m3a_import_isolation.py` |
| 15 | M3C one-pass record identity | payload lacked `_record_index` | fail | one-pass payloads receive deterministic record indices | M9_REGRESSION | `test_m3c_canonical_config_boundary.py` |
| 16 | M3C selected linkage | publication retained only selected summaries | fail | publication carries complete selected payloads for linkage validation | M9_REGRESSION | `test_selected_exemplar_adversarial_linkage.py` |
| 17 | production CLI config shape | test expected the retired nested parser model | fail | test uses canonical flat `ProductionBuildConfig` fields | OBSOLETE_LEGACY_CLI_EXPECTATION | `test_public_cli_happy_path.py` |
| 18 | M4 resume terminal status | resumed finalization was expected to report fresh-build `pass` | fail | test asserts intentional `resumed_finalization` status and flags | ACCEPTED_AUTHORITY_ALIGNMENT_REQUIRES_TEST_UPDATE | `test_m4_live_canonical_execution.py` |
| 19 | M5 field inventory | characterization expected the retired field count | fail | current canonical inventory is recorded without changing Contract | ACCEPTED_AUTHORITY_ALIGNMENT_REQUIRES_TEST_UPDATE | `test_m5a_contract_characterization.py` |
| 20 | M5 authority hash/config | expected hash used obsolete normalization fields | fail | canonical full-width controls and batch-one preset are included | ACCEPTED_AUTHORITY_ALIGNMENT_REQUIRES_TEST_UPDATE | `test_m5b_canonical_config_contract.py`, `test_m5c_configuration_normalization.py` |

The compact dynamic payload and mask fixes are confined to the canonical or
explicit compatibility boundaries. Historical Contract-v4 validation remains
available through its historical path. The preserved P6 receipt was not
rewritten: its historical count remains 60, while the current generator is
qualified against its observed 54 distinct examples from 64 coordinates.

Ruff was 216 violations at accepted M8 and 205 at the original M9 tip. The
continuation corrects the inherited violations, with no blanket ignores and no
historical evidence rewrite; `ruff check .` and `ruff format --check .` pass.

The original dirty worktree remains preserved at
`/home/nyx/radjax-tome` with its `bible.md` and
`src/radjax_tome/backends/hf_torch.py` changes untouched. Those changes were
not copied because they describe a pre-existing compatibility correction rather
than an M9 regression. This continuation is the clean, reviewable repair line.
