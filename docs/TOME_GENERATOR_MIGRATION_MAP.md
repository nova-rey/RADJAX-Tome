# Tome Generator Migration Map

## Summary

Spec 3 is blocked. The extraction audit found a narrow TeacherTextbook migration, not a complete Tome Generator extraction.

| Metric | Count |
| --- | ---: |
| intentionally_omitted | 32 |
| migrated | 46 |
| missing | 36 |
| missing_tests | 9 |
| mixed_producer_consumer_files | 23 |
| needs_human_review | 0 |
| new_equivalent_tests | 80 |
| partial | 258 |
| producer_cli_files | 43 |
| producer_core_files | 18 |
| producer_relevant_old_files | 372 |
| producer_relevant_old_tests | 89 |
| total_old_candidate_files | 384 |

Note: the current audit still treats quarantine `.txt` references as path
matches, so `partial`, `duplicate_or_merged`, and `new_equivalent_tests` include
retained quarantine evidence. Active Spec 2.9 promotion status is recorded in
`docs/TOME_GENERATOR_QUARANTINE_SURGERY_LEDGER.json`, which processed 273
quarantine-backed paths: 3 `promoted`, 78 `split_promoted`, 5
`kept_quarantined`, 32 `belongs_student`, 2 `belongs_contract`, 1 `deprecated`,
151 `deferred`, and 1 `waived`.

## Short-Term Roadmap

- Spec 2.5 - Extraction completeness audit. DONE.
- Spec 2.6 - Audit triage and producer migration map. DONE.
- Spec 2.7 - Migrate highest-priority producer schemas/stores/validators/tests. DONE.
- Spec 2.8 - Bulk producer migration with quarantine. DONE.
- Spec 2.9 - Surgical split of quarantined mixed producer/student files. DONE.
- Spec 2.10 - Audit closure, A/B expansion, waivers, and Spec 3 gate check. NEXT.
- Spec 3 - Contract-valid Tome emission with cover_page.json, only after the gate passes.

Previous micro-migration roadmap:

- Spec 2.8 - Migrate real-teacher/HF/corpus/source-identity producer paths.
- Spec 2.9 - Migrate behavioral fingerprint / corridor / exemplar producer artifact paths.
- Spec 2.10 - Re-run extraction audit and A/B parity; reduce blockers to zero or explicit waivers.
- Spec 3 - Only then implement Contract-valid Tome emission with cover_page.json.

## Bucket Definitions

- `must_migrate_tome_before_spec3`
- `migrate_tome_before_full_burn`
- `migrate_tome_deferred`
- `belongs_contract`
- `belongs_student`
- `mixed_requires_split`
- `historical_deprecated`
- `duplicate_or_merged`
- `waive_with_reason`
- `needs_human_review`

## Triage Summary Counts

| Metric | Count |
| --- | ---: |
| belongs_contract | 8 |
| belongs_student | 14 |
| duplicate_or_merged | 258 |
| migrate_tome_before_full_burn | 14 |

## High-Risk Blocker Summary

| Metric | Count |
| --- | ---: |
| belongs_contract | 3 |
| belongs_student | 3 |
| duplicate_or_merged | 46 |
| migrate_tome_before_full_burn | 14 |

## Missing Test Summary

| Metric | Count |
| --- | ---: |
| belongs_contract | 2 |
| belongs_student | 7 |

## Ordered Migration Chunks

### Spec 2.7 - Producer Schemas, Stores, Validators, and Core Tests

Migrate or waive producer schemas, target stores, validators, export/inspection CLIs, and core tests required before cover-page work.

Risk: `high`

Must include:
- teacher artifact schemas and manifests
- target-store and multishard producer helpers
- export/inspect/validate teacher target CLIs
- dense/top-k/cascaded producer tests

Acceptance criteria:
- Spec 2.5 audit has no unwaived producer-core Spec 3 blockers
- legacy A/B parity still passes

### Spec 2.8 - Corpus, Source Identity, Real-Teacher/HF Producer Paths

Restore producer-side corpus/source identity, HF/local-files-only, and real-teacher capture entry points needed by Tome generation.

Risk: `high`

Must include:
- prompt corpus loading and splitting
- source/example_id preservation
- HF teacher export and specimen smoke
- real-teacher producer capture boundaries

Acceptance criteria:
- source identity is covered by tests
- HF/local-files-only producer path is tested or explicitly waived

### Spec 2.9 - Behavioral Fingerprint, Corridor, Exemplar Producer Artifacts

Split and migrate producer-side behavioral fingerprint, exemplar, corridor, and capture-summary artifact paths.

Risk: `high`

Must include:
- fingerprint artifact schemas and validators
- exemplar target/exemplar artifact writers
- capture_summary.json style outputs
- stat bands and provenance metadata

Acceptance criteria:
- mixed producer/student files have explicit split destinations
- producer artifact tests are ported or waived

### Spec 2.10 - Audit Closure, A/B Expansion, and Explicit Waivers

Rerun extraction audit and A/B parity, close or explicitly waive every remaining blocker, and lock the Spec 3 entry gate.

Risk: `medium`

Must include:
- updated extraction audit
- updated migration map
- expanded A/B cases if migrated behavior requires them
- waiver register

Acceptance criteria:
- Spec 3 gate passes
- no untriaged high-risk producer blockers remain

## Spec 3 Gate

Passed: `False`

- The regenerated audit still reports 45 high-risk producer items as quarantined
  or duplicate/merged path evidence, but the Spec 2.9 surgery ledger records no
  remaining quarantine entry as a Spec 3 blocker.
- legacy A/B parity must still pass after migration waves
- extraction audit must be interpreted in Spec 2.10 with active promotion
  separated from retained quarantine evidence
- Contract-valid `cover_page.json` Tome emission remains unimplemented

## Open Questions / Human Review

- Which contract-bound schemas need RADJAX-Contract issues before Tome migration continues?

## High-Risk Path-Level Detail

The generated local `migration_map.json` contains the complete table. The current high-risk table is summarized here:

- `docs/RADLADS2_FLA_KVM_RESEARCH_INTAKE.md` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/docs/RADLADS2_FLA_KVM_RESEARCH_INTAKE.md.txt`
- `docs/TINY_OVERFIT_REHEARSAL.md` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/docs/TINY_OVERFIT_REHEARSAL.md.txt`
- `docs/VOCAB_CONTRACT.md` -> `belongs_contract`; blocks_spec3=`False`; destination=`vocab compatibility and contract schemas`
- `scripts/build_fingerprint_artifact.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/build_fingerprint_artifact.py.txt`
- `scripts/build_real_teacher_fingerprint_artifact.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/build_real_teacher_fingerprint_artifact.py.txt`
- `scripts/create_fake_targets.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/generate_multiscale_configs.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/inspect_fingerprint_exemplars.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/inspect_fingerprint_targets.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/plan_model_scale.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_adaptive_corridor_pass.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/run_adaptive_corridor_pass.py.txt`
- `scripts/run_corridor_aggressiveness_calibration.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/run_corridor_aggressiveness_calibration.py.txt`
- `scripts/run_corridor_measurement.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/run_corridor_measurement.py.txt`
- `scripts/run_distill_stage.py` -> `belongs_student`; blocks_spec3=`False`; destination=`student training or evaluation workflow`
- `scripts/run_exemplar_pass.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_export_smoke.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_fingerprint_arc2_report.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_fingerprint_baseline_comparison.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/run_fingerprint_baseline_comparison.py.txt`
- `scripts/run_fingerprint_held_out_evaluation.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_fingerprint_quality_per_byte_experiment.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/run_fingerprint_quality_per_byte_experiment.py.txt`
- `scripts/run_fingerprint_smoke.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_fingerprint_trained_baseline_comparison.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/run_fingerprint_trained_baseline_comparison.py.txt`
- `scripts/run_first_serious_burn.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/run_first_serious_burn.py.txt`
- `scripts/run_full_distillation_crossover.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/run_full_distillation_crossover.py.txt`
- `scripts/run_hf_teacher_specimen_smoke.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/run_hf_teacher_specimen_smoke.py.txt`
- `scripts/run_hf_teacher_specimen_swap_smoke.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/run_hf_teacher_specimen_swap_smoke.py.txt`
- `scripts/run_mini_eval_harness.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/run_mini_eval_harness.py.txt`
- `scripts/run_mode_plateau_controller_smoke.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_multiscale_shape_dry_run.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_quality_per_byte_experiment.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/run_quality_per_byte_experiment.py.txt`
- `scripts/run_real_student_fingerprint_forward_smoke.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_real_teacher_fingerprint_training_rehearsal.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/run_real_teacher_fingerprint_training_rehearsal.py.txt`
- `scripts/run_two_cycle_experiment.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/run_two_cycle_experiment.py.txt`
- `scripts/tpu_distill_smoke.py` -> `belongs_student`; blocks_spec3=`False`; destination=`student training or evaluation workflow`
- `scripts/train_student_smoke.py` -> `belongs_student`; blocks_spec3=`False`; destination=`student backend/runtime`
- `scripts/validate_local.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/validate_local.py.txt`
- `scripts/validate_student_artifact.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/scripts/validate_student_artifact.py.txt`
- `scripts/write_fingerprint_provenance.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `src/qrwkv_xla/artifacts/fingerprint.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`src/radjax_tome/fingerprint/artifacts.py`
- `src/qrwkv_xla/artifacts/fingerprint_exemplars.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/artifacts/fingerprint_exemplars.py.txt, src/radjax_tome/fingerprint/exemplars.py`
- `src/qrwkv_xla/artifacts/student_artifact.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/artifacts/student_artifact.py.txt`
- `src/qrwkv_xla/burn/first_serious_burn.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/burn/first_serious_burn.py.txt`
- `src/qrwkv_xla/contracts/compatibility.py` -> `belongs_contract`; blocks_spec3=`False`; destination=`shared artifact/schema validation`
- `src/qrwkv_xla/contracts/vocab.py` -> `belongs_contract`; blocks_spec3=`False`; destination=`vocab compatibility and contract schemas`
- `src/qrwkv_xla/distill/losses.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/distill/losses.py.txt`
- `src/qrwkv_xla/distill/target_dispatch.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/distill/target_dispatch.py.txt`
- `src/qrwkv_xla/eval/mini_eval.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/eval/mini_eval.py.txt`
- `src/qrwkv_xla/fingerprint/arc_report.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/fingerprint/arc_report.py.txt`
- `src/qrwkv_xla/fingerprint/baseline_comparison.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/fingerprint/baseline_comparison.py.txt`
- `src/qrwkv_xla/fingerprint/budgeted_artifact.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/fingerprint/budgeted_artifact.py.txt`
- `src/qrwkv_xla/fingerprint/capture.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/fingerprint/capture.py.txt`
- `src/qrwkv_xla/fingerprint/exemplar_pass.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/fingerprint/exemplar_pass.py.txt`
- `src/qrwkv_xla/fingerprint/held_out_evaluation.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/fingerprint/held_out_evaluation.py.txt`
- `src/qrwkv_xla/fingerprint/quality_per_byte.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/fingerprint/quality_per_byte.py.txt`
- `src/qrwkv_xla/fingerprint/radjax_crossover_backend.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/fingerprint/radjax_crossover_backend.py.txt`
- `src/qrwkv_xla/fingerprint/real_teacher.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/fingerprint/real_teacher.py.txt`
- `src/qrwkv_xla/fingerprint/trained_baseline.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/fingerprint/trained_baseline.py.txt`
- `src/qrwkv_xla/fingerprint/training_rehearsal.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/fingerprint/training_rehearsal.py.txt`
- `src/qrwkv_xla/readiness/big_burn.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/readiness/big_burn.py.txt`
- `src/qrwkv_xla/smoke/colab_tpu.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/smoke/colab_tpu.py.txt`
- `src/qrwkv_xla/targets/real_teacher_consumption.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/targets/real_teacher_consumption.py.txt`
- `src/qrwkv_xla/teacher_export/hf.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/teacher_export/hf.py.txt`
- `src/qrwkv_xla/training/fingerprint_exemplar_loss.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/training/fingerprint_exemplar_loss.py.txt`
- `src/qrwkv_xla/training/fingerprint_reports.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/training/fingerprint_reports.py.txt`
- `src/qrwkv_xla/training/fingerprint_smoke.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/training/fingerprint_smoke.py.txt`
- `src/qrwkv_xla/training/real_teacher_overfit.py` -> `duplicate_or_merged`; blocks_spec3=`False`; destination=`quarantine/qrwkv_xla/src/qrwkv_xla/training/real_teacher_overfit.py.txt`
