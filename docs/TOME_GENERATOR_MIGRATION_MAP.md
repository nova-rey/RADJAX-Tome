# Tome Generator Migration Map

## Summary

Spec 3 is blocked. The extraction audit found a narrow TeacherTextbook migration, not a complete Tome Generator extraction.

| Metric | Count |
| --- | ---: |
| intentionally_omitted | 32 |
| migrated | 37 |
| missing | 301 |
| missing_tests | 82 |
| mixed_producer_consumer_files | 23 |
| needs_human_review | 0 |
| new_equivalent_tests | 7 |
| partial | 2 |
| producer_cli_files | 43 |
| producer_core_files | 18 |
| producer_relevant_old_files | 372 |
| producer_relevant_old_tests | 89 |
| total_old_candidate_files | 384 |

## Short-Term Roadmap

- Spec 2.5 - Extraction completeness audit. DONE.
- Spec 2.6 - Audit triage and producer migration map. DONE.
- Spec 2.7 - Migrate highest-priority producer schemas/stores/validators/tests. THIS SPEC.
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
| duplicate_or_merged | 2 |
| migrate_tome_before_full_burn | 94 |
| migrate_tome_deferred | 97 |
| mixed_requires_split | 26 |
| must_migrate_tome_before_spec3 | 42 |
| needs_human_review | 20 |

## High-Risk Blocker Summary

| Metric | Count |
| --- | ---: |
| belongs_contract | 3 |
| belongs_student | 3 |
| migrate_tome_before_full_burn | 41 |
| mixed_requires_split | 22 |
| must_migrate_tome_before_spec3 | 3 |

## Missing Test Summary

| Metric | Count |
| --- | ---: |
| belongs_contract | 2 |
| belongs_student | 7 |
| defer | 26 |
| must_port_before_spec3 | 16 |
| must_port_with_associated_feature | 31 |

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

- 3 high-risk Tome producer items must migrate before Spec 3
- 22 high-risk mixed producer/student items need splits
- 16 missing producer tests must port before Spec 3
- legacy A/B parity must still pass after migration waves
- extraction audit must rerun with no untriaged producer-core blockers

## Open Questions / Human Review

- Which contract-bound schemas need RADJAX-Contract issues before Tome migration continues?
- For mixed fingerprint/burn files, which producer outputs belong in Tome and which execution paths belong in Student?
- Which deferred docs/fixtures are still useful as regression evidence?

## High-Risk Path-Level Detail

The generated local `migration_map.json` contains the complete table. The current high-risk table is summarized here:

- `docs/RADLADS2_FLA_KVM_RESEARCH_INTAKE.md` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split producer artifact logic from consumer/runtime logic`
- `docs/TINY_OVERFIT_REHEARSAL.md` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split producer artifact logic from consumer/runtime logic`
- `docs/VOCAB_CONTRACT.md` -> `belongs_contract`; blocks_spec3=`False`; destination=`vocab compatibility and contract schemas`
- `scripts/build_fingerprint_artifact.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/build_real_teacher_fingerprint_artifact.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/create_fake_targets.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/generate_multiscale_configs.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/inspect_fingerprint_artifact.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/inspect_fingerprint_exemplars.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/inspect_fingerprint_targets.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/plan_model_scale.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/resolve_qwen_policy.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_adaptive_corridor_pass.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_corridor_aggressiveness_calibration.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_corridor_measurement.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_distill_stage.py` -> `belongs_student`; blocks_spec3=`False`; destination=`student training or evaluation workflow`
- `scripts/run_exemplar_pass.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_export_smoke.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_fingerprint_arc2_report.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_fingerprint_baseline_comparison.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_fingerprint_held_out_evaluation.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_fingerprint_quality_per_byte_experiment.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_fingerprint_smoke.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_fingerprint_trained_baseline_comparison.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_first_serious_burn.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split burn/distillation orchestration into producer inputs vs student execution`
- `scripts/run_full_distillation_crossover.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split burn/distillation orchestration into producer inputs vs student execution`
- `scripts/run_hf_teacher_specimen_smoke.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_hf_teacher_specimen_swap_smoke.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_mini_eval_harness.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_mode_plateau_controller_smoke.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_multiscale_shape_dry_run.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_quality_per_byte_experiment.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_real_student_fingerprint_forward_smoke.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_real_teacher_fingerprint_training_rehearsal.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/run_two_cycle_experiment.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/tpu_distill_smoke.py` -> `belongs_student`; blocks_spec3=`False`; destination=`student training or evaluation workflow`
- `scripts/train_student_smoke.py` -> `belongs_student`; blocks_spec3=`False`; destination=`student backend/runtime`
- `scripts/validate_fingerprint_artifact.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/validate_local.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `scripts/validate_student_artifact.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split producer artifact logic from consumer/runtime logic`
- `scripts/write_fingerprint_provenance.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer CLI surface`
- `src/qrwkv_xla/artifacts/fingerprint.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer fingerprint artifact surface`
- `src/qrwkv_xla/artifacts/fingerprint_exemplars.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer fingerprint artifact surface`
- `src/qrwkv_xla/artifacts/fingerprint_summary.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer fingerprint artifact surface`
- `src/qrwkv_xla/artifacts/student_artifact.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split producer artifact logic from consumer/runtime logic`
- `src/qrwkv_xla/burn/first_serious_burn.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split burn/distillation orchestration into producer inputs vs student execution`
- `src/qrwkv_xla/contracts/compatibility.py` -> `belongs_contract`; blocks_spec3=`False`; destination=`shared artifact/schema validation`
- `src/qrwkv_xla/contracts/vocab.py` -> `belongs_contract`; blocks_spec3=`False`; destination=`vocab compatibility and contract schemas`
- `src/qrwkv_xla/distill/losses.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`RADJAX-Tome producer surface`
- `src/qrwkv_xla/distill/target_dispatch.py` -> `must_migrate_tome_before_spec3`; blocks_spec3=`True`; destination=`teacher target store/export/validation surface`
- `src/qrwkv_xla/eval/mini_eval.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`RADJAX-Tome producer surface`
- `src/qrwkv_xla/fingerprint/arc_report.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split fingerprint producer artifacts from student training/eval`
- `src/qrwkv_xla/fingerprint/baseline_comparison.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split fingerprint producer artifacts from student training/eval`
- `src/qrwkv_xla/fingerprint/budgeted_artifact.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer fingerprint artifact surface`
- `src/qrwkv_xla/fingerprint/capture.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`producer fingerprint artifact surface`
- `src/qrwkv_xla/fingerprint/exemplar_pass.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split fingerprint producer artifacts from student training/eval`
- `src/qrwkv_xla/fingerprint/held_out_evaluation.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split fingerprint producer artifacts from student training/eval`
- `src/qrwkv_xla/fingerprint/provenance.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split fingerprint producer artifacts from student training/eval`
- `src/qrwkv_xla/fingerprint/quality_per_byte.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split fingerprint producer artifacts from student training/eval`
- `src/qrwkv_xla/fingerprint/radjax_crossover_backend.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split fingerprint producer artifacts from student training/eval`
- `src/qrwkv_xla/fingerprint/real_teacher.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split fingerprint producer artifacts from student training/eval`
- `src/qrwkv_xla/fingerprint/trained_baseline.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split fingerprint producer artifacts from student training/eval`
- `src/qrwkv_xla/fingerprint/training_rehearsal.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split fingerprint producer artifacts from student training/eval`
- `src/qrwkv_xla/readiness/big_burn.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split burn/distillation orchestration into producer inputs vs student execution`
- `src/qrwkv_xla/smoke/colab_tpu.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split producer artifact logic from consumer/runtime logic`
- `src/qrwkv_xla/targets/real_teacher_consumption.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`teacher target store/export/validation surface`
- `src/qrwkv_xla/teacher_export/hf.py` -> `must_migrate_tome_before_spec3`; blocks_spec3=`True`; destination=`teacher backend/export producer surface`
- `src/qrwkv_xla/teachers/hf_specimen_smoke.py` -> `must_migrate_tome_before_spec3`; blocks_spec3=`True`; destination=`teacher backend/export producer surface`
- `src/qrwkv_xla/training/fingerprint_exemplar_loss.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split fingerprint producer artifacts from student training/eval`
- `src/qrwkv_xla/training/fingerprint_reports.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split fingerprint producer artifacts from student training/eval`
- `src/qrwkv_xla/training/fingerprint_smoke.py` -> `mixed_requires_split`; blocks_spec3=`True`; destination=`split fingerprint producer artifacts from student training/eval`
- `src/qrwkv_xla/training/real_teacher_overfit.py` -> `migrate_tome_before_full_burn`; blocks_spec3=`False`; destination=`RADJAX-Tome producer surface`
