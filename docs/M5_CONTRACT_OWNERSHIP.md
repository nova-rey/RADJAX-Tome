# M5 Contract Ownership and Characterization Ledger

## Baseline and scope

This ledger is the M5A pre-change inventory for `main` at
`69ee0ab5be374967cd839e95ea118c0cb570f46f`. The only pre-existing worktree
entry is an unrelated untracked `.DS_Store`; M5 must preserve it.

The earlier assessment described the flat production configuration as
approximately 61 fields. The authoritative current `ProductionBuildConfig`
dataclass has **67** fields. M5 uses that concrete count, and its
characterization test requires this ledger to account for every field.

## Configuration ownership

| Current fields | Classification | Current producers and consumers | M5 canonical owner |
|---|---|---|---|
| `teacher_model`, `tokenizer_id`, `dataset_path`, `corpus_manifest_path`, `teacher_model_provenance_path`, `output_dir` | User intent; provenance/input identity. `output_dir` is a destination, not semantic identity. | CLI `production-build`, programmatic config; preflight, backend builder, reports | Input/provenance and output sections of resolved build config |
| `teacher_backend`, `runtime_mode`, `target_policy`, `sequence_length`, `vocab_size`, `top_k`, `num_buckets`, `dynamic_top_k_min`, `dynamic_top_k_max`, `dynamic_mass_threshold` | Canonical behavior/teacher intent; selected fields are authority inputs | CLI/config; backend config, score pass, run plan, delivery | Teacher, token, and behavioral-surface sections |
| `long_tail_warning_k`, `very_long_tail_warning_k`, `perverse_tail_warning_k`, `reject_perverse_exemplars`, `primary_selected_exemplar_budget`, `long_tail_side_board_cap`, `perverse_tail_side_board_cap`, `include_long_tail_in_primary`, `include_perverse_tail_in_primary`, `include_perverse_tail_in_student` | Canonical selection policy; not current selection-authority inputs | CLI/config; leaderboard and package/profile writers | Exemplar policy section |
| `gpu_batch_size_mode`, `gpu_batch_size_preset`, `gpu_batch_size_custom`, `gpu_batch_size_auto_min`, `gpu_batch_size_auto_max`, `shard_size_examples`, `max_examples` | User execution intent; resolved batch and shard plan are derived | CLI/config; run-plan and backend builder | Execution controls, with derived plan separated |
| `resume`, `overwrite`, `strict_provenance`, `fail_on_plan_warnings`, `no_build_if_plan_warn`, `max_artifact_bytes` | Runtime/validation controls, not Tome semantic identity | CLI/config; preflight, resume, run plan | Execution and validation controls |
| `run_plan_path`, `production_report_path`, `parity_left`, `parity_report_path`, `run_manifest_path`, `progress_log_path`, `progress` | Derived output locations, operational state, or compatibility diagnostics | CLI/config; reports and parity utilities | Output/report routing; `parity_*` remains compatibility-only |
| `exemplar_delivery_path`, `exemplar_selection_enabled`, `exemplar_leaderboard_capacity`, `selected_exemplar_budget`, `selected_exemplar_fraction`, `retain_unselected_exemplar_payloads`, `exemplar_score_policy`, `selected_rerun_batch_size`, `track_delivery_timing`, `selection_integration_policy`, `total_selected_exemplar_budget`, `fingerprint_corridor_budget_fraction`, `fingerprint_corridor_budget_max`, `fingerprint_corridor_mode_cap`, `fingerprint_corridor_candidate_pool_cap`, `require_full_selected_budget` | Canonical exemplar/selection policy; listed authority fields are protected | CLI/config; native Path B, C2-C5, delivery, authorities | Exemplar, selection, and rerun policy sections |
| `corridor_feature_jsonl_path`, `global_board_supply_path`, `c4_claims_path`, `c5_selection_path`, `source_passports_path` | Explicit compatibility/external-authority overrides | Programmatic config; C2-C5 and authority validators | Isolated compatibility override adapter |

## Protected selection-authority projection

`builder.production._selection_integration_hash` currently hashes exactly 25
keys. M5 must preserve the key names, values, string/path treatment, schema
literals, compact sorted JSON encoding, and SHA-256 result until an explicitly
approved migration:

```text
selection_integration_policy, teacher_model, tokenizer_id, dataset_path,
corpus_manifest_path, target_policy, sequence_length, vocab_size, top_k,
num_buckets, dynamic_top_k_min, dynamic_top_k_max,
dynamic_mass_threshold, selected_rerun_batch_size,
total_selected_exemplar_budget, fingerprint_corridor_budget_fraction,
fingerprint_corridor_budget_max, fingerprint_corridor_mode_cap,
fingerprint_corridor_candidate_pool_cap, require_full_selected_budget,
c2_schema, c3_schema, c4_schema, c5_schema, delivery_path
```

This is distinct from authority-hash v1/v2. The M5 configuration boundary may
provide this payload from resolved configuration, but it may not alter the
existing 25-field semantics.

## Artifact, cover, package, and transport ownership

| Surface | Current writer | Reader/validator | M5 disposition |
|---|---|---|---|
| Cover-page v2 (`cover_page_version: 2`) | `tome.cover_page.write_cover_page` during production/finalization | directory validator and `.rtome` bundle validator | Historical reader; map explicitly to canonical descriptor |
| Package cover v1 (`radjax_tome_package_cover_v1`) | `tome.packaging._write_package_cover_page` after profile materialization | package validator and student reader | Historical reader; map explicitly to canonical descriptor |
| Package manifests v1 | package writer | package validator/audit | Strong current file/profile inventory; evolve to canonical manifest v2 |
| Unpacked directory | production and package writers | producer/cover/package validators | Canonical transport representation |
| `.rtome` uncompressed tar | `tome.bundle.pack_tome_bundle` | bundle validator | Compatibility transport behind one M5 transport abstraction |
| `tgz` package archive | package writer | external extraction then package validator | Compatibility transport behind the same abstraction |
| Golden v1/v2 contracts | Golden capture code | Golden validators/comparators | Frozen; no M5 fixture rewrite or authority reinterpretation |

## M5 target ownership and contradictions

M5B defines three distinct configuration representations: user intent,
resolved canonical configuration, and derived execution plan. Runtime progress,
report locations, backend probing, and compatibility override paths are not
canonical intent. A legacy flat configuration is an explicit adapter, never an
opaque payload inside canonical config.

M5's canonical public cover is `radjax_tome_cover_v3`. Its nested sections are
identity, training, package, manifests, authority, provenance, and validation.
The canonical Tome semantic identity is computed from a profile-neutral,
training-authoritative projection. Profile manifests may differ because debug
or provenance inventory differs, but a student package and full-debug package
for the same training payload must have the same Tome identity. Tar/gzip
wrapping is likewise nonsemantic.

The content manifest must exclude its cover page so the cover can reference
the manifest digest without a circular hash. Directory, `.rtome`, and `tgz`
validators will converge on the same semantic validation path in M5D; M5A
only records this target and changes no writer.

## M5C normalization implementation

M5C activates the typed configuration boundary defined in M5B. The canonical
default intent, named presets, explicit advanced overrides, validation,
resolved configuration, derived execution plan, and protected selection
authority now have one owner: `builder.config`. The current CLI uses this
normalizer and `build_production_gpu_tome` adapts a resolved request into the
flat execution configuration only at the preserved production/Path-B boundary.

The default/override order is exactly preset, explicit overrides, validation,
execution-plan derivation, then 25-field authority projection. See
`docs/M5_CONFIGURATION_NORMALIZATION.md` for the preset values and the
compatibility adapter boundary. M5C does not modify cover, manifest, package,
transport, historical-reader, authority-hash, or Golden writer behavior.
