# T4 Tome Rehearsal

This runbook is a manual GPU rehearsal. It is not part of ordinary CI, and a
successful code/test run must not be reported as a successful T4 run.

## Prepare

1. Install RADJAX-Tome with the GPU teacher and development extras.
2. Install a CUDA-enabled PyTorch wheel compatible with the T4 driver.
3. Keep the teacher model, tokenizer, corpus, corpus manifest, source
   passports, strict corridor features, and production global-board supply
   local. Production mode does not download or verify over the network.
4. Run `radjax-tome doctor` for the teacher/backend configuration and confirm
   CUDA is available, the selected device is a T4, and the intended model and
   corpus provenance files are present.

## Run

Plan first with `radjax-tome plan`, using the same backend, corpus, teacher
provenance, sequence length, dynamic-top-k controls, and C6 policy/budget
flags. Then run a streaming Path B build:

```text
radjax-tome production-build \
  --teacher-backend gpu_torch --runtime-mode cpu_gpu \
  --selection-integration-policy corridor_first_global_backfill_v1 \
  --total-selected-exemplar-budget 1024 \
  --corridor-feature-jsonl /path/to/strict-corridor-features.jsonl \
  --global-board-supply /path/to/production-global-supply.json \
  --source-passports /path/to/source-passports.json \
  --exemplar-delivery-path two_pass_rerun_selected \
  --exemplar-selection-enabled --no-retain-unselected-exemplar-payloads \
  --resume --progress \
  --teacher-model /path/to/model --tokenizer-id /path/to/tokenizer \
  --dataset /path/to/corpus.jsonl \
  --corpus-manifest /path/to/corpus_manifest.json \
  --teacher-model-provenance /path/to/teacher_provenance.json \
  --output /path/to/output
```

Use the actual local paths and approved budget. Do not use compatibility
proxies or the development selector manifest.

## Inspect and validate

Confirm that production and delivery reports are `pass`, then inspect:

- `reports/fingerprint_corridor_coverage.json`;
- `reports/c6_integrated_selection_validation.json`;
- `selected_linkage_audit.json`;
- `leaderboards/selected_exemplars.json`;
- `selected_exemplars/selected-exemplars-00000.json`;
- `cover_page.json`, `run_manifest.json`, and `production_progress.json`.

Check that C5 unique count, payload count, source-passport count, curriculum
union, audit count, cover summary, and student-package count agree. If both
delivery paths were run from the same source authority, run
`exemplar-delivery-parity` and inspect coordinates, payload identities,
multi-role metadata, and passport linkage.

Package only after the unpacked artifact validates. Validate the full-debug and
student packages independently and verify hashes and package-local summaries.

## Failure and resume

Preserve `production_progress.json`, `run_manifest.json`, failure reports, and
the C6 checkpoint directory for diagnosis. Resume only with the identical
selection policy, C1-C5 inputs, budget configuration, delivery path, and
source hashes. A changed policy/configuration must be rejected by the resume
hash. Remove an intentionally abandoned output only after preserving its
reports and recording the reason.

This rehearsal is PyTorch CPU/NVIDIA GPU producer validation. TPU/JAX execution
is outside RADJAX-Tome and is not part of this test.
