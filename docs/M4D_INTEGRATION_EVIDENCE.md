# M4D Integration Evidence

## Local proof — pass

This evidence applies to the post-`da807cc` canonical-execution correction on
branch `m3-m4-canonical-path-b-refactor`. The earlier local proof did not
exercise production through typed slices two through five; it must not be read
as proof of that integration. The current gate executes the real ordered
canonical callbacks and preserves the immutable fixture. The working tree is
clean apart from the pre-existing untracked `.DS_Store`.

| Gate | Command | Result |
|---|---|---|
| Complete non-GPU suite | `python3 -m pytest -q` | `784 passed, 22 skipped in 95.98s` |
| Native, import, delivery, validation, linkage, reconciliation, live canonical execution, resume assembly, and Golden focus | See the exact command below | `225 passed in 35.83s` |
| Immutable Golden fixture | `PYTHONPATH=src python3 -m radjax_tome.cli.main golden validate --fixture tests/fixtures/golden_t4_1k` | `pass`; count `256`; semantic root `sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba` |
| Static and whitespace checks | `ruff check src tests`; `ruff format --check src tests`; `git diff --check` | pass |

The focused command was:

```bash
python3 -m pytest -q \
  tests/test_m3a_import_isolation.py \
  tests/test_native_path_b_api.py \
  tests/test_native_path_b_contracts.py \
  tests/test_native_path_b_orchestrator.py \
  tests/test_native_path_b_resume.py \
  tests/test_m4b_production_stage_integration.py \
  tests/test_m4_live_canonical_execution.py \
  tests/test_m4c_resume_assembly.py \
  tests/test_m3a_runtime_characterization.py \
  tests/test_production_build.py \
  tests/test_selected_exemplar_delivery.py \
  tests/test_selected_exemplar_adversarial_linkage.py \
  tests/test_c6_integration.py \
  tests/test_golden_t4_1k_fixture.py \
  tests/test_golden_contract_compare.py \
  tests/test_golden_projection_truth_gate.py
```

## Reviewed T4 Golden 1K proof — not executed

This checkout contains only the immutable semantic fixture, not a terminal
canonical artifact suitable for `golden compare`. The local host has no
`nvidia-smi`, CUDA-capable T4, `torch`, or `transformers`; it cannot perform or
stand in for the rental proof. No fixture was regenerated or treated as an
observed artifact.

The reviewed T4 run must use the same 1K corpus/model/provenance identity as
the fixture, sequence length `128`, vocabulary `262144`, dynamic top-k range
`32..262144`, mass threshold `0.99`, rerun batch `8`, exact native C6 policy,
and selected count `256`. The general rehearsal's `1024` budget is not a
substitute for this Golden 1K comparison.

From a clean checkout with CUDA PyTorch and a confirmed T4, run:

```bash
radjax-tome doctor \
  --teacher-backend gpu_torch --runtime-mode cpu_gpu \
  --target-policy corridor_exemplar_v1 \
  --teacher-model /path/to/model --tokenizer-id /path/to/tokenizer \
  --sequence-length 128 --batch-size 8 --vocab-size 262144 --top-k 32 \
  --exemplar-selection-enabled \
  --exemplar-fulfillment-policy rerun_selected_capture \
  --gpu-batch-size-mode preset --gpu-batch-size-preset 8 \
  --fallback-policy error --write-report /path/to/t4-doctor.json

radjax-tome production-build \
  --teacher-backend gpu_torch --runtime-mode cpu_gpu \
  --target-policy corridor_exemplar_v1 \
  --selection-integration-policy corridor_first_global_backfill_v1 \
  --total-selected-exemplar-budget 256 \
  --exemplar-delivery-path two_pass_rerun_selected \
  --exemplar-selection-enabled --no-retain-unselected-exemplar-payloads \
  --sequence-length 128 --vocab-size 262144 --top-k 32 \
  --dynamic-top-k-min 32 --dynamic-top-k-max 262144 \
  --dynamic-mass-threshold 0.99 --selected-rerun-batch-size 8 \
  --gpu-batch-size-mode preset --gpu-batch-size-preset 8 \
  --max-examples 1000 --strict-provenance --progress \
  --global-board-supply /path/to/production-global-supply.json \
  --source-passports /path/to/source-passports.json \
  --teacher-model /path/to/model --tokenizer-id /path/to/tokenizer \
  --dataset /path/to/corpus.jsonl \
  --corpus-manifest /path/to/corpus_manifest.json \
  --teacher-model-provenance /path/to/teacher_provenance.json \
  --output /path/to/t4-golden-1k-output

radjax-tome golden compare \
  --fixture tests/fixtures/golden_t4_1k \
  --artifact /path/to/t4-golden-1k-output
```

Before approving the reviewed result, retain the doctor report, the exact
redacted command/input identifiers, passing terminal reports, and raw compare
JSON. The artifact must have passing production, validation, delivery,
selected-linkage, and C6 reconciliation reports; 256 unique selected/payload
coordinates; no blockers; and a compare result with status `pass`, the frozen
root above, and empty `differences` and `storage_only_differences` lists.
Any CUDA/T4, provenance/configuration, count, terminal-report, or comparison
difference is a stop condition: preserve evidence, do not merge `main`, and do
not alter the fixture.
