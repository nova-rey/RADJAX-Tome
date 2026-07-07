# Tome Runtime Backends

Spec 3.3A defines the runtime/backend vocabulary before backend migration
begins. It is a capability model, not a new backend implementation.

Runtime mode chooses where computation happens. Target policy chooses what comes back. The capability matrix says what is supported. Metadata proves what actually happened. The writer stays backend-neutral.

## Internal Contract Wall

The internal contract wall separates CPU orchestration, runtime backend work,
target policy, and artifact writing. The writer should accept valid target
payloads and metadata without knowing whether logits or reductions came from
CPU, GPU, TPU, a fake backend, or a future backend family.

CPU front-end/orchestrator responsibilities:

- corpus loading
- batching
- CPU scheduling
- ordered output writing
- validation
- cover page and bundle metadata
- progress and reporting

Runtime backend responsibilities:

- teacher execution
- target reduction
- runtime-specific memory and transfer policy
- runtime diagnostics

## Backend Contract Skeleton

Spec 3.3B introduces the contract skeleton only. The active public builder has not migrated to this backend contract yet.

The skeleton is intentionally small:

- `TeacherBackendConfig` carries backend ID, runtime mode, CPU orchestration
  mode, target policy, model/tokenizer IDs, shape settings, local-files policy,
  and fallback policy.
- `TeacherBatchInput` carries batch example IDs and texts.
- `TeacherEmissionResult` returns backend-neutral `input_ids`,
  `attention_mask`, payload arrays, and JSON-ish runtime metadata.
- `BackendCapability` records backend ID, backend family, runtime mode, target
  policy, support status, optimization status, implementation status, and
  notes.
- `TeacherEmissionBackend` exposes `capabilities()`, `emit_batch(...)`,
  `metadata()`, and `close()`.

The registry provides deterministic `register_backend(...)`,
`create_backend(...)`, and `list_backend_capabilities()` helpers. Duplicate
backend IDs fail clearly, unknown backend IDs fail clearly, and the registry
does not import torch, transformers, jax, CUDA, MPS, or TPU dependencies.

The default registered backend is `fake_numpy`. It proves the contract wall for
`dense_logits` on `cpu` with `supported_debug` status and deterministic NumPy
arrays. It does not replace the current fake TeacherTextbook builder.

## CPU Reference Backend

Spec 3.3C adds `cpu_reference` as the first real backend-contract
implementation. It is serial/reference, not optimized. It emits deterministic
CPU payloads for `dense_logits`, `topk_with_tail_v0`,
`cascaded_soft_labels_v1`, `dynamic_cascaded_soft_labels_v1`, and
`corridor_exemplar_v1`.

The CPU reference backend does not migrate the public builder. It does not
implement staged CPU orchestration. It does not implement HF, GPU, or TPU
runtime backends.

Corridor/exemplar is CPU-supported as a serial/reference path. GPU/TPU are
future acceleration paths, not exclusive ownership paths. This CPU
corridor/exemplar implementation is a deterministic reference path for
backend-contract correctness. It is not a claim of optimized historical parity.

Payload summary:

- `dense_logits`: `logits`
- `topk_with_tail_v0`: `top_token_ids`, `top_log_probs`, `top_probs`,
  `top_mass`, `tail_mass`, `teacher_entropy`
- `cascaded_soft_labels_v1`: the top-k/tail fields plus `bucket_masses`
- `dynamic_cascaded_soft_labels_v1`: dynamic top-k explicit head arrays,
  `top_selection_mask`, `effective_top_k`, and bucketed tail masses
- `corridor_exemplar_v1`: production behavioral/fingerprint payload with
  `corridor_records`, `corridor_summary`, `exemplar_records`,
  `exemplar_summary`, `mode_records`, `source_policy_summary`,
  `schema_metadata`, corridor token/entropy/confidence arrays, source-policy
  arrays, and entropy-selected exemplar positions

The cascaded reference reducer removes top-k tokens, sorts the remaining tail
probabilities descending, partitions them into fixed contiguous buckets, and
sums probability mass per bucket. This is a correctness baseline for payload
shape and mass accounting, not a performance path.

Spec 3.3F6 adds the CPU reference and contract shape for
`dynamic_cascaded_soft_labels_v1`. It is dynamic top-k explicit head plus
bucketed tail, not simple dynamic top-k with one tail mass. The payload is
fixed and padded to the effective dynamic max K. `top_selection_mask` is
authoritative: masked top slots have token ID 0, probability 0.0, and log
probability 0.0, and consumers must ignore them. The CPU reference records
`effective_top_k`, threshold/min/max metadata, bucket metadata, and observed
effective-k statistics.

Spec 3.3F8 locks `corridor_exemplar_v1` as a production
behavioral/fingerprint payload, not another soft-label compression format. It
records `schema_version=corridor_exemplar_v1`,
`corridor_payload_flavor=production_v1`,
`production_corridor_schema=true`, `historical_parity_claimed=false`, and a
source-policy-aware summary. Allowed exemplar source policies are
`dense_logits`, `cascaded_soft_labels_v1`, and
`dynamic_cascaded_soft_labels_v1`; dynamic cascaded is the preferred future
compact source. The CPU reference path remains deterministic/proxy math with
`historical_reference_source=cpu_reference_proxy`. F9 owns active GPU corridor/exemplar
acceleration.

Spec 3.3F9 implements `gpu_torch` corridor/exemplar emission against that F8
production schema. It supports `dense_logits`, `cascaded_soft_labels_v1`, and
`dynamic_cascaded_soft_labels_v1` as exemplar source policies, keeps dense
logits on device even for dense source mode, and transfers only compact
production arrays plus CPU-built record summaries. The public builder has not
migrated to `gpu_torch`.

Spec 3.3F9.1 names the current capture behavior
`one_pass_candidate`. In this mode, the backend emits compact candidate data
for every batch example in one teacher pass. This is compute-efficient and
simple, but may become storage-heavy for huge corpora. It is not final
corpus-level exemplar pruning.

Spec 3.3F9.2 adds explicit `two_pass_sparse_exemplar` capture mode. In this
mode, pass 1 emits [B]-scale `score_pass` summaries for every example; a later
selector can choose examples/positions; pass 2 reruns selected examples and
emits F8 production-shaped exemplar payloads with
`exemplar_capture_stage=selected_exemplar_pass`. This trades extra teacher
inference for lower transfer and storage. The public builder has not migrated,
and TPU/JAX remain out of scope.

Spec 3.3F9.3 adds `exemplar_capture_mode=auto`. Manual
`one_pass_candidate` and `two_pass_sparse_exemplar` settings still win. Auto
estimates one-pass candidate bytes, two-pass score bytes, selected-pass bytes,
expected selected fraction, and any available disk budget, then records
`exemplar_capture_policy=auto_exemplar_capture_policy_v1`,
`manual_override_used`, `auto_policy_reason`, and
`auto_policy_inputs_missing` in metadata. The policy is an estimate; reducer
semantics do not change.

Spec 3.3F9.4 adds GPU batch-size policy and guardrail metadata. Batch size N
means N examples enter the backend and one batched result with leading
dimension N comes back. The backend does not stream examples back one by one
and does not secretly split oversized batches; the future builder/orchestrator
owns probing and scheduling. The policy modes are `preset`, `custom`, and
`auto`: presets allow 1/2/4/8/16/32/64, `custom` preserves the user-specified
positive batch size, and `custom >64` is allowed with warning metadata. Auto
uses `exponential_probe_v1`, chooses the last good batch from probe results,
and can use midpoint refinement. Metadata records estimated-vs-measured byte
caveats and measured compact payload bytes when arrays are available. F9.4 is
single-device only; multidevice vocabulary is future-reserved with
`multidevice_enabled=false` and `batch_partition_strategy=single_device`.

## CPU Orchestration Runner

Spec 3.3D adds a backend batch runner above backend emission. The runner
accepts ordered `BackendBatchEnvelope` values, calls a
`TeacherEmissionBackend`, and returns ordered results plus run-level metadata.

`serial` is the correctness/debug path. It processes one batch at a time and
sorts results by `sequence_id` before returning.

`staged` is a deterministic pipeline-shaped path, not yet a performance claim.
It records staged orchestration metadata while preserving output order. It does
not port historical queue, prefetch, or ordered-commit machinery.

`auto` chooses `serial` for tiny workloads and `staged` otherwise. The current
heuristic uses batch count first, then total example count, and records an
`auto_reason` such as `tiny_batch_count`, `tiny_example_count`, or
`normal_workload`.

Backend emission metadata describes backend behavior. Run-level orchestration
metadata describes how the CPU front end scheduled batches. The public builder
has not migrated to the runner yet.

## HF Torch Backend

Spec 3.3E adds `hf_torch` behind the `TeacherEmissionBackend` contract. It is
CPU-runtime-first and loads `torch` and `transformers` lazily, only when model
availability or emission is requested.

`hf_torch` supports real HF causal LM emission for `dense_logits`,
`topk_with_tail_v0`, and `cascaded_soft_labels_v1`. Dense logits are returned as
NumPy arrays. Compact payloads are CPU reductions from real HF logits, not GPU
compact reduction.

`hf_torch` lists `dynamic_cascaded_soft_labels_v1` as planned after the Spec
3.3F6 CPU reference contract shape. It does not emit dynamic cascaded payloads
in F6.

The backend does not claim CUDA, MPS, TPU, or optimized accelerator behavior.
The public builder has not migrated to `hf_torch` by default.

## GPU Torch Backend

Spec 3.3F1 adds `gpu_torch` behind the `TeacherEmissionBackend` contract. It is
an accelerator-shaped backend with `runtime_mode=cpu_gpu` and lazy
`torch`/`transformers` imports.

`gpu_torch` detects Torch accelerator devices in deterministic order: CUDA
first, then MPS. Explicit `gpu_torch` requests fail clearly when neither device
is available; they do not silently route to CPU or `hf_torch`.

Spec 3.3F1 implements `dense_logits` as `supported_debug`. The path loads an HF
causal-LM model/tokenizer lazily, runs inference on the selected accelerator,
moves full dense logits back to host as NumPy payloads, and records metadata
that says the dense debug path and host transfer were used.

Spec 3.3F2 adds `topk_with_tail_v0` as an optimized compact GPU reduction path.
After model forward, logits remain as Torch tensors on the selected accelerator
while log-softmax, top-k probability, tail mass, and entropy are computed. Only
the compact payload arrays move back to host.

Spec 3.3F3 adds `cascaded_soft_labels_v1` as an optimized compact GPU
reduction path. It extends the top-k/tail reducer with `bucket_masses` computed
on the selected accelerator using contiguous descending tail-probability
buckets. Only compact payload arrays move back to host.

Spec 3.3F4 adds optional vocab-axis chunking for compact GPU reducers through
`gpu_enable_vocab_chunking` and `gpu_vocab_chunk_size`. Dense debug remains
unchunked and still transfers full logits to host. Compact top-k/tail supports
effective vocab-axis chunking and records requested/effective chunking, chunk
counts, compact host transfer bytes, dense-equivalent byte estimates, and
deterministic reducer workspace estimates. These are estimates, not measured
peak GPU memory.

Spec 3.3F4 also removes duplicate full-vocab softmax/probability work from the
non-chunked cascaded path by sharing the probability workspace used for top-k
and bucket-mass reduction.

Spec 3.3F4.1 fixes cascaded chunking metadata truthfulness. Cascaded
soft-label reduction still uses exact contiguous descending tail-probability
buckets. When vocab chunking is requested for cascaded, current metadata records
`vocab_chunking_requested=true` but `vocab_chunking_used=false` with
`vocab_chunking_reason=exact_bucket_policy_requires_full_probability_workspace`
because exact bucket construction reconstructs a full probability workspace on
device. It does not transfer full dense logits to host, but it also does not
claim chunk-sized reducer workspace.

Spec 3.3F5 adds runtime diagnostics and error hardening. `gpu_torch`
diagnostics report missing `torch`, missing `transformers`, no CUDA/MPS
accelerator, missing local model/tokenizer files, unsupported target policies,
and invalid config without requiring network downloads. Successful emissions
record `fallback_used=false`, `fallback_policy`, `failure_stage=none`, and
`failure_reason=null`.

`gpu_torch` never silently falls back to CPU. `fallback_policy=auto` is an
orchestrator signal only: a higher-level runner may choose another backend
later, but `gpu_torch` itself does not call `hf_torch` or `cpu_reference` and
does not emit CPU results for an explicit `cpu_gpu` request.

Spec 3.3F6 defines `dynamic_cascaded_soft_labels_v1` in the CPU reference
backend. Spec 3.3F7 adds the optimized `gpu_torch` reducer for that policy:
dynamic top-k explicit head plus bucketed tail, with a mass-threshold cutoff
bounded by dynamic min/max K. The reducer keeps logits on the selected
accelerator, computes dynamic head selection and tail buckets with Torch
tensors, and transfers only compact payload arrays back to host. The dynamic
cascaded payload can later be used as a corridor/exemplar source policy.

Spec 3.3F7.1 keeps the F7 dynamic cascaded payload and metadata contract but
vectorizes dynamic explicit-head selection to reduce Python-loop overhead.
Exact bucket semantics remain unchanged.

Runtime fallback/error hardening is in place for implemented `gpu_torch`
policies. Spec 3.3F10 adds explicit builder routing for backend-contract
emissions, including `gpu_torch` when `runtime_mode=cpu_gpu`.

## GPU Builder Integration Gate

Spec 3.3F10 adds a medium integration gate for the builder, not a new reducer.
The build command can explicitly select `teacher_backend=gpu_torch` with
`runtime_mode=cpu_gpu`, route batches through `TeacherEmissionBackend`, and
write valid Tome artifacts for backend-emitted policies.

Builder-routed artifacts preserve backend metadata in the target-store
metadata, `cover_page.json`, `teacher_manifest.json`, and
`emission_config.json`. The propagated fields include requested/effective
runtime, backend identity, fallback policy, capability status, optimized-path
flag, GPU compact metadata, exemplar-capture metadata, auto capture policy
metadata, and batch-size policy metadata.

The builder recognizes the backend artifact schemas
`dynamic_cascaded_soft_labels_v1`, `corridor_exemplar_v1`, and
`corridor_exemplar_score_pass_v1`. The two-pass score-pass schema is writeable
for deterministic local smoke coverage, but F10 does not implement a
production global two-pass selector.

Batch size remains batch-in/batch-out: the builder may slice the input corpus
into batches using its configured batch size, but the backend receives one
batch and returns one batched result. The backend does not secretly split
oversized batches. Artifacts record configured, actual, and effective GPU
batch-size metadata.

Example GPU-routed dynamic cascaded build:

```bash
python -m radjax_tome.cli.main build \
  --output out/tome_dynamic_gpu \
  --teacher-backend gpu_torch \
  --runtime-mode cpu_gpu \
  --target-policy dynamic
```

Example GPU-routed corridor build:

```bash
python -m radjax_tome.cli.main build \
  --output out/tome_corridor_gpu \
  --teacher-backend gpu_torch \
  --runtime-mode cpu_gpu \
  --target-policy corridor \
  --exemplar-capture-mode one_pass_candidate
```

An explicit `gpu_torch` request never silently falls back to CPU, `hf_torch`,
or `cpu_reference`. `fallback_policy=auto` is still only an orchestrator
signal; the builder records the request but does not perform a hidden backend
swap. F10 also does not add real auto batch probing, builder hydra behavior,
multidevice scheduling, TPU/JAX support, or production-readiness claims.

## Multi-Leaderboard Exemplar Selection

Spec 3.3F10.1 adds the shared
`multi_leaderboard_exemplar_selector_v1` harness above corridor/exemplar
emission. It is capture-mode-agnostic: `one_pass_candidate` and
`two_pass_sparse_exemplar` produce common candidate records, then candidates
compete for bounded high-score boards. At the end, board winners are
deduplicated and written to `exemplar_selection_manifest.json`.

Path A, `one_pass_candidate`, is debug/small-run oriented. Rich candidate
payloads already exist, so fulfillment uses `select_from_existing_capture` and
marks retained examples/positions from the existing artifact. The full debug
artifact may remain intact with `retain_all_candidates_debug`.

Path B, `two_pass_sparse_exemplar`, is production-shaped for large corpora and
storage-sensitive runs. Cheap score-pass summaries feed the same selector, and
fulfillment uses `rerun_selected_capture`: the manifest is a rerun requisition
with enough source addressing to rerun selected examples in a later selected
pass.

F10.1 does not implement semantic embeddings, a utility-calibrated selector, a
claimed optimal production selector, real auto batch probing, multidevice
scheduling, TPU/JAX, or reducer math changes. Backend emission capability
statuses are unchanged.

Spec 3.3F10.1.1 refines the multi-leaderboard dedupe pass with
`rank_aware_board_assignment_with_backfill_v1`. If one candidate appears on
several boards, it stays on the board where it ranked strongest, is suppressed
from weaker boards, and those boards scan runner-up pools to backfill open
slots. Budget trimming is `score_aware_assigned_board_rank_v1`, so retained
examples prefer better assigned-board rank and score rather than alphabetical
example IDs. This remains the same selector architecture and makes no semantic
embedding, utility-calibrated, production global selector, TPU/JAX, or backend
capability claim.

## Runtime Doctor And Metadata Sanity

Spec 3.3F11 adds operational polish around the existing runtime/backend
surface. `radjax-tome doctor` now emits a backend availability summary and a
JSON-serializable `runtime_doctor_report_v1` preflight report. It records the
requested backend/runtime/target policy, model/tokenizer IDs, local-files and
download flags, fallback policy, capability status, optional dependency
availability, accelerator availability, `can_emit`, failure stage/reason, and
remediation hints.

`radjax-tome inspect --metadata-sanity` and
`radjax-tome validate --metadata-sanity --write-report` now expose
`artifact_metadata_sanity_report_v1`. The report normalizes stringified
artifact metadata for reporting only, summarizes backend/effective-backend
routing, compact GPU metadata, exemplar capture metadata, selector metadata,
and batch-size metadata, then flags contradictory claims such as score-pass
artifacts that claim final corridor schema, `gpu_torch` requested/effective
mismatches without explicit fallback metadata, false future-selector claims,
or multidevice metadata that is not `single_device`.

F11 changes report and CLI diagnostics only. It adds no new reducer math, no
new selector policy, no real auto batch probing, no production global selector,
no multidevice scheduler, and no TPU/JAX backend.

## Runtime Modes

`cpu` means CPU-side orchestration plus CPU teacher execution and reduction. It
is the universal correctness and reference path. Spec 3.3F6 adds CPU reference
support for dynamic cascaded soft labels.

`cpu_gpu` means CPU-side orchestration plus GPU-backed teacher execution and/or
GPU-backed reduction. Spec 3.3F1 implements a dense debug/smoke HF Torch path
on CUDA or MPS. Spec 3.3F2 adds compact top-k/tail reduction on the accelerator.
Spec 3.3F3 adds compact cascaded soft-label reduction on the accelerator.
Spec 3.3F4 adds optional vocab chunking and memory/workspace metadata for
compact reducers. Spec 3.3F4.1 corrects cascaded chunking metadata so exact
bucket construction does not overclaim effective chunked workspace. Spec 3.3F5
adds structured diagnostics and no-silent-CPU-fallback error hardening. Spec
3.3F6 defines the dynamic cascaded CPU contract shape, and Spec 3.3F7 adds the
optimized GPU dynamic cascaded reducer. Spec 3.3F10 adds explicit builder
routing for supported `gpu_torch` policies without silent CPU fallback.

`cpu_tpu` means CPU-side orchestration plus TPU/JAX/XLA-backed teacher
execution and/or TPU-backed reduction. This is a future backend family; no
active TPU implementation is claimed.

## CPU Orchestration Modes

`auto` is the default user-facing choice. A future implementation may choose
serial for tiny/debug workloads and staged for normal workloads.

`serial` is the simple reference/debug/small-smoke path.

`staged` is the future multicore or pipelined orchestration path. It may overlap
corpus loading, tokenization and prefetch, teacher inference, reduction, and
ordered commit/write.

## Target Policies

`dense_logits` stores full-resolution teacher logits. Dense logits are useful
for reference, debug, and very small corpora. Dense logits are not the main
optimization target for large Tome generation.

`topk_with_tail_v0` stores fixed top-k probabilities plus tail mass. CPU
reference support exists, and GPU/TPU optimized reduction should target this
compact format before trying to make dense logits fast at scale.

`cascaded_soft_labels_v1` stores cascaded bucket soft-labels. CPU reference
support exists, and accelerator work should optimize compact reduction for this
representation rather than assume dense output is the final payload.

`dynamic_cascaded_soft_labels_v1` stores a dynamic top-k explicit head plus
bucketed tail. It is not simple dynamic top-k with one tail mass. It uses
padded explicit-head arrays, `top_selection_mask` to identify selected slots,
`effective_top_k` per position, and the same contiguous descending
tail-probability bucket policy as fixed cascaded soft labels.

`corridor_exemplar_v1` is the behavioral corridor plus exemplar-oriented target
family. As of Spec 3.3F8, it has a locked production payload flavor and is
source-policy-aware. Dense, fixed-cascaded, and dynamic-cascaded source
policies feed its behavioral fields. Spec 3.3F9 adds `gpu_torch` acceleration
for the behavioral/fingerprint policy while preserving compact-only host
transfer and leaving public builder migration out of scope.
Spec 3.3F9.1 records `one_pass_candidate` capture metadata with
`exemplar_candidate_scope=batch_all_examples` and
`corpus_level_exemplar_finalization=false`.
Spec 3.3F9.2 adds `two_pass_sparse_exemplar`: a compact `score_pass` emits
only [B]-scale score summaries, then a `selected_exemplar_pass` can rerun
chosen examples and emit the F8 production schema. F9.2 does not implement
public builder migration or TPU/JAX.
Spec 3.3F9.3 adds `auto` capture selection. Manual overrides win, and auto
records estimated byte counts, expected selected fraction, disk budget if
known, missing inputs, and the reason for the effective mode.
Spec 3.3F9.4 adds `gpu_batch_size_policy_v1` guardrail metadata. `preset`,
`custom`, and `auto` modes resolve an effective batch size without changing the
backend batch-in/batch-out reducer semantics. `exponential_probe_v1` synthetic
probe results choose the last good batch, `custom >64` emits a warning, and
estimated-vs-measured byte fields are recorded without claiming measured GPU
peak memory. Public builder migration, multidevice scheduling, and TPU/JAX
remain out of scope.
Spec 3.3F10.1 adds the shared multi-leaderboard exemplar selector. Path A and
Path B use the same selector; only fulfillment differs.
Spec 3.3F10.1.1 makes that selector rank-aware: duplicate candidates are
assigned to their strongest board, weaker boards backfill from runner-up pools,
and budget trimming is score-aware rather than alphabetical.
Spec 3.3F11 adds runtime doctor/preflight reports and artifact metadata sanity
reports. Backend emission semantics, selector behavior, capability statuses,
auto-batch behavior, multidevice support, and TPU/JAX remain unchanged.

## Support Statuses

`unsupported` means the combination is not supported and is not currently
planned in the matrix.

`planned` means the combination is planned but not implemented.

`supported` means the combination is implemented as a normal path.

`supported_debug` means the combination is implemented but intended for debug,
reference, or small workloads.

`optimized` means the combination is implemented as an optimized path for that
runtime.

`historical_reference_exists` means historical QRWKV-XLA work exists and can
guide future migration, but active RADJAX-Tome does not claim equivalent
behavior.

## Fallback Policy

Explicit runtime requests for unsupported or unimplemented combinations should
fail. Auto runtime may choose a supported fallback. Fallback must be recorded in
metadata when implemented. There must be no silent accelerator-to-CPU fallback.
For `gpu_torch`, `fallback_policy=auto` does not authorize backend-local CPU
emission; orchestrator fallback must happen outside `gpu_torch` and be recorded
there.

## Metadata Policy

Future runtime metadata should record requested and effective values so
consumers can prove what actually happened:

- requested_runtime_mode
- effective_runtime_mode
- requested_cpu_orchestration_mode
- effective_cpu_orchestration_mode
- requested_backend_family
- effective_backend_family
- requested_target_policy
- effective_target_policy
- fallback_used
- fallback_reason
- optimized_path_used
- runtime_kind
- device_kind
- capability_status

## Design Notes

Dense logits are useful for reference, debug, and very small corpora. They
remain supported because they are the simplest correctness target. Dense logits
are not the main optimization target because large dense teacher outputs are
expensive to store, move, and consume.

GPU and TPU optimization should target compact formats first:
`topk_with_tail_v0`, `cascaded_soft_labels_v1`, and `corridor_exemplar_v1`.
Optimizing dense logits alone would preserve the largest payload shape and
delay the actual Tome-size reduction work.

TPU support must not be designed as CUDA-with-different-spelling. JAX/XLA and
TPU execution have different compilation, sharding, transfer, and shape
constraints, so the contract should describe runtime capability without baking
in CUDA assumptions.
