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
`cascaded_soft_labels_v1`, and `corridor_exemplar_v1`.

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
- `corridor_exemplar_v1`: `corridor_records`, `corridor_summary`,
  `exemplar_records`, `exemplar_summary`, corridor token/entropy/confidence
  arrays, and deterministic high-entropy exemplar selections

The cascaded reference reducer removes top-k tokens, sorts the remaining tail
probabilities descending, partitions them into fixed contiguous buckets, and
sums probability mass per bucket. This is a correctness baseline for payload
shape and mass accounting, not a performance path.

The corridor/exemplar reference reducer derives token-level behavior scores
from deterministic CPU logits, summarizes confidence and entropy per batch, and
selects exemplar positions with the stable
`deterministic_high_entropy_top_n_v1` policy.

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

The backend does not claim CUDA, MPS, TPU, or optimized accelerator behavior.
The public builder has not migrated to `hf_torch` by default.

## GPU Torch Backend

Spec 3.3F1 adds `gpu_torch` behind the `TeacherEmissionBackend` contract. It is
an accelerator-shaped debug backend with `runtime_mode=cpu_gpu` and lazy
`torch`/`transformers` imports.

`gpu_torch` detects Torch accelerator devices in deterministic order: CUDA
first, then MPS. Explicit `gpu_torch` requests fail clearly when neither device
is available; they do not silently route to CPU or `hf_torch`.

Spec 3.3F1 implements only `dense_logits` as `supported_debug`. The path loads
an HF causal-LM model/tokenizer lazily, runs inference on the selected
accelerator, moves dense logits back to host as NumPy payloads, and records
metadata that says the dense debug path and host transfer were used.

`topk_with_tail_v0`, `cascaded_soft_labels_v1`, and `corridor_exemplar_v1`
remain historical-reference/future GPU compact-reduction work. The public
builder has not migrated to `gpu_torch`.

## Runtime Modes

`cpu` means CPU-side orchestration plus CPU teacher execution and reduction. It
is the universal correctness and reference path.

`cpu_gpu` means CPU-side orchestration plus GPU-backed teacher execution and/or
GPU-backed reduction. Spec 3.3F1 implements only a dense debug/smoke HF Torch
path on CUDA or MPS; high-throughput compact GPU reduction remains future 3.3F
work.

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
reference support is planned, and GPU/TPU optimized reduction should target
this compact format before trying to make dense logits fast at scale.

`cascaded_soft_labels_v1` stores cascaded bucket soft-labels. CPU reference
support is planned, and accelerator work should optimize compact reduction for
this representation rather than assume dense output is the final payload.

`corridor_exemplar_v1` is the behavioral corridor plus exemplar-oriented target
family. It may map to multiple artifact substructures later. GPU and TPU
optimization should prioritize this compact behavioral format once the backend
contract exists.

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
