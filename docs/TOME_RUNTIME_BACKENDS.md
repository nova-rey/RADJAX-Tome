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

## Runtime Modes

`cpu` means CPU-side orchestration plus CPU teacher execution and reduction. It
is the universal correctness and reference path.

`cpu_gpu` means CPU-side orchestration plus GPU-backed teacher execution and/or
GPU-backed reduction. This is the future high-throughput path for HF Torch
CUDA/MPS-style acceleration.

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
