# Spec 3 Roadmap

Spec 3 resumes RADJAX-Tome planning after the 2.14-2.18 cleanup arc. These
are roadmap arcs, not necessarily single-shot implementation specs.

## Arcs

| Arc | Title | Status |
|---|---|---|
| 3.0 | Optimization Handoff Inventory and Roadmap Lock | complete when this document and inventory land |
| 3.1 | Cover Page v1 for unpacked Tome directory | complete once cover-page generation and validation land |
| 3.2 | Tome Bundle Container v1 | complete once deterministic .rtome pack/inspect/validate/unpack land |
| 3.3 | Teacher Backend Runtime Modes: CPU, CPU+GPU, CPU+TPU | planned |
| 3.4 | Dynamic Top-K Compression Policy | planned |
| 3.5 | Final CLI Polish / Optional TUI | planned |

## Recommended Ordering

1. Bookmark optimization handoff.
2. Implement cover page for unpacked Tome.
3. Implement bundle container.
4. Implement backend runtime modes.
5. Implement dynamic top-k.
6. Polish CLI / optional TUI.

## Ordering Rationale

Cover page defines the artifact contract before optimized generator machinery
targets it. Backend runtime abstraction should exist before porting
CUDA-specific optimizations. Dynamic top-k should wait until artifact metadata
and backend reduction policy surfaces exist. TUI is optional polish after
functional readiness.

Spec 3.1 adds `cover_page.json` for unpacked Tome directories so optimized
generation can target a contract-shaped artifact instead of forcing the contract
to follow an optimization-specific layout later.

Spec 3.2 adds `.rtome` bundle v1 as a deterministic tar packaging layer around
the cover-page-described files. It is packaging, not a new semantic compression
policy.

## Spec 3.3 Mini-Roadmap

Spec 3.3 is split into runtime/backend roadmap units:

| Unit | Title | Status |
|---|---|---|
| 3.3A | Runtime Mode Capability Model | complete |
| 3.3B | Backend Contract + Registry Skeleton | complete |
| 3.3C | CPU Reference Backend | complete once the CPU reference backend lands |
| 3.3C.1 | CPU Corridor / Exemplar Reference Policy | complete once the capability correction lands |
| 3.3D | CPU Orchestration Modes: auto / serial / staged | complete once the backend batch runner lands |
| 3.3E | HF Torch Backend Behind the Contract | complete once the HF Torch contract backend lands |
| 3.3F | GPU Compact Reduction Migration | in progress through sub-roadmap |
| 3.3G | TPU/JAX Backend Skeleton | planned |
| 3.3H | Runtime Metadata + CLI/Doctor Polish | planned |

3.3A defines vocabulary.

3.3B defines the internal backend contract wall.

Spec 3.3B adds the backend contract and registry skeleton only. It does not complete CPU/GPU/TPU runtime implementation or migrate the public builder.

3.3C establishes the CPU correctness baseline.

Spec 3.3C adds a serial CPU reference backend only; staged CPU orchestration
remains planned for 3.3D.

Spec 3.3C.1 corrects `cpu_reference` corridor/exemplar support as a
CPU-supported serial/reference path.

3.3D adds CPU orchestration and staged execution.

Spec 3.3D adds deterministic backend batch orchestration for `auto`, `serial`,
and `staged`; staged is not yet a historical optimizer port.

3.3E moves current HF Torch behavior behind the contract.

Spec 3.3E adds CPU-runtime-first `hf_torch` behind the backend contract; 3.3F is
where GPU compact reduction begins.

3.3F ports GPU compact/chunked reduction after the contract exists.

Spec 3.3F is split into smaller GPU Torch migration units:

| Unit | Title | Status |
|---|---|---|
| 3.3F1 | GPU Torch Backend Detection + Dense Debug Smoke | complete once the gpu_torch dense smoke path lands |
| 3.3F2 | GPU Top-K/Tail Compact Reducer | complete once the gpu_torch top-k/tail reducer lands |
| 3.3F3 | GPU Cascaded Soft-Label Reducer | complete once the gpu_torch cascaded reducer lands |
| 3.3F4 | Chunked Vocab Reduction + Memory Metadata | complete once chunking metadata lands |
| 3.3F4.1 | Cascaded Chunking Metadata Truth Fix | complete once cascaded chunking overclaim is fixed |
| 3.3F5 | GPU Runtime Fallback / Error Hardening | complete once diagnostics and fallback hardening land |
| 3.3F6 | Dynamic Cascaded Soft Labels CPU Reference + Contract Shape | complete once the CPU reference contract shape lands |
| 3.3F7 | GPU Dynamic Cascaded Soft Labels Reducer | complete once the gpu_torch dynamic reducer lands |
| 3.3F7.1 | GPU Dynamic Cascaded Reducer Vectorization Rehearsal | complete once dynamic head selection vectorization lands |
| 3.3F8 | Corridor/Exemplar Production Schema Lock | complete once the production corridor schema lands |
| 3.3F9 | GPU Corridor/Exemplar Acceleration | complete once gpu_torch emits the F8 production schema |
| 3.3F10 | GPU Builder Integration Gate | planned |
| 3.3F11 | GPU Runtime Final Polish / Doctor Metadata | planned |

Spec 3.3F1 adds `gpu_torch` as a CUDA/MPS-detecting dense debug backend. It
does not implement compact GPU reduction or public builder migration.

Spec 3.3F2 adds the first compact GPU reducer: `gpu_torch` computes
`topk_with_tail_v0` on CUDA or MPS and transfers compact payload arrays back to
host. It does not complete cascaded reduction, chunked vocab reduction, or
public builder migration.

Spec 3.3F3 adds optimized `cascaded_soft_labels_v1` reduction for `gpu_torch`,
including GPU-computed `bucket_masses` and compact payload transfer. It does
not complete chunked vocab reduction, memory hardening, corridor acceleration,
or public builder migration.

Spec 3.3F4 adds optional vocab-axis chunking and memory/workspace metadata for
`gpu_torch` compact reducers, plus the cascaded duplicate-softmax cleanup. It
does not complete OOM recovery, runtime fallback hardening, corridor
acceleration, or public builder migration.

Spec 3.3F4.1 corrects cascaded chunking metadata: requested cascaded vocab
chunking is preserved, but current exact bucket construction does not claim
effective chunked workspace because it requires a full probability workspace.

Spec 3.3F5 hardens `gpu_torch` runtime failures with structured diagnostics,
clear missing dependency/accelerator/model-load reasons, wrapped device failure
context, and no backend-local CPU fallback.

Spec 3.3F6 adds `dynamic_cascaded_soft_labels_v1` as a CPU reference contract:
dynamic top-k explicit head plus bucketed tail, with padded mask-driven payload
shape. It does not add GPU support; Spec 3.3F7 owns the optimized GPU dynamic
cascaded reducer.

Spec 3.3F7 adds the optimized `gpu_torch` reducer for
`dynamic_cascaded_soft_labels_v1`, matching the F6 payload shape while keeping
dynamic head selection and bucketed tail reduction on Torch tensors before
compact host transfer.

Spec 3.3F7.1 is a narrow vectorization rehearsal for that dynamic reducer. It
keeps the same payload and metadata contract, vectorizes dynamic explicit-head
selection across batch/sequence positions, and preserves exact bucketed tail
semantics without claiming measured speedups.

Spec 3.3F8 locks `corridor_exemplar_v1` as a production behavioral/fingerprint
schema. It is source-policy-aware across `dense_logits`,
`cascaded_soft_labels_v1`, and `dynamic_cascaded_soft_labels_v1`, with dynamic
cascaded as the preferred future compact source. GPU corridor/exemplar
acceleration remains planned for F9.

Spec 3.3F9 implements `gpu_torch` acceleration for `corridor_exemplar_v1`
against the F8 production schema. The GPU path supports dense, fixed-cascaded,
and dynamic-cascaded exemplar source policies, emits
`compact_corridor_exemplar` metadata, and transfers compact production arrays
only. It does not migrate the public builder.

Spec 3.3F9.1 formalizes that behavior as `one_pass_candidate` capture mode:
the backend emits compact candidate data for every batch example, with
`exemplar_candidate_scope=batch_all_examples`, and does not perform final
corpus-level exemplar pruning.

The official post-F5 path finishes meaningful `gpu_torch` optimization before
TPU: F6 dynamic cascaded CPU reference, F7 GPU dynamic cascaded reducer, F7.1
dynamic reducer vectorization rehearsal, F8 corridor/exemplar production
schema lock, F9 GPU corridor/exemplar acceleration, F10 GPU builder integration
gate, F11 runtime final polish and doctor metadata, then 3.3G TPU/JAX backend
skeleton.

3.3G adds TPU/JAX shape without CUDA assumptions.

3.3H exposes backend status through CLI/doctor polish.
