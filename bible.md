# RADJAX-Tome Project Ledger

Earlier history in this root ledger was reconstructed from current repository
state because no root `bible.md` existed when Spec 3.1 landed. Existing
historical notes remain in `docs/BIBLE.md`; future spec commits should append
here unless a spec explicitly says not to.

## 2026-07-01 — Cleanup Arc Catch-Up And Spec 3.1 Cover Page

The cleanup arc from 2.14 through 2.18 is complete on `main`: archive/mainline
hygiene, public CLI happy path, shared report rendering and thin capability
script, narrowed fingerprint API boundary, and shared test fixture helpers are
all represented in the current repository state.

Spec 3.0 locked the post-cleanup roadmap and preserved the historical
optimization handoff in repository-local docs and deterministic inventory JSON.

Spec 3.1 implements the first unpacked Tome directory front door:
`cover_page.json`. Fake/offline builds now write the cover page beside existing
TeacherTextbook sidecars, public validation checks it when present, and inspect
prints cover-page summary fields. This does not implement the Spec 3.2 bundle
container, compression layer, dynamic top-k, or CPU/GPU/TPU runtime modes.

## 2026-07-01 — Spec 3.2 Tome Bundle Container

Spec 3.2 adds `.rtome` as a deterministic tar bundle for moving and storing an
unpacked Tome directory as one file. The public CLI now supports `pack`,
bundle-aware `inspect`, bundle-aware `validate`, and safe `unpack`.

The bundle is packaging only: it keeps `cover_page.json` at archive root, packs
the cover-page-listed files, validates hashes and sizes without extraction, and
does not impose a compression requirement. Dynamic top-k and backend runtime
optimization remain future Spec 3 arcs.

## 2026-07-01 — Spec 3.3A Runtime Mode Capability Model

Spec 3.3A defines the runtime mode capability model before backend migration:
`cpu`, `cpu_gpu`, and `cpu_tpu` runtime modes; `auto / serial / staged` CPU
orchestration modes; target policies for `dense_logits`, `topk_with_tail_v0`,
`cascaded_soft_labels_v1`, and `corridor_exemplar_v1`; and a deterministic
runtime capability matrix.

This is intentionally vocabulary, documentation, and inventory only. It does
not implement the backend contract, migrate the active builders, port GPU
optimization, add TPU support, change target shards, change `cover_page.json`,
or change `.rtome` bundles.

## 2026-07-01 — Spec 3.3B Backend Contract And Registry Skeleton

Spec 3.3B adds the backend contract and registry skeleton for future
teacher-side Tome target emission backends. The new contract vocabulary includes
`TeacherBackendConfig`, `TeacherBatchInput`, `TeacherEmissionResult`, and
`BackendCapability`, with a deterministic registry for creating backends and
listing capabilities.

The default registered proof backend is `fake_numpy`, which emits deterministic
`dense_logits` through the new contract. There is no builder migration yet: the
active public builder behavior, HF path, GPU optimization, TPU support, target
shards, `cover_page.json`, and `.rtome` bundle behavior remain unchanged.

## 2026-07-01 — Spec 3.3C CPU Reference Backend

Spec 3.3C adds the CPU reference backend, `cpu_reference`, as the serial/reference
correctness baseline behind the backend contract. It emits deterministic
payloads for `dense_logits`, `topk_with_tail_v0`, and
`cascaded_soft_labels_v1` without adding heavy runtime dependencies.

There is no public builder migration, no staged orchestration, and no GPU/TPU
implementation in this spec. The backend is intentionally boring: it exists so
future accelerated runtimes have a deterministic CPU target to compare against.

## 2026-07-01 — Spec 3.3C.1 CPU Corridor / Exemplar Reference Policy

Spec 3.3C.1 corrects the `cpu_reference` capability model for
`corridor_exemplar_v1`. Corridor/exemplar generation is now represented as a
CPU-supported serial/reference path through the backend contract, with
deterministic corridor summaries and high-entropy exemplar selections.

This is a capability matrix correction and CPU reference implementation only:
there is no builder migration, no staged orchestration, and no GPU/TPU
implementation.

## 2026-07-01 — Spec 3.3D CPU Orchestration Modes

Spec 3.3D adds CPU orchestration modes for backend emission:
`auto / serial / staged`. The new backend batch runner preserves deterministic
ordering by `sequence_id` and records run-level metadata for requested/effective
orchestration mode, batch counts, example counts, sequence ranges, and auto
resolution.

This does not migrate the public builder, port HF/GPU/TPU runtimes, or claim
that staged mode is performance optimized. It creates the scheduling lane for
future backend work.

## 2026-07-01 — Spec 3.3E HF Torch Backend Behind The Contract

Spec 3.3E adds `hf_torch` as an HF Torch backend implementing
`TeacherEmissionBackend`. It keeps torch/transformers imports lazy and is CPU
runtime first. When optional local HF dependencies and model files are
available, it can emit `dense_logits`, `topk_with_tail_v0`, and
`cascaded_soft_labels_v1` through the backend contract.

This does not implement GPU compact optimization, CUDA/MPS acceleration,
TPU/JAX, or public builder migration.

## 2026-07-01 — Spec 3.3F1 GPU Torch Detection And Dense Debug Smoke

Spec 3.3F1 starts the GPU Torch sub-roadmap by adding `gpu_torch` as a
`TeacherEmissionBackend` with `runtime_mode=cpu_gpu`. It lazily detects Torch
accelerators in CUDA-then-MPS order, loads HF Torch dependencies only when
availability or emission is requested, and can emit `dense_logits` as a debug
smoke path on an available accelerator.

This is deliberately not the compact GPU reducer. Dense logits are transferred
back to host and metadata records that the path is unoptimized, debug-oriented,
and not using compact reduction. `topk_with_tail_v0`,
`cascaded_soft_labels_v1`, and `corridor_exemplar_v1` remain future GPU
reduction work, with historical QRWKV-XLA code only as migration reference.

## 2026-07-01 — Spec 3.3F2 GPU Top-K / Tail Compact Reducer

Spec 3.3F2 adds the first real `gpu_torch` compact reducer:
`topk_with_tail_v0`. The backend keeps HF Torch logits on the selected CUDA or
MPS device, computes top-k probabilities, log-probabilities, top mass, tail
mass, and teacher entropy as Torch tensors, then transfers only the compact
payload arrays back to host as a compact payload.

The `dense_logits` debug path still transfers full logits to host and remains
unoptimized. This spec does not implement cascaded GPU reduction, chunked vocab
reduction, public builder migration, or TPU/JAX support.

## 2026-07-02 — Spec 3.3F3 GPU Cascaded Soft-Label Reducer

Spec 3.3F3 adds `cascaded_soft_labels_v1` as an optimized `gpu_torch` compact
path. The GPU compact cascaded reducer builds on the top-k/tail reducer,
computes `bucket_masses` on the selected CUDA or MPS device with contiguous
descending tail-probability buckets, and transfers only compact payload arrays
back to host.

This does not implement chunked vocab reduction, GPU corridor/exemplar
acceleration, public builder migration, or TPU/JAX support.

## 2026-07-02 — Spec 3.3F4 Chunked Vocab Reduction And Memory Metadata

Spec 3.3F4 adds optional vocab chunking to `gpu_torch` compact reducers and
records memory metadata, including `estimated_reducer_workspace_bytes`, dense
equivalent byte estimates, compact transfer bytes, requested/effective chunk
size, and chunk counts.

Compact top-k/tail and cascaded soft-label reduction still avoid full dense
host transfer. The cascaded path now reuses the shared probability workspace
instead of duplicating full-vocab softmax/probability work, while preserving
`bucket_masses`.

This does not migrate the public builder, implement GPU corridor/exemplar
acceleration, claim measured peak GPU memory, or add TPU/JAX support.

## 2026-07-02 — Spec 3.3F4.1 Cascaded Chunking Metadata Truth Fix

Spec 3.3F4.1 fixes metadata truth for `cascaded_soft_labels_v1`. When vocab
chunking is requested for cascaded exact bucket construction, metadata now
preserves `vocab_chunking_requested=true` but records `vocab_chunking_used=false`
with `exact_bucket_policy_requires_full_probability_workspace`, because the
current exact `bucket_masses` path needs a full probability workspace on device.

Top-k chunking remains effective. This metadata truth fix does not migrate the
public builder, add TPU/JAX support, or change target artifacts.

## 2026-07-02 — Spec 3.3F5 GPU Runtime Fallback / Error Hardening

Spec 3.3F5 hardens `gpu_torch` runtime diagnostics and fallback behavior.
Diagnostics now report missing `torch`, missing `transformers`, no CUDA/MPS
accelerator, missing local model/tokenizer files, unsupported targets, and
invalid chunk config without requiring network downloads.

Explicit `gpu_torch` / `cpu_gpu` execution still has no silent CPU fallback.
`fallback_policy=auto` is recorded as an orchestrator-level signal, not a
backend-local path to `hf_torch` or `cpu_reference`. Device transfer, model
forward, reduction, and compact host-transfer failures are wrapped with
accelerator context. This does not migrate the public builder, add TPU/JAX, or
implement GPU corridor/exemplar acceleration.

## 2026-07-02 — Spec 3.3F6 Dynamic Cascaded CPU Reference Contract

Spec 3.3F6 adds `dynamic_cascaded_soft_labels_v1` as a CPU reference contract
shape. The payload is dynamic top-k explicit head plus bucketed tail, with
`top_selection_mask` marking selected head slots and `effective_top_k`
recording the per-position selected count.

The selection policy is `mass_threshold_v1`: choose enough sorted probability
mass to meet the configured threshold, bounded by configured dynamic min/max K,
then bucket the non-selected tail. This creates the reference oracle for future
`gpu_torch` F7 dynamic cascaded optimization and gives future
corridor/exemplar schema work a possible exemplar source policy. It does not
migrate the public builder, add TPU/JAX, or implement corridor/exemplar
production schema.

## 2026-07-02 — Spec 3.3F7 GPU Dynamic Cascaded Reducer

Spec 3.3F7 adds optimized `gpu_torch` support for
`dynamic_cascaded_soft_labels_v1`. The reducer computes the dynamic top-k
explicit head and bucketed tail with Torch tensors, using the
`mass_threshold_v1` policy and preserving `top_selection_mask` plus
`effective_top_k` in the compact payload.

This path uses compact payload transfer only:
`dense_logits_transferred_to_host=false` for dynamic cascaded emission. Dynamic
cascaded can later serve as a corridor/exemplar source policy, but this spec
does not migrate the public builder, add TPU/JAX, or implement
corridor/exemplar production schema.

## 2026-07-02 — Spec 3.3F7.1 GPU Dynamic Cascaded Vectorization Rehearsal

Spec 3.3F7.1 keeps `gpu_torch` support for
`dynamic_cascaded_soft_labels_v1` on the same payload contract and same metadata
contract while adding vectorized dynamic head selection across batch/sequence
positions. The bucketed tail is preserved with exact contiguous descending tail
probability masses, and dynamic cascaded emission still uses no dense host
transfer.

This is only a vectorization rehearsal. It does not implement
corridor/exemplar production schema or acceleration, does not migrate the
public builder, and does not add TPU/JAX support.

## 2026-07-02 — Spec 3.3F8 Corridor/Exemplar Production Schema Lock

Spec 3.3F8 locks `corridor_exemplar_v1` as a production behavioral/fingerprint
schema. CPU reference emissions now record
`production_corridor_schema=true`, `corridor_payload_flavor=production_v1`, and
`historical_parity_claimed=false`, with `historical_reference_source` set to
`cpu_reference_proxy` for the deterministic reference implementation.

The schema is source-policy-aware through `exemplar_source_policy`. Allowed
source policies are `dense_logits`, `cascaded_soft_labels_v1`, and
`dynamic_cascaded_soft_labels_v1`, with dynamic cascaded serving as the
preferred future compact source for optimized work. The production payload
includes source policy summary and schema metadata alongside corridor,
exemplar, mode, and source arrays.

This does not implement GPU corridor/exemplar acceleration; future `gpu_torch`
F9 owns that. It also does not migrate the public builder and does not add
TPU/JAX support.
