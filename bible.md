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
