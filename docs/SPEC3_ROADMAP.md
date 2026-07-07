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
| 3.3F10 | GPU Builder Integration Gate | complete once backend-routed builder artifacts land |
| 3.3F10.1 | Multi-Leaderboard Exemplar Selection Harness | complete once shared selector manifests land |
| 3.3F10.1.1 | Rank-Aware Leaderboard Deduplication Backfill | complete once rank-aware dedupe and backfill land |
| 3.3F11 | GPU Runtime Final Polish / Doctor Metadata | complete once runtime doctor and artifact metadata sanity reports land |

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

Spec 3.3F9.2 adds `two_pass_sparse_exemplar` as an explicit storage/transfer
saving mode. The first `score_pass` emits [B]-scale score summaries for all
examples; the `selected_exemplar_pass` reruns chosen examples and emits the F8
production schema. This does not migrate the public builder or add TPU/JAX
support.

Spec 3.3F9.3 adds `auto` exemplar capture selection. Manual
`one_pass_candidate` and `two_pass_sparse_exemplar` overrides win; auto records
estimated one-pass candidate bytes, two-pass score bytes, selected-pass bytes,
expected selected fraction, disk budget when known, missing inputs, and an
explicit policy reason in metadata.

Spec 3.3F9.4 adds GPU batch size policy guardrails before public builder
integration. `preset`, `custom`, and `auto` modes resolve an effective batch
size without changing backend batch-in/batch-out behavior. Auto uses
`exponential_probe_v1` synthetic probe results to choose the last good batch,
custom values above 64 are allowed with warning metadata, and
estimated-vs-measured byte caveats are recorded. This is single-device only,
future-reserves multidevice vocabulary, and does not migrate the builder or add
TPU/JAX support.

Spec 3.3F10 adds the GPU Builder Integration Gate. The public build command can
explicitly route through the backend contract, including `gpu_torch` with
`runtime_mode=cpu_gpu`, and write artifacts for
`dynamic_cascaded_soft_labels_v1`, `corridor_exemplar_v1`, and
`corridor_exemplar_score_pass_v1` schema recognition. Metadata propagation is
the gate: runtime/backend/fallback/capability, optimized-path, GPU compact,
exemplar-capture, auto-policy, and batch-size policy fields flow into artifact
metadata and cover pages. It preserves no silent CPU fallback and does not add
a production global two-pass selector, real auto batch probing, builder hydra
behavior, or TPU/JAX support.

Spec 3.3F10.1 adds `multi_leaderboard_exemplar_selector_v1`, a
capture-mode-agnostic selector shared by `one_pass_candidate` and
`two_pass_sparse_exemplar`. Candidates compete for bounded leaderboards, then
the union of winners is deduplicated into `exemplar_selection_manifest.json`.
Path A fulfills with `select_from_existing_capture` for debug/small-run
inspection. Path B fulfills with `rerun_selected_capture` as a
production-shaped rerun requisition for selected examples. This does not add
semantic embeddings, a utility-calibrated selector, TPU/JAX, or backend
capability status changes.

Spec 3.3F10.1.1 refines selector deduplication with
`rank_aware_board_assignment_with_backfill_v1`. A duplicate candidate is kept
on the board where it has the strongest rank, removed from weaker boards, and
those boards backfill from runner-up pools when possible. Budgets are applied
after assignment using score-aware rank/score ordering instead of alphabetical
example ID ordering.

Spec 3.3F11 adds runtime final polish and doctor metadata without changing
backend math. `radjax-tome doctor` emits a `runtime_doctor_report_v1`
preflight report with dependency, accelerator, capability, fallback, failure,
and remediation fields. `inspect --metadata-sanity` and
`validate --metadata-sanity --write-report` emit
`artifact_metadata_sanity_report_v1` summaries for backend routing, compact
GPU metadata, exemplar capture, selector metadata, and batch-size metadata.
F11 adds no new reducer math, no new selector policy, no real auto batch
probing, no production global selector, no multidevice scheduler, and no
TPU/JAX.

Spec 4.7 adds `radjax-tome production-build` as the one-command production GPU
Tome path. It composes existing doctor, planner, streaming builder,
validation, cover-page, and optional parity surfaces into a single local-only,
no-download workflow with `production_build_report_v1`.

Spec 4.7.a adds a separate experimental `multi-gpu-path-b` harness for Path B
candidate scheduling. It requires explicit devices, keeps worker outputs
disjoint, merges candidate records deterministically on CPU, and keeps
single-GPU `production-build` as the recommended path.

The official post-F5 path finishes meaningful `gpu_torch` optimization before
TPU: F6 dynamic cascaded CPU reference, F7 GPU dynamic cascaded reducer, F7.1
dynamic reducer vectorization rehearsal, F8 corridor/exemplar production
schema lock, F9 GPU corridor/exemplar acceleration, F10 GPU builder integration
gate, F10.1 multi-leaderboard exemplar selection, F10.1.1 rank-aware
deduplication backfill, F11 runtime final polish and doctor metadata, F12
optional parity/deathmatch harness, then 3.3G TPU/JAX backend skeleton.

3.3G adds TPU/JAX shape without CUDA assumptions.

3.3H exposes backend status through CLI/doctor polish.

## Phase 4

Phase 4 is the Production GPU Tome Pipeline. Spec 4.1 starts that phase with a
local deterministic corpus builder and provenance contract. It writes
`corpus.jsonl`, `corpus_manifest.json`, and `corpus_build_report.json`, records
content/source/corpus/manifest hashes, and lets generated Tomes cite
`source_corpus_hash` plus manifest provenance.

Spec 4.1 does not scrape the internet, clone GitHub repositories, download
teacher models, add semantic filtering, plan GPU runs, or touch TPU/JAX.

Spec 4.1.1 cleans up corpus format truth: `.json` is intentionally unsupported
until a structured JSON import spec exists, `.jsonl` text rows remain
supported, `created_at` is a real UTC timestamp, and
`manifest_hash_policy=exclude_self_hash_and_created_at_v1` keeps manifest
hashes stable by excluding both `manifest_hash` and `created_at`.

Spec 4.2 adds local teacher model provenance and setup UX. `radjax-tome model
inspect` writes `teacher_model_provenance_v1` from local files, hashes config,
tokenizer, and weight files, infers Hugging Face cache identity only from local
path shape, supports user-declared identity, and lets Tome builds cite the
validated sidecar. It does not download teacher models, perform network
verification, change backend emission capability statuses, plan GPU runs, or
touch TPU/JAX.

Spec 4.3 adds a post-build parity / A-B deathmatch harness. `radjax-tome
parity` compares two generated Tome artifact directories, writes
`tome_parity_report_v1`, and checks required sidecars, target-store metadata,
shard arrays, finite values, numeric tolerance metrics, selector manifests,
corpus provenance, teacher model provenance, metadata truth, and forbidden
claims. It does not change backend math, selector behavior, GPU planning,
model acquisition, network verification, or TPU/JAX support.

Spec 4.4 improves GPU install and dependency UX. `gpu-teacher` names the
HF/PyTorch GPU teacher optional dependency path, `docs/GPU_INSTALL.md` explains
fresh venv setup and CUDA wheel caveats, and `radjax-tome doctor` now reports
GPU install diagnostics plus actionable remediation. It does not change
backend capability statuses, reducer math, selector behavior, model download
policy, GPU run planning, or TPU/JAX support.

Spec 4.5 adds `radjax-tome plan` and `gpu_run_plan_v1` for GPU run preflight.
The planner reuses doctor diagnostics, validates supplied corpus/model
provenance, writes rough memory/artifact estimates, records capture-mode
implications, and can perform bounded tiny auto-batch probes for `gpu_torch`.
It does not run production builds, download models, perform network
verification, add streaming/resume, add multidevice scheduling, or touch
TPU/JAX.

Spec 4.6 adds streaming backend build orchestration and safe resume.
`radjax-tome build --streaming` reads corpus JSONL incrementally, writes shards
atomically, records `run_manifest.json` plus `progress_log.jsonl`, writes
`failure_report.json` on failure, and resumes only when config and completed
shard hashes still match. It refuses corpus-global exemplar selection rather
than pretending a streaming global selector ran. It does not change backend
math, selector behavior, model download policy, network policy, multidevice
scheduling, or TPU/JAX support.
