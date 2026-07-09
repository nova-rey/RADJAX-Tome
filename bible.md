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

## 2026-07-02 — Spec 3.3F9 GPU Corridor/Exemplar Acceleration

Spec 3.3F9 implements `gpu_torch` support for `corridor_exemplar_v1` against
the F8 production schema. Successful GPU corridor emission records
`gpu_reduction_mode=compact_corridor_exemplar`,
`dense_logits_transferred_to_host=false`, `historical_parity_claimed=false`,
and `historical_reference_source=gpu_torch_production`.

The GPU path is source-policy-aware through `exemplar_source_policy` and
supports `dense_logits`, `cascaded_soft_labels_v1`, and
`dynamic_cascaded_soft_labels_v1`, with dynamic cascaded remaining the default
compact source. It transfers compact production arrays only and builds record
summaries after transfer.

This does not migrate the public builder and does not add TPU/JAX support.

## 2026-07-02 — Spec 3.3F9.1 One-Pass Candidate Formalization

Spec 3.3F9.1 formalizes current `corridor_exemplar_v1` behavior as
`one_pass_candidate`. CPU reference and `gpu_torch` backends emit compact
candidate data for every example in the batch and record
`exemplar_candidate_scope=batch_all_examples`.

This mode is not final corpus-level exemplar pruning:
`corpus_level_exemplar_finalization=false` and
`requires_second_pass_for_final_exemplars=false`. It does not implement
two-pass sparse exemplar capture, does not migrate the public builder, and does
not add TPU/JAX support.

## 2026-07-02 — Spec 3.3F9.2 Two-Pass Sparse Exemplar Capture

Spec 3.3F9.2 adds `two_pass_sparse_exemplar` as a storage/transfer-saving
corridor-exemplar capture mode. Pass 1 emits [B]-scale `score_pass` summaries
for all examples; pass 2 reruns selected examples and emits F8
production-shaped exemplar payloads with `selected_exemplar_pass` metadata.

One-pass mode remains available. No auto policy yet, no builder migration, no
TPU/JAX.

## 2026-07-02 — Spec 3.3F9.3 Exemplar Capture Auto Policy

Spec 3.3F9.3 adds `exemplar_capture_mode=auto` for choosing between
`one_pass_candidate` and `two_pass_sparse_exemplar`. Manual capture-mode
settings still win.

Auto records `exemplar_capture_policy=auto_exemplar_capture_policy_v1`,
`manual_override_used`, `auto_policy_reason`,
`estimated_one_pass_candidate_bytes`, `estimated_two_pass_score_bytes`,
`estimated_two_pass_selected_bytes`, `estimated_two_pass_total_bytes`,
`expected_selected_exemplar_fraction`, `available_disk_budget_bytes`, and
`auto_policy_inputs_missing`. This does not change reducer semantics, migrate
the public builder, or add TPU/JAX.

## 2026-07-02 — Spec 3.3F9.4 GPU Batch Size Policy Guardrails

Spec 3.3F9.4 adds a gpu batch size policy with
`gpu_batch_size_policy_v1` guardrail metadata and `preset`, `custom`, and
`auto` modes. Presets are
bounded to 1/2/4/8/16/32/64, `custom` preserves the requested batch size, and
`custom >64` is allowed with warning metadata.

Auto uses `exponential_probe_v1` synthetic probe results to choose the last
good batch, with optional midpoint refinement. Metadata records the estimated
vs measured bytes caveat, actual compact payload bytes when arrays are
available, and no measured GPU peak memory claim unless measured.

F9.4 preserves batch-in/batch-out backend behavior, remains single-device only,
future-reserves multidevice vocabulary, does not migrate the public builder,
and does not add TPU/JAX.

## 2026-07-02 — Spec 3.3F10 GPU Builder Integration Gate

Spec 3.3F10 adds the GPU Builder Integration Gate. The builder now supports
gpu_torch builder routing through the `TeacherEmissionBackend` contract for
explicit `teacher_backend=gpu_torch` and `runtime_mode=cpu_gpu` requests, with
no silent CPU fallback.

The artifact path recognizes `dynamic_cascaded_soft_labels_v1`
artifact/schema recognition, `corridor_exemplar_v1` artifact/schema
recognition, and `corridor_exemplar_score_pass_v1` score-pass artifacts.
Metadata propagation preserves runtime/backend/fallback/capability fields,
optimized-path evidence, GPU compact fields, exemplar-capture metadata
propagation, auto-policy fields, and batch-size metadata propagation into the
artifact metadata and cover page.

This is not a production global two-pass selector, no real auto batch probing,
no builder hydra, and no TPU/JAX. It is a builder integration gate for
backend-routed artifacts, not a new reducer or production readiness claim.

## 2026-07-07 — Spec 3.3F10.1 Multi-Leaderboard Exemplar Selection Harness

Spec 3.3F10.1 adds `multi_leaderboard_exemplar_selector_v1`, a
capture-mode-agnostic selector shared by `one_pass_candidate` and
`two_pass_sparse_exemplar`. Both paths produce common candidate records, feed
bounded leaderboards, and deduplicate winners into
`exemplar_selection_manifest.json` while preserving winning boards and
selection reasons.

Only fulfillment differs. Path A uses `select_from_existing_capture` for
debug/small-run inspection and can retain the full rich candidate artifact.
Path B uses `rerun_selected_capture` as a production-shaped rerun requisition
for selected examples.

This adds no semantic embeddings, no utility-calibrated selector, and no
TPU/JAX work. It also does not change reducer math or backend capability
statuses.

## 2026-07-07 — Spec 3.3F10.1.1 Rank-Aware Leaderboard Deduplication Backfill

Spec 3.3F10.1.1 refines `multi_leaderboard_exemplar_selector_v1` with
`rank_aware_board_assignment_with_backfill_v1`. Duplicate suppression now
assigns a candidate to the board where it ranks strongest, removes it from
weaker boards, and lets those boards perform runner-up backfill from retained
candidate pools.

The same `exemplar_selection_manifest.json` records assigned boards,
suppressed duplicate boards, rank-by-board evidence, duplicate/backfill counts,
and score-aware budget trimming through `score_aware_assigned_board_rank_v1`.

This adds no semantic embeddings, no utility-calibrated selector, no production
global selector, and no TPU/JAX work. Backend reducer math and Path A / Path B
capture semantics stay unchanged.

## 2026-07-07 — Spec 3.3F11 GPU Runtime Final Polish / Doctor Metadata

Spec 3.3F11 adds GPU Runtime Final Polish around the existing backend
contract. `radjax-tome doctor` now produces a runtime doctor preflight report
with backend availability summary, dependency/accelerator status,
`can_emit`, failure stage/reason, fallback fields, and remediation hints.

Artifacts can now be inspected with an artifact metadata sanity report. The
report summarizes backend/effective-backend routing, compact GPU metadata,
exemplar capture state, selector metadata sanity, and batch-size metadata sanity.
It flags contradictory claims such as a score pass pretending to be
final production corridor output, a gpu_torch request with no explained
fallback, future selector claims, or multidevice metadata without
`single_device`.

This is a report/doctor polish step only: no new reducer math, no new selector policy,
no real auto batch probing, no production global selector, no multidevice, and
no TPU/JAX.

## 2026-07-07 — Spec 4.1 Corpus Builder and Provenance Contract

Spec 4.1 begins Phase 4, the Production GPU Tome Pipeline, by adding a
first-class local corpus builder and provenance contract. The builder turns
local source files into deterministic normalized corpus records, writes
`corpus.jsonl`, `corpus_manifest.json`, and `corpus_build_report.json`, and
computes content hashes, source hashes, `corpus_hash`, and `manifest_hash`.

The corpus artifact records normalization policy, chunking policy,
deduplication policy, source discovery policy, source counts, example counts,
and source summaries. `radjax-tome corpus build`, `radjax-tome corpus inspect`,
and `radjax-tome corpus validate` expose the workflow through the public CLI.

Tome builds can now accept `--corpus-manifest` and record
`source_corpus_hash`, `source_corpus_manifest_hash`, corpus schema, corpus
counts, normalization policy, chunking policy, deduplication policy, and
manifest path in target metadata, `teacher_manifest.json`,
`emission_config.json`, and `cover_page.json`.

Spec 4.1 does not scrape the internet, does not clone GitHub, does not
download teacher models, does not add semantic filtering, does not implement
license/legal judgment, does not plan GPU runs, and does not touch TPU/JAX.

## 2026-07-07 — Spec 4.1.1 Corpus Format Truth Cleanup

Spec 4.1.1 removes ambiguous `.json` corpus source support. The corpus builder
now supports `.txt`, `.md`, `.markdown`, `.py`, and `.jsonl` rows with a string
`text` field. Structured `.json` import is intentionally not supported yet
because arbitrary JSON extraction needs a separate contract.

Corpus manifests now record a real UTC `created_at` and
`manifest_hash_policy=exclude_self_hash_and_created_at_v1`. `corpus_hash`
still hashes canonical `corpus.jsonl` bytes. `manifest_hash` hashes canonical
manifest JSON while excluding both `manifest_hash` and `created_at`, so
identical corpus content and stable build configuration can retain the same
manifest hash across different build times.

This patch does not add structured JSON import, internet scraping, GitHub
cloning, model downloading, teacher emission changes, GPU work, JAX, or TPU.

## 2026-07-07 — Spec 4.2 Teacher Model Provenance and Setup UX

Spec 4.2 adds first-class teacher model provenance for local teacher files.
`radjax-tome model inspect` writes `teacher_model_provenance_v1`, hashing
recognized config, tokenizer, and weight files with per-file records plus
`config_hash`, `tokenizer_hash`, `weights_hash`, and `model_directory_hash`.
`radjax-tome model validate` recomputes those hashes and rejects tampered local
files.

The provenance sidecar records identity confidence honestly. Friendly identity
may be verified from local config, inferred from a local Hugging Face cache
snapshot path, declared by the user, or left unknown. HF repo/revision inference
is local path inference only, not upstream or network verification.

Tome builds can now accept `--teacher-model-provenance` and record a compact
teacher model provenance summary in target metadata, `teacher_manifest.json`,
`emission_config.json`, and `cover_page.json`; full file inventories remain in
the sidecar.

This patch does not silently download teacher models, does not perform network
verification, does not add GPU run planning, does not add parity/deathmatch
harnesses, and does not touch JAX or TPU.

## 2026-07-07 — Spec 4.3 Parity / A-B Deathmatch Harness

Spec 4.3 adds a post-build Tome parity harness. `radjax-tome parity` compares
two generated Tome artifact directories and writes `tome_parity_report_v1` to
`parity_report.json`.

The report checks required sidecars, target-store metadata, shard array fields,
array shapes and dtypes, finite floating values, numeric tolerance metrics,
selector manifest policy/truth fields, selected exemplar overlap, corpus
provenance, teacher model provenance, metadata sanity, cover-page linkage, and
forbidden truth claims.

Parity does not require byte-identical artifacts or exact floating equality.
Floating arrays record max/mean absolute difference, max relative difference,
and within-tolerance fraction under declared `rtol`/`atol`. Shape, dtype,
schema, sidecar, finite-value, provenance hash, and metadata-truth violations
remain hard failures.

This patch does not change backend reducer math, does not change selector
behavior, does not download teacher models, does not perform network
verification, does not add GPU run planning, and does not touch JAX or TPU.

## 2026-07-07 — Spec 4.4 GPU Install / Dependency UX

Spec 4.4 improves the setup and diagnostic path for GPU teacher emission. The
package now exposes a `gpu-teacher` optional dependency extra, currently
matching the `teacher-hf` Torch/Transformers dependencies while naming the
GPU-oriented workflow explicitly.

`radjax-tome doctor` now includes additive GPU install diagnostics in
`runtime_doctor_report_v1`: Python/platform status, RADJAX-Tome import status,
Torch and Transformers availability/version, CUDA availability, CUDA device
count and names, Torch CUDA version, MPS availability, JAX availability, and
recommended install extra. Doctor summaries include actionable remediation
hints and recommended next commands for model provenance, corpus build, GPU
build, and parity comparison.

`docs/GPU_INSTALL.md` documents fresh venv setup, editable installs,
`teacher-hf` and `gpu-teacher` extras, PyTorch CUDA wheel caveats, doctor
usage, local teacher model provenance, corpus building, fake smokes, tiny local
GPU smokes, and parity comparison.

This patch does not install NVIDIA drivers, does not silently download teacher
models, does not perform network model verification, does not add real auto
batch probing, does not add GPU run planning, does not change backend reducer
math, does not change selector behavior, and does not touch JAX or TPU.

## 2026-07-07 — Spec 4.5 Real GPU Run Planner and Auto Batch Probe

Spec 4.5 adds `radjax-tome plan`, which writes `gpu_run_plan_v1` to
`run_plan.json` before a large GPU Tome build. The plan includes doctor-derived
environment diagnostics, dataset summary, corpus manifest validation when
provided, teacher model provenance validation when provided, requested and
resolved GPU batch policy, memory estimates, artifact estimates, capture-mode
implications, recommended build command, and explicit non-claims.

When `gpu_batch_size_mode=auto` is requested for `gpu_torch`, the planner runs
bounded tiny local probes over exponential candidate batch sizes, using the
same local model/tokenizer load path and target reducer path. It records
per-candidate pass/fail details, observed memory fields when available, the
largest passing batch size, the first failing batch size, and the selected
effective batch size.

The planner treats missing provenance as a warning by default and invalid
supplied provenance as a blocker. Memory and artifact estimates are explicitly
rough planning estimates, not contractual output sizes.

This patch does not run a production build, does not download models, does not
perform network verification, does not add streaming/resume, does not add
multidevice scheduling, does not change backend reducer math, does not change
selector behavior, and does not touch JAX or TPU.

## 2026-07-07 — Spec 4.5.1 Run Planner Hash Truth Fix

Spec 4.5.1 fixes run-planner corpus hash truth. `radjax-tome plan` now compares
the supplied `corpus.jsonl` hash to `corpus_manifest.json` using the same
`sha256:<hex>` string format as the 4.1 corpus builder, so valid corpus
artifacts are not falsely rejected.

Corpus provenance status now fails when the supplied manifest hash does not
match the dataset, instead of reporting `corpus_provenance.status=pass` while
also emitting a corpus blocker. Failed auto batch probes with no passing
candidate no longer present a fallback effective batch as runnable, and their
recommended command is omitted.

The generic rough-estimate caveat moved from a warning into `estimate_notes`;
estimate sections still record `estimate_confidence=rough`.

## 2026-07-07 — Spec 4.6 Streaming Large-Run Builder and Resume

Spec 4.6 adds a streaming backend build path behind
`radjax-tome build --streaming`. The builder reads corpus JSONL incrementally,
preserves corpus order, emits backend batches into bounded shard groups, and
writes each shard through a temporary file plus fsync and rename before marking
it complete.

Streaming builds write `run_manifest.json` with
`streaming_run_manifest_v1`, `progress_log.jsonl` with append-only run/shard
events, and `failure_report.json` on failure. Completed shard records include
example ranges and final shard `sha256:<hex>` hashes.

Resume is enabled with `--resume`. It verifies `resume_config_hash`, completed
shard existence, completed shard hashes, and removes stale temporary shard
files before continuing from the next incomplete shard. Config, dataset, or
corpus drift is refused. Completed shards are preserved after failure.

Streaming metadata records `streaming_build=true`, `resume_supported=true`,
`run_manifest_path`, `progress_log_path`, shard size, completed example count,
`resume_config_hash`, and atomic write policy. The cover page includes a
streaming summary. The streaming path refuses corpus-global exemplar selection
for now rather than claiming selector finalization that did not happen.

This patch does not add one-command production orchestration, does not change
backend reducer math, does not change selector behavior, does not download
models, does not perform network verification, does not add multidevice
scheduling, and does not touch JAX or TPU.

## 2026-07-07 — Spec 4.7 One-Command Production GPU Tome Build

Spec 4.7 adds `radjax-tome production-build` as the high-level production GPU
Tome command. It validates corpus and teacher-model provenance, runs doctor and
planner preflights, writes `run_plan.json`, passes the effective batch size
into the streaming builder, validates the artifact, writes `cover_page.json`,
optionally runs post-build parity, and emits `production_build_report_v1`.

The production path defaults to `gpu_torch`, `cpu_gpu`,
`corridor_exemplar_v1`, streaming output, resumability, strict local files, no
downloads, and error-on-fallback behavior. This is orchestration and reporting
only: it does not add new reducer math, multidevice scheduling, TPU/JAX, model
downloads, network verification, or silent CPU fallback.

## 2026-07-07 — Spec 4.7.1 Production Resume / Fail-Fast Truth Polish

Spec 4.7.1 fixes the completed-resume production path. When
`production-build --resume` finds a complete run manifest and the existing
artifact validates, it now writes `production_build_report.json` and returns
`pass` before doctor, planner, or streaming build reruns. If the completed
artifact is invalid, it fails from validation blockers without invoking the
planner or builder.

The low-level build CLI keeps `--fail-fast` hidden from users because there is
no distinct non-fail-fast continuation mode to advertise or record. This patch
does not add new production behavior, downloads, network verification,
multidevice scheduling, TPU/JAX support, reducer math changes, or selector
behavior changes.

## 2026-07-07 — Spec 4.7.a Experimental Multi-GPU Path B Candidate Harness

Spec 4.7.a adds `radjax-tome multi-gpu-path-b` as an opt-in experimental Path B
candidate scheduling harness. It requires explicit device IDs, assigns shard
ranges round-robin to candidate workers, writes worker-local outputs, keeps the
coordinator in charge of `multi_gpu_worker_manifest.json` and
`multi_gpu_path_b_report.json`, and merges candidate records deterministically
on CPU.

The accepted 4.7.a path is a fake-worker scheduler harness for GPU-free testing
and report/manifest truth. Single-GPU `production-build` remains the
recommended production path. This patch does not add DDP, model parallelism,
combined VRAM, network verification, model downloads, TPU/JAX support, full
multi-GPU burn validation, reducer math changes, or selector scoring changes.

## 2026-07-08 — HF Dry-Run Heavy Import Test Isolation

The HF specimen dry-run test now asserts the actual contract: dry-run execution
must not introduce new heavy imports with `jax`, `torch`, or `transformers`
prefixes. The assertion is isolated around the dry-run call so earlier tests or
test harness setup that already imported a heavy module do not create a false
failure. Production code is unchanged.

## 2026-07-08 — P4.8B Selected-Only Exemplar Delivery Harness

Spec P4.8B adds an opt-in selected-only delivery harness to
`radjax-tome production-build` for `corridor_exemplar_v1`. The new CLI flags
select Path A (`one_pass_pruned_candidate`) or Path B
(`two_pass_rerun_selected`), enable selection, size leaderboards and budgets,
and explicitly control non-selected exemplar payload retention.

Selected-only runs materialize broad corridor evidence, shared leaderboards,
`leaderboards/selected_exemplars.json`, compressed selected exemplar payload
shards, and `delivery_report.json`. Production reports now surface
`delivery_path`, `num_examples_scored`, `num_selected_exemplars`,
`selected_exemplar_payload_retained`, and
`non_selected_exemplar_payload_retained`.

Validation now fails selected-only artifacts that claim non-selected payload
retention, omit selected compressed payload fields, select zero exemplars when
selection is enabled, report invalid Path B rerun counts, or retain temporary
candidate payload directories. `radjax-tome exemplar-delivery-parity` compares
Path A and Path B selected IDs, positions, ranks, scores, mode keys, selected
payload shapes, retained bytes, rerun counts, and retention status.

Defaults remain unchanged unless exemplar selection is explicitly enabled. This
patch does not modify the experimental `multi-gpu-path-b` harness, does not
claim student training quality, does not retain dense logits, and does not add
silent CPU fallback.

## 2026-07-08 — P4.8B Selected Rerun Payload Correction

The selected-only delivery harness now treats Path B selected exemplar payloads
as backend output, not local synthesis. After the score/corridor pass selects
winners, production-build reruns the configured teacher backend only for the
selected example IDs, using the planner-selected effective batch size, and
requests `dynamic_cascaded_soft_labels_v1` emission for the selected pass.

Final selected exemplar payload shards slice `top_token_ids`, `top_log_probs`,
`top_probs`, `top_selection_mask`, `effective_top_k`, `top_mass`, `tail_mass`,
`bucket_masses`, and `teacher_entropy` from that backend emission at the
selected positions. Path B main artifacts remain score-pass artifacts and do
not first retain broad per-example full exemplar payloads. Tests now fail if
selected payload values are fabricated locally or if the selected rerun invokes
the backend for examples other than the selected winners.

## 2026-07-08 — P4.8B Path A Capture Parity Correction

Path A selected-only delivery now materializes selected exemplar payloads from
the already-captured one-pass candidate shard arrays instead of rerunning the
teacher backend. The main pass captures dynamic top-k token IDs, log-probs,
probs, selection masks, bucket masses, and related mass metadata long enough to
slice selected winners, then prunes those broad candidate payload arrays from
the final shards when non-selected retention is disabled.

Path B keeps the selected backend rerun behavior. The parity harness and tests
now prove Path A has `teacher_rerun_count=0`, Path B has
`teacher_rerun_count=selected_example_count`, and selected IDs, positions,
scores, and payload shapes match across both delivery paths.

## 2026-07-08 — P4.8B GPU bfloat16 NumPy Transfer Guard

The `gpu_torch` tensor-to-NumPy transfer helper now casts floating tensors to
`float32` before moving through `.to("cpu").numpy()`. This prevents bfloat16
teacher outputs, including Gemma 3 270M GPU tensors, from hitting NumPy's
unsupported bfloat16 conversion path during compact payload materialization.

## 2026-07-08 — P4.8B Canonical Parity Scoring and Path A Pruning

Selected-only Path A now asks the selector for the same canonical score fields
as Path B: selected-position entropy, max entropy, mean entropy, confidence,
position bucket, length bucket, and source policy ID. Path A no longer lets
tail-mass or effective-top-k-only boards participate in selected-only parity
selection unless Path B emits matching score-pass fields.

When non-selected exemplar retention is disabled, Path A now prunes all
`exemplar_source_*` arrays from final shards after selected payloads are
materialized. Validation fails selected-only artifacts that retain any
`exemplar_source_*` broad candidate array. A 1000-example Path A/B parity test
now covers selected IDs, positions, score ranks, mode keys, Path A zero reruns,
Path B selected reruns, and final Path A shard pruning.

## 2026-07-08 — P4.8B Score-Surface Parity and Timing Instrumentation

Path A one-pass corridor shards now carry compact `score_*` fields alongside
temporary captured exemplar payload arrays. Selected-only delivery builds both
Path A and Path B leaderboards from the shared score-pass schema, eliminating
rank drift caused by separate precision paths while keeping Path A payload
materialization shard-backed and Path B payload materialization backend-rerun.

`production-build --track-delivery-timing` now records optional informational
timing in production, delivery, and exemplar-delivery parity reports. Timing
fields include production/preflight/main-pass/selection/payload/pruning/rerun
wall seconds, simple throughput rates, faster-path summaries, and explicit
non-claims. Timing is environment-specific and never affects parity pass/fail
without a future explicit performance gate.

## 2026-07-08 — P4.8B Mixed GPU Compact Payload Preservation

The GPU compact payload converter now handles mixed one-pass
`corridor_exemplar_v1` payloads before score-only payloads. When a reducer emits
both corridor/source capture arrays and canonical `score_*` fields, conversion
returns the union so Path A can select from the shared score surface and still
materialize selected payloads from one-pass `exemplar_source_*` candidate shard
data before final pruning.

## 2026-07-09 — P4.11 First-Class Corridor Artifact Export

Selected-only `corridor_exemplar_v1` production builds now emit explicit
`corridors/` artifacts: `corridor_fingerprints.json`, `corridor_modes.json`,
`mode_assignments.json`, `corridor_summary.json`, and a human-readable
`corridor_summary.txt`. Path A and Path B share the same logical artifact shape;
Path A still uses captured one-pass payload arrays before pruning, while Path B
reruns only selected examples for selected exemplar payloads.

The corridor exporter groups canonical score-surface observations by top token,
entropy bucket, confidence bucket, and position bucket. Reports and validation
now expose direct corridor yes/no fields and counts, and selected exemplar
records/payloads are linked back to corridor fingerprint and mode IDs.

## 2026-07-09 — P4.12 Full-Corpus Corridor Observation Basis

Fingerprint corridor export now prefers full per-token-position corridor arrays
instead of score-selected rows. Two-pass score shards retain compact
`corridor_top_token_ids`, `corridor_teacher_entropy`, `corridor_confidence`, and
`corridor_lengths`, while still avoiding dense logits and broad non-selected
exemplar payloads.

`corridor_summary.json`, delivery reports, production reports, validation
reports, and the human corridor summary now state the observation basis,
positions available, positions used, and whether the export is degraded.
Score-selected-only corridor export is explicitly marked degraded and rejected
by happy-path validation.

## 2026-07-09 — P4.12 GPU Score-Pass Compact Branch Fix

The GPU compact payload converter now distinguishes P4.12 score-pass payloads
from full one-pass corridor/exemplar payloads. Score-pass payloads can include
compact full-surface corridor arrays without requiring `corridor_top_probs` or
`exemplar_source_*` candidate payload arrays, preserving the full-corpus
corridor evidence while avoiding one-pass candidate retention.
