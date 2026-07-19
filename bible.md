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

## 2026-07-09 — Production Dynamic Top-K CLI Controls

`radjax-tome production-build` now exposes `--dynamic-top-k-min`,
`--dynamic-top-k-max`, and `--dynamic-mass-threshold`. Production build config,
backend config construction, emission metadata, target params, and
`production_build_report.json` all record the requested dynamic top-k controls
so selected-only burns can test larger exemplar caps such as 128.

## 2026-07-09 — P4.12 Stat-Band Corridor Mode Export

Fingerprint corridor export now separates diagnostic fingerprints from training
corridor modes. CPU and GPU corridor score payloads retain compact per-position
stats for entropy, top1 margin, top8 mass, top32 mass, and tail mass without
dense logits or dense probabilities.

`corridor_modes.json` now uses the original QRWKV-XLA-style `stat_bands_v0`
policy keyed by entropy, top1-margin, and top32-mass bins, bounded by the
default 256-mode cap. Mode records include min/max/mean bounds for the five
tracked stats, full token-position mode assignments are retained, selected
exemplars link to stat-band `corridor_mode_id` values, and validation rejects
legacy `fingerprint_group_v1` pseudo-mode artifacts.

## 2026-07-09 — P4.12 Corridor Stat Support and Packed Assignments

Stat-band corridor export now requires real top-32 probability support before
computing `top32_mass` and `tail_mass`. CPU and GPU corridor reducers use an
internal top-32 stat source for corridor statistics while keeping selected
exemplar payload top-k controls separate, and reports record both
`corridor_stat_top_k` and `min_corridor_stat_top_k`.

Full token-position corridor assignments now use `packed_numpy_v1` storage under
`corridors/mode_assignments/` with int32 position/example/mode arrays and
float32 weights. `mode_assignments.json` is a small manifest instead of a giant
assignment list, and validation checks packed array paths, dtypes, shapes,
mode-id ranges, position ranges, example-index ranges, and nonnegative finite
weights.

## 2026-07-09 — GPU Corridor Reducer Top-32 Test Fixtures

GPU corridor reducer contract tests now use top-32-capable logits so optional
Torch environments exercise the production corridor stat requirement instead of
failing on under-depth fixtures. The explicit K<32 failure test remains the
coverage for inputs that cannot compute real `top32_mass` and `tail_mass`.

## 2026-07-09 — Production Build Progress Sidecar

`radjax-tome production-build` now emits visible progress by default and writes
an atomically replaced `production_progress.json` sidecar. Score pass updates
report processed examples, throughput, elapsed time, ETA, and shard count;
selected reruns report selected-example throughput; corridor export reports
position counts, mode/fingerprint discovery, and assignment storage; validation
and report writing also publish phase updates.

## 2026-07-10 — P1.5 Production Tome Contract Alignment

Cover-page v2 is now the complete semantic front door for production Tomes. It
indexes every durable core, corridor, packed-assignment, and selected-exemplar
file by role with deterministic paths, hashes, sizes, required flags, and file
classifications. Generic corridor and exemplar surface declarations replace a
single-payload assumption, and the recommended plan records corridor then
exemplar as declarative checkpointed passes.

A deterministic eight-example fake producer fixture exercises multiple
stat-band modes, packed assignments, diagnostic fingerprints, and varying
dynamic top-k exemplar payloads without network, JAX, Torch, or Transformers.
Contract owns the canonical frozen fixture and shared interpretation; Tome owns
the reproducible generation recipe and producer-side validation.

## 2026-07-10 — Selected Exemplar Score-Pass Linkage

Selected-only delivery now preserves the canonical score-pass tuple from
candidate extraction through leaderboard records and selected payload shards:
selected example id, selected position, selected-position entropy score,
top-token id, source shard id, and source row. Path B selected rerun payloads
are validated against those persisted score-pass shard fields, and validation
fails with an explicit selected exemplar linkage mismatch when records or
payloads drift from the source row.

The CPU reference backend fixture is now row-invariant so selected-only reruns
can be validated like real teacher models: the same example produces the same
teacher behavior regardless of selected-rerun batch placement. The deterministic
production contract fixture was updated to emit the same linkage metadata and
score-consistent selected payloads as production selected-only builds.

## 2026-07-10 — Source-Coordinate Selected Exemplar Linkage

Selected exemplar linkage validation now keys both Path A and Path B to the
canonical source candidate coordinate instead of assuming every selected
exemplar must equal `score_selected_position`. Selected records and payloads
carry `source_shard_id`, `source_row`, `source_position`, `source_score`,
`source_top_token_id`, and `source_score_policy`; `selected_position` and
`selected_score` are aliases of that source coordinate for the retained
candidate.

Path A one-pass delivery now selects and materializes from one-pass candidate
coordinates, including a guarded candidate-rank path for compact candidate
layouts. Path B keeps the stricter score-pass alias check, where source
position/score/top-token must also match `score_selected_position`,
`score_selected_position_entropy`, and `score_top_token_id`. Validation now
fails with a source-coordinate mismatch when records, payloads, or corridor
arrays disagree.

## 2026-07-10 — Path A Payload-Ref Candidate Slot Preservation

Path A one-pass selected-only delivery now preserves a non-null
`one_pass_candidate_v1` payload reference from candidate extraction through
leaderboard records and selected payload materialization. The reference records
the source shard, source row, source position, candidate rank, source top token,
and source score so compact candidate payload arrays are sliced from the
selected candidate slot rather than from leaderboard order or row-local rank
accidentally.

Selected payload materialization now verifies compact candidate slots by both
source position and source top-token id, searches candidate ranks if a stored
rank is stale, and raises the selected linkage mismatch when no slot matches.
Path A validation also fails when compact selected records or payloads lose
their payload reference, closing the real 1K mismatch where the record tuple was
correct but `top_token_ids[0]` came from the wrong one-pass candidate slot.

## 2026-07-10 — Path A Source-Coordinate Diagnostics and Full-Sequence Resolution

Path A selected delivery now classifies one-pass source arrays before resolving
the payload coordinate. Full-sequence arrays are always sliced at the canonical
`source_position`; only compact candidate-rank arrays use the preserved
`candidate_rank` and rank search. The record source shard and row are now the
authoritative shard coordinate, while `payload_ref` must agree with every source
coordinate field before it can identify a compact payload slot.

Any selected-delivery linkage failure now carries a structured diagnostic with
the selected record, payload reference, array shapes, storage classification,
full-sequence check, and compact candidate-slot checks. Production reports keep
base `validation_status` separate from `selected_delivery_status`, and record
the selected-delivery failure stage and diagnostic instead of reducing it to an
opaque blocker.

## 2026-07-10 — Path A Payload Token Authority

Path A now distinguishes corridor diagnostics from selected exemplar payload
identity. `score_top_token_id` remains the corridor/stat token used for score
metadata, while `source_top_token_id` is taken from
`exemplar_source_top_token_ids` at the full-sequence source position or compact
candidate rank that supplies the retained training payload. This permits the
two reduction surfaces to diverge without making a valid selected exemplar fail
linkage validation.

Path A validation checks the emitted payload token against the payload-source
array before pruning and no longer requires it to equal `corridor_top_token_ids`.
Path B remains stricter: its score-pass token, corridor token, and rerun payload
token must still agree.

## 2026-07-10 — Path B Score-Pass Tuple and Rerun Diagnostics

Path B score-pass candidates now retain the complete authoritative source tuple
in `corridor_exemplar_score_pass_v1` payload references: shard, row, position,
selected-position entropy, and top token. The score-pass reducer uses the same
second-pass source policy for its corridor and score token IDs as the selected
rerun uses for its compressed payload, while compact corridor statistics remain
separate.

Before any selected rerun starts, delivery verifies every selected record against
its score-pass shard. After emission, it verifies the rerun payload token and
entropy against that source tuple. A failure report now includes the selected
record, score-pass shard values, corridor values, selected-record order, rerun
input order, rerun row, and mismatched rerun values, so pre-rerun record drift is
distinguished from actual backend-emission drift.

## 2026-07-10 — Adversarial Selected-Linkage Audit

Selected exemplar artifacts now have a reusable strict linkage auditor and the
`audit-selected-linkage` CLI. The audit treats each selected record as a source
passport, verifies record and payload order across every selected payload shard,
checks explicit source shard and row coordinates, applies Path A payload-source
authority and Path B score-pass authority independently, and confirms every
selected coordinate links to its packed stat-band corridor mode assignment.
Errors retain the complete record, payload, source values, and mismatch fields
instead of collapsing linkage failures into a boolean.

The adversarial suite uses uneven source shards, misleading example IDs,
candidate ranks that differ from token positions, duplicate scores and selected
records, rerun input reordering and deduplication, split payload shards, and
single-field mutations across the complete passport. Deterministic integration
builds audit both delivery paths. Delivery parity now treats exact selected
identity as an explicit controlled-fixture requirement while always enforcing
artifact shape, mode-table, assignment-linkage, payload-shape, and retention
compatibility.

## 2026-07-10 — Tome Packaging Profiles and Student Trust Contract

Completed producer artifacts can now be exported without changing production
semantics through two explicit profiles. `full_debug_provenance` carries the
full retained producer package, source shards, packed corridor targets, selected
payloads, linkage audit, and externalized content, shard, corridor-assignment,
and selected-payload manifests. `student` carries only the portable training
contract: packed inputs and masks, packed corridor assignments and modes,
selected exemplar payloads, provenance sidecars, and the same self-verifying
manifests, with raw producer shards and debug surfaces excluded.

Student packages export `examples_input_ids.npy` aligned with corridor example
metadata, so corridor and exemplar batches can be built without reading source
shards. The profile-specific linkage audit validates internal selected passports,
mode assignments, and input resolvability while explicitly reporting that
producer-shard authority is unavailable. Package creation stages into a temporary
directory, writes hash manifests and a manifest-oriented cover page, validates
the staged result, then atomically publishes a directory or `.tgz` archive.
Machine-local absolute paths are retained only as explicitly marked non-portable
provenance fields in student packages.

## 2026-07-11 — Package Cover Summary Truth

Package cover pages now resolve their top-level summary only from files retained
inside the package. Corridor totals and delivery-path fall back to
`corridor_summary.json`, assignment totals come from the corridor-assignment
manifest, selected counts come from the selected-payload manifest, and audit and
validation statuses come from their packaged reports. Full/debug packages may
also use their retained delivery and production reports, while student packages
never depend on those omitted producer reports. Both profile summaries explicitly
state their package profile and producer-shard authority.

## 2026-07-11 — Dynamic Top-K Long-Tail Diagnostics

Selected exemplars now retain a dynamic top-k diagnostic passport in both the
leaderboard record and compressed payload: effective support, top mass, dynamic
mass threshold and cap, saturation state, vocab fraction, a classified tail
shape, and deterministic warnings. Delivery, production, selected-payload
manifests, and package cover diagnostics aggregate the same distribution into a
long-tail summary so a larger dynamic cap can be evaluated per exemplar rather
than treated as a static global target shape.

The default is observational. Long and suspicious tails are retained with
warnings, leaving the existing canonical selection surface intact. The explicit
`reject_perverse_exemplars` control filters only suspicious-flat or full-vocab
candidates before selection, so the selector promotes the next eligible
candidate without changing Path A capture, Path B rerun semantics, corridor
modes, or selected-linkage authority.

## 2026-07-11 — Long-Tail Observations Are Not Build Warnings

Long-tail classes are experimental diagnostic observations, not quality verdicts.
Delivery and production reports remain `pass` when linkage, validation, and
retention contracts are clean, even if selected exemplars are classified as
long-tail, suspicious-flat, or full-vocabulary. Their aggregate messages live
under `long_tail_observations`, while `warnings` remains reserved for actual
artifact, selection-budget, metadata, or validation defects.

The compact GPU two-pass score helper remains the only score payload contract
that carries `score_effective_top_k` and `score_top_mass`; the full
corridor/exemplar production reducer intentionally does not claim those
score-pass-only fields.

## 2026-07-11 — GPU Score-Pass Long-Tail Test Contract

The compact GPU two-pass score payload contract explicitly includes
`score_effective_top_k` and `score_top_mass`. The reducer test asserts both
scalar-per-example shapes and integer/float dtypes so long-tail diagnostics keep
their compact score-pass inputs without expanding the payload into dense targets.

## 2026-07-11 — Long-Tail Mass Reporting Polish

Production reports now copy `long_tail_observations` from the completed delivery
report, keeping build-level and delivery-level experiment diagnostics aligned.
Diagnostic `top_mass` is clamped to the valid probability range while preserving
`raw_top_mass` and `top_mass_clamped`; numeric overshoot remains visible without
presenting impossible mass as a normal probability or turning a valid build into
a warning or failure.

## 2026-07-11 — Selected Exemplar Curriculum Boards

The existing rank-aware multi-score-board selector remains the sole selection
mechanism. After it chooses candidates and long-tail diagnostics are attached,
the delivery layer routes finished exemplars into curriculum boards: `primary`,
`long_tail_uncertainty`, or `perverse_tail_diagnostic`. Score-board provenance
is retained through `rank_by_board`, `scores_by_board`, and a board-summary
count, so curriculum routing never replaces the selection policy or its
deduplication behavior.

Student packages retain primary and auxiliary long-tail records by default,
while the perverse diagnostic board requires explicit producer opt-in. Every
retained board record preserves the same source passport and corridor linkage as
the flat selected-exemplar list, allowing the existing audit to validate the
complete packaged curriculum without special-case trust paths.

## 2026-07-11 — Student Board Summary Truth

Student package filtering is a content transformation, not only a file copy.
After removing excluded curriculum boards, leaderboard documents and payload
shards must recompute `long_tail_summary`, `selected_board_summary`, and grouped
board records from the retained flat list. Package manifests, cover diagnostics,
and selected-linkage audits therefore describe the same student-visible record
set rather than producer-side counts.

## 2026-07-11 — C1 Corridor Archetype Scoring

C1 adds the pure `fingerprint.corridor_archetypes` contract: typed candidate
features, validated thresholds and weights, ordered eligibility reason codes,
bounded membership/centrality/difficulty components, and deterministic utility
only for candidates that first pass corridor-core eligibility. It is deliberately
not wired into production selection or artifact emission; C2 owns offline
per-corridor micro-leaderboards.

## 2026-07-11 — C2 Offline Corridor Micro-Leaderboards

C2 adds a deterministic offline fingerprint-corridor candidate leaderboard
artifact. Explicit feature provenance is required for production-grade output;
compatibility proxies require a visible developer override. Each observed mode
retains a bounded pool ranked by utility, membership, centrality, useful
difficulty, and stable coordinate. Duplicate/conflicting coordinates, mode
support conflicts, malformed scores, pool overflow, hashes, and count
arithmetic are validated. The offline CLI accepts explicit compact candidate
JSONL and fails closed rather than fabricating features from selected payloads.
C2 stops before global corridor budgets, production selection, curriculum, and
payload materialization; C3 owns final corridor budgeting.

## 2026-07-11 — C3 Bounded Corridor Coverage Budget

C3 adds a deterministic offline allocator over validated C2 mode pools. It
floors Decimal budget fractions, applies an optional hard maximum, limits each
mode by its retained eligible capacity and mode cap, and preserves the exact
remaining global budget. Breadth-first round-robin water filling precedes
per-mode depth; severe first-round oversubscription uses top utility,
membership, centrality, support, and mode-ID priority. Every observed mode and
zero-allocation reason remains visible. Coverage plans retain C2 provenance and
hashes, contain no candidate coordinate claims or payloads, and are independently
hash-validated. C4 owns coordinate claims, collision handling, and global
backfill.

## 2026-07-11 — C2.1 Strict Provenance and Streaming State

C2.1 closes two acceptance blockers. JSONL feature loading is incremental and
fidelity-aware: explicit and derived records must carry real numeric corridor
features, derived records must identify their derivations, and only explicitly
marked compatibility proxies may use C1 adapter defaults. The builder detects
duplicate and conflicting coordinates through a temporary disk-backed SQLite
index, retaining only bounded per-mode pools and compact counters in memory.
Artifact `production_grade` now reflects observed provenance, so enabling a
proxy override does not downgrade a run that contains only real explicit or
derived features; actual proxy observations remain warned/non-production.

## 2026-07-11 — C4 Corridor-First Claims and Global Backfill

C4 adds an offline, deterministic coordinate-claim stage over validated C2
leaderboards, a validated C3 coverage plan, and an explicit ranked global-board
supply. Corridor representatives are claimed first in mode order; global
boards then backfill the remaining budget while preserving collision
obligations and replacement lineage. The artifact is atomic, hash-validated,
JSONL-based, and payload-free. C4 does not modify production selection, run
teacher inference, or define the C5 multi-role training schema.

## 2026-07-11 — C4.1 FIFO Backfill Lineage

C4.1 makes global backfill lineage one-to-one and auditable. Pending collision
and ineligible events are held in FIFO order; each accepted replacement consumes
at most one pending event, while any remaining events are explicitly unresolved.
Validation rejects repeated skipped ranks, repeated replacement seats, and
replacement references that do not correspond to selected global claims. The
stable global-board supply remains an offline contract; production integration
must provide a production-grade exporter rather than routing through the
development selector-manifest adapter.

## 2026-07-11 — C5 Durable Multi-Role Selected-Exemplar Records

C5 adds a payload-free offline projection from validated C4 claims to one rich
record per unique coordinate. Records preserve every C4 obligation, retain the
C4 primary claim and canonical selection order, derive corridor/global role
lists independently, carry a verified source passport, and expose one stable
coordinate payload identity marked `not_materialized_in_c5`. A deterministic
legacy flat projection remains available for existing consumers. C5 does not
rerun selection, change production output, route curriculum boards, or
materialize teacher targets; C6 owns those integrations.

## 2026-07-11 — C6 Corridor-First Production Integration

C6 adds the opt-in `corridor_first_global_backfill_v1` production integration.
C5 rich records are the authoritative unique coordinate set for delivery,
reports, audit, curriculum projection, cover metadata, and package summaries.
The strict integration validator compares those surfaces and requires real
source passports, while global-only production remains the default. A
production-grade ranked global-board exporter now emits the stable C4 supply
contract and rejects development selector manifests; strict C2 JSONL features
remain incremental and disk-backed with compatibility proxies rejected in
production. Policy and budget settings are included in plans, emission metadata,
streaming resume hashes, and production reports. C6 coverage and validation
reports are copied into full-debug/student packages and surfaced in cover-page
diagnostics. The T4 runbook is documented, but no real T4 rehearsal was
executed in this change.

## 2026-07-12 — C6.1 Integrated Evidence Closure

C6.1 replaces placeholder integration evidence with actual producer surfaces.
The production path now derives strict C2 features from the current packed
corridor assignments, mode bounds, and shard statistics, recording hashes and
normalization derivations instead of accepting a free-standing feature JSONL.
It writes `curriculum/selected_routes.json` from current consumption-board
routing, validates the coordinate union separately from route multiplicity, and
extends the selected-linkage audit with C5-aware parity. Both Path A and Path B
materialize the C5 coordinate set; Path B recognizes an authoritative C5 source
coordinate without pretending it was the score-pass argmax. Student/full-debug
packaging now revalidates C5, legacy, payload, passport, curriculum, and audit
parity locally, including zero-byte C4 lineage files. C6 coverage includes
zero-allocation modes and explicit reasons, reports a direct T4 status of
`not_executed`, and the T4 runbook reserves `--resume` for recovery only.

## 2026-07-12 — C6.2 Native Path B Authority Orchestration

C6.2 makes `corridor_first_global_backfill_v1` a native one-command
production flow. After its single full score/discovery pass, production builds
the packed corridor artifact, exports strict C2 features, a production ranked
global supply, and a hash-bound source-passport JSONL authority from the same
score surface. C2-C5 then consume those internal authorities before Path B
reruns only the final selected examples and emits one payload per C5
coordinate. Global supply and passports are no longer normal-path CLI
prerequisites; supplied files are fail-closed checkpoints tied to the current
score-pass authority hash. The authority manifest records producer hashes and
selection configuration, source-passport export is bounded through C4 by
loading only final C5 coordinates, and production reports distinguish the
full-pass and selected-rerun work. Global-only behavior and C1-C5 semantics
remain unchanged. The T4 rehearsal remains `not_executed`.

## 2026-07-12 — C6.2 Checkpoint Override Truth

Optional C6 global-supply and source-passport checkpoints now become the
authority actually consumed by C2-C5 only after exact comparison with the
current score-pass authority hash. The authority manifest records the paths in
use and whether an external override was used; a mismatched checkpoint fails
closed. The normal production path continues to use the internally generated
Stage 2 authorities.

## 2026-07-12 — C6.3 Native Path B Execution Boundary

Native C6 Path B now declares `native_c6_path_b_v1` from its score pass through
delivery. The score pass produces only the bounded authority surfaces needed by
C2-C5; it does not run a legacy selected-payload selector or pre-rerun. C5
coordinates are frozen and checked for the required unique budget before a
second-pass backend is created. The selected rerun is batched, keeps canonical
passport order after batch processing, and reports teacher/compression timing,
batch counts, and host/device memory peaks. Production and progress reports
also record phase-level RSS checkpoints, C6 budget/overlap diagnostics, zero
legacy reruns, and one native rerun when delivery succeeds. An interrupted
running progress sidecar is marked stale on the next invocation; the T4
rehearsal remains `not_executed`.

## 2026-07-12 — C6.3 Rerun Batching and Payload Lifetime Correction

Native Path B selected reruns now have an explicit
`--selected-rerun-batch-size` independent of score-pass batch policy, recorded
in selection hashes and delivery/production reports. Native C6 serializes each
compressed selected payload to its own shard as it is produced, keeps only
scalar payload summaries and references in memory, indexes records by example
within each batch, and releases batch emissions immediately. C6 final
reconciliation consumes the scalar payload index rather than a full payload
shard. Budget diagnostics now use canonical coordinate sets for unique corridor
and global supply, overlap, within-role duplication, requested allocation, and
Jaccard arithmetic. No T4 rehearsal was executed by this corrective patch.

## 2026-07-12 — C6.3.2 Selected-Position Transactional Delivery

Native C6 selected delivery now carries frozen selected positions through the
teacher batch contract and gathers only those rows before dynamic compression.
Path B has an independently configurable selected-rerun batch size, recognizes
only recoverable CUDA out-of-memory failures for deterministic retry, and
records retry, batch, memory, source-example, and selected-coordinate
telemetry. Native payloads are written incrementally to authority- and
coordinate-bound staging shards, validated by payload hash, resumable when
valid, and promoted only after the exact selected-coordinate set is complete;
public partial shards and indexes are cleared on a new attempt. Payload indexes
and delivery reports use atomic replacement. Production stops before artifact
validation, linkage audit, or cover generation when selected delivery fails,
and the progress sidecar records the structured failure. CPU and GPU backends
share the selected-position request semantics, while legacy delivery remains
compatible. No T4 rehearsal was executed by this corrective patch.

## 2026-07-12 — C6.3.3 Quantization-Aware Parity and Resume Evidence

Selected delivery parity now requires exact selected identity, source
coordinates, source passport coordinates, and top-token identity while allowing
the documented `0.00390625` entropy quantization step. Parity reports expose
the absolute entropy delta, allowed tolerance, parity status, coordinate exact
match, and top-token exact match; nonfinite or materially divergent entropy
still fails. Path B score-pass diagnostics resolve evidence by the exact
selected position and source passport row, including multiple positions from
one example, and preserve the score evidence row in failures. Native resumable
delivery retains authority-bound valid staged shards, quarantines invalid
staged files with explicit paths/counts, and reports preserved and quarantined
staging evidence after failure. No T4 rehearsal was executed by this patch.

## 2026-07-12 — C6.3.3 Live Rerun Entropy Gate Correction

The live Path B selected-rerun validator now uses the same documented entropy
quantization tolerance as artifact parity. A one-step `0.00390625` teacher
entropy difference passes the rerun payload gate, while nonfinite values and
meaningful divergence remain hard failures. Failure diagnostics expose the
live entropy delta, allowed tolerance, and parity status. Regression coverage
now exercises the live record-versus-rerun comparison directly. No T4
rehearsal was executed by this corrective patch.

## 2026-07-12 — C6.3.4 Streaming Linkage Audit and Finalization Resume

Selected-linkage auditing now shares one quantization-aware entropy helper
with live Path B rerun validation. Audits retain exact selected coordinates,
source passport fields, authority hashes, top-token identity, and payload
hashes while reporting per-record entropy deltas and tolerance. Payload shards
are read one at a time and reconciled against the coordinate-keyed payload
index; diagnostics retain scalar linkage state instead of full probability
arrays. Student package board filtering recomputes native shard and payload
index hashes after filtering. A completed score and selected-delivery surface
can resume C6 finalization without invoking the teacher again. No T4 rehearsal
was executed by this corrective patch.

## 2026-07-12 — C6.3.5 CPU-Only Finalization Resume Gate

Resumable native C6 Path B artifacts now undergo a read-only, authority-bound
complete-delivery probe before accelerator doctor and run-plan construction.
Eligible resumes validate score completion, C6 authorities, configuration
bindings, coordinate sets, payload-index references, per-shard authority, and
payload hashes, then finalize on CPU without model loading, teacher work, or
selected reruns. Reports preserve original teacher/backend provenance and add
explicit finalization-only, skipped-accelerator, and CPU-finalization fields.
Incomplete or configuration-mismatched artifacts remain fail-closed and use
the normal accelerator-required resume path. No T4 rehearsal was executed by
this corrective patch.

## 2026-07-12 — C6.3.5.1 Legacy Native Metadata Compatibility

Legacy native-C6 delivery surfaces are now recognized by their explicit
native execution, envelope, index schema, and shard structure. Before CPU-only
finalization, migration verifies the canonical score-pass authority, every
existing shard envelope hash and payload authority, then atomically backfills
missing exact-coordinate payload-index hashes and delivery metadata. Authority
manifests record the rewritten index hash. Corrupt shards fail closed without
index, delivery, or payload-body changes. Migration records its source schema,
backfill count, zero payload-body modifications, and zero teacher work. No T4
rehearsal was executed by this corrective patch.

## 2026-07-12 — M1 Mainline Canonization Inventory

Recorded the native two-pass fingerprint-corridor Path B direction, the golden
1K behavioral lock, an exhaustive tracked-surface disposition ledger, and the
research-status map. This documentation-only milestone preserves research and
records existing package-initializer imports of research modules as M3/M6
dependency-boundary remediation; it makes no runtime import changes.

## 2026-07-12 — M2A Golden 1K Contract Machinery

Added an offline semantic contract, read-only terminal-artifact capture,
fixture validation, and quantization-aware comparison tooling for the native
two-pass fingerprint-corridor Path B golden run. The repository intentionally
contains no fabricated T4 coordinates, passports, or payload semantics: M2
remains capture_pending until the authoritative rental artifact is exported.

## 2026-07-12 — M2A Native C6 Projection Correction

Corrected golden capture to derive selected obligations from the rich C5
multi-role authority, join native streamed `payload_index.selected_exemplars`,
and bind C2-C5 allocation, claims, dedupe/backfill, coverage, budget, and C6
authority records. The capture remains read-only and capture_pending until the
real terminal T4 artifact supplies the authoritative 256-coordinate fixture.

## 2026-07-12 — M2A Native C6 Projection Correction II

Aligned the C3 reader with `coverage_plan.json` plus its validation report,
made selection index authoritative across C5 obligation, passport, and payload
records, and removed absolute model/tokenizer paths from semantic identity in
favor of provenance and corpus-policy fields. Capture remains capture_pending.

## 2026-07-12 — M2A Native Authority Projection Correction III

Golden input identity now reads teacher and corpus provenance from the native
teacher manifest or emission configuration, while the run manifest contributes
only corroborating hashes. C4 corridor/global claims, collisions, selected
coordinates, and backfill lineage are projected from their JSONL authorities.
No real golden fixture was created; status remains capture_pending.

## 2026-07-16 — M2A Golden Capture Truth Gate

Golden capture now treats prefixed `source_corpus_*` provenance and the logical
teacher `model_name` as canonical identity, resolves native delivery and rerun
batch aliases from final reports, and rejects null corpus, teacher-hash,
execution-mode, delivery, selection-policy, or dynamic-top-k authority. C4
semantic projection now excludes storage manifests and physical file hashes in
favor of the captured claim, obligation, selected-coordinate, and backfill
rows. No real golden fixture was created; status remains capture_pending.

## 2026-07-19 — Golden Fixture Staging Directory Compatibility

Golden fixture writing now accepts the pre-created staging directory returned
by `tempfile.mkdtemp`, matching capture's atomic-write flow. This is a writer
compatibility repair only; golden schemas and semantic projection are unchanged.

## 2026-07-19 — M2A Sparse Golden Payload Projection

Golden payload semantics now retain only active dynamic-top-k entries in rank
order, with padded backend arrays and selection masks excluded. Fixture
validation rejects dense fields, malformed sparse arrays, nonfinite values,
duplicate active tokens, and oversized records; comparison streams JSONL rows.
M2A remains capture_pending until the corrected sparse fixture is recaptured.

## 2026-07-19 — M2A Streaming Selected Payload Capture

Golden capture now projects selected payload shards one at a time, validates
every source coordinate against C5 and the payload index, releases each dense
shard, and retains only compact semantics ordered by C5 selection index.
Comparison through `golden compare --artifact` inherits this bounded capture
path. No real fixture was created; M2A remains capture_pending.

## 2026-07-19 — M2A Digest-Only Payload Semantics

Golden payload records now retain scalar target summaries and versioned,
ordered active-value digests instead of active token/probability arrays. Raw
payload storage is validated before hashing but discarded immediately, keeping
full-vocabulary selections compact and fixture-size bounded. M2A remains
capture_pending until a corrected fixture is captured from the terminal Tome.

## 2026-07-19 — M2A Binary Active-Payload Digests

Active payload digests now use versioned, chunked canonical binary encoding:
big-endian signed int64 token IDs, big-endian IEEE-754 float64 probabilities
and log-probabilities, explicit active counts, and normalized signed zero. The
combined digest binds the component digests without per-entry JSON serialization.
M2A remains capture_pending.

## 2026-07-19 — M2A/M2B Golden Fixture Portability Gate

Golden board-summary projection now excludes storage-valued artifact locator
IDs while preserving logical IDs and hashes. A final recursive portability gate
rejects POSIX, Windows, UNC, file URI, and home-relative storage locations in
every contract object and JSONL record. M2A/M2B remain capture_pending; no
fixture was committed.

## 2026-07-19 — M2B Canonical Golden T4 1K Fixture Closure

M2A golden capture machinery and M2B canonical fixture closure are complete.
Commit `371a60541aa6c73dddcde510203064c8882c935e` captured the portable native
two-pass fingerprint-corridor Path B Golden 1K contract at
`tests/fixtures/golden_t4_1k`. Its semantic root is
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`
with 256 selected coordinates. The working production head remains native
two-pass fingerprint-corridor Path B; research-frozen paths remain frozen.
The committed fixture has passed contract and portability validation and is the
mandatory semantic regression baseline for future canonical-pipeline changes.

## 2026-07-19 — M3A Corridor Phase Characterization

M3A adds a focused CPU characterization of native Path B's two corridor writes:
the early score-surface export has zero selected records and no selected-link
claim, while the post-rerun export overwrites the public corridor summary with
selected linkage and emits detailed corridor progress. The characterization
also proves its observable order and a current finalization-only resume probe
after removing only the C6 validation report. No runtime algorithm, schema,
artifact path, or golden fixture changed.

## 2026-07-19 — M3A Root Gate Evidence

The M3A root gate verified the full suite at `714 passed, 22 skipped` and the
focused M3A/import/runtime set at `23 passed`. Golden validation retained count
`256` and semantic root
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
Ruff check/format and `git diff --check` passed.

## 2026-07-19 — M3B Slice One Import Isolation

M3B slice one makes the root, backend, and audit facades resolve compatibility
exports lazily; builder, reports, and fingerprint isolation is deferred to
slice two. The focused import/refactor gate passed `46` tests. Golden validation
retained count `256` and semantic root
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
Ruff check/format and `git diff --check` passed.

## 2026-07-19 — M3B Final Import Isolation and Hydra Repair

M3B finalizes builder, reports, and fingerprint isolation and repairs the
Hydra inventory for the lazy-export helper and M3A boundary documents. Public
facade names and direct compatibility-leaf paths remain preserved, while fresh
root, canonical, parser, and help imports leak neither optional ML stacks nor
classified research modules. The full gate passed `731 passed, 22 skipped`; the
focused final import gate passed `37` tests. Golden validation retained count
`256` and semantic root
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
Ruff check/format and `git diff --check` passed.

## 2026-07-19 — M3C Typed Native Path B Boundary

M3C adds a typed exact native Path B configuration adapter and delegation seam
while preserving the existing production executor, artifacts, progress, and
CLI/parser behavior. Global-only and partial/alias routes remain non-native.
The full gate passed `747 passed, 22 skipped`; M3C integration passed `7`
tests; direct native API passed `9` tests and import isolation passed `37`
tests. Golden validation retained count `256` and semantic root
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
Ruff check/format and `git diff --check` passed.

## 2026-07-19 — M4A Typed Stage Contracts and Evidence Readers

M4A adds typed contracts and read-only evidence readers without changing
schemas or paths. Early provisional corridor evidence is rejected as final;
evidence derives hashes from existing JSON and performs no writes. The full
gate passed `751 passed, 22 skipped` before mechanical format; the post-format
focused native contract/API/runtime/M3C set passed `21` tests. Golden validation
retained count `256` and semantic root
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
Ruff check/format and `git diff --check` passed.

## 2026-07-19 — M4B Slice One Preflight and Score Adapters

M4B slice one introduces typed callback-driven preflight-to-score adapters.
Score execution is blocked after preflight failure; the adapters write no
artifacts, progress, or reports and preserve schemas and paths. Later corridor
stages remain for later slices. The initial focused gate passed `171` tests;
post-format broad focused coverage passed `131` tests and the orchestrator
suite passed `4` tests. Golden validation retained count `256` and semantic
root `sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
Ruff check/format and `git diff --check` passed.

## 2026-07-19 — M4B Slice Two Provisional Corridor and Authority Adapters

M4B slice two proves callback order from score pass through a zero/unlinked
provisional corridor to fingerprint and global authority export. A failed early
corridor stops both authority callbacks, and the adapter writes no selection,
rerun, late-corridor, or evidence schemas. The focused gate passed `173` tests.
Golden validation retained count `256` and semantic root
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
Ruff check/format and `git diff --check` passed. This does not claim production
facade integration.

## 2026-07-19 — M4B Slice Three Typed C2-C5 Selection Handoff

M4B slice three adds a typed C2-C5 selection handoff that consumes both
fingerprint and global authority proofs and exposes explicit C2, C3, C4, and
C5 stage evidence. Selection failure stops before rerun or finalization. The
focused gate passed `176` tests. Golden validation retained count `256` and
semantic root
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
Ruff check/format and `git diff --check` passed. This does not claim production
facade integration.

## 2026-07-19 — M4B Slice Four Selected Delivery and Assembly

M4B slice four strictly orders C5 selection, selected rerun, final
selected-linked corridor, and assembly. Provisional corridor evidence is
rejected; any later-stage failure stops downstream promotion or assembly; the
adapters write no validation, reconciliation, or report state. The focused gate
passed `179` tests. Golden validation retained count `256` and semantic root
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
Ruff check/format and `git diff --check` passed. This does not claim production
facade integration.

## 2026-07-19 — M4B Slice Five Typed Terminal Finalization

M4B slice five strictly orders typed assembly, validation/linkage,
reconciliation/cover, and final reporting handoffs; every failure stops later
callbacks. Terminal `NativePathBRunResult` failures are preserved, and the
adapters add no persistent state. The focused gate passed `182` tests. Golden
validation retained count `256` and semantic root
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
Ruff check/format and `git diff --check` passed. This does not claim production
facade integration.

## 2026-07-19 — M4B Canonical Production Slice-One Integration

The exact canonical Path-B production route now runs the existing preflight
and score-pass operations through the typed slice-one adapter before entering
the unchanged early provisional corridor and later selected-linked corridor
continuation. Global-only production bypasses the adapter. Callback exception
propagation is opt-in only for that production seam, preserving the established
runtime doctor behavior while standalone adapters remain failure-normalizing.
The focused integration and compatibility gate passed `132` tests; the complete
suite passed `769 passed, 22 skipped`. Golden validation retained count `256`
and semantic root
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
Ruff check/format and `git diff --check` passed.

## 2026-07-19 — M4C Evidence-Derived Resume and Failure Normalization

M4C adds a read-only canonical Path-B resume classifier with no new persistent
stage schema. It derives the earliest repairable stage from the existing run
plan, score manifest, distinct early/late corridor evidence, authority,
selection, delivery, validation, reconciliation, and production-report files.
Fresh, partial, delivery-pending, finalization-only, terminal, stale, corrupt,
and full-config hash-mismatch cases are covered. A provisional early corridor
cannot be treated as selected-linked final evidence; after the late overwrite,
corridor corruption is attributed to late finalization. Existing compatibility
migration remains production-owned. The M4C matrix and native/import focused
gate passed `75` tests. Golden validation retained count `256` and semantic
root `sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
Ruff check/format and `git diff --check` passed.

## 2026-07-19 — M4D Local Integration Proof

M4D local proof passed without changing runtime code or the immutable fixture:
the full non-GPU suite passed `779 passed, 22 skipped`, the native/import/
delivery/validation/linkage/reconciliation/Golden focus passed `220` tests,
and static checks passed. Golden validation retained count `256` and semantic
root `sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
`docs/M4D_INTEGRATION_EVIDENCE.md` records the exact reviewed T4 Golden 1K
commands and acceptance criteria. T4 rental proof is not executed on this
host: it has no CUDA/T4, Torch, Transformers, or terminal canonical artifact.
This is an external merge gate, not a local pass claim; `main` remains
untouched and the fixture was not regenerated.
