# RADJAX-Tome Project Ledger

## 2026-07-31 — M7A Payload Sharding Characterization

M7 begins from an explicit characterization of the current selected-payload
surface. Canonical Path B emits one-record
`selected_exemplar_payload_shard_v1` JSON files and uses transactional native
staging/promotion, but package and public student-reader paths retain eager
payload collections. The current v3 semantic identity hashes physical selected
payload files, so regrouping unchanged records changes its digest. The approved
M7 solution is a versioned v4 / semantic-identity-v2 contract; v3 remains
historical compatibility evidence and is not reinterpreted. The complete
pre-M7 payload-field census is locked in a characterization test for M7B
classification.

## 2026-07-31 — M7B Tome-Local Streaming Contract Proposal

The repository now carries a proposed, not-yet-published `radjax_tome/v2`
contract tree for review. It defines the closed v4 cover, identity v2,
streamed content-manifest graph, JSONL selected-payload index, bounded
record-count sharding, exact 38-field selected-payload projection, errors,
profiles, compatibility declarations, digest vector, and checksum inventory.
The proposal preserves the v1 publication unchanged and is intentionally
stopped before any RADJAX-Contract mutation.

## 2026-07-31 — M7B Contract Correction

The v2 inventory exclusion now correctly names `cover_page.json`, and each
JSONL payload-index record explicitly binds both its coordinate-derived logical
ID and raw payload digest. This resolves the draft index ambiguity before the
portable streaming validator and conformance corpus are accepted. Contract and
Student remain untouched.

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

## 2026-07-19 — M4D Evidence Inventory Repair

The committed M4D evidence document is now recorded in the authoritative Hydra
disposition inventory, restoring the tracked-doc ledger gate required by CI.
This is metadata-only; it changes no runtime, artifact, fixture, or T4 proof
claim. The focused disposition gate passes locally before publication.

## 2026-07-19 — M4B/M4C Canonical Execution Correction

The previous local M4D note did not establish live execution of typed slices
two through five, because production stopped after the slice-one adapter. The
canonical Path-B facade now executes the real ordered callbacks for provisional
early corridor export, split fingerprint/global authority export, C2-C5
selection, selected rerun, late selected-linked corridor finalization, payload
assembly, validation/linkage, reconciliation/cover, and final reporting.
`materialize_selected_exemplar_delivery` remains the identical public
three-phase composition while the canonical callbacks use the same phases
directly. Global-only stays on the compatibility path. Resume now derives and
uses an explicit artifact-assembly boundary from existing files before
validation; no persistent stage schema was added. Focused delivery, canonical
boundary, characterization, production, contracts, orchestration, C6, and
resume gates passed `119` tests after the correction; Ruff, format, and diff
checks passed. This corrects the local-proof scope and does not claim T4 proof.

## 2026-07-19 — M4D Refreshed Local Integration Proof

After the canonical-execution correction, the complete non-GPU suite passed
`784 passed, 22 skipped in 95.98s`; the native/import/delivery/validation/
linkage/reconciliation/live-resume/Golden focus passed `225` tests in `35.83s`.
Ruff check, format, and diff checks passed. Immutable Golden validation remains
`pass` with count `256` and semantic root
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
`docs/M4D_INTEGRATION_EVIDENCE.md` now corrects the earlier slice-one-only
scope; reviewed T4 Golden 1K comparison remains unexecuted and is still the
external merge gate.

## 2026-07-26 — M4D Versioned Authority-Hash and Golden Contract Migration

The accepted M4D diagnosis is implemented without modifying the immutable T4
Golden v1 fixture or running GPU inference. Historical
`radjax.c6.score_pass_authority.v1` remains the exact raw-byte recipe, while
new artifacts emit explicit v2 semantic authority, retain their v1 lineage
hash, and record raw digests for metadata, packed-assignment manifest, corridor
modes, and production selector inputs. Golden contracts now have distinct v1
and v2 schema/digest domains; cross-version comparisons are explicitly
incompatible, and a v2 capture can project a historical v1 artifact read-only
after verifying its recorded v1 authority. The complete contract is in
`docs/AUTHORITY_HASH_V2_MIGRATION.md`. Focused authority/M3/M4/Golden/Hydra
coverage passed `133 passed, 1 skipped in 18.97s`; the complete local suite
passed `794 passed, 23 skipped in 92.78s`. The conditional July 19/July 24
source-artifact comparison skipped because neither artifact is mounted locally;
it runs only when both explicit artifact paths are supplied. Ruff check/format,
`git diff --check`, CLI Golden help, and frozen v1 fixture validation passed;
the fixture root remains
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.

## 2026-07-26 — M4D Closure and Mainline Approval

The reviewed historical-artifact v2 comparison has closed the final M4D gap.
`tests/test_authority_hash_contract_v2.py::test_july_t4_artifacts_compare_under_v2_when_both_are_available`
passed in `168.86s` against
`/teamspace/studios/this_studio/radjax_t4_path_b_1k/c6_3_2_native_clean` and
`/teamspace/studios/this_studio/radjax_t4_path_b_1k/m4d_refactor_proof/tome`.
It proves semantic equality under authority-hash v2 while preserving distinct
raw integrity digests. This closure records the reviewed read-only comparison;
it neither reruns GPU inference nor changes either source artifact or the
immutable v1 fixture. The final bounded roadmap and scope review found all
M3A–M4D checkpoint criteria satisfied, no corridor-order collapse, no frozen
fixture mutation, no unauthorized scope expansion, and no merge-policy
blocker. The final local suite passed `794 passed, 23 skipped in 84.38s`.
The closure also repairs the Hydra inventory record for the committed
`builder/authority_hashes.py`; its focused ledger gate passes. M4 is complete
and the reviewed `m3-m4-canonical-path-b-refactor` branch is approved for
normal merge to `main`.

## 2026-07-30 — M5A Contract Ownership and Characterization

M5A records the pre-change ownership of the mainline configuration and Tome
contract surface in `docs/M5_CONTRACT_OWNERSHIP.md`. The current
`ProductionBuildConfig` contains 67 fields, replacing the earlier approximate
61-field assessment; the M5A characterization test pins the complete ordered
field surface. It also pins the existing 25-field selection-integration
authority projection to
`sha256:c7bdbfe538c007db6b65c7fc87850b29355dfeef5300c5bd4fc6efb178e987ab`
for a fixed request, asserts every authority-bearing request field changes the
hash, and asserts a non-authority long-tail policy does not. Cover-page v2 and
`radjax_tome_package_cover_v1` remain distinct historical contracts. No
production writer, reader, artifact schema, Golden fixture, authority recipe,
or GPU behavior changed. Focused M5 config/cover/package/bundle/Golden
coverage passed `87 passed, 1 skipped`; the complete local suite passed
`798 passed, 23 skipped in 90.45s`. The unrelated pre-existing `.DS_Store`
remains untracked and untouched.

## 2026-07-30 — M5E Historical Adapter Closure

`tome.compatibility` is the isolated, dependency-light compatibility reader for
cover-page v2 and `radjax_tome_package_cover_v1`. The pure adapter maps only
known historical facts into a deliberately incomplete descriptor: v2 target
settings and an explicitly non-profile-complete inventory claim, or v1 package
profile/layout and manifest references. It leaves identity, authority, and any
unproven section absent. The path reader validates directory inputs under their
native historical validator before mapping; it safely materializes an outer
legacy `tgz` into a temporary directory and performs the same native
validation. Standalone JSON covers cannot claim artifact validity, and unknown
schemas, layouts, profiles, malformed inventories, and unsafe archives fail
closed. New writers remain v3 only; no historical artifact, authority recipe,
Path-B stage, or Golden fixture was rewritten. Focused M3/M5/cover/package/
Hydra coverage passed `123 passed in 20.87s`; the native-reader package/adapter
regression passed `24 passed in 7.46s`; the cache-cleared full local suite
passed `851 passed, 23 skipped` (874 collected). Ruff, formatting, JSON, and
diff checks passed. The unrelated pre-existing `.DS_Store` remains untracked
and untouched.

## 2026-07-30 — M5B Canonical Configuration and Tome Contracts

M5B adds a dependency-light canonical configuration boundary with explicit
`TomeBuildIntent`, `ResolvedTomeBuildConfig`, and `TomeExecutionPlan` forms,
plus a legacy adapter that copies the inventoried 67-field
`ProductionBuildConfig` without rerouting production. The unchanged 25-field
selection-integration projection is available from resolved configuration and
is characterized as byte-for-byte equivalent to the existing production hash.
`radjax_tome_cover_v3` is defined as a nested pure contract with identity,
training, package, manifests, authority, provenance, and validation sections.
Its semantic identity contains only the training-authoritative payload,
training contract, and authority binding: profile-specific inventory and
directory/`.rtome`/`tgz` transport cannot change it. Package manifests retain
raw-byte digests and may therefore differ between student and full-debug
packages; the cover is excluded from the manifest to prevent circular hashing.
Historical v2/v1 readers, production writers, CLI behavior, native Path-B
routing, authority-hash v1/v2, and the frozen Golden fixture are unchanged.
Focused M5/M3/M4/cover/package/Golden/Hydra coverage passed
`147 passed, 1 skipped in 21.68s`; the complete local suite passed
`807 passed, 23 skipped in 93.82s`. Ruff check, formatting, and
`git diff --check` passed. The unrelated pre-existing `.DS_Store` remains
untracked and untouched. M5B is ready for its mandatory review gate; M5C has
not begun.

## 2026-07-30 — M5B Compatibility-Mapping Review Correction

The M5B contract document now tabulates the explicit, non-inferential mapping
from cover-page v2 and `radjax_tome_package_cover_v1` to the nested v3
concepts. The table separates known teacher/training descriptors, profile
inventory and raw-integrity facts, provenance/runtime metadata, and validation
claims; it records what each historical format cannot establish and requires a
future M5E adapter to leave unknown information absent. This is
documentation-only: no production, writer, reader, routing, authority, or
fixture code changed. The already-completed M5B focused suite
(`147 passed, 1 skipped`) and full suite (`807 passed, 23 skipped`) remain the
applicable code evidence; the ledger/Hydra/M5 contract documentation tests and
Ruff checks pass after this correction. M5C has not begun.

## 2026-07-30 — M5B Closed-Shape Contract Hardening

The approved M5B correction adds standalone validators for semantic identity
and canonical content manifests. They recompute their digests, require exact
contract keys, require exact lowercase SHA-256 syntax, and reject stale,
unsorted, duplicate, traversing, malformed, or type-invalid nested records.
The v3 cover validator now rejects unexpected top-level/package/manifest
sections and validates identity plus manifest contracts before checking their
cross-references. Identity comparison validates both inputs before equality,
so stale matching digest strings cannot compare successfully. The contract
documentation now records the closed core shapes and no-extension policy.
No production writer, CLI route, native Path B stage, historical reader,
authority recipe, or Golden fixture changed. Focused M5/M3/M4/cover/package/
Golden/Hydra/ledger coverage passed `162 passed, 1 skipped in 22.59s`; the
complete local suite passed `821 passed, 23 skipped in 92.09s`. Ruff check,
formatting, and `git diff --check` passed. The unrelated pre-existing
`.DS_Store` remains untracked and untouched. M5C has not begun.

## 2026-07-30 — M5C Canonical Configuration Normalization and Presets

M5C makes the M5B `TomeBuildIntent` / `ResolvedTomeBuildConfig` boundary the
authoritative public production configuration path. `production-build` now
uses the single normalizer, which applies a named preset before only explicitly
provided advanced overrides, validates before execution, derives a separate
execution plan, and then derives the unchanged 25-field selection authority.
`--print-resolved-config` exposes that complete result before preflight, model
loading, or artifact writing. Supported presets are `smoke`, `t4-1k`,
`t4-10k`, and `production-100k`; the T4 size presets preserve the reviewed T4
semantic settings and differ only in `max_examples` (1K, 10K, or 100K).

Legacy `ProductionBuildConfig` remains a fully copied, explicit adapter and
continues to preserve its immutable object identity at the native Path-B
boundary after canonical validation. Canonical requests use the same explicit
flat execution adapter only at that preserved boundary. The native route gate,
early provisional and late selected-linked corridor order, authority recipes,
historical readers, covers, manifests, transports, and frozen Golden fixtures
are unchanged. New canonical CLI requests default to not retaining unselected
exemplar payloads; retention remains an explicit advanced compatibility
override. `docs/M5_CONFIGURATION_NORMALIZATION.md` records the contract,
preset values, compatibility seam, and non-claims.

Focused M3/M4/M5/production/Golden coverage passed `155 passed, 1 skipped in
17.69s`; the full available local suite passed `829 passed, 23 skipped in
92.40s`. Ruff check, formatting, and `git diff --check` passed. No GPU
inference ran and no fixture was regenerated. The unrelated pre-existing
`.DS_Store` remains untracked and untouched.

## 2026-07-30 — M5C Normalization Validation Hardening

The canonical normalization boundary now validates concrete nested-section and
field types before any preflight/runtime work. Integer and numeric controls
reject booleans; required strings, paths, enums, finite numeric ranges, and
canonical corridor-fraction decimal text are fail-closed. The latter retains
historically authority-bearing spellings such as `"0.50"` without rewriting
the fixed 25-field projection. Validation now enforces backend/runtime pairs,
resume-versus-overwrite exclusion, strict canonical selection/Path-B
dependencies, and the resolved-config envelope schema plus metadata. The
explicit legacy flat adapter retains only its intentional compatibility path.

Focused validation, legacy-adapter, M3/M4 routing, Golden, and production
coverage passed `72 passed, 1 skipped in 9.43s`; the complete local suite
passed `838 passed, 23 skipped in 93.16s`. Ruff, formatting, and diff checks
passed. No GPU work, fixture regeneration, authority projection change, or
Path-B stage-order change occurred; the unrelated `.DS_Store` remains
untracked and untouched.

## 2026-07-30 — M5D Source-Derived Identity Seam

M5D begins with a pure source-artifact derivation seam for the canonical v3
cover. It separates timestamp-insensitive training semantic identity from
profile-specific raw inventory and preserves raw integrity digests for the
authority-relevant source files. The focused v3 contract tests pass; writer,
validator, profile, and transport routing remain the next M5D integration
slice. No existing public writer or historical reader changed in this commit.

## 2026-07-30 — M5D Canonical Package Cover Migration

Package writers for both `student` and `full_debug_provenance` now emit the
closed nested v3 cover and validate its canonical raw inventory against the
materialized directory. Source identity is computed once before profile
materialization, so the two packages share a Tome identity while retaining
distinct manifest digests. The former package-v1 cover is captured only under
v3 provenance for compatibility diagnostics; existing manifest-level and
student/full-debug contract validation is reused through explicit internal
references, not as a coequal public cover. Focused profile, bundle, canonical
contract, and identity coverage passed `41 passed`; no GPU work or fixture
mutation occurred.

## 2026-07-30 — M5D Canonical Directory and Unified Transport

The unpacked Tome writer now emits the same closed `radjax_tome_cover_v3`
consumer contract as package writers and validates its complete raw inventory
against the directory. Its former v2 cover is retained only as explicit
provenance; the native v2 validator remains available for historical inputs.
The bundle abstraction now owns both deterministic uncompressed `.rtome` and
deterministic gzip transport. New profile `.tgz` output uses that common bundle
writer at archive root instead of a separate tar implementation, while its
v3 profile/inventory agreement is validated before archival. Directory and
archive validation both use the v3 cover/manifest validation path; archives
also prove exact member inventory, raw SHA-256, byte sizes, and deterministic
tar metadata. A focused cover/bundle/package/CLI/production fixture suite
passed `43 passed in 17.20s`. No GPU execution, authority recipe change,
Golden fixture mutation, or Path-B stage change occurred. The unrelated
pre-existing `.DS_Store` remains untracked and untouched.

## 2026-07-30 — M5D Validation Completion

The source identity projection now binds the complete known training payload:
canonical JSON sidecars with runtime timestamps excluded, selected payload and
curriculum JSON, assignment metadata and arrays, and shard NPZ members by
sorted uncompressed content. A mutation to an authoritative assignment payload
changes the semantic identity; a timestamp-only mutation still changes raw
integrity but not identity. The streaming completed-resume path now refreshes
the v3 cover only after writing its final progress event, preserving exact
inventory agreement. Focused M5D cover/bundle/package/streaming coverage passed
`39 passed in 13.07s`; the broader writer/provenance/streaming regression group
passed `75 passed in 10.69s`; the cache-cleared full local suite passed
`842 passed, 23 skipped` (865 collected). Ruff, formatting, and diff checks
passed. No GPU execution, Golden fixture mutation, authority-projection change,
or Path-B stage change occurred. The unrelated pre-existing `.DS_Store`
remains untracked and untouched.

## 2026-07-30 — M5F Canonical Tome Contract Closure

M5 is complete as a product-contract milestone. The paved road is typed build
intent to resolved configuration to derived execution plan, then the preserved
production/Path-B facade, source-derived profile-neutral Tome identity, closed
v3 cover and manifest, and deterministic directory/`.rtome`/gzip transport.
Student and full-debug packages retain distinct raw inventories while agreeing
on source identity. Historical v2 and v1 inputs validate natively and adapt
only known facts into an incomplete descriptor; new writers emit v3 only.
The closure matrix, compatibility limits, contract versions, exact verification
commands, intentional deferrals, and non-claims are recorded in
`docs/M5_CLOSURE_REPORT.md`.

The final M5F static gates passed: `ruff check src tests`,
`ruff format --check src tests` (`205 files already formatted`), JSON syntax,
and `git diff --check`. The final cache-cleared full local suite passed
`851 passed, 23 skipped` (874 collected). This leaves the immutable Golden v1
fixture, fixed 25-field selection authority, authority-hash v1/v2 contracts,
canonical Path-B early/late corridor order, and M4D proof unchanged. No GPU
inference, fixture regeneration, main merge, M6 work, performance work, or
corpus/model/runtime expansion occurred. The unrelated pre-existing
`.DS_Store` remains untracked and untouched.

## 2026-07-31 — M6A Publication Inventory and Ownership

M6 begins with a checked ownership ledger for the M5 v3 cover, semantic
identity, content manifest, profiles, transport, historical descriptors, and
protected M4/M5 authority and Golden boundaries. It separates deterministic
producer transport requirements from consumer safety and consumer canonicality
reporting, and records that the independent portable validator will be created
once in M6B and reused for M6D parity. The existing v2-first cover-page guide
is recorded as documentation drift to correct in M6B. No writer, CLI, Path-B,
authority, Golden fixture, or external repository changed; the unrelated
`.DS_Store` remains untouched.

## 2026-07-31 — M6B Portable Contract Source and Validator

M6B adds the Tome-local, implementation-neutral v1 contract source: closed
v3 identity, manifest, and cover schemas; canonical digest and transport
recipes; profile, error, and historical compatibility definitions; checksum
inventory; and a stdlib-only independent conformance validator. Native writers
and CLI routing remain untouched. The current cover-page documentation now
identifies v3 as the front door and v2 as historical compatibility only. The
portable validator is created once here and will be reused for M6D parity.

## 2026-07-31 — M6C Conformance Corpus and Digest Vectors

M6C adds a compact, checksum-pinned conformance catalog and independently
reproducible canonical JSON/SHA-256 vector. The corpus exercises valid v3
directories and deterministic archive transports, stale semantic identity,
unsafe archive paths, historical nonpromotion, and the deliberate distinction
between unsafe transport and safe-but-noncanonical metadata. Fixture artifacts
are generated only in temporary test directories; no model inference or Golden
fixture regeneration is involved.

## 2026-07-31 — M6D Native and Portable Conformance

M6D reuses the single M6B portable validator against canonical directory,
`.rtome`, gzip, student, and full-debug producer outputs. It proves profile
identity agreement with permitted manifest differences and documents the only
intentional validator difference: native bundle validation enforces producer
canonical transport metadata, while portable consumer validation reports a
safe noncanonical container and rejects it only in strict mode. Unsafe or
integrity-invalid transport remains fail-closed in both paths.

## 2026-07-31 — M6E Contract 0.2.0 Publication Pin

The approved portable asset tree and corpus are published byte-identically in
RADJAX-Contract `0.2.0` at tag `v0.2.0` and commit
`147ca371c78a98dbc82de1ea93deb4f3ae27f399`. Tome pins verification to that
release and retains its tree only as a checksum-enforced offline mirror. Tests
compare the mirror to Contract source when available; release verification also
proved the installed wheel assets identical. Production writers and CLI remain
independent of Contract runtime imports.

## 2026-07-31 — M6F Contract Publication Closure

M6 closes the Tome-local publication and conformance milestone. Contract 0.2.0
is the released authority for the static v3 assets; Tome's identical asset tree
is an offline checksum-enforced mirror. The closure report records exact
version, tag, commit, checksum, downstream verification commands, proven
transport/identity behavior, and intentional deferrals. No Student work,
inference, accelerator work, Golden fixture regeneration, or merge occurred.

## 2026-07-31 — M6 Publication-Pin Dependency Clarification

The M6 publication pin is documented as a pin of Tome's pre-existing
RADJAX-Contract dependency from `main` to `v0.2.0`, not as a new
development-only dependency. Existing production modules retain their
established Contract API imports; M6 introduces no production import of the
v3 publication resource API, whose assets serve verification and conformance.

## 2026-07-31 — M6A Production Boundary Characterization

The roadmap's original M6 now begins from the approved publication base.  A
checked ownership ledger and AST-backed characterization pin the public
production façade, M4 native Path-B state-machine seam, packaging façade, and
the required inward dependency direction before implementation movement.  The
ledger distinguishes the completed v3 publication milestone from this
production-boundary work, preserves Contract `v0.2.0` and its offline mirror,
and explicitly defers M7--M12 work.  No production behavior, authority recipe,
Golden fixture, v3 contract, Contract repository, or Student repository changed.

## 2026-07-31 — M6B Native Path-B Callback Composition Extraction

Post-score callback composition now lives in the production-stage integration
module rather than the public `builder.production` façade.  That module binds
existing callbacks to the already canonical M4 Path-B slices two through five;
it adds no second state machine, persistent workflow format, or changed stage
semantics.  The live canonical traversal test follows the new private seam and
continues to prove the exact ordered slices.  Focused native/orchestration,
production, C6, authority, and M6A boundary coverage passed `72 passed, 1
skipped`; Ruff, format, and diff checks passed.  No GPU work, Golden mutation,
authority/configuration change, Contract change, or Student work occurred.

## 2026-07-31 — M6C Delivery Typed-Handoff Extraction

Delivery configuration, failure types, report vocabulary, and the immutable
prepared-delivery handoff now have one dependency-light owner in
`builder.exemplar_delivery_contracts`.  The established `exemplar_delivery`
module forwards the exact same public types while retaining its existing
rerun, staging, assembly, and validation behavior, avoiding a flag-day import
change.  Focused delivery, adversarial linkage, native resume/orchestrator,
production, and live-canonical tests passed `123 passed`; Ruff, formatting,
and diff checks passed.  No delivery semantics, artifact schema/path, M4
ordering, GPU run, Golden fixture, Contract, or Student work changed.

## 2026-07-31 — M6D Packaging Descriptor and Validation Boundary

Package materialization now receives an explicit `ValidatedTomeArtifact`
source handoff and `tome.packaging` no longer directly imports builder or audit
internals.  Existing producer-only full-debug, linkage, C6, and long-tail
checks are delegated through a Tome-side validation adapter with lazy imports
to preserve the established writer import cycle boundary.  Native validators
remain authoritative; packaging output, public APIs, v3 covers/manifests,
profiles, archive behavior, and historical support are unchanged.  Focused
package/bundle/M5/M6 parity and publication-pin coverage passed `66 passed`;
Ruff, formatting, and diff checks passed.  No Contract or Student work, GPU
execution, Golden mutation, or semantic change occurred.

## 2026-07-31 — M6E Initializer and CLI Façade Isolation

The Hydra-recorded initializer violations are resolved: research and frozen
lazy export registries now live in package-local compatibility modules, while
`backends`, `builder`, and `reports` preserve their supported lazy public
exports without direct research edges.  CLI validation now imports
`write_cover_page` through the Tome façade.  Import-isolation, native resume,
production, configuration, publication-pin, and disposition coverage passed
`106 passed`; Ruff, formatting, JSON, and diff checks passed.  No public symbol
was removed and no v3, M4/M5, authority, Golden, Contract, Student, or
accelerator behavior changed.

## 2026-07-31 — M6F Production Boundary Closure

The roadmap's original production-boundary M6 is complete. Public production
facades, typed M4 handoffs, package descriptor/validation boundaries, and
compatibility export registries now state and enforce ownership without
changing the M4 state machine or M5/v3 semantics. The cache-free closure suite
passed `876 passed, 23 skipped in 89.32s`; immutable Golden validation passed
with count `256` and semantic root
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
Compile, Ruff, formatting, JSON, and diff checks passed. The v3 Contract
`v0.2.0` pin/offline mirror, authority v1/v2, fixed 25-field projection,
historical formats, public CLI, artifact contracts, and all intentional M7+
deferrals remain unchanged. No Contract/Student work, accelerator inference,
or fixture regeneration occurred; the unrelated `.DS_Store` remains untouched.

## 2026-07-31 — M6 Corrective Stage and Delivery Ownership

Independent review established that the earlier M6F claim was premature. This
corrective checkpoint moves substantive preflight, score-pass, authority,
selection, delivery, assembly, verification, and terminal-report operations
into focused `builder.production_stages` modules; `builder.production` remains
the public compatibility, configuration, and sole-Path-B routing façade.
Selected-delivery behavior is physically partitioned into rerun, staging,
payload, assembly, validation, parity, and reporting owners, while
`builder.exemplar_delivery` is a narrow import-compatible façade. The M4
native orchestrator remains the only state machine; no artifact semantics,
authority recipe, Golden fixture, Contract, Student, or accelerator work
changed. Focused native/production and selected-delivery characterization is
recorded with this correction; broader dependency and closure evidence remains
open until the validation-boundary and graph-policy checkpoints complete.

## 2026-07-31 — M6 Corrective Validation Boundary and Graph Policy

Reusable teacher-textbook, C6 selection, long-tail, corridor, and selected
delivery/linkage validation now has a Builder-independent
`artifact_validation` owner. Builder compatibility modules forward retained
imports while Tome packaging and producer validation reach the shared owner
without a Builder edge. Package materialization converts its emitted canonical
directory into a complete `ValidatedTomeArtifact` containing the cover,
semantic identity, manifest, profile/inventory, validation evidence, and
authority references before proceeding. A static transitive import graph now
reports the complete forbidden path, detects cycles in production layers, and
includes a forwarding-module regression. Focused architecture, packaging,
descriptor, validation, delivery, and production tests passed during this
checkpoint. No M4/M5/v3 semantics, authority projection, Golden fixture,
Contract, Student, or accelerator behavior changed; final full-suite and
closure evidence remain pending.

## 2026-07-31 — M6 Corrective Closure

The independent-review deficiencies are closed by corrective commits
`4091464` and `5203465`: production stages have real owners, selected delivery
has a behavioral split behind its compatibility façade, reusable validation is
Builder-independent, and package/validation routes cannot transitively import
Builder. The AST graph policy reports complete forbidden paths, detects cycles,
and proves a forwarding module cannot hide a prohibited edge. The corrected
closure record documents retained façade responsibility, removal conditions,
and M7 deferrals without claiming a line-count-only victory. Cache-cleared full
tests, focused M3--M6/Golden/Contract-pin coverage, compile, Ruff, formatting,
JSON, diff, lightweight CLI help, and immutable Golden validation completed;
the Golden result remains selected count `256` and semantic root
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
No Contract, Student, Golden fixture, published v3 asset, authority recipe,
fixed 25-field projection, M4 state-machine behavior, or accelerator workload
changed.

## 2026-07-31 — M6 Final Integration Cleanup

The final integration cleanup corrects the closure status and the measured
`builder.exemplar_delivery` façade size to 42 lines. The static outward-policy
now evaluates every descendant of the governed production, delivery, artifact
validation, and Tome packaging/validation namespaces, rather than treating a
clean package initializer as sufficient. Its new synthetic regression proves a
forbidden dependency in a descendant module is reported with the complete path.
No runtime behavior, public API, Contract pin/mirror, Golden fixture, v3 asset,
Contract repository, or Student repository changed.

## 2026-07-31 — M7B Portable Streaming Validator Correction

The proposed Tome-local v2 sharding contract now has black-box proof rather
than only schema/checksum proof. Its standard-library directory validator
streams the payload index and JSONL shards with scalar digest state, verifies
the acyclic cover/header/inventory graph, fixed paths, raw digests, contiguous
count-based shard ranges, index-to-row bindings, per-shard and whole-sequence
digests, the compact v2 semantic identity, and all 38 semantic payload fields.
The identity binds the sequence digest and count rather than an eager payload
array. Opaque extensions now preserve a declared value and its canonical digest
without interpreting it; unknown profile/capability/digest contracts still fail
closed. New generated package cases prove that stale layout counts, index
addresses, shard sequence digests, raw-envelope-refreshed payload tampering,
and rehashed stale identities are rejected. This remains a Tome-local proposed
contract: RADJAX-Contract, Student, M4/M5 semantics, authority recipes, Golden
fixtures, and accelerator execution are untouched pending the explicit M7
Contract review gate.

## 2026-07-31 — M7C Tome-Local Conformance Corpus

The checksum-pinned v2 conformance catalog now names the generated valid and
adversarial streaming packages and their stable fail-closed categories. It
also explicitly retains v3 as native historical evidence and v2/package-v1 as
incomplete non-inferential descriptors; no historical artifact is promoted to
the new v4 identity. The corpus stays compact and reproducible: payload bytes
are generated in temporary test directories, while the reviewed schemas,
vectors, catalog, and checksum inventory are checked in. Contract, Student,
Golden fixtures, authority behavior, and accelerator work remain untouched.

## 2026-07-31 — M7D Additive V4 Streaming Writer

The new `tome.payload_sharding_v4` writer is a separate, transactional v4
artifact boundary. It streams selected records to compact canonical JSONL with
strict record-count shard boundaries, disk-backed duplicate logical-ID checking,
per-record/index/shard/sequence integrity, a compact sequence-bound semantic
identity, and the acyclic cover/header/inventory graph. Its deterministic
regrouping proof shows physical layout hashes change while semantic identity
does not. The legacy M4 selected-rerun staging, v3 package writer, archive
writer, public v3 readers, authority, Golden fixtures, Contract, and Student
remain unmodified. Further M7D work still needs the source-adapter, transport,
and native/portable parity boundaries before the Contract review gate.

## 2026-07-31 — M7D Bounded-Memory Duplicate Detection

The portable v2 validator now uses a temporary on-disk uniqueness table for
logical IDs instead of retaining an in-memory set. Together with its streaming
JSONL readers and sequence digest sinks, validation retains one encoded record
plus fixed buffers and scalar state, while still rejecting duplicate IDs. The
writer uses the same disk-backed uniqueness approach. No v3 validator, writer,
archive behavior, Contract, Student, Golden fixture, or production runtime
path changed.

## 2026-07-31 — M7D Legacy Adapter and V4 Transport

The v4 writer now has an explicit read-only legacy-artifact adapter: it derives
the established training/authority context, projects only the closed selected
payload fields, and proves the v3 source bytes are unchanged. A separate v4
tar/gzip transport writer fixes member ordering, timestamps, ownership, modes,
and gzip mtime; it is byte-deterministic. The portable validator safely streams
an archive into temporary disk before directory validation, rejecting traversal,
duplicate, special, corrupt, and unsupported members without loading a package
into memory. Legacy bundle APIs and v3 semantics remain untouched.

## 2026-07-31 — M7D Streaming Shard Index

The v4 layout no longer embeds a package-size-proportional shard array. It
references a checksum-bound `payload-shards.jsonl` index, which the writer,
portable validator, and archive path consume sequentially. This makes the
declared bounded-memory model precise: one maximum encoded payload record,
one shard-index record, and fixed digest/database/I/O state; payload and shard
collections are not materialized. The schema/checksum corpus, docs, and
black-box fixtures moved together. No legacy v3 artifact, Contract, Student,
Golden fixture, authority recipe, or production-stage behavior changed.

## 2026-07-31 — M7D Complete Legacy-Artifact Packaging Boundary

The additive v4 package adapter now builds one coherent package from a complete
legacy artifact: it copies profile-permitted non-payload members into staging,
omits legacy cover/manifests/selected wrappers, emits v4 selected shards and
the acyclic v4 manifest graph, validates, and promotes atomically. Student
packages omit raw producer shards; full-debug packages retain permitted source
provenance, while the selected payload identity remains profile-independent.
Tests prove source immutability, complete-package member behavior, legacy
selected-payload projection, and portable validation. The v3 public package
and archive routes remain compatibility paths, not inputs to v4 semantics.

## 2026-07-31 — M7D Tome-Wide Semantic Identity Correction

The v4 semantic identity now includes the sorted non-selected v3 training
payload projection alongside training contract, authority, selected count, and
the streamed selected-payload sequence digest. Only the grouping-sensitive
legacy `selected_exemplars/*` entries are replaced by the sequence projection;
the remainder of the Tome’s training-authoritative semantic boundary is still
bound. This prevents sharding from accidentally narrowing canonical Tome
identity to selected records. The source adapter supplies that projection from
the native v3 identity, while direct synthetic writer callers may explicitly
provide none. No v3 identity recipe, profile behavior, Golden fixture, Contract,
Student, or authority semantics changed.

## 2026-07-31 — M7E Streaming Contract Publication Pin

RADJAX-Contract `0.3.1`, tag `v0.3.1`, commit
`f8ca8c0885d7c539a51d1594ba7a38c4d457b4d` is now the normative owner of the
M7 v4 streaming contract. Tome pins its established Contract dependency to
that release and retains `contracts/radjax_tome/v2` as a checksum-enforced,
offline verified mirror. Source, installed-wheel, and mirror asset parity are
tested byte-for-byte. M7 introduces no production writer or CLI import of the
new Contract-resource API; Contract remains a verification/conformance boundary.
The M6 v1 mirror and v0.2.0 historical evidence remain unchanged.

## 2026-07-31 — M7F Portable Validator Ownership Correction

The reusable M7 portable validator is now Contract-owned at `v0.3.1`; Tome's
tool is a thin compatibility command. This removes the last producer-only
implementation of consumer-visible validation while preserving no new runtime
dependency for Tome production writers or CLI routes.

## 2026-07-31 — M7F Closure Inventory Correction

The Hydra disposition ledger now explicitly owns the M7 v4 streaming writer,
characterization, portable contract, and Contract-publication pin documents.
This is an inventory-only closure correction required by the repository's
tracked-surface audit; it does not change payload, identity, authority, archive,
Contract, Student, Golden, or runtime behavior.

## 2026-07-31 — M7F Closure and Downstream Handoff

M7 closes with the Contract v0.3.0 streaming publication, Tome's verified
offline mirror, v4 source/package adapters, deterministic direct transport,
and portable streaming validation. The closure report records the exact pin,
consumer discovery APIs, claims, nonclaims, and cache-disabled reproduction
commands. M7 makes no claims about Student training, random archive seeks,
accelerator execution, M8 batching, UX, corpus work, or model performance.

## 2026-07-31 — M7 Corrective Characterization

Independent review rejected the earlier M7 closure because its v4 writer was
additive rather than the native Path-B paved road and its archive validation
spooled complete input. The corrective branch now locks three reproduced
baseline failures before implementation: a missing required semantic field was
published successfully, a `.tgz` retaining a `directory` cover declaration
validated successfully, and the deterministic archive writer emitted that
wrong declaration. The focused characterization command failed `3` tests at
the verified `f0b1e2e` / `v0.3.1` baseline for those exact reasons. This is
not a closure claim; production integration, transactional resume, direct
streaming, indexed access, adversarial coverage, and bounded-memory proof are
still pending independent review.

## 2026-07-31 — M7 Corrective Writer Publication Boundary

The v4 writer now rejects a record missing any declared required semantic field
while it is still in its private staging directory, before a successful return
can expose an invalid package. Deterministic archive emission now synthesizes
the cover's physical transport declaration (`tgz` or `rtome`) without changing
the unpacked directory cover, inventory, raw member digests, or layout-
independent semantic identity. The intentionally malformed archive regression
remains pending the Contract-side transport mismatch enforcement; no closure
or native Path-B integration claim is made here.

## 2026-07-31 — M7 Corrective Portable Streaming Boundary

Tome now pins its verification and conformance boundary to the untagged,
additive RADJAX-Contract `0.3.2` candidate commit
`78ba36300f201d75b016b2fdcf5720e467310815`; no published tag was changed.
The checked-in v2 mirror was updated byte-for-byte for the expanded portable
error vocabulary. The Tome-facing Student adapter delegates direct archive
iteration, verification state, and strict/permissive canonicality to the
Contract-owned reader. It reports `fully_verified` only after exhaustion and
`closed_early` after intentional partial consumption. An extracted-directory
adapter delegates indexed `(shard_id, row)` reads to Contract and rejects
archive random access. Focused writer, mismatch, streaming, and indexed-reader
tests pass. Native Path-B publication, resume staging, the full adversarial
matrix, and bounded-memory closure evidence remain pending.

## 2026-07-31 — M7 Corrective Native Publication and Staging

The ordinary canonical Path-B report now identifies a deterministic sibling
v4 directory and `.tgz` only after the retained score, selected-rerun, late
corridor, reconciliation, and validation callbacks complete. The legacy tree
remains the historical/resume input, while the reported v4 path is the paved
consumer artifact. A delivery-owned v4 staging primitive provides atomic
sealed JSONL shards, receipt-backed contiguous-prefix reuse, interruption
cleanup, and rejection of gaps, overlaps, reordering, and capacity mismatch.
The legacy adapter projects declared v4 semantic fields while omitting
legacy-only delivery receipts, and the deterministic tar writer emits the
cover/header/inventory prelude before inventory-governed members for direct
sequential Contract validation. Focused native publication, writer, and
staging tests pass. This remains corrective implementation only; complete
resume wiring, full adversarial/memory coverage, and independent closure review
are still pending.

## 2026-07-31 — M7 Corrective Formatting and Import Hygiene

The corrective streaming/publication surfaces were formatted and their public
imports normalized after integration. Focused native Path-B, staging, writer,
and streaming tests remain green; this is mechanical hygiene only and changes
no contract, artifact, authority, or Golden behavior.

## 2026-07-31 — M7 Corrective Status Record

The historical M7 closure document has been corrected: its `f0b1e2e` closure
claim was rejected by independent review, the preserved v0.3.1 publication is
distinguished from the untagged v0.3.2 Contract candidate, and no official
milestone closure is claimed. The Hydra ownership ledger now records the
terminal native v4 publication stage. This documentation/ledger checkpoint
does not change published v0.3.1 assets, Student, Golden fixtures, authority
contracts, or production semantics.

## 2026-07-31 — M7 Corrective Bounded-Memory Evidence

The direct archive reader is now guarded by a local tracemalloc test with
fixed maximum record size and 8-versus-64-record synthetic payloads. Warmed
measurements were 1,119,345 bytes for 531,792 logical JSONL bytes and
1,148,017 bytes for 4,254,552 logical JSONL bytes; the enforced envelope is
under 1,250,000 bytes, non-scaling by record count, and under half the large
payload. A tracked tar source proves first yield before 80 percent of archive
input, rejects unbounded reads, and proves early close consumes no remainder
or false full-verification state. These are synthetic local correctness tests;
they make no performance or accelerator claim.

## 2026-07-31 — M7 Corrective Transactional Native Publication

The native v4 publisher now consumes the delivery-owned sealed-shard staging
transaction instead of bypassing it. It verifies a receipt-backed contiguous
prefix against native source-order records on resume, seals only missing shards,
then constructs Contract-validated directory and archive candidates before
atomic promotion. Existing final directory/archive pairs are Contract-validated
before resume accepts them; corrupt completed output now fails closed. Focused
native interruption tests cover after staging prepare, after shard sealing,
after staging completion, before archive packing, and before final promotion;
they prove no final partial archive is accepted. The public writer separately
validates strict Contract conformance before directory promotion and writes an
archive through a temporary sibling before validation and replacement. This is
corrective implementation evidence only; independent closure review remains
required.

## 2026-07-31 — M7 Corrective Review Packet Update

The corrective review record now enumerates the implemented native transaction,
resume validation, strict pre-promotion Contract validation, and temporary
archive promotion behavior. It records the adversarial interruption cases
actually exercised and keeps the status explicit: corrective implementation is
complete, but independent M7 closure review is pending. Cache-cleared Tome
validation completed with `939 passed, 23 skipped`; no published v0.3.1 asset,
Student file, Golden fixture, authority recipe, or fixed selection projection
changed.

## 2026-07-31 — M7 Corrective Configuration Boundary

`payload_records_per_shard` is now an explicit execution-only control with a
default of `128`. It flows through legacy intent adaptation, canonical intent,
resolved configuration, execution planning, CLI overrides, the flat execution
adapter, and the native delivery configuration. The focused characterization
proves the value survives every boundary, rejects non-positive/non-integer
values, and is absent from the fixed 25-field selection-authority payload and
hash. This checkpoint changes no physical writer, resume behavior, identity,
authority, or Golden artifact.

## 2026-07-31 — M7 Corrective Native Path-B v4 Publication

The ordinary canonical Path-B terminal reporting callback now publishes the
validated completed native artifact through the v4 adapter only after the
existing late-corridor, linkage-validation, and reconciliation proof.  It
reports one obvious consumer-facing sibling directory and deterministic `.tgz`
archive, including their v4 identity and shard facts; the retained legacy tree
is solely the historical/resume source for that transactional projection.
The focused native integration test proves a configured two-record shard
capacity reaches the final v4 layout and both physical forms pass the portable
validator.  This preserves the sole M4 state machine and does not claim M7
closure; staging-prefix resume and direct-streaming evidence remain separate
corrective work.

## 2026-08-01 — Tome CI Formatting Repair

GitHub Actions run `30701169362` failed only at Ruff's formatting check for a
single Python snippet in `docs/M7_CLOSURE_REPORT.md`; lint and every earlier
CI step passed on both Python 3.11 and 3.12. The snippet now uses Ruff's
canonical single-line form. This documentation-only repair changes no runtime,
contract, artifact, authority, Golden fixture, or M7 claim.

## 2026-08-01 — Native-v3 Student-Consumption v2 Integration Candidate

Tome now pins the published RADJAX-Contract `v0.4.1` authority at
`a6877178d5f07d68f5e0bc28419d0e8e1a58890e` and carries its byte-verified,
offline-capable v2 asset mirror.  The ordinary `student` package path emits a
closed `radjax_tome_cover_v3_student_consumption_v2` declaration and derives
the required NPZ/JSON sidecars inside the package transaction.  The base
native-v3 semantic root remains unchanged; Contract validates the independent
consumption identity and every inventory-backed resource before directory or
archive promotion.  Historical v3 and full-debug packages retain their
existing cover family and behavior.  This is a Tome integration candidate;
the Tome fixture receipt and independent cross-repository closure review remain
pending.

## 2026-08-01 — Native-v3 Student-Consumption v2 Fixture Receipt

The ordinary CPU reference production path generated the current
native-v3 Student fixture, then produced both directory and deterministic
`.tgz` Student deliveries.  The published Contract `v0.4.1` resolver admitted
both forms with `native_v3_student_v2`; the student and full-debug packages
shared the same base native-v3 root while the derived sidecars bound a separate
consumption digest.  `docs/C3_NATIVE_V3_STUDENT_CONSUMPTION_V2_RECEIPT.md`
records the exact commits and run digests.  This receipt is integration
evidence, not an official closure declaration; independent review remains
required.

## 2026-08-01 — Student-Consumption CLI Contract-Parity Repair

Subprocess helpers now preserve an inherited `PYTHONPATH` after Tome's source
root.  This keeps the public CLI path aligned with the caller's pinned
Contract source/wheel during Contract-parity testing rather than silently
falling back to a globally installed older Contract.  The change affects test
execution isolation only; it neither changes the runtime dependency declared
by `pyproject.toml` nor changes package, identity, authority, or fixture
semantics.

## 2026-08-01 — Native-v3 Student-Consumption v3 Integration Candidate

New ordinary native Path-B Student packages emit the explicit
`native_v3_student_v3` declaration and the closed row-range, delivery-receipt,
and authority-reference evidence bodies. Tome pins Contract `v0.5.1` at
`f9c9278b6a467a6ba7a3972e1644bfc3d13abd6b`; its checked-in v3 assets are an
offline checksum-verified mirror, not a second authority. The package boundary
uses Contract validation before directory or `.tgz` promotion. V2 remains an
untouched historical profile with no v3-to-v2 fallback, and base native-v3
semantic identity, authority recipes, and M7 behavior remain unchanged.

## 2026-08-01 — Native-v3 Student-Consumption v3 Historical-Cover Repair

The v3 package-validator integration retains explicit recognition of the
published v2 canonical cover family. New production emits v3 only, but a v2
canonical package continues through the canonical validation path rather than
being demoted to the legacy package-v1 reader. The added regression locks this
compatibility boundary without changing v2 assets or semantics.

## 2026-08-01 — Native-v3 Student-Consumption v3 CPU Receipt

An ordinary canonical Path-B CPU-reference package and its deterministic `.tgz`
delivery were admitted by Contract `v0.5.1` under the explicit v3 profile. The
receipt records matching base and v3 consumption semantic identities across
the two transports, with raw `.tgz` integrity kept separate. This is
cross-repository integration evidence only; it neither changes the immutable
Golden fixture nor authorizes Student implementation.

## 2026-08-01 — Native-v3 Student-Consumption v3 CI Repair

The v3 receipt maps the legacy one-pass producer label
`one_pass_pruned_candidate` to the Contract-owned semantic delivery class
`one_pass_full`; two-pass remains explicitly `two_pass_rerun_selected`.
The Hydra disposition ledger now owns the v3 materializer and receipt. This
restores ordinary native Path-B coverage without weakening profile selection,
authority, or historical v2 validation.

## 2026-08-01 — Native-v3 Student-Consumption v4 Candidate

New Student packages now explicitly emit `native_v3_student_v4` and pin
Contract `v0.6.0` at `b1209f21fef9405776a757f1a5749d3152bbc3c6`; the checked-in
v4 assets are a byte-verified offline mirror. Delivery-receipt and
authority-reference sidecars remain inventory-bound and Contract-validated,
but their body digests are excluded from the v4 batch-semantic identity. The
paired Path A/Path B regression proves identical payloads retain one
consumption digest while preserving distinct receipt evidence. Published v2
and v3 profile handling remains historical and explicit; base native-v3
identity, authority recipes, M7 transport behavior, and Golden data are
unchanged.

## 2026-08-01 — Phase 5 T1 Language/Tokenizer Binding Pin And Capture

Tome pins published RADJAX-Contract `v0.7.0` at
`cac3dd21e0d56df5a9e6fd50b20267e0b8960995` and carries its closed v5 assets
as a checksum-verified offline mirror.  `language_tokenizer_binding.py`
captures Contract-v5-compatible binding and canonical vocabulary evidence from
the instantiated Smoke or explicit fast Hugging Face tokenizer used by the
tokenization loop.  Incomplete vocabulary, normalization/configuration,
added/special-token, or immutable-revision evidence fails closed; metadata is
not promoted into a binding.  V4 remains untouched historical behavior.

## 2026-08-01 — Phase 5 T2 v5 Package Transaction

New Student packages emit the explicit `native_v3_student_v5` binding manifest,
inventory-backed vocabulary resource, and v5 cover declaration.  The package
transaction first requires Contract-admitted source capture and verifies every
source payload token ID is in its declared vocabulary domain.  Directory and
archive candidates are strictly admitted before atomic public promotion; the
archive path validates its deterministic inventory before safe extraction for
the Contract's directory-based v5 resolver.  `smoke_tokenizer` is a separate
CPU-only producer whose instantiated SmokeTokenizer both encodes the payloads
and supplies capture.  Historical CPU/fake artifacts have no inferred v5
binding and fail closed; their v4 behavior is not rewritten or reinterpreted.

## 2026-08-01 — Phase 5 T3 Deterministic v5 Fixture And Receipt

The committed `native_v3_student_v5_smoke_v1` fixture is generated by the
ordinary `smoke_tokenizer` production builder and ordinary transactional
Student package path, not a synthetic Contract fixture writer. Its
machine-readable receipt binds Contract `v0.7.0` at
`cac3dd21e0d56df5a9e6fd50b20267e0b8960995`, Tome `3861c23` at production,
the generic binding digest, profile, exact strict Contract validator entry
point, fixed producer configuration, and fixture semantic/raw/tree evidence.
The T3 test regenerates twice and proves byte-identical binding and vocabulary
resources plus equal generic-binding and package semantic digests; physical
raw/tree digests remain explicitly non-semantic evidence. Contract and Tome
validators admit the committed package. This does not modify historical v4,
V64/T5, Golden, runtime, training, accelerator, or quality claims.

## 2026-08-02 — M7 Corrective Acceptance Status Reconciliation

The earlier `f0b1e2e` M7 closure claim remains correctly recorded as rejected
while corrective work was incomplete. The corrective implementation completed
the native transaction and resume obligations at `43d7816`, recorded its
evidence at `77fd342`, and was accepted into `main` by merge `48a7b2f`
("Merge M7 payload sharding and streaming validation"). With no affirmative
post-merge M7 blocker, M7 is now recorded as closed. The historical untagged
Contract `0.3.2` candidate is not rewritten as a release; the M7 v0.3.1/v2
mirror remains historical evidence while current Tome uses a later Contract
pin. This wording-only reconciliation changes no production behavior, schema,
Contract asset, Golden fixture, authority recipe, fixed 25-field projection,
Student surface, or M8 scope.

## 2026-08-02 — M8A Selected-Pass Measurement Readiness

M8A adds a private, benchmark-only measurement seam around the existing
canonical Path-B selected-delivery rerun. The control is not public build
configuration: it requires an immutable post-C5 checkpoint digest, a distinct
temporary output root, and a cap of 1, 2, 4, or 8 while leaving the requested
selected-rerun batch size and all authority inputs unchanged. The replay guard
hashes checkpoint files before and after the canonical rerun, requires frozen
C5 records, and copies rather than hard-links upstream files so measurement
writes cannot alter score or selection evidence. Additive
`selected_pass_execution_v1` diagnostics record phase accounting, source and
coordinate shape observations, process/optional CUDA resources, OOM/reload
events, and explicit compilation-not-authorized state; the deterministic
64-source largest-remainder sampler is marked approximation-only. Focused
fake-backend tests are nonrepresentative readiness evidence, not accelerator
or performance proof. This commit changes no public CLI/configuration,
selection/authority hash, Contract, M7 publication, Golden fixture, Student,
production batch policy, or M8B behavior; canonical-anchor and representative
T4 evidence remain pending independent M8A review.

## 2026-08-02 — M8A Instrumentation and Checkpoint Remediation

The M8A readiness seam now attaches its observer privately to the actual
`gpu_torch` backend and emits phase-separated host-wall metadata only for that
private measurement path: model/tokenizer load, tokenization, H2D, forward,
selected-position index/gather, compact reduction, and compact D2H. The
metadata records eager execution plus explicit compilation-not-authorized and
CUDA-event-not-available state; no backend configuration, authority, or public
metadata contract gains a measurement flag. Post-C5 replay capture now fails
closed unless a persisted, content-addressed manifest binds named score,
corridor, authority, C2-C5, passport, model, tokenizer, corpus, and resolved
config evidence. Focused mocked GPU and fake-artifact checks prove observer
attachment, manifest tamper rejection, cap-isolated exact payload sequence,
and replay ownership; they remain nonrepresentative. This remediation does not
begin M8B, run a T4 anchor, change production selection or M7 semantics, or
make any accelerator/performance claim.

## 2026-08-02 — M8A Final Observer Phase Reconciliation

The private M8A observer now keeps selected-position index preparation and
selected-row gather as distinct phase names without changing canonical
selection, batching, reduction, or payload computation. The staging owner
measures index preparation, while `gpu_torch` measures the existing selected
row gather. Its attached-only metadata now records shape and dtype facts for
input, logits, gathered logits, and the compact result without an extra device
transfer or materialization; mocked GPU tests cover both phase separation and
those fields. This scoped diagnostic remediation remains private to
benchmark-only M8A measurement: it does not start M8B or alter authority,
production behavior, M7, Contract, Golden, Student, model, corpus, or
performance claims.

## 2026-08-02 — M7 Selected-Rerun Entropy Accumulation Correction

A representative current-main T4 anchor exposed a reproducible strict Path-B
linkage failure: score-pass and selected-rerun top-token identities agreed,
but entropy differed by two quantization steps for a frozen C5 coordinate.
The underlying Gemma logits were identical across the two batch compositions.
The corrective change makes the dynamic selected-rerun reducer use the same
log-probability representation and chunk-wise entropy accumulation as the
score reducer, while retaining the existing one-step linkage tolerance. A
focused GPU reducer regression asserts exact score/rerun entropy agreement
under chunked vocabulary reduction. This is a narrow M7 correctness repair;
it does not relax validation, alter selection, authority, requested batch
size, Contract, Golden fixtures, Student, or M8B policy. A fresh canonical T4
anchor remains required before M7 or M8A acceptance is claimed.

## 2026-08-02 — M8A Private T4 Replay Evidence Driver

The M8A driver is a development-only script with no package or production CLI
entry point. It rejects a non-passing canonical anchor, snapshots only the
post-C5 score store, C2-C5/corridor authority evidence, source passports, and
model/tokenizer/corpus/config identities into a content-addressed checkpoint,
then invokes the existing selected-rerun owner in fresh system-temporary
outputs. Each replay rehashes the snapshot, asserts zero score and selection
writer invocations, preserves requested batch size eight, checks the five
percent phase-accounting gate, and compares authority, frozen records, and
payload hashes across the required full and capped runs. Raw evidence remains
outside the repository. This adds no production path, policy, M7 semantic
change, or M8B work; representative results remain pending the corrected
canonical anchor's terminal acceptance.

## 2026-08-02 — M8A Exclusive Phase Accounting Correction

The first representative replay correctly rejected its own measurement report:
the observer had counted each native payload write both as an explicit
hash/JSON/atomic-write phase and inside the enclosing payload-conversion
phase. The correction subtracts the measured nested write interval from the
conversion/linkage interval, leaving the two additive and the selected-pass
denominator unchanged. Focused selected-delivery and M8A readiness tests pass
after the change. This is a private measurement-accounting repair only: it
does not change the canonical rerun's payload bytes, selection, authority,
batching, M7 behavior, Contract, or any production output.

## 2026-08-02 — M8A Batch-Shape Equivalence Evidence Handling

The representative M8A cap matrix reached its intended cross-cap comparison:
each replay independently passed immutable-checkpoint, zero score/selection
writer, authority, linkage, and accounting gates, while the canonical payload
hash sequence differed across at least one execution cap. That is an
equivalence-rejection result, not a measurement failure to suppress. The
private driver now records the exact cross-cap comparison outcome in its raw
report while continuing to fail closed if repeated runs at the same cap differ.
It does not alter payloads, hashes, tolerance, batching, selection, authority,
M7, Contract, or production behavior. The result excludes a batch-shape change
from M8B consideration; M8A remains a measurement/evidence milestone only.

## 2026-08-02 — M8A Representative T4 Baseline Evidence

The accepted fresh T4 anchor completed its canonical production sequence,
strict selected linkage, and Contract-validated v4 directory/archive
publication, yielding 256 selected records and semantic identity
`sha256:9d5796a7ff2d4db5000b9a128502cc68d0933f13be4786c4cefc1982a615dea7`.
The content-addressed post-C5 checkpoint bound 43 required upstream files.
Three 213-source/256-coordinate full cap-eight replays and the complete
64-source cap 1/2/4/8 warm-up-plus-three matrix all proved zero score/selection
writer invocation, unchanged checkpoint evidence, fixed requested batch eight,
shared authority, no OOM/fallback, and exact phase reconciliation.

Measured full selected-pass wall times were 365.188, 358.474, and 357.258
seconds. Hash/JSON/atomic staging dominated at approximately 324 seconds per
run (about 90 percent); payload conversion/linkage was about 23 seconds and
teacher forward 2.3–5.4 seconds. The 64-source approximation found cap one
faster but its exact payload hash sequence differed from the cap-eight
reference, whereas caps two, four, and eight matched. This is an
equivalence-rejection fact, so M8A selects no production batch-shape change
and authorizes neither M8B nor M8C. Raw path-bearing evidence remains outside
the repository; the committed measurement document records the reproducible,
bounded conclusion. M8A stops for independent review.

## 2026-08-02 — M8A External Evidence Receipt

The private T4 raw baseline report, post-C5 manifest, canonical production
report, validation report, and selected-linkage audit now have a committed
sanitized SHA-256 receipt. This preserves the required no-private-path/no-raw
artifact boundary while giving an authorized independent reviewer exact bytes
to verify. The receipt changes no benchmark result, authority, production
behavior, M7 claim, or M8B status.

## 2026-08-02 — M7 Regression Test Format Normalization

The focused M7 dynamic-reducer regression test was normalized with the
repository formatter after the M8A full-suite validation. This is a
format-only follow-up: no test assertion, production code, authority,
measurement evidence, or milestone status changed.

## 2026-08-02 — B3 Explicit v6 Composition

Tome now has an explicit, opt-in native_v3_student_v6 package path pinned to
the immutable Contract v0.8.0 commit. It composes the existing v5
language/tokenizer binding with closed behavioral resource projections and
requires Contract admission before publication. The default remains v5; this
does not change M8A, start M8B or B4, alter Student, create a canonical
fixture, or infer missing full-grid assignments.

## 2026-08-02 — B3 v6 Native M7 Binding Repair

The v6 package adapter now discovers the accepted native Path-B M7 sibling
`output_dir.v4.tgz` and binds its exact bytes as the selected-exemplar
authority resource with `m7_tome_archive` encoding. It preserves the M7
archive rather than flattening or re-sharding it, and Contract admission runs
before either directory or archive promotion. Focused tests cover both
transports, bounded verified opening states, and forced Contract-admission
failure with no promoted output. No default-profile, M8B, B4, Student, or
Contract change is made.

## 2026-08-02 — B3 v6 M7 Semantic-Rejection and Pin Repair

The Tome B3 suite now rebuilds the real native M7 sibling with an internally
consistent inner exemplar mutation, verifies the M7 archive itself still
admits, and proves v6 rejects it at exemplar semantics with BRC027 rather
than raw integrity. Historical v5 mirror provenance remains pinned to its
v0.7.0 source commit while the active dependency assertion separately pins
the immutable Contract v0.8.0 commit. No default-profile, M8B, B4, Student,
or Contract change is made.

## 2026-08-02 — B4 Ordinary v6 Fixture

An ordinary CPU smoke production run now supplies a committed explicit-v6
fixture, its native M7 sibling, directory and archive package forms, and a
machine-readable receipt. The evidence resolves every declared resource via
Contract and records distinct behavioral, composition, package, and raw
domains. This does not migrate the default profile or alter Student, Contract,
M8B, or B4-external behavior.

## 2026-08-02 — B4 Fixture Inventory Completion

The B4 fixture command and verifier are recorded in the required Hydra
inventory. This completes the repository-owned fixture evidence inventory
without changing artifact authority, default selection, Contract, Student,
M8B, or B4 scope.

## 2026-08-02 — B4 Fixture Verification Strengthening

The B4 receipt now binds the native M7 sibling to the packaged M7 member and
the fixture builder validates both directory and archive transports before
recording evidence. Repeated ordinary production proves the recorded v6
identity domains stable, while the verifier exercises every Contract opener.
No Contract, Student, default profile, M8B, or B4-external change is made.

## 2026-08-02 — B4 Public Router and Receipt Audit Repair

The fixture now exercises the public generic v6 resource dispatcher for every
non-M7 role and verifies every recorded descriptor identity and resource
declaration against a fresh Contract resolution. M7 remains on its dedicated
bounded stream opener. This is evidence-only and leaves Contract, Student,
the default profile, M8B, and B4 scope unchanged.

## 2026-08-02 — B4 JSONL Opener Closure Repair

The fixture now exercises ordinary JSONL roles through the whole-resource
iterator and both public verified-byte openers, while M7 remains dedicated to
its stream opener. Receipt verification now independently compares both
directory and archive resource registries. No behavioral, Contract, Student,
default-profile, M8B, or B4 scope change is made.

## 2026-08-02 — P5.U1 Leakage-Free v6 Fixture Refresh

The ordinary `native_v3_student_v6_smoke` production configuration now selects
three exemplars through the canonical builder and normal Student packager. Its
committed directory, archive, M7 sibling, manifests, and receipt are refreshed
from that run. The verification binds the exact deterministic coordinates
`corpus_000000003:0`, `corpus_000000003:3`, and `corpus_000000001:2`, proves
at least two exemplar-bearing example identities, and requires all five public
identity domains to agree across strict directory/archive resolution. Contract,
Student, the default profile, M8B, and Tome behavior outside this ordinary
proof fixture remain unchanged.

## 2026-08-03 — M8B.1 Frozen Selected-Staging Statistics

M8B now fixes its private three-run benchmark definitions before collecting
new representative evidence: median, spread, combined spread, noise and
material-regression predicates, and host/device peak limits. The receipt
validator rejects altered formula projections. This is measurement-only: it
does not alter the canonical selected-delivery path, authority, artifacts,
batch policy, Contract, Student, Golden evidence, or v5 default.

## 2026-08-03 — M8B.1 Initial-Staging Diagnostic Split

Private selected-pass diagnostics now separate canonical body encoding/hash,
pretty staging JSON encoding, temporary write, close, and atomic replacement.
The historical M8A aggregate is retained and is not relabeled. Production has
no measurement control and therefore preserves its former path and bytes; no
writer strategy, authority, payload, batch policy, Contract, Student, Golden,
or v5 behavior changes in this measurement-only commit.

## 2026-08-03 — M8B.1 Current-Base Baseline Driver

The private M8B driver reuses only the immutable post-C5 checkpoint and the
canonical selected-rerun owner for three requested/effective cap-eight runs.
It binds the frozen statistical projection, validates same-cap payload parity,
and writes path-bearing raw evidence outside the repository. This driver is
not a public CLI, does not rerun score or selection work, and does not change
production behavior, artifacts, authority, batch policy, Contract, Student,
Golden evidence, or the v5 default.

## 2026-08-06 — M8B.1 Representative T4 Current-Base Evidence

The frozen three-run cap-eight T4 current-base replay completed at
`31b94ad8cea639d23eeb7b569a3c5041d52e742f` against the immutable 43-file
post-C5 checkpoint. Median selected-pass wall time was 262.905 seconds and
median initial staging was 236.268 seconds, or 89.868 percent of selected-pass
time; the required 50-percent M8B.1 gate passes. Every replay preserved
same-cap payload parity, batch-eight authority, immutable checkpoint evidence,
zero score/selection invocations, no CUDA OOM retry, and exact phase
accounting. The committed receipt binds the path-bearing raw report by digest
only. This authorizes the bounded M8B.2 staging implementation, but makes no
performance-success, batch-policy, Contract, Student, Golden, publication, or
production-behavior claim.
## 2026-08-06 — M8B.2 Bounded Canonical Payload Staging

The canonical native selected-payload staging owner now encodes each initial
payload once through a bounded standard-library JSON stream. It hashes the
exact compact canonical object while retaining only grammar state and encoder
chunks, appends the hash member atomically, and reopens the completed file to
verify the existing payload-hash rule. Grammar tests exercise outer-object and
string/escape chunk boundaries, while transaction checks preserve an existing
target on replacement failure and clean orphaned non-resumable temporaries.
This is an internal staging strategy only: the selected owner, authority,
record order, later synchronization, promoted bytes, batch policy, Contract,
Student, Golden evidence, v5 default, and M7 behavior remain unchanged.

## 2026-08-06 — M8B.3 Selected-Staging Regression Closure

The M8B.2 streaming implementation passed its grammar, measurement, selected
delivery, M7 resume/streaming, B4 v6 receipt, and Hydra-ledger regression
suite, plus the repository-wide regression run. The test boundary includes the
existing canonical v5/v6 and native M7 checks; no fixture or Golden artifact
was regenerated. The implementation remains a private execution strategy of
the sole canonical selected-delivery owner, with no Contract or Student change,
no v5-default change, no batch-policy change, and no Candidate-3 rewrite
avoidance.

## 2026-08-06 — M8B.4 Candidate Replay Protocol

The private M8B driver can now perform the required isolated cap-eight warm-up
and three-run candidate replay against the retained digest-bound M8B.1 raw
receipt. It refuses a different immutable checkpoint and records the baseline
report digest plus frozen staging and selected-pass comparisons. The script
remains private, invokes only the canonical selected-rerun owner, and changes
no production path, authority, batch size, Contract, Student, fixture, Golden,
or public CLI.

## 2026-08-06 — M8B.4 Negative Result and M8B.2 Revert

The representative T4 candidate receipt showed median initial staging of
712.649 seconds and selected-pass wall time of 842.997 seconds, versus the
M8B.1 baselines of 236.268 and 262.905 seconds. The bounded streaming writer
therefore regressed both required metrics and failed the frozen noise-aware
improvement gate. The M8B.2 production implementation is reverted in this new
append-only commit; its private benchmark evidence remains retained. This
records no justified production optimization and does not authorize batching,
Candidate 3, Contract or Student work, or M9.

## 2026-08-06 — M8B.4 Sanitized Candidate Receipt

The negative candidate report and baseline report are committed as exact raw
evidence digests only. The T4 studio was stopped after collection. M8B now
stops for independent review with the restored pre-optimization production
staging path; no M9 or further engineering starts here.

## 2026-08-06 — Program C Contract v3 Adoption Pin

The isolated `adopt/tome-artifact-contract-v3` branch pins the explicit v3
Tome artifact path to RADJAX-Contract release `0.9.0`, peeled commit
`1fa43e1aea2e198511db86dafb0aeefa525d48c7`, and mirrors the byte-verified v3
asset tree under `contracts/radjax_tome/v3`. The default artifact Contract v2,
Student v5/v6 contracts, M7/v4 transport, Golden evidence, M8, and M9 remain
unchanged. This commit only establishes the release identity and offline
verification surface; no production v3 emission or default migration is
claimed yet.

## 2026-08-06 — Program C Explicit v3 Production Projection

The canonical Path-B terminal handoff now supports an opt-in
`--artifact-contract-version v3` projection. It snapshots the finalized
selected records and payloads after late corridor linkage, projects only the
released Contract v3 closed record fields, and publishes independently
validated directory and `.v3.tgz` artifacts. The existing v2/default and M7/v4
publication paths remain unchanged; this commit makes no performance or
default-migration claim.

## 2026-08-06 — Program C v3 Verification and Transactional Publication

The opt-in v3 path now exposes a thin `verify-artifact` adapter for standard,
governed, and externally attested Contract validation, with external evidence
required outside the artifact under test. Directory and archive publication
are separate private journaled transactions using the released Contract state
machine; each validates before promotion and records no journal state in public
artifacts. The v2 default, historical v4/v5/v6 paths, Student, Golden data,
M8, and M9 remain unchanged.

## 2026-08-06 — Program C v3 Finalized Payload Handoff

Canonical opt-in v3 publication now receives the complete already-materialized
selected payload values through a typed extension of the existing finalized
handoff. It does not reread staged or v4 files, repeat score/selection logic, or
alter the v2 summary-only default path. The handoff also carries the existing
selection-authority identity explicitly; all Contract v3 validation remains
owned by the released Contract.

## 2026-08-06 — Program C v3 Fixture and Compatibility Proof

The opt-in Contract v3 canonical path produced the committed ordinary-CPU
four-record, two-shard fixture and separately validated directory/archive
transports. Standard integrity, governed comparison, external attestation,
archive receipt, raw/graph corruption, pre-yield streaming, resharding, and
coherent-replacement behavior are covered by focused tests. The v2 default,
historical v4/v5/v6 paths, Student, Golden data, M8, and M9 remain unchanged;
this is adoption evidence only and makes no default-migration or performance
claim.

## 2026-08-06 — Program C Historical Regression Inventory Reconciliation

The full-suite inventory expectations now include the authorized opt-in
`verify-artifact` command and the `artifact_contract_version` configuration
field. Historical Student v4/v5 tests continue to validate their frozen
mirrors while recognizing the released Contract v3 commit as the active
dependency pin. No historical package behavior, v2 default, Student path, or
Golden evidence was changed.

## 2026-08-06 — Program C Ledger Regression Surface

Hydra validation now recognizes the repository's existing append-only record
kinds and historical `docs:` aliases while including the new v3 verification
command and fixture surfaces. This is ledger/test reconciliation only; no
production behavior or released artifact was changed.

## 2026-08-06 — Program C Hydra Verification Surface

Hydra now records the opt-in `cli:verify-artifact` command as a canonical
Program-C surface. The append-only inventory continues to distinguish
historical compatibility entries and special `docs:` aliases; no default CLI,
released behavior, or prior artifact contract was altered.

## 2026-08-06 — Program C Formatting Gate

The source-bound Ruff format gate was applied to the v3 configuration adapter,
Contract pin test, and Hydra inventory test. The changes are formatting-only;
the v3 handoff, default v2 behavior, historical compatibility, and all artifact
identities remain unchanged.

## 2026-08-06 — Program C v3 `.rtome` Transport Proof

The adopted v3 path now provides a transport-only wrapper for an already
promoted v3 directory and proves Contract acceptance of `.rtome`. It does not
emit an additional canonical sibling or construct a second semantic package;
directory and `.tgz` remain the canonical publication outputs.

## 2026-08-06 — Program C Transactional Publication Hardening

The opt-in v3 publisher now separates directory and archive journal receipts,
seals bytes before receipt commits, validates archive candidates before public
visibility, uses deterministic transport metadata, refuses stale private
transactions, and exposes bounded PC39–PC47 fault-injection boundaries. A
promoted directory can be resumed into an independently validated archive;
the pair is never reported as atomically visible. The v2 default and all
historical paths remain unchanged.

## 2026-08-06 — Program C Fixture Mutation and Rebuild Proofs

The ordinary-CPU v3 fixture now records its expanded build and public-member
receipt projection, deterministic archive identity, root-change metamorphic
checks, corruption/index mutation rejection, assurance-mode outcomes,
streaming guards, and archive-only resume behavior. These are opt-in adoption
proofs only; no Golden artifact, v2 default, Student path, M8 state, or M9
work was changed.

## 2026-08-06 — Program C Released Contract Parity Evidence

The v3 adoption branch now records sanitized release receipt and asset-hash
metadata for the immutable RADJAX Contract v0.9.0 release. Source, release
asset, and offline mirror identities are checked without introducing network
or mutable dependency assumptions; no Contract, Student, Golden, M8, or M9
behavior changed.

## 2026-08-06 — Program C Archive-Resume Journal Closure

Archive-only resume now records its own private Contract journal transitions,
reuses a matching interrupted journal, validates the promoted directory before
creating archive bytes, and removes private state only after the archive is
durably promoted. This closes the transactional recovery correction without
changing any released format, default v2 behavior, Student, Golden, M8, or M9.

## 2026-08-07 — Program C Archive Journal Validation and Idempotent Cleanup

Archive recovery now validates every discovered private journal object through
the released Contract journal APIs and compares its Tome-owned binding with the
promoted directory, intended archive, authority, policy, configuration,
semantic root, transaction relationship, and sealed receipts before any
overwrite or resume work. Matching pre-existing archives are validated and
completed without rewriting; conflicting, stale, malformed, mixed-run,
cross-authority, cross-configuration, or unreceipted state fails closed.
Bound private staging is fsynced and removed before the journal root only after
both independently promoted outputs are valid, so completion-marker and
interrupted-cleanup recovery are idempotent. This bounded correction leaves
the v3 public identity, Contract release, v2 default, historical paths,
Student, Golden evidence, M8, and M9 unchanged.

## 2026-08-07 — Program C Recovery Topology and Private Ownership Correction

The v3 recovery dispatcher now recognizes only explicitly bound private
topologies: canonical directory-only state after PC45--PC47, canonical
directory-plus-archive state, and fresh archive-only repack state. Directory
promotion is validated and completed through the released Contract restart
disposition before archive publication begins; archive-only state never
fabricates a directory journal. All private roots, journals, staging paths,
receipts, markers, and nested components are inspected without following
symlinks before mutation or cleanup. Directory and archive remain independent
transactions, public v3 bytes and identities are unchanged, and the v2 default,
historical paths, Student, Golden evidence, M8, and M9 remain untouched.

## 2026-08-07 — Program C Deterministic Private Identity Correction

The v3 publication path now derives directory and canonical archive transaction
identities from validated non-journal facts, derives fresh archive-only identity
from the validated public semantic root and explicit topology, and derives
journal-root and role-specific staging names from those identities. Recovery
rejects coherently rewritten transaction or staging claims, foreign sibling
staging, and stripped canonical journals relabeled as archive-only. Cleanup is
authorized only for exact derived private paths under Tome's documented local
ownership model. Contract 0.9.0 remains authoritative for journal structure,
receipts, transitions, and restart disposition; public v3 bytes and identities,
the v2 default, historical paths, Student, Golden evidence, M8, and M9 remain
unchanged.

## 2026-08-08 — Path-B M7 Byte-Reproducibility Correction

Fresh ordinary Path-B builds from one shared input set reproduced identical
selected records and shard bytes, but the native v4 M7 archive diverged first
in copied runtime/debug members: wall-clock timestamps, machine-local source
paths, staging paths, and selected-rerun timing/host measurements. The
behavior-bearing selected payload was already equal; the archive inventory was
binding incidental producer diagnostics. The terminal v4 writer now gives the
Student profile a deterministic semantic/core projection: private C6,
reports, run/progress, timing, production-report, linkage-audit, and
side-board diagnostics remain in the legacy/full-debug surfaces but are not
M7 Student members; required core timestamps use a fixed nonsemantic epoch and
retained absolute paths become source-relative or stable external-basename
references. The normal writer then seals the same deterministic v4 directory
and archive under Contract v0.8.3/a33a0a0d; legacy production, v2 default,
Student, Contract, Golden evidence, M8, and M9 remain untouched.
The explicit v6 Student package path applies the same projection before
materializing Contract v6 resources and manifests, so the native M7 sibling
and v6 directory/archive package forms reproduce from the same declared
inputs. No committed P6.U1 worktree fixture was modified.

## 2026-08-08 — Tome-Owned P6.U1 Reduced-Burn Authority Checkpoint

The bounded P6.U1 authority checkpoint publishes a maintained ordinary CPU
smoke-tokenizer producer command, a complete normalized declared-input record,
and committed input bytes. It uses Tome's current Contract `0.9.0` pin at
`1fa43e1aea2e198511db86dafb0aeefa525d48c7`; the stale `0.8.3`/`a33a0a0d`
worktree material remains non-authoritative. The producer derives its build
configuration from the declared record, reuses the exact path-bound corpus
manifest, and records strict v6 admission, source/passport linkage, resource
identities, and pair-reproducibility evidence. No Student or Contract files,
v2 defaults, public identity semantics, Golden evidence, M8, or M9 are changed.

## 2026-08-08 — P6.U1 Authority Fixture Generated

The committed reduced-burn fixture and receipt were generated after producer
checkpoint `340998ee62e9608ae69c10cb9fd21f6ab72d2584` through the published
command, using Contract `radjax-contract` `0.9.0` at
`1fa43e1aea2e198511db86dafb0aeefa525d48c7`. Two fresh builds first matched
canonical declared-input bytes
(`sha256:0cf34d02a192790f8c010446e5e7daa900a10bf867b4863b580e769dfcb6822c`)
and then matched native M7 bytes, Student directory bytes, and Student archive
bytes. The generated evidence qualifies 64 stable examples, 4096 valid tokens,
and 60 selected IDs; no output was hand-edited or postprocessed. Student and
Contract remain unchanged and P6.U1 is not a Student-side acceptance claim.

## 2026-08-08 — P6.U1 Declared-Input Closure

The authority preflight now closes the Contract package, input-root, tokenizer
identifier, and teacher model-name bindings against their committed sources;
mutating those identity-bearing inputs is rejected before production. Hydra
classifies the normalized input record and input directory as fixture surfaces,
and no public artifact bytes or Contract/Student files changed.

The authority tests also exercise closed-shape mutations for the Contract
package, input root, and tokenizer identifier, in addition to the teacher model
provenance mutation; each is rejected before output production.

## 2026-08-15 — M8C Selected-Pass Staging Diagnostic

The measurement-only M8C checkpoint recovered the exact retained post-C5
checkpoint from the dedicated Tome Lightning Studio, transferred it to a
Modal Tesla T4, and ran three fresh cap-eight replays on Tome
`6a6c65378cfd86a190e44e861ed9323927c2acc8`. The existing private
`selected_pass_execution_v1` observer remained disabled by default and no
production behavior changed. The fresh median selected-pass wall was
305.824600515 seconds and initial staging was 275.400979866 seconds (90.05%).
Staging JSON encoding (193.809392 seconds median) and canonical body
encoding/hash (79.466049 seconds) dominated; the T4 was idle during the CPU
serialization path. Corridor reread/rehash, post-linkage evidence hashing,
archive creation, Contract validation, and packaging were explicitly outside
or unmeasured in this boundary and are not claimed to be zero-cost. The raw
report and analysis are retained under `docs/evidence/` and
`docs/M8C_SELECTED_PASS_STAGING_DIAGNOSTIC.md`. No optimization, Contract or
Student change, default change, or M9 work was performed; a later checkpoint
must establish byte/read/hash counters and crash-safe staging invalidation
before selecting receipt propagation.

The follow-up evidence-completeness correction publishes a maintained Modal
wrapper with explicit source, Contract, checkpoint, model, and expected-commit
inputs, and extends the disabled-by-default observer with operation-count and
byte-count fields for canonical hashing, JSON encoding, temporary writes, and
atomic replacement. Focused M8 tests cover the new schema. Corridor rereads,
post-linkage evidence, archive creation, Contract validation, and packaging
remain explicitly unmeasured and are not treated as zero-cost; this checkpoint
still selects no optimization.

The maintained Modal wrapper binds the exact expected source commit through a
narrow remote `git rev-parse HEAD` shim because the source mount omits `.git`;
it does not mount or trust mutable provider history.

The runner passes the expected source commit as an explicit remote function
argument as well, because provider containers do not inherit the local shell
environment used to construct the mounts.

The remote module import now defers the required-commit check to the local
entrypoint; the remote function receives that value explicitly, avoiding an
ambient provider environment dependency.

Hydra classifies the committed sanitized M8C raw report as a fixture evidence
surface, matching the repository's closed disposition vocabulary.

The maintained runner writes the fresh report to the established
`m8b_selected_staging_baseline_current.json` volume path, replacing only the
prior diagnostic copy after a successful complete run.

The maintained Modal runner now commits start, failure, and completion markers
and also writes a uniquely named diagnostic copy, making remote persistence
observable without changing the selected-pass producer.

It also returns a compressed report through the authenticated Modal result and
materializes it locally, providing an independent transfer path when a remote
volume snapshot races the final commit.

The fresh counter-bearing Modal series could not be promoted: one completed
summary without a retrievable raw report, and a second lost its gRPC heartbeat
before result transfer. The checkpoint therefore remains measurement-blocked;
the historical raw report was not rewritten or relabeled.

The runner now refuses to publish a completed report unless every run carries
the selected-pass operation-count ledger, and it supports a bounded post-run
hold for direct container transfer when the client result stream is impaired.

The runner also mounts the terminal production anchor separately from the
43-file immutable checkpoint, matching the M8 authority layout where the
production report is intentionally outside the checkpoint manifest.

Modal volume commit was observed to return `Operation not supported by device`
on the diagnostic mount. The runner therefore treats the private `/tmp`
report plus authenticated result/direct-container transfer as authoritative
for this measurement and no longer blocks completion on that volume.

The volume mount itself was also removed after container inspection reproduced
the same provider filesystem error even without a commit operation.

2026-08-15 — M8C fresh controlled baseline evidence

The measurement-only M8C checkpoint now retains the counter-bearing three-run
Modal T4 report and sanitized environment record for Tome `5e6f9a7`, using the
restored M8B checkpoint and separately mounted production anchor. The raw
report was transferred from the held container after the diagnostic volume
proved unsupported; no production optimization or trust shortcut was added.
The measured median selected pass is 308.435139861 seconds, with 274.397374801
seconds initial staging. Operation counts establish 768 canonical hashes,
6,109,840,830 canonical bytes read, 13,358,155,263 JSON bytes written, and
768 atomic replacements across the three runs. Corridor, typed-evidence,
archive, Contract-validation, and packaging boundaries remain explicitly
unmeasured and are recorded as next-checkpoint gaps. Lightning Studio remains
stopped, Modal resources were torn down, and Contract/Student behavior is
unchanged.

2026-08-15 — M8D payload anatomy diagnostics

M8D adds disabled-by-default diagnostics at the existing selected-exemplar
staging and corridor-synchronization seams. The observer records dynamic-K and
retained-mass distributions, bounded field-size samples, cumulative canonical
and pretty-byte totals, reread/rehash/rewrite counts, and post-linkage timing.
It does not alter payload construction, dynamic top-K/CSL behavior, selection,
corridor semantics, checkpointing, or publication defaults. A bounded
full-lifecycle replay is required before any optimization candidate is chosen.

The first M8D remote attempt was stopped after instrumentation overhead was
observed to dominate progress: encoding every field independently multiplied
the existing JSON work. The hook was narrowed to three bounded exact field
samples plus reused operation-ledger byte totals and cheap all-record K/mass
facts before another lifecycle run.

The first bounded full-lifecycle replay completed its selected staging and
publication work but could not emit its diagnostic receipt because the frozen
M8B three-run summary helper rejected the intentionally single-run lifecycle
mode. The receipt writer now records explicit single-run observations without
pretending to provide an M8B median.

2026-08-15 — M8D complete selected-exemplar lifecycle evidence

The corrected one-run Modal T4 lifecycle replay at Tome
9b046aedb7ce7d22eda2884f9e7703ad55b4dfb2 records payload anatomy, dynamic-K
and retained-mass distributions, bounded field-size samples, cumulative
canonical and pretty-JSON work, and the post-linkage corridor reread/rehash/
rewrite cycle. 145 of 256 exemplars are full vocabulary width; initial staging
is 273.075659 seconds and the post-selected corridor synchronization rewrite is
578.148473 seconds. The report and environment record are retained under
docs/evidence/M8D_*.json. No optimization was implemented. Archive, Contract
validation, typed post-linkage evidence, and packaging remain explicitly
unmeasured boundaries for the next authorized design checkpoint.

2026-08-15 — M8D rewrite-byte counter correction

The lifecycle diagnostic now binds the post-linkage anatomy
`bytes_rewritten` field to the final atomic output size, matching the existing
operation ledger without adding another payload serialization. A fresh normal
Modal writer run at 7f5ff7f produced the corrected report (one reported run,
three allocated output roots); its sanitized raw digest is
f870dc490e84f20a8a6acab0a184c5c90eebc3c84217621ba2b688d025cc9349.

The regenerated report and environment record identify the exact source
commit 7f5ff7f8251c6cb51cd6b976a299090e51cc297c; the diagnostic prose uses
the same identity and the corrected 5,083,039,446-byte lifecycle total. The
candidate matrix was refreshed to the regenerated 327.002139828-second run
and its 591.313904348-second post-linkage rewrite.

2026-08-15 — M8E selection-system audit

An offline, measurement-only audit records the canonical C1-C6 selection call
graph, current leaderboard ranking and bounded-pool behavior, coordinate-keyed
deduplication and reason preservation, and the de facto perverse-tail routing
of full-width exemplars. The retained M8D report confirms 145/256 full-width
selected exemplars and vocabulary-width physical arrays even for a small-K
sample; no dynamic-top-K or score-pass behavior was changed. A deterministic
analyzer simulates exact rational C_full(N)=max(1,floor(N/3)) composition,
order-independent full-width displacement, and 1/4, 1/3, 1/2, and uncapped
sensitivity. No production policy, Contract, Student, Golden evidence, or
default changed; future implementation requires selection-authority and
backfill decisions.

M8F sub-checkpoint 2 closes the legacy within-board duplicate-coordinate gap
by retaining only the best governed rank for a repeated selected coordinate.
The complete ranked-reserve/backfill redesign remains explicitly unclaimed for
the next checkpoint.

The canonical C6 C2 path now opts into complete ranked candidate retention,
while compatibility callers retain their bounded micro-pool behavior. Exact
full-width ratio authority propagates through SelectionIntent, resolved
configuration, C2, and C4. Compact payload and immutable-body Contract work
remain design-only.

2026-08-15 — M8F sub-checkpoint 1 exact full-width policy

Added a Tome-owned exact-rational `{numerator: 1, denominator: 3}` composition
primitive and bound the pair into the production selection integration hash.
Full-width exemplars remain diagnostically observable but are ordinary eligible
exemplars and are no longer excluded by default through the Tome long-tail
classification defaults. This checkpoint does not yet wire composition into
the C2-C6 flow; ranked reserves and backfill remain the next bounded commit.

The follow-up evidence correction derives the duplicate-reason and
same-source controlled scenarios from deterministic in-memory records and
regenerates the machine-readable report; the audit document intentionally
does not pin a stale report digest.

The audit simulator was corrected to keep the full-width allowance hard even
when the narrow candidate pool is exhausted. It now reports category shortfall
separately from total eligible-pool exhaustion, with derived controlled
fixtures covering duplicate coordinates and exhaustion.

2026-08-15 — M8G Contract amendment design

Published a design-only, Contract-pinned amendment for compact dynamic-top-K
storage and immutable exemplar body/linkage manifests. It defines closed field
registries, FV3-framed identity preimages, raw-versus-semantic digest domains,
package inventory binding, and a crash-resumable body/manifest journal matrix.
No RADJAX-Contract or Tome production implementation was made for this design.

2026-08-15 — M8F/M8G completion corrections

Completed the canonical C4 full-width composition wiring: the exact rational
allowance is now applied to corridor claims with unused full-width allowance
filled by ranked narrow candidates, and global-board full-width metadata
survives export/load round trips. Tightened the design-only Contract amendment
with a closed, non-recursive selection-obligation tuple registry. Compact
dynamic-top-K storage and immutable exemplar-body production remain unstarted.

2026-08-15 — M8G design registry tightening

Closed the design-only manifest obligation representation as a fixed tuple with
explicit role and collision enums and a count field; arbitrary recursive
objects are not part of the proposed Contract amendment. No Contract or
production payload code changed.

2026-08-15 — M8G amendment byte-level closure

Expanded the design-only Contract amendment with a normative padded-to-compact
projection, profile-code mapping, exact domain-label and inventory framing,
closed receipt field types/state codes, and restart-state matrix. No Contract,
Student, or Tome payload implementation was started.

2026-08-15 — M8F score-pass width binding

Bound target-store vocabulary size into canonical score-pass candidate records
so the production global selector can classify full-width candidates before
export, reload, and C4 cap enforcement. Added focused extraction coverage.

2026-08-15 — M8F complete-reserve validator correction

When canonical C2 retains a complete ranked candidate reserve, artifact
validation now permits that reserve to exceed the compatibility micro-pool cap;
bounded compatibility callers retain the original cap validation. Final C4
composition remains responsible for the exact full-width allowance.

2026-08-15 — M8F selection-authority parity correction

Bound the exact full-width composition pair into the production facade’s
selection hash as well as the shared stage hash, eliminating an authority
disagreement during packaging and resume validation.

2026-08-15 — M8F diagnostic-board characterization

Updated Tome packaging characterization to assert that the retained
`perverse_tail_diagnostic` artifact is diagnostic-only: selected exemplars are
not placed in a separate semantic board or excluded from ordinary profiles.

2026-08-15 — M8F global-supply cap round-trip regression

Added an integration regression proving top-level full-width metadata survives
global-board export/load and the canonical C4 cap admits only the exact
allowance before ranked narrow backfill.

2026-08-15 — M8F validation formatting

Applied repository Ruff formatting to the new global-supply cap regression;
behavior and authority surfaces are unchanged.

2026-08-15 — M8G identity and receipt wire closure

Made proposed digest fields uniformly raw 32-byte values, bound schema/profile
to receipts, enumerated legal state edges, and separated header CRC from the
payload CRC in the design-only amendment. No Contract implementation began.
2026-08-16 — M8G compact physical payload adapter

Added the opt-in Tome adapter that projects one legacy padded selected
exemplar payload into the coordinated Contract compact body resource. The
legacy padded path remains the default; logical K, ordering, mass, and CSL are
unchanged.

2026-08-16 — M8G immutable body transaction adapter

Added an opt-in transaction writer that validates and atomically promotes a
compact immutable body before committing a separate manifest. Legacy staging
and payload defaults remain unchanged; no Contract or Student runtime code
was modified.

2026-08-16 — M8G manifest-byte binding correction

The immutable transaction now derives the canonical closed manifest encoding
and rejects caller-supplied bytes that do not exactly match the validated
manifest identity.

2026-08-16 — M8G transaction preflight correction

Manifest canonical-byte and body-binding checks now complete before any body
promotion, preventing failed transactions from publishing orphan bodies.

2026-08-16 — M8G journaled reservation and recovery correction

Added transaction-private reservations, Contract receipt emission at durable
body/manifest states, content-addressed body reuse, private staging, no-follow
symlink rejection, deterministic orphan diagnostics, and preflight conflict
checks. The manifest remains the semantic commit point.

2026-08-16 — M8G parent-chain path hardening

Transaction roots and target parents now walk existing components with lstat
and reject symlink substitution before staging or promotion.

2026-08-16 — M8G contiguous journal-state correction

Receipts now record every Contract journal state from generation through
commit, including manifest validation, inventory binding, and package
validation. Recovery reports persisted state sequences and committed runs
remove private temporary body and manifest files.

2026-08-16 — M8G receipt-chain recovery correction

Recovery now reconstructs receipt fields, validates each receipt through the
Contract validator, enforces contiguous state numbering, rejects symlinked
transaction entries, and quarantines malformed or incomplete journals.
Atomic publication uses exclusive hard-link promotion to avoid overwrite races.

2026-08-16 — M8G receipt/object and inventory binding correction

Recovery now checks JSON receipts against their binary canonical counterparts,
binds transaction IDs and promoted body/manifest digests to actual resources,
and requires committed manifests to appear in the validated inventory.

2026-08-16 — M8G inventory and path-binding correction

Recovery now rejects absolute or escaping receipt paths, validates inventory
schema and member digests against promoted resources, and preflights existing
inventory before publication.

2026-08-16 — M8G partial-state resume correction

Valid journals before manifest promotion now reconstruct and validate the
staged manifest, reuse the promoted body, complete manifest promotion,
inventory binding, and remaining Contract receipts. Pre-body partial journals
return to a clean restart state instead of being quarantined.

2026-08-16 — M8G committed-recovery state filtering correction

Inventory reconciliation now applies only to receipt states that have a
manifest path, avoiding false quarantine from earlier body-only states while
retaining strict digest and member checks for committed resources.

2026-08-16 — M8G recovery and package-boundary correction

Manifest preflight is now persisted before body publication so a valid
BODY_PROMOTED crash can resume. Recovery repairs valid post-manifest states
through inventory and deterministic transaction-archive validation, while
binary receipts remain authoritative and missing JSON mirrors are regenerated.
The opt-in transaction archive contains only validated committed resources;
fault boundaries and structured recovery assessment/plan surfaces are exposed
for controlled interruption tests.

2026-08-16 — M8G archive conflict-integrity correction

Transaction archive publication now uses exclusive no-replace promotion and
validates archived body and manifest bytes against the canonical inventory
digests. A conflicting pre-existing archive is preserved and causes recovery
quarantine rather than overwrite.

2026-08-16 — M8G configuration and archive portability correction

Receipt configuration identity is now an explicit transaction input with a
deterministic profile-derived default, independent of manifest authority.
Archive metadata is normalized for deterministic cross-environment bytes, and
system /tmp and /var aliases on macOS are not mistaken for caller-controlled
transaction symlinks.

2026-08-16 — M8G final validation fixture correction

The configuration-swap recovery fixture is formatted and included in the
focused validation set; Ruff, format, compile, and diff checks remain clean.

2026-08-16 — M8G deterministic archive container correction

The opt-in transaction archive now writes gzip with an explicit zero mtime in
addition to normalized tar member metadata, making repeated recovery archive
finalization byte-stable and preventing false immutable-resource conflicts.

2026-08-16 — M8G configuration-binding correction

Transaction IDs now include the independent configuration identity, and
recovery rejects receipt chains authored under another configuration. A
configuration-swap regression fixture preserves the fail-closed behavior.

2026-08-16 — M8G recovery blocker traceability

Added the blocker-resolution table mapping partial-state recovery, receipt
authority, archive conflict handling, configuration binding, and fault tests
to their implementation and evidence. The opt-in M8G resource archive is
explicitly distinguished from the full canonical Tome producer package.

2026-08-16 — M8G workload authority recovery

The historical M8 replay authority was recovered from the documented retained
external artifact, copied to the durable local artifact store, and verified as
the 43-file verify-checkpoint root with 213 selected sources, 256 coordinates,
matching checkpoint/authority/corpus/model/tokenizer digests, and complete
source-passport coverage. A path-independent bundle manifest and byte-level
validator were added; raw provider metadata and binaries remain outside Git.

2026-08-16 — M8G canonical materialization modes

Added explicit legacy-padded, compact-K monolithic, and compact-K immutable
body mode authority to canonical configuration and replay. CPU/GPU selected
evidence reducers can emit direct variable-length retained arrays for compact
modes; legacy retains its rectangular mask path. Reports carry requested and
executed mode plus physical/logical allocation counters.

2026-08-16 — M8G canonical physical publication

Compact selected payloads now omit the legacy vocabulary-width mask at the
publication boundary, use the explicit compact shard schema, and immutable
mode publishes each logical-K body through the accepted transaction with a
closed manifest. GPU and CPU reducers share the direct compact evidence shape;
legacy selected payloads remain unchanged.

2026-08-16 — M8G workload resource hardening

The authoritative bundle no longer declares an unused recovery-anchor root;
checkpoint and model trees reject symlinked resources during verification, and
the preparation record identifies the 43-file replay root as the sole
checkpoint authority.

2026-08-16 — M8G workload validator closure

The workload validator now verifies the complete 43-file checkpoint tree,
canonical selected-record identity, duplicate-free coordinate rows and bounds,
source-count/passport coverage, corpus authority, and retained model,
tokenizer, configuration, weights-file, and provenance digests. Validation
requires the separately governed artifact root and remains independent of
provider-local paths.

2026-08-16 — M8G workload identity hardening

Bundle validation now recomputes the checkpoint identity and model category
hashes, verifies the complete retained model-file manifest, binds the corpus
manifest digest, rejects layout escape paths, and checks selected coordinates
against source passports and corpus rows. The durable artifact validates with
213 sources and 256 coordinates.

2026-08-16 — M8G workload tree closure

The validator now rejects extra or missing checkpoint/model files, duplicate
model manifest entries, and normalized weight/config identity drift. The
bundle remains an external-byte artifact with a path-independent committed
manifest.

2026-08-16 — M8G direct compact mode authority

Explicit compact representation modes now control the CPU dynamic materializer
without requiring an incidental selected-source metadata flag; corridor and
score-pass paths retain their governed legacy materialization behavior.

2026-08-16 — M8G canonical mode formatting

The canonical mode regression test is formatted and remains part of the
focused validation gate for explicit legacy and compact backend modes.

2026-08-16 — M8G authority separation

Representation mode remains in normalized execution and resume authority while
the historical selection authority projection stays byte-compatible, so mode
changes do not alter governed selection semantics.

2026-08-16 — M8G frozen-selection replay entrance

Production now accepts an explicitly paired verified replay root and workload
bundle manifest. Adoption validates the complete historical authority, creates
a private invocation-owned copy of required replay records, and routes frozen
selected records into the existing C6 selected-source materialization path
without rebuilding C1--C5 or changing ordinary external-checkpoint rejection.

2026-08-16 — M8G replay adoption resume binding

Replay adoption is idempotent only for the same bundle, checkpoint, selected
record identity, and intact private member digests; incomplete, substituted, or
cross-authority private adoption roots fail closed before teacher execution.

2026-08-16 — M8G replay fail-closed guard coverage

Replay authority inputs reject symlink substitution before resolution, and the
existing external C4/C5 checkpoint rejection is covered alongside replay
configuration tests.

2026-08-16 — M8G replay authority precedes preflight

Verified frozen-selection adoption now occurs before ordinary input and teacher
preflight. Preflight therefore validates only the private closure-checked
replay inputs, while ordinary production retains its existing order and guards.

2026-08-16 — M8G adopted-member containment

Idempotent replay adoption validates every metadata-declared member as an
owned relative regular path before hashing, rejecting traversal and symlink
substitution without reading outside the private adoption root.

2026-08-20 — M8G portable workload finalization

Added the canonical finalization entry point for completed 1K M8G generation
evidence, including portable path projection, source-row closure derivation,
checkpoint and workload authority records, finalization receipts, and the
Contract pin for the public workload authority API.

Pinned the finalizer to the reviewed Contract workload-authority commit.

2026-08-20 — M8G closed authority records

Updated finalization evidence to emit explicit public workload record types,
checkpoint cross-bindings, truthful finalization receipt evidence, and CPU-only
replay preflight records for each approved representation mode. No generation,
materialization, GPU, or benchmark work was performed.
