# M8D Selected-Exemplar Payload Lifecycle Diagnostic

This measurement-only checkpoint starts at Tome `9b046aedb7ce7d22eda2884f9e7703ad55b4dfb2`, on the M8 diagnostic branch descended from `de31224915b0144a2d8b733ef13dc0724c60e51c`. It does not implement an optimization and does not change payload meaning, dynamic top-K, CSL, selection, batching, defaults, or validation.

The supplied official TOME Generation Glossary is used as terminology authority: the score pass is the corpus-wide compact-measurement pass; a selected source contains one or more selected coordinates; the selected-source pass reruns only selected sources; a corridor is a numbered lane in the behavioral fingerprint; and an exemplar is the high-resolution teacher output at one frozen selected coordinate. “Corridor generation” is not used for selected-exemplar staging.

## Authority and run

The exact Contract authority is `radjax-contract 0.9.0` at `1fa43e1aea2e198511db86dafb0aeefa525d48c7`. The restored post-C5 checkpoint is `sha256:a850ad85f3a0ca95a000dda239ff7f46296de7b411fe5db20b844bfa78ad6e12` with manifest `sha256:0ea34090bf7b2e6331644d2eaa57b738c771ca48e62dc76527878a6b9b915a39`, authority `sha256:e000f5b79c7c8d8f7b59de5f33964bef04a4340f21f216aef0933895b69482bf`, and selected-record digest `sha256:e471535e04db268c3fd852bc5b8dc746bfc4ef26a29baca75f17652262fca3a0`.

One fresh full lifecycle run was required because retained M8C evidence did not contain payload anatomy or corridor rewrite counts. It used Modal Tesla T4 app `ap-7aORm7YXOg8kGQcwhwb3GF` (container `ta-01M03F0YA066GWYC17TF18DAZR`), three fresh output roots, 213 selected sources, 256 selected coordinates, requested source batch 8, fixed source-count batching, and OOM halving only. Lightning AI was not used. The raw report was transferred and sanitized only to remove a private staging pathname; the committed generated report is `docs/evidence/M8D_SELECTED_PAYLOAD_LIFECYCLE_RAW.json` with SHA-256 `sha256:617920f965e17e23b382a343a77c8f2dd6e00494919392cc7abfc55c257f843e`.

Command (with authority-bearing roots supplied explicitly) was:

```text
M8C_TOME_ROOT=/absolute/Tome M8C_CONTRACT_ROOT=/absolute/RADJAX-Contract \
M8C_CHECKPOINT_ROOT=/absolute/checkpoint M8C_ANCHOR_ROOT=/absolute/checkpoint-anchor \
M8C_MODEL_ROOT=/absolute/model M8C_EXPECTED_TOME_COMMIT=9b046aedb7ce7d22eda2884f9e7703ad55b4dfb2 \
M8D_FULL_LIFECYCLE=1 M8D_RUN_COUNT=1 PYTHONUNBUFFERED=1 \
uvx --python 3.12 --with cbor2==5.6.5 --from modal \
  modal run scripts/run_m8c_modal_baseline.py::main
```

The one-run report has selected-pass wall `324.638876462 s`, initial staging `273.075658968 s`, and lifecycle output `5,083,039,434` bytes. It is not a replacement for the retained three-run M8C baseline (`308.435139861 s` median); it is a detailed lifecycle diagnostic on the same controlled workload.

## Payload anatomy

Across 256 selected coordinates, effective dynamic top-K has minimum 32, median/p90/p95/p99/max 262,144, total 38,129,241, and 65 distinct widths. 47 coordinates have K=32. 145/256 (56.640625%) are full vocabulary width (and also at least 95% of vocabulary width). This is governed output, not a compression failure: broad-uncertainty exemplars legitimately retain full width under the dynamic top-K and CSL rules.

Retained top mass is min 0.98046875, median 0.99609375, p90/p95/p99/max 1.0, total 254.73046875. Tail mass is min 0, median 0.00390625, p90/p95 0.015625, p99/max 0.01953125, total 1.26953125. The exact per-coordinate K and mass arrays are retained in the raw report.

Three bounded field-size observations (one small-K and two full-width) show why K alone is not a size predictor. At K=38, canonical/pretty payloads are 4,196,726/9,439,908 bytes. Its dense fields are `top_selection_mask` 1,572,827/2,359,260, `top_log_probs` 1,048,751/1,835,184, `top_probs` 1,049,086/1,835,519, and `top_token_ids` 524,403/1,310,836 bytes, each with 262,144 elements. At full width, sampled payloads are 10,800,589/16,043,784 and 10,795,323/16,038,518 bytes. Small metadata, provenance, scores, source position, board/corridor fields, and four-element `bucket_masses` are comparatively negligible.

The selected payload does not retain dense score-pass logits. The score pass retains compact measurements; the selected-source pass constructs high-resolution exemplar arrays. The field observations show that this implementation’s selected exemplar includes dense vocabulary-width arrays for token IDs, probabilities, log-probabilities, and the selection mask even when effective K is small; this is factual anatomy, not a recommendation to remove them.

## Byte and traversal accounting

Initial staging processed 2,036,613,610 canonical bytes and encoded 4,452,718,421 pretty-JSON bytes. Post-linkage synchronization reread 4,452,718,421 bytes, produced 2,036,638,668 canonical bytes, and rewrote 4,452,749,623 pretty-JSON bytes. Thus initial plus post-linkage cumulative canonical work is 4,073,252,278 bytes and cumulative pretty encoding is 8,905,468,044 bytes; cumulative whitespace/format expansion is 4,832,215,766 bytes. The pretty counter is larger because indentation, separators, and decimal formatting expand the canonical representation, and because the payload is encoded again after linkage injection.

The direct operation ledger reports, per 256-coordinate pass: 256 canonical hashes, 256 initial pretty encodings, 256 temporary files, 256 atomic replacements, then 256 post-linkage rereads, validations, hashes, rewrites, and replacements. The first pass therefore writes 4,452,718,421 bytes and the post-linkage pass writes 4,452,749,623 bytes. Lifecycle output is 5,083,039,434 bytes. Full Python-object traversal counts are inferred from the normal construction and post-linkage parse/rebuild path; archive and Contract validation traversals were not separately instrumented and are explicitly not claimed as zero.

The non-overlapping selected-pass timing view is: initial staging 273.075659 s; teacher forward 7.228341 s; tokenization 0.176889 s; selected-row gather 0.135199 s; compact reduction 0.211212 s; compact device-to-host 0.600435 s; payload/linkage validation 18.339746 s; temporary writes/close/replacement about 2.36 s. Initial staging is approximately 84.1% of this run’s 324.639-second selected-pass wall. The post-selected corridor synchronization reread/rehash/rewrite was 578.148473 s and is measured separately, not added to selected-pass wall. Its ledger is 256 reads, validations, hashes, and replacements. Typed post-linkage evidence hashing, archive validation, Contract validation, and package materialization were not separately timed by this checkpoint.

CPU evidence identifies staging JSON encoding (192.066697 s) and canonical body encoding/hash (78.650225 s) as dominant; teacher forward was 7.228341 s. No profiler was run because a function-level profiler would materially distort this multi-gigabyte serialization path; the stage timers and byte/counter ledger are the authoritative attribution. RSS rose from 515,313,664 to a 4,846,374,912-byte peak (incremental 4,331,061,248 bytes); peak CUDA allocation was 1,423,718,400 bytes. The high-water RSS occurs while selected payload objects, canonical buffers, pretty buffers, and staging output coexist; allocator retention means the incremental figure is an upper bound on live payload-only memory.

## Candidate matrix (analysis only)

| candidate class | measured surface | conservative view |
| --- | --- | --- |
| Serialize finalized payload once | 192.1 s initial pretty encoding plus later rewrite | Plausibly above the 51.406 s net threshold, but requires crash-consistent publication and linkage semantics. |
| Separate immutable exemplar body from small linkage metadata | 578.1 s post-linkage reread/rehash/rewrite and 4.45 GB reread | Best-supported single class; potentially gate-crossing, but requires a transaction/schema design and exact identity rules. |
| Reuse governed canonical representation | 78.7 s canonical encoding/hash | Plausible in isolation, but must still handle linkage mutation and raw integrity. |
| Alternate deterministic JSON encoder | 192.1 s encoding | Requires byte/identity proof; replacement overhead and determinism risk are unknown. |
| Concurrent independent exemplar serialization | staging CPU time | May reduce wall time, but 4.33 GB incremental RSS and filesystem contention make the benefit unproven. |
| Reduce coexistence of representations | 4.33 GB incremental RSS | Memory benefit is clear; throughput benefit is not established. |
| One traversal for hash and storage | canonical/hash and write path | Could save a traversal, but needs a safe immutable representation and recovery proof. |

The best-supported next candidate is separating the immutable exemplar body from small mutable corridor/linkage metadata, treated as one staging-transaction redesign rather than a stack of unrelated optimizations. The next implementation checkpoint must answer whether Contract/package identities can bind the body and linkage independently while preserving raw integrity, semantic identity, directory/archive equivalence, crash consistency, resume, and mandatory full verification at the untrusted boundary. This checkpoint selects no candidate and implements none.

## Validation and limitations

Focused M8 measurement, anatomy, staging, driver, and Hydra tests pass at the final checkpoint. The diagnostic is disabled by default and output-equivalent when enabled. The full Tome suite retains the known local PyTorch/NumPy ABI failures (`RuntimeError: Numpy is not available`); these were reproduced on the unmodified starting environment and are not attributed to the measurement hooks. Historical M8B.1 and rejected M8B.4 evidence remain unchanged. No Contract or Student files changed. No optimization, trust receipt, batching change, score-pass retention, dynamic-top-K change, or public semantic change was made.

Unmeasured boundaries are retained explicitly: typed post-linkage evidence, archive/Contract validation, and package traversal. They must be instrumented before a later candidate claims end-to-end savings. Modal resources were stopped after report transfer and checksum inspection; Lightning AI was not used.
