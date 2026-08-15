# M8C Selected-Pass Staging Diagnostic

This is a measurement-only checkpoint on the restored Tome baseline
`6a6c65378cfd86a190e44e861ed9323927c2acc8`, instrumented and run at
`5e6f9a74f465b501e310054efb323471f33e69e4`. It does not change production
behavior, Contract defaults, payload meaning, or validation. The existing
private `selected_pass_execution_v1` observer and M8B baseline driver were
used; no optimization was implemented.

## Authority and controlled run

The exact checkpoint was recovered from the dedicated `radjax-tome-1`
Lightning Studio and verified locally before transfer:

* Contract: `radjax-contract` 0.9.0,
  `1fa43e1aea2e198511db86dafb0aeefa525d48c7`;
* post-C5 checkpoint: `sha256:a850ad85f3a0ca95a000dda239ff7f46296de7b411fe5db20b844bfa78ad6e12`;
* checkpoint manifest: `sha256:0ea34090bf7b2e6331644d2eaa57b738c771ca48e62dc76527878a6b9b915a39`;
* workload: 213 selected sources, 256 selected coordinates, requested/effective
  source batch 8, fixed source-count batching, OOM halving only;
* authority hash: `sha256:e000f5b79c7c8d8f7b59de5f33964bef04a4340f21f216aef0933895b69482bf`;
* record digest: `sha256:e471535e04db268c3fd852bc5b8dc746bfc4ef26a29baca75f17652262fca3a0`.

The final instrumented run used Modal app `ap-oLTCbPgXz3E36YNXZ5Z6bx`,
container `ta-01M037F9VV8CZK1X84VQFA8C5R`, a Tesla T4, and three fresh output
roots. The raw report digest was
`sha256:abf74c5555f3d4bd92fb38fb1811a80ffee4a7bf326ec9ecdf4d644dd9244239`;
the committed sanitized report digest is
`sha256:64f56cc1614567797b5dcd819650b6a11bb1a80c675f34af35e31088e49273f6`.
Only private staging path spelling was removed from the committed copy. The
matching environment and run-identity record is
`docs/evidence/M8C_CURRENT_BASELINE_ENVIRONMENT.json`.

The maintained remote wrapper is `scripts/run_m8c_modal_baseline.py`. It takes
all authority-bearing roots through explicit environment variables. A fresh
run is launched with:

```text
M8C_TOME_ROOT=/absolute/Tome \\
M8C_CONTRACT_ROOT=/absolute/RADJAX-Contract \\
M8C_CHECKPOINT_ROOT=/absolute/checkpoint \\
M8C_ANCHOR_ROOT=/absolute/checkpoint-anchor \\
M8C_MODEL_ROOT=/absolute/model \\
M8C_EXPECTED_TOME_COMMIT=5e6f9a74f465b501e310054efb323471f33e69e4 \\
M8C_HOLD_SECONDS=600 \\
PYTHONUNBUFFERED=1 uvx --python 3.12 --with cbor2==5.6.5 --from modal \\
  modal run scripts/run_m8c_modal_baseline.py::main
```

The maintained runner mounts the terminal production anchor separately from
the immutable checkpoint because `production_build_report.json` is outside the
43-file checkpoint manifest. The mounted diagnostic invoked
`scripts/run_m8b_selected_staging_baseline.py`
with that separate anchor, the checkpoint's corpus and manifest, the
transferred Gemma 3 270M model/tokenizer, and `--expected-tome-commit
5e6f9a74f465b501e310054efb323471f33e69e4`. The source tree and Contract tree
were mounted from the exact verified commits. The private git shim only
supplied `rev-parse HEAD` because the source mount intentionally omitted `.git`;
it did not alter production code. The report was transferred directly from
the held container's private `/tmp` output because the legacy diagnostic
volume returned `Operation not supported by device`; no volume is part of the
final runner. The dedicated Lightning CPU Studio was stopped afterward; the
DreamStream Studio was not touched.

## Results

Schema `m8b_selected_staging_baseline_v1` produced three valid runs. Selected
pass walls were 321.673433458, 308.435139861, and 303.518228375 seconds; the
median is **308.435139861 s**. Initial staging medians were
**274.397374801 s** (89.0% of selected-pass wall), and the existing
initial-staging gate passed. The correct 20%-throughput removable-time
threshold is `T - T/1.2 = 51.4058566435 s` for this fresh median.

The phase ledger is inclusive and therefore is not a non-overlapping sum;
the report's accounting reconciliation and unattributed-control fields remain
authoritative. Median phase timings (percentage of selected-pass wall shown
only as orientation) were:

| phase | median seconds | orientation |
| --- | ---: | ---: |
| staging JSON encoding | 193.117270 | 62.62% |
| canonical body encoding/hash | 78.766513 | 25.54% |
| payload conversion/linkage validation | 18.727755 | 6.07% |
| teacher forward | 4.157908 | 1.35% |
| model/tokenizer load | 2.622324 | 0.85% |
| temporary-file write | 2.446317 | 0.79% |
| compact D2H transfer | 0.570000 | 0.18% |
| all remaining measured phases | 0.34 | <0.12% each |

The selected-rerun compression and teacher timings remain broader driver
metrics and must not be added to the selected-pass phase ledger.

The report records no measured corridor reread/rehash/rewrite (`not_measured`),
typed post-linkage evidence hashing (`not_measured`), archive creation,
Contract validation, or packaging inside this selected-pass boundary. Those
are explicit measurement gaps, not zero-cost claims. Per-run staging payload
count was 256 and selected-rerun batch count was 27. The selected-pass observer
reported 256 coordinate records and zero score-pass/selection-writer
invocations, no OOM retries, and identical 256-element payload-hash sequences
across all runs.

Peak process RSS remained approximately 5.34 GB and peak CUDA allocated
memory approximately 1.33 GiB. During active staging the T4 was idle while
the CPU was near saturation; temporary output reached approximately 4.5 GB.
This directly confirms that JSON materialization and filesystem staging, not
teacher inference, dominate this workload.

## Decision

Initial staging is the dominant measured component. Its JSON encoding and
canonical-body encoding/hash alone are greater than the 51.41-second minimum
removable time. Receipt propagation remains plausible in principle, but the
current evidence still does not establish which portion is legally removable.
No candidate is selected here. The next authorized checkpoint must instrument
the remaining corridor/linkage boundary and design a crash-consistent staging
transaction that preserves full verification at the untrusted boundary and
checkpoint invalidation on any governed-byte mutation.

The rejected streaming-staging candidate remains untouched and negative
evidence. No optimization, trust receipt, batching change, Contract change,
default change, or public semantic change was made in this checkpoint.

## Operation ledger and remaining measurement gaps

The final report carries `selected_pass_operation_counts_v1` for all three
runs. Per run, the ledger records 256 canonical hashes over
2,036,613,610 encoded bytes, 256 staging records writing 4,452,718,421 bytes,
256 temporary-file opens/writes, and 256 atomic replacements. Across the three
runs that is 768 hashes, 6,109,840,830 canonical-body bytes read,
13,358,155,263 JSON bytes written, and 768 replacements. These are direct
instrument counters, not inferred timings.

Corridor reread/rehash/rewrite, typed post-linkage evidence hashing, archive
creation, Contract validation, and packaging remain explicitly
`not_measured` inside this selected-pass boundary. They are measurement gaps,
not zero-cost claims. The fresh report is now retained with its exact raw and
sanitized digests, so a later checkpoint can extend the ledger without losing
controlled baseline authority.
