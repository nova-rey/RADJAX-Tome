# M8C Selected-Pass Staging Diagnostic

This is a measurement-only checkpoint on Tome commit
`6a6c65378cfd86a190e44e861ed9323927c2acc8`. It does not change production
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

The run used Modal app `ap-6Aiz7uBiFJ3l67RgyK0PYo`, a Tesla T4, and three
fresh output roots. The persisted source report is
`docs/evidence/M8C_CURRENT_BASELINE_RAW.json`, whose sanitized local digest is
`sha256:93fd6a6fdc7527addbb2a45bd0618101a9f73ef28b855da1d8e38165de383279`.
The unsanitized remote report digest was
`sha256:7dfb0fffdc9418656df639294f0a7617dfbab645b1847a90bcbd457ac09325a`;
only private staging path spelling was removed from the committed copy.

The maintained remote wrapper is `scripts/run_m8c_modal_baseline.py`. It takes
all authority-bearing roots through explicit environment variables. A fresh
run is launched with:

```text
M8C_TOME_ROOT=/absolute/Tome \\
M8C_CONTRACT_ROOT=/absolute/RADJAX-Contract \\
M8C_CHECKPOINT_ROOT=/absolute/checkpoint \\
M8C_MODEL_ROOT=/absolute/model \\
M8C_EXPECTED_TOME_COMMIT=6a6c65378cfd86a190e44e861ed9323927c2acc8 \\
PYTHONUNBUFFERED=1 uvx --python 3.12 --with cbor2==5.6.5 --from modal \\
  modal run scripts/run_m8c_modal_baseline.py::main
```

The original run used an equivalent disposable wrapper before this maintained
wrapper was committed; the exact driver invocation and all input values were
the same. The mounted diagnostic invoked
`scripts/run_m8b_selected_staging_baseline.py`
with the checkpoint anchor, its corpus and manifest, the transferred Gemma
3 270M model/tokenizer, and `--expected-tome-commit
6a6c65378cfd86a190e44e861ed9323927c2acc8`. The source tree and Contract tree
were mounted from the exact verified commits. The private git shim only
supplied `rev-parse HEAD` because the source mount intentionally omitted `.git`;
it did not alter production code. The Modal container terminated after the
report was committed to the `radjax-m8b-evidence` volume. The dedicated
Lightning CPU Studio was stopped afterward; the DreamStream Studio was not
touched.

## Results

Schema `m8b_selected_staging_baseline_v1` produced three valid runs. Selected
pass walls were 323.518603620, 305.824600515, and 305.544781195 seconds; the
median is **305.824600515 s**. Initial staging medians were **275.400979866 s**
(90.0519% of selected-pass wall), and the existing initial-staging gate passed.
The correct 20%-throughput removable-time threshold is
`T - T/1.2 = 50.9707667525 s` for this fresh median.

The phase ledger is inclusive and therefore is not a non-overlapping sum;
the report's accounting reconciliation and unattributed-control fields remain
authoritative. Median phase timings (percentage of selected-pass wall shown
only as orientation) were:

| phase | median seconds | orientation |
| --- | ---: | ---: |
| staging JSON encoding | 193.809392 | 63.37% |
| canonical body encoding/hash | 79.466049 | 25.98% |
| payload conversion/linkage validation | 18.494613 | 6.05% |
| teacher forward | 4.198435 | 1.37% |
| model/tokenizer load | 2.232704 | 0.73% |
| temporary-file write | 2.156643 | 0.71% |
| compact D2H transfer | 0.571035 | 0.19% |
| all remaining measured phases | 0.34 | <0.12% each |

The selected-rerun compression medians were 294.966660094,
294.517..., and 294.004... seconds (median 294.517 s); teacher medians were
20.654262138, 11.159..., and 11.402... seconds (median 11.402 s). The
compression aggregate is a broader driver metric and must not be added to the
selected-pass phase ledger.

The report records no measured corridor reread/rehash/rewrite (`not_measured`),
typed post-linkage evidence hashing (`not_measured`), archive creation,
Contract validation, or packaging inside this selected-pass boundary. Those
are explicit measurement gaps, not zero-cost claims. Per-run staging payload
count was 256 and selected-rerun batch count was 27. The selected-pass observer
reported 256 coordinate records and zero score-pass/selection-writer
invocations, no OOM retries, and identical 256-element payload-hash sequences
across all runs.

Peak process RSS median was approximately 5.34 GB and peak CUDA allocated
memory approximately 1.33 GiB. During active staging the T4 was idle while
the CPU was near saturation; temporary output reached approximately 4.5 GB.
This directly confirms that JSON materialization and filesystem staging, not
teacher inference, dominate this workload.

## Decision

Initial staging is the dominant measured component. Its JSON encoding and
canonical-body encoding/hash alone are greater than the 50.97-second minimum
removable time, so receipt propagation remains plausible in principle, but the
current report does not establish which portion is legally removable. No
candidate is selected here. The next authorized checkpoint must instrument
read/write/hash/validation counts and the corridor/linkage boundary, then design
a crash-consistent staging transaction that preserves full verification at the
untrusted boundary and checkpoint invalidation on any governed-byte mutation.

The rejected streaming-staging candidate remains untouched and negative
evidence. No optimization, trust receipt, batching change, Contract change,
default change, or public semantic change was made in this checkpoint.

## Fresh instrumented-series transfer status

Two fresh Modal T4 series were attempted after the private operation-counter
hooks and maintained runner were added. One completed and returned a summary
(median initial staging 278.416 s; median selected-pass wall 310.464 s), but
its raw report was not present in the durable volume after the app stopped and
could not be retrieved from the completed container. A second run at commit
`6044287` encountered a Modal gRPC heartbeat failure before its compressed
report result returned. The durable volume therefore still contains only the
historical raw report (`sha256:7dfb0ff...`), which predates the counter hooks.

This is an evidence-transfer blocker: the committed report does not contain
the required read/write/hash/file-operation counters, and the fresh summaries
cannot be promoted to raw governed evidence without the corresponding report
bytes. No fresh instrumented baseline is claimed, and no optimization is
authorized or selected until a provider run can transfer and verify its raw
report.
