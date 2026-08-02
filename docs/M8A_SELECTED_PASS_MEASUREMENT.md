# M8A Selected-Pass Measurement Readiness

M8A adds a private benchmark-only measurement seam around the existing
canonical selected-delivery rerun. `SelectedPassMeasurementControl` is not a
Tome build option: it requires `benchmark_only=True`, accepts only execution
caps `1`, `2`, `4`, or `8`, binds a content-addressed post-C5 checkpoint, and
requires a distinct temporary output root. It is absent from resolved build
configuration, the fixed selection projection, authority hashes, package
metadata, Contract inputs, and the production CLI.

The replay helper persists a content-addressed checkpoint manifest and rejects
capture unless it explicitly binds score, corridor, authority, C2, C3, C4,
C5, source passports, model, tokenizer, corpus, and resolved-config evidence.
It rehashes the named evidence tree before and after the canonical rerun and
rejects a non-frozen C5 handoff. It copies checkpoint files into a fresh
temporary measurement root rather than hard-linking them, so a replay write
cannot mutate upstream evidence. The harness calls the existing
`run_selected_delivery_rerun` owner and does not invoke score or selection
writers.

When the private control is present, `selected_pass_execution_v1` records the
selected-pass wall-time denominator, phase ledger, 5-percent accounting
reconciliation, effective source sizes, tensor shapes and dtypes, process and
optional CUDA observations, OOM/reload events, and explicit
`not_authorized` compilation status. The real GPU Torch backend emits its
private phase metadata only while that observer is attached: selected-position
index preparation is measured by the canonical staging owner and selected-row
gather by the GPU backend. Each observed GPU batch reports input, logits,
gathered-logits, and compact-result shapes and dtypes without materializing an
additional tensor. CUDA event timing is explicitly unavailable because M8A
does not authorize it. M7 publication,
Contract validation,
archive creation, and v5 packaging remain separately marked as unmeasured
post-selected-pass phases. A deterministic largest-remainder source sampler
reports the 64-source approximation only; it is not closure evidence.

This is instrumentation readiness, not representative M8A evidence. It makes
no optimization, accelerator, performance, compilation, new production batch
policy, M7, Contract, Golden, tokenizer, Student, corpus, UX, or model-quality
claim. A canonical anchor and T4 replay evidence remain required for M8A
acceptance, and M8B is not started here.

## Representative T4 baseline evidence

The representative evidence was captured on a Tesla T4 (15,636,037,632 bytes),
CUDA 13.0, PyTorch 2.13.0+cu130, Python 3.12.3, with the accepted canonical
anchor checkpoint `sha256:a850ad85f3a0ca95a000dda239ff7f46296de7b411fe5db20b844bfa78ad6e12`.
The raw report is retained outside this repository because it contains private
machine paths and run-level details.

The canonical anchor passed the full production sequence, including strict
selected linkage, v4 directory/archive Contract admission, and a 256-record
canonical Tome with semantic identity
`sha256:9d5796a7ff2d4db5000b9a128502cc68d0933f13be4786c4cefc1982a615dea7`.
The anchor and all replays used 213 selected sources and 256 coordinates; the
measured replay authority was consistently
`sha256:e000f5b79c7c8d8f7b59de5f33964bef04a4340f21f216aef0933895b69482bf`.
Every replay recorded zero score-pass and selection-writer invocations, no OOM
or fallback event, unchanged checkpoint evidence, and exact phase accounting.

Three complete requested/effective-batch-eight replays measured 365.188,
358.474, and 357.258 seconds respectively (0.583, 0.594, and 0.596 selected
sources/s). The full denominator included model load and staging but excluded
checkpoint validation; the latter remained separately measured at roughly 2.6
to 2.8 seconds before and after each replay. Hash/JSON/atomic staging consumed
324.2 to 324.8 seconds, about 90 percent of the full denominator. Payload
conversion/linkage consumed 23.0 to 23.8 seconds; teacher forward was 2.3 to
5.4 seconds. Thus serialization/staging, rather than a safe GPU batch change,
is the measured dominant candidate mechanism.

The deterministic 64-source/73-coordinate approximation had a maximum
proportional multiplicity deviation of 0.00925. It used the required warm-up
plus three measured iterations at each private cap. Requested batch size stayed
eight for every run. Median selected-source throughput was approximately 0.777
at cap 1, 0.638 at cap 2, 0.641 at cap 4, and 0.639 at cap 8. These sample
rates are mechanism evidence only, not closure performance evidence.

Exact M7 semantic comparison is decisive: the sample payload-hash sequence at
cap 1 differs from the cap-8 reference; caps 2, 4, and 8 match exactly. The
authority hash and frozen record digest remain identical for every cap. Cap 1
is therefore not an admissible production-equivalent batch shape, regardless
of its higher sample rate. No M8B policy, resource contract, optimization, or
production batching change is selected from this evidence. M8A stops here for
independent review.
