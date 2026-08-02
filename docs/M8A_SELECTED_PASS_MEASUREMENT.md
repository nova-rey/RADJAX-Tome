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
