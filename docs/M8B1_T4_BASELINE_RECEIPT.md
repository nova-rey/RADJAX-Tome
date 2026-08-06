# M8B.1 T4 Current-Base Baseline Receipt

This receipt commits content-addressed, sanitized evidence for the private
M8B.1 T4 current-base measurement. The complete raw report remains outside the
repository because it contains private machine paths and detailed runtime
observations. Its digest enables authorized verification of the exact source
bytes without promoting the raw evidence or treating this summary as a
production artifact.

| Evidence | SHA-256 |
| --- | --- |
| M8B.1 raw baseline report | `7b39ce34efe8dbf47022b24ec5052998c4767d3fdca47d8692ea84dcefe8871f` |
| Immutable post-C5 manifest | `0ea34090bf7b2e6331644d2eaa57b738c771ca48e62dc76527878a6b9b915a39` |

The raw report uses schema `m8b_selected_staging_baseline_v1` and was measured
at Tome commit `31b94ad8cea639d23eeb7b569a3c5041d52e742f`. It reuses the
immutable post-C5 checkpoint with digest
`sha256:a850ad85f3a0ca95a000dda239ff7f46296de7b411fe5db20b844bfa78ad6e12`
and 43 bound files. The environment was a Tesla T4 (15,360 MiB), driver
`580.173.02`, PyTorch `2.13.0+cu130` with CUDA `13.0`, and Transformers
`5.14.1`.

Three canonical requested/effective cap-eight selected-stage replays each
processed 213 sources and emitted 256 selected records. The selected-pass wall
times were 308.737, 262.905, and 262.091 seconds (median 262.905 seconds;
spread 46.645 seconds). Initial staging had a median of 236.268 seconds with
23.431 seconds spread, or 89.868% of the selected-pass median (6.006 percentage
points spread). The frozen M8B.1 gate therefore passes: initial staging is at
least 50% of selected-pass wall time.

Every replay preserved the immutable checkpoint, used the canonical selected
rerun owner, retained requested/effective batch eight, had zero score-pass and
selection-writer invocations, no CUDA OOM retries, and exact phase accounting.
The raw report also records exact same-cap payload-hash sequences, source
multiplicity, shapes, resource observations, and separated private timing
phases. It does not authorize a batch-policy change, a score/selection rerun,
publication, Contract or Student change, Golden fixture change, or any other
production behavior change.

This is the baseline evidence that authorizes the bounded M8B.2
grammar-safe staging implementation under the approved plan; it is not an M8B
performance-success claim.
