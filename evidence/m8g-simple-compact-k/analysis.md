# M8G simple compact-K storage

Dataset: `M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1` (169 valid
surviving records; logical root `sha256:51eba04fbb2a6a2ee6d20a8c7d65b0027c3c559e38cd91f952c2176656ed933c`).
No teacher, tokenizer, selection, replay, GPU, or Modal work was performed.

## Results

| mode | total median (s) | representation median (s) | linkage median (s) | output bytes | archive bytes |
|---|---:|---:|---:|---:|---:|
| legacy padded | 30.348 | 25.218 | 1.135 | 1,208,094,516 | 65,411,719 |
| compact K | 49.737 | 37.442 | 10.572 | 237,336,636 | 40,712,947 |
| compact K + simple body store | 49.281 | 37.133 | 10.578 | 237,336,636 | 40,712,983 |

Raw values for all nine ordered samples are in `raw_three_round_report.json`.
Compact K reduced persistent output by 80.4% and level-1 archive bytes by 37.8%,
with a 63.9% end-to-end time increase in this focused implementation. The body
store kept body bytes unchanged across linkage; only `metadata.jsonl` was
rewritten. All 169 packed Contract bodies in every compact sample validated and
matched the shared logical root.

## Recommendation

Retain compact K as the canonical storage shape: it removes vocabulary-width
padding and persistent masks and preserves exact logical evidence. Revise the
linkage/package hot path before claiming an end-to-end speedup: the measured
simple body store has lower storage but higher CPU time than legacy on this
derived component corpus. Keep immutable-body transaction mode experimental;
the simple body/metadata split is the default compact path.

These are component-boundary measurements, not complete v19 replay or M8D
comparisons. M8D used a different historical workload and remains context only.
