# Pipelined compact representation benchmark

Dataset: M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1 (169 records).

The setup decoded source JSON and performed exactly 169 compact projections outside timed phases.

| Mode | representation | linkage | validation | archive | total | output bytes | archive bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy_padded_monolithic | 24.3296s | 0.8625s | 1.0531s | 2.4299s | 28.7409s | 1,208,094,516 | 65,411,745 |
| compact_k_monolithic | 31.8243s | 0.0013s | 0.3216s | 1.2429s | 33.4294s | 237,336,636 | 40,712,981 |
| compact_k_immutable_body | 31.7539s | 0.0013s | 0.3267s | 1.2540s | 33.3947s | 237,336,636 | 40,712,952 |

Worker sweep (simple split):
- 1 workers: total [32.9738, 33.3528, 32.7869]s; median 32.9738s; representation median 31.3051s; producer blocked [0.0, 0.0, 0.0]s
- 2 workers: total [32.8047, 33.1176, 32.6871]s; median 32.8047s; representation median 31.2302s; producer blocked [0.0, 0.0, 0.0]s
- 4 workers: total [33.0879, 33.1999, 33.335]s; median 33.1999s; representation median 31.5764s; producer blocked [0.0, 0.0, 0.0]s

Corrected baseline medians: legacy 29.0361s total; compact 33.6394s; compact plus simple split 32.3993s.

This run measured compact monolithic at 33.4294s (-0.62% vs corrected compact baseline) and simple split at 33.3947s (3.07% vs corrected split baseline). Relative to this run's legacy median, compact monolithic changed 16.31% and simple split changed 16.19%.

The selected default from the bounded sweep is one worker: it had the lowest median total. Queue high-water was approximately 3.2 MB and producer blocked time was effectively zero; no body rereads or rewrites occurred.

All mode outputs were checked against the shared logical-evidence root; see logical_equivalence.json.