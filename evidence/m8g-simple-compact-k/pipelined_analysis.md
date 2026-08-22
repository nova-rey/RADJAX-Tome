# Pipelined compact representation benchmark

Dataset: M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1 (169 records).

The setup decoded source JSON and performed exactly 169 compact projections outside timed phases.

| Mode | representation | linkage | validation | archive | total | output bytes | archive bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy_padded_monolithic | 23.8216s | 0.8747s | 1.0932s | 2.4140s | 28.4214s | 1,208,094,516 | 65,411,757 |
| compact_k_monolithic | 32.1621s | 0.0013s | 0.3557s | 1.2639s | 33.7239s | 237,336,636 | 40,712,808 |
| compact_k_immutable_body | 32.0562s | 0.0013s | 0.3307s | 1.2548s | 33.6239s | 237,336,636 | 40,712,925 |

Worker sweep (simple split):
- 1 workers: total [33.1654, 33.2926, 33.041]s; median 33.1654s; representation median 31.4671s; producer blocked [0.0, 0.0, 0.0]s
- 2 workers: total [33.1673, 32.9491, 33.4848]s; median 33.1673s; representation median 31.5410s; producer blocked [0.0, 0.0, 0.0]s
- 4 workers: total [33.5748, 33.9681, 33.5575]s; median 33.5748s; representation median 32.0311s; producer blocked [0.0, 0.0, 0.0]s

Corrected baseline medians: legacy 29.0361s total; compact 33.6394s; compact plus simple split 32.3993s.

This run measured compact monolithic at 33.7239s (0.25% vs corrected compact baseline) and simple split at 33.6239s (3.78% vs corrected split baseline). Relative to this run's legacy median, compact monolithic changed 18.66% and simple split changed 18.30%.

The selected default from the bounded sweep is one worker: it had the lowest median total. Queue high-water was approximately 3.2 MB and producer blocked time was effectively zero; no body rereads or rewrites occurred.

All mode outputs were checked against the shared logical-evidence root; see logical_equivalence.json.