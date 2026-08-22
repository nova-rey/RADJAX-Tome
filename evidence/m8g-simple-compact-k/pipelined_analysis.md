# Pipelined compact representation benchmark

Dataset: M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1 (169 records).

The setup decoded source JSON and performed exactly 169 compact projections outside timed phases.

| Mode | representation | linkage | validation | archive | total | output bytes | archive bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy_padded_monolithic | 23.9049s | 0.7112s | 1.0845s | 2.4301s | 28.4221s | 1,208,094,516 | 65,411,713 |
| compact_k_monolithic | 31.8292s | 0.0013s | 0.2861s | 1.2379s | 33.3746s | 237,336,636 | 40,712,909 |
| compact_k_immutable_body | 31.2641s | 0.0013s | 0.3469s | 1.2587s | 32.8731s | 237,336,636 | 40,712,987 |

Worker sweep (simple split):
- 1 workers: total [32.7367, 32.5215, 32.7412]s; median 32.7367s; representation median 31.1867s; producer blocked [0.0, 0.0, 0.0]s
- 2 workers: total [32.6772, 33.2022, 33.688]s; median 33.2022s; representation median 31.6271s; producer blocked [0.0, 0.0, 0.0]s
- 4 workers: total [33.5989, 33.1221, 33.1793]s; median 33.1793s; representation median 31.5569s; producer blocked [0.0, 0.0, 0.0]s

Corrected baseline medians: legacy 29.0361s total; compact 33.6394s; compact plus simple split 32.3993s.

This run measured compact monolithic at 33.3746s (-0.79% vs corrected compact baseline) and simple split at 32.8731s (1.46% vs corrected split baseline). Relative to this run's legacy median, compact monolithic changed 17.42% and simple split changed 15.66%.

The selected default from the bounded sweep is one worker: it had the lowest median total. Queue high-water was approximately 3.2 MB and producer blocked time was effectively zero; no body rereads or rewrites occurred.

All mode outputs were checked against the shared logical-evidence root; see logical_equivalence.json.