# Pipelined compact representation benchmark

Dataset: M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1 (169 records).

The setup decoded source JSON and performed exactly 169 compact projections outside timed phases.

| Mode | representation | linkage | validation | archive | total | output bytes | archive bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy_padded_monolithic | 24.1633s | 1.0477s | 1.0412s | 2.4302s | 28.8863s | 1,208,094,516 | 65,411,714 |
| compact_k_monolithic | 31.5908s | 0.0013s | 0.3241s | 1.2682s | 33.2828s | 237,336,636 | 40,713,002 |
| compact_k_immutable_body | 31.9858s | 0.0013s | 0.2985s | 1.2531s | 33.5431s | 237,336,636 | 40,712,983 |

Worker sweep (simple split):
- 1 workers: total [33.0798, 32.8229, 32.9893]s; median 32.9893s; representation median 31.3789s; producer blocked [0.0, 0.0, 0.0]s
- 2 workers: total [33.2979, 33.0662, 33.2718]s; median 33.2718s; representation median 31.6106s; producer blocked [0.0, 0.0, 0.0]s
- 4 workers: total [33.9022, 33.3512, 33.1104]s; median 33.3512s; representation median 31.6319s; producer blocked [0.0, 0.0, 0.0]s

Corrected baseline medians: legacy 29.0361s total; compact 33.6394s; compact plus simple split 32.3993s.

This run measured compact monolithic at 33.2828s (-1.06% vs corrected compact baseline) and simple split at 33.5431s (3.53% vs corrected split baseline). Relative to this run's legacy median, compact monolithic changed 15.22% and simple split changed 16.12%.

The selected default from the bounded sweep is one worker: it had the lowest median total. Queue high-water was approximately 3.2 MB and producer blocked time was effectively zero; no body rereads or rewrites occurred.

All mode outputs were checked against the shared logical-evidence root; see logical_equivalence.json.