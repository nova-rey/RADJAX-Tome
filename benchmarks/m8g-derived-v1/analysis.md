# M8G derived representation benchmark

Dataset: `M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1`.

This benchmark uses 169 surviving valid payloads from the incomplete v19 workload. It is component-performance evidence, not a complete v19 replay result.

| Mode | Median total (s) | Median representation (s) | Package bytes | Archive bytes |
|---|---:|---:|---:|---:|
| legacy_padded_monolithic | 50.636 | 32.359 | 1,245,188,388 | 61,160,520 |
| compact_k_monolithic | 186.086 | 35.378 | 887,413,318 | 63,443,505 |
| compact_k_immutable_body | 336.134 | 331.972 | 37,080,714 package surface; 998,534,114 complete transaction tree | 1,810,534 |

Raw totals by round:

- Legacy: 49.811, 50.636, 51.085 seconds.
- Compact-K monolithic: 186.086, 185.601, 186.637 seconds.
- Compact-K immutable body: 335.140, 336.134, 337.342 seconds.

Compact-K reduced the measured package surface by 28.7% versus legacy, but increased end-to-end time by 267.3%, dominated by archive construction. Immutable-body separation reduced the complete transaction tree by approximately 19.8% versus the legacy package, but increased total time by 563.1% because of per-body transaction and publication work.

Logical equivalence was checked on all 169 records: coordinates, logical K, token IDs, probabilities, and log probabilities were exact across legacy and compact outputs; all 169 immutable body identities matched the expected codec bodies. No percentage threshold was applied.

Recommendations:

- Compact-K alone: revise/retain based on whether storage reduction outweighs serialization/archive cost.
- Immutable-body separation: revise; storage improved, but transaction overhead was substantial in this run.

The result is paired and valid only for this derived surviving-payload component corpus. It must not be presented as a complete v19 replay or as a direct historical M8D comparison.
