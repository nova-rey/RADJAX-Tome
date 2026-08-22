# Corrected M8G simple compact-K benchmark

Dataset: M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1 (169 records).

Setup decoded and compacted each source exactly once outside measured phases.

| Mode | median representation (s) | median linkage (s) | median validation (s) | median archive (s) | median total (s) | output bytes | archive bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy_padded_monolithic | 24.1667 | 0.8584 | 1.6967 | 2.4700 | 29.0361 | 1,208,094,516 | 65,411,729 |
| compact_k_monolithic | 32.0710 | 0.0013 | 0.3072 | 1.2433 | 33.6394 | 237,336,636 | 40,712,995 |
| compact_k_immutable_body | 30.8407 | 0.0013 | 0.3174 | 1.2317 | 32.3993 | 237,336,636 | 40,712,905 |

Compact linkage counters are zero for source-payload reads, body reads, body hashes, and body rewrites; only metadata is read and rewritten.

Prior invalid measurement (source reread included in compact linkage): legacy median 30.348 s; compact median 49.737 s; compact linkage 10.572 s; compact representation 37.442 s.

Corrected timers exclude source JSON read/parse and one-time padded-to-compact projection from all mode-specific timers.
