# M8 hashing-cost measurement

This bounded CPU measurement used M8G_DERIVED_VALID_EXEMPLAR_COMPONENT_BENCHMARK_V1 (169 records) on the pushed compact pipeline. Evidence is retained at /home/nyx/m8g/evidence/M8_HASH_COST_M8G.

## Lifecycle

run_selected_source_rerun -> assemble_selected_exemplars -> write_compact_body_store_pipelined_from_compact -> compact_body_from_buffers -> encode_compact_body_packed_from_buffers -> body_raw_digest -> private body write/fsync/rename -> metadata digest -> metadata-only linkage -> output validation -> inventory -> deterministic archive.

Each sample produced 169 distinct immutable bodies. Each body was hashed exactly once in memory (169 body hashes, 237,281,120 bytes); body reread hashes and body rewrites were zero. Linkage read and rewrote only metadata (about 55 KB), with zero body reads, body hashes, or body rewrites.

## Three samples

| sample | total (s) | body hash (s) | validation phase (s) | representation (s) | archive (s) | output bytes | archive bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.3710 | 2.6541 | 0.3213 | 6.8086 | 1.2378 | 237336636 | 40706067 |
| 2 | 8.2684 | 2.2438 | 0.3757 | 6.6209 | 1.2688 | 237336636 | 40706087 |
| 3 | 7.9133 | 2.6809 | 0.3845 | 6.2927 | 1.2328 | 237336636 | 40705970 |
| median | 8.2684 | 2.6541 | 0.3757 | 6.6209 | 1.2378 | 237336636 | 40706067 |

A non-sample post-hoc probe measured the cryptographic portion of output-validation SHA-256 at approximately 0.115 s per 237,336,636-byte output. Cryptographic hashing is measurable (roughly one third of end-to-end wall time when combined with body hashing), but the measured body work is required initial identity creation, not redundant trusted-lifecycle hashing.

## Hash classification

- body_raw_digest: REQUIRED_INITIAL_IDENTITY; once per immutable body.
- metadata SHA-256: REQUIRED_INITIAL_IDENTITY; once for the metadata record.
- output validation SHA-256: REQUIRED_TRUST_BOUNDARY_VERIFICATION; verifies the completed output.
- body reread/re-hash after publication or linkage: REDUNDANT_WITHIN_TRUSTED_LIFECYCLE; observed zero.
- archive: NEGLIGIBLE_OR_NONSCALING for hashing; archive compression itself remains separately timed.

No qualifying redundant hashing operation was measured. No production patch was justified; Contract bytes and logical evidence were left unchanged.

