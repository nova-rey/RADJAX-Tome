# M7 Payload Sharding Contract

M7 introduces the Tome-local proposed `radjax_tome/v2` publication contract.
It is not a reinterpretation of native v3: v3 remains grouping-sensitive and
historical. Version 2 gives a shard-size-independent selected-payload identity.

## Streaming manifest graph

`cover_page.json` hashes a compact content-manifest header. The header hashes a
JSONL inventory. The inventory includes every remaining member but excludes
the cover, header, and itself. Readers may therefore verify cover, then header,
then inventory in one direction without a manifest cycle.

## Selected payloads

Payload layout specifies the payload and shard-index references, sequence
digest, selected count, and per-shard record capacity. Both indexes are JSONL:
the shard index carries shard integrity records, and each payload-index row
maps a logical record to `(shard,row)` and binds raw and semantic hashes. The
layout reference owns the sole record count. Validators must stream lines and
shards; they must not materialize the complete selected payload set.

The v2 semantic projection explicitly preserves all 38 M7A producer fields:
`selected_example_id`, `selected_position`, `selected_score`,
`score_selected_position_entropy`, `score_top_token_id`, `source_shard_id`,
`source_row`, `source_position`, `source_score`, `source_top_token_id`,
`source_score_policy`, `payload_ref`, `selected_policy`, `source_delivery_path`,
`top_token_ids`, `top_log_probs`, `top_probs`, `top_selection_mask`,
`effective_top_k`, `top_mass`, `tail_mass`, `bucket_masses`, `teacher_entropy`,
`sequence_length`, `vocab_size`, `num_buckets`, `dynamic_top_k`,
`dynamic_mass_threshold`, `dynamic_top_k_max`, `top_k_saturated`,
`long_tail_class`, `long_tail_warnings`, `effective_top_k_fraction_of_vocab`,
`semantic_tail_tag`, `selected_board`, `corridor_mode_id`,
`corridor_fingerprint_id`, and `corridor_assignment_status`.

All are required semantic fields; none is a layout/integrity exception. Null,
non-finite numeric values, undeclared fields, and undeclared opaque extensions
are invalid. See the v2 recipe and schemas for canonical digest inputs.

The compact identity binds `payload_sequence_digest` and `selected_count`, not
an eagerly materialized payload list. Its digest input is the exact object
formed by `schema_version`, `payload_sequence_digest`, `selected_count`,
the sorted `nonselected_training_payload` v3 projection, `training_contract`,
and `authority`; `semantic_digest` is the resulting
digest, not an input. Opaque values within a recognized profile are retained
and hashed through their declared `{schema_id, value, semantic_digest}`
envelope. Unknown profiles, execution capabilities, digest methods, and closed
core fields fail closed.

## Student-profile byte boundary

The ordinary `student` v4 writer is the governed M7 byte boundary. It retains
the closed semantic/core members needed by the v4 identity and selected
consumer, then writes its own deterministic inventory and cover. Legacy
runtime/debug evidence is intentionally not copied into this profile:
`c6/`, `reports/`, run/progress manifests, delivery timing reports, production
reports, linkage audits, and side-board-only diagnostics remain available in
the legacy producer tree and in `full_debug_provenance`, but are not Student
M7 members. The core `metadata.json` and `teacher_manifest.json` timestamps are
required legacy fields, so the v4 writer emits the fixed epoch marker for
those nonsemantic fields. Absolute machine-local path strings in retained JSON
are reduced to source-relative (or stable external-basename) references.

This projection occurs during the normal terminal v4 writer before the
directory and archive are sealed; it is not a fixture rewrite or a post-build
normalization step. It does not alter selected record order, semantic framing,
shard assignment, Contract validation, the v2 default, or the legacy artifact
used for resume and historical verification. The explicit debug profile keeps
the complete legacy evidence surface for diagnostics.

The explicit v6 Student package writer applies the same boundary to its copied
Path-B source members before it materializes the Contract-owned v6 resources
and manifests. This keeps the native M7 sibling and the v6 directory and
archive forms byte-reproducible from identical declared inputs while leaving
the producer/debug tree available for diagnostics and resume. Historical v5
and v6 fixture bytes are not rewritten in place; the projection is selected
only by the explicit v6 package path.
