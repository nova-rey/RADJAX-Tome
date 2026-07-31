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

Payload layout specifies the index, sequence digest, selected count, per-shard
record capacity, and shard integrity records. The index is JSONL: each row
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
