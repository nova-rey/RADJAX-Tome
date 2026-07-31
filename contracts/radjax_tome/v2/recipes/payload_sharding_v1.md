# Payload sharding v1

The cover references and hashes only `manifests/content-manifest-header.json`.
The header references and hashes only `manifests/content-manifest-inventory.jsonl`.
The inventory lists every other package member, never itself, the header, or the
cover. This directed `cover -> header -> inventory` chain is acyclic.

`selected_exemplars/payload-layout.json` binds layout/version, index reference,
the selected-record sequence digest, selected count, shard capacity, and shard
integrity records. `payload-index.jsonl` is one index record per line: it maps
`logical_id` to `(shard,row)`, a raw `payload_sha256`, and a semantic digest.
The layout index reference owns the only index record count; JSONL records do
not repeat it.

Canonical JSON is UTF-8, recursively key-sorted, compact (`separators=(',',
':')`), and SHA-256 prefixed with `sha256:`. Sequence digest input is exactly
`{"schema_version":"selected_exemplar_payload_sequence_v1","records":[{"logical_id":...,"payload_semantic_digest":...},...]}` in strict
`selection_index` order. A shard semantic digest uses the same formula over its
contiguous subsequence. Regrouping records changes layout/integrity hashes but
not the sequence or v2 identity digest.

Every listed M7A producer field is required and semantic; null is forbidden.
Numbers must be finite before canonical JSON encoding. The semantic payload
digest is canonical JSON over exactly those 38 fields plus, when present, an
`opaque_extensions` map. It excludes wrapper, shard/address, raw-byte, and
transport fields. Each opaque extension key is lower-snake-case and its value
is `{schema_id, semantic_digest}`; it is opaque only when that declaration is
present, otherwise it is rejected.
