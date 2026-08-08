# Public proof graph recipe

The only public regular members are the fixed cover, manifest header, inventory,
provenance documents, layout, two indexes, and shard paths declared by the
shard index. Every discovered regular member must be fixed or be inventoried
exactly once. The cover, header, and inventory are excluded from inventory.

Raw-digest direction is acyclic: `cover -> header -> inventory -> all remaining
members`. Layout references the two indexes. References are closed
`{path, sha256, size_bytes, schema_version}` objects; index references add
`record_count`. Digest receipt direction is never reversed and no public object
may reference the cover.

Member paths are nonempty relative POSIX ASCII names matching
`[A-Za-z0-9][A-Za-z0-9._/-]*`, with no absolute path, empty/dot/dot-dot segment,
backslash, duplicate, symlink, hardlink, device, FIFO, or shadowing member.
JSON is strict UTF-8 without BOM, duplicate keys, nonfinite numbers, or trailing
content. JSONL is strict UTF-8 without BOM/CR/blank lines; each object ends in
exactly one LF and nonempty data requires final LF.
