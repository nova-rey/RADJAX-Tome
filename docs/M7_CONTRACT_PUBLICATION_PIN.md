# M7 Streaming Contract Publication Pin

The normative M7 streaming contract is RADJAX-Contract release `0.3.0`, Git
tag `v0.3.0`, source commit `1b8c6f79bdfa32464a55aa93c8b3626aaa1a0047`. It publishes the byte-identical
`radjax_tome/v2` assets approved in Tome: v4 cover, manifest graph, payload and
shard indexes, semantic identity, canonicalization recipes, compatibility
descriptors, error vocabulary, vectors, and conformance catalog.

Tome pins its established Contract dependency to `v0.3.0`. M7 introduces no
production writer or CLI import of the new publication-resource API; it is used
only for contract verification and conformance. The checked-in
`contracts/radjax_tome/v2` tree is an offline-capable verified mirror, not an
independently editable authority. Tests fail if it differs from Contract source
or installed package resources.

The v1 mirror and its `v0.2.0` pin remain historical M6 evidence. They do not
reinterpret v3 artifacts as v4.
