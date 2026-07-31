# M7 Streaming Contract Publication Pin

The normative M7 streaming contract is RADJAX-Contract release `0.3.1`, Git
tag `v0.3.1`, source commit `f8ca8c0885d7c539a51d1594ba7a38c4d457b4d`. It publishes the byte-identical
`radjax_tome/v2` assets approved in Tome: v4 cover, manifest graph, payload and
shard indexes, semantic identity, canonicalization recipes, compatibility
descriptors, error vocabulary, vectors, and conformance catalog.

Tome pins its established Contract dependency to `v0.3.1`. M7 introduces no
production writer or CLI import of the new publication-resource API; it is used
only for contract verification and conformance. The shared portable validator
is Contract-owned; Tome's tool is a forwarding compatibility command. The checked-in
`contracts/radjax_tome/v2` tree is an offline-capable verified mirror, not an
independently editable authority. Tests fail if it differs from Contract source
or installed package resources.

The v1 mirror and its `v0.2.0` pin remain historical M6 evidence. They do not
reinterpret v3 artifacts as v4.
