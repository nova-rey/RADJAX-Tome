# M7 Streaming Contract Publication Pin

The historical published M7 streaming-contract baseline is RADJAX-Contract
release `0.3.1`, Git tag `v0.3.1`, source commit
`f8ca8c0885d7c539a51d1594ba7a38c4d457b4d`. During correction, the branch
verified against untagged additive `0.3.2` commit
`78ba36300f201d75b016b2fdcf5720e467310815`; no `0.3.2` release or tag was
created. Merge `48a7b2f` nevertheless accepted the completed Tome corrective
implementation. That acceptance does not recast the candidate as a historical
Contract release.

The `0.3.1` baseline remains immutable and publishes the byte-identical
`radjax_tome/v2` assets approved in Tome: v4 cover, manifest graph, payload and
shard indexes, semantic identity, canonicalization recipes, compatibility
descriptors, error vocabulary, vectors, and conformance catalog. This M7 pin is
historical evidence; Tome's current dependency has subsequently advanced to
Contract `v0.7.0` at `cac3dd21e0d56df5a9e6fd50b20267e0b8960995`.

M7 introduced no production writer or CLI import of the publication-resource
API; it is used only for contract verification and conformance. The shared
portable validator is Contract-owned; Tome's tool is a forwarding compatibility
command. The checked-in `contracts/radjax_tome/v2` tree is an offline-capable
verified mirror, not an independently editable authority. Tests fail if it
differs from its historical Contract source or installed package resources.

The v1 mirror and its `v0.2.0` pin remain historical M6 evidence. They do not
reinterpret v3 artifacts as v4.
