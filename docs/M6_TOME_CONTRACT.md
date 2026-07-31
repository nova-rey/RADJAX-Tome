# M6 Portable Tome Contract

The portable v1 contract is the checked-in tree at
`contracts/radjax_tome/v1`. It is sufficient to validate canonical v3
identity, profile inventory, cover binding, directory inventory, and safe
transport without importing RADJAX-Tome.

## Paved road

1. Read `cover_page.json` without guessing unindexed files.
2. Require `radjax_tome_cover_v3` and validate its closed envelope.
3. Recompute semantic identity and content-manifest digests using the published
   compact sorted JSON recipe.
4. Validate every manifest member's normalized path, raw digest, and size.
5. Treat `student`, `full_debug_provenance`, and `unpacked` inventories as
   profile-specific while comparing their identity independently of profile or
   transport.

The independent implementation is `tools/validate_radjax_tome_contract.py`.
It is stdlib-only conformance tooling, not a production writer or runtime.

## Transport rules

Producer conformance requires deterministic tar/gzip metadata and byte output.
Consumer safety rejects traversal, duplicates, links, special members,
corruption, unsupported formats, and inventory disagreement. A safe archive
with noncanonical container metadata is reportable as
`transport_noncanonical`; strict conformance may reject it, but semantic
identity never changes because of wrapping.

## Version and compatibility policy

Core v3 objects are closed and unknown core fields fail closed. Within a
recognized profile, opaque semantic-map values may be preserved and hashed but
not interpreted. Unknown profile IDs, required capability IDs, digest methods,
execution-required payload formats, or schema versions fail closed.

Historical `cover_page_v2` and `radjax_tome_package_cover_v1` validate only
through their native readers and yield explicitly incomplete descriptors. They
must never be promoted to v3 identity or authority claims.

The contract tree is Tome-local through M6D. RADJAX-Contract publication is a
separate, approval-gated transfer after byte-equivalence and parity proof.
