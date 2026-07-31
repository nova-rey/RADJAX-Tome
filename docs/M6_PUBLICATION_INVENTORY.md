# M6 Publication Inventory and Ownership Ledger

## Scope

M6 publishes the smallest portable contract needed to validate and consume a
canonical Tome artifact without importing RADJAX-Tome. It does not publish a
student runtime, producer runtime, model behavior, or Golden replacement.

## Ownership ledger

| Surface | Current producer | Native validator | M6 disposition |
| --- | --- | --- | --- |
| `radjax_tome_cover_v3` | `tome.cover_page`, `tome.packaging` | `tome.contracts`, directory/bundle validators | Normative portable cover contract |
| `radjax_tome_semantic_identity_v1` | `tome.canonical_artifact` | `tome.contracts` | Normative identity envelope and digest recipe |
| `tome_content_manifest_v2` | `tome.canonical_artifact` | `tome.contracts` | Normative profile inventory and raw-integrity contract |
| Profiles and classifications | package/artifact writers | `tome.contracts`, `tome.packaging` | Normative enums and inventory rules |
| Directory, `.rtome`, and gzip wrapping | `tome.bundle` | `tome.bundle` | Normative safety and transport contract; producer determinism is separate |
| v2 cover and package-v1 cover | legacy writers only | native validators, `tome.compatibility` | Compatibility-only, explicitly incomplete descriptors |
| Artifact-visible training and authority references | `tome.canonical_artifact` | v3 cover validator | Normative recognized-profile registry; opaque values are not inferred |
| Typed configuration and execution plan | `builder.config` | builder validation | Private producer configuration except artifact-visible output fields |
| Selection authority, authority v1/v2, Golden data | production and Golden modules | existing regression suites | Preserved references, not a second portable recipe |
| Model loading, CLI routing, writer orchestration | builders and CLI | implementation tests | Private |

## M6 boundary rules

The v3 identity is derived only from training-authoritative semantic payload.
Profile inventory, raw manifest digest, archive bytes, timestamps, and
container metadata are not semantic identity. Raw inventory remains
integrity-significant and must be validated separately.

Canonical writers must create deterministic transport bytes: sorted archive
members, normalized tar metadata, and deterministic gzip metadata. Consumer
safety is different: consumers reject unsafe paths, duplicate members, links,
special files, corruption, unsupported formats, and inventory mismatches.
Safe but noncanonical container metadata is a reportable canonicality finding,
not inherently a safety failure.

The closed v3 envelope has no ignored-field extension mechanism. Within a
recognized semantic profile, opaque semantic-map values can be preserved and
hashed without interpretation. Unknown profile IDs, required capability IDs,
digest methods, execution-required payload formats, or schema versions fail
closed.

## Current documentation correction

`docs/TOME_COVER_PAGE.md` documents the historical v2 cover as if it were the
current front door. M5 writers now emit v3 and retain v2 only under provenance.
M6B replaces that guidance with v3 contract material and marks v2 historical.

## Single-source strategy

M6B will add `contracts/radjax_tome/v1` as the Tome-local normative source.
Native code is an implementation checked against published schemas, recipes,
vectors, and fixtures. The only portable validator is introduced in M6B; M6D
uses it for parity and does not add another portable validator. RADJAX-Contract
remains untouched until the M6D review gate.
