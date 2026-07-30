# M5 Canonical Configuration and Tome Contract (M5B)

## Status and scope

This document defines the M5B public contract boundary.  It is deliberately
pure and dependency-light: M5B neither reroutes `production-build` nor changes
the historical production writer, `CanonicalPathBConfig`, cover-page v2,
package-cover v1, `.rtome`, `tgz`, authority-hash v1/v2, or the frozen Golden
fixture.  M5C--M5F require the separate post-M5B review approval.

## Configuration contract

`radjax_tome.builder.config` has three explicit representations:

1. `TomeBuildIntent` is the typed caller request.  Its sections are `teacher`,
   `corpus`, `behavior`, `corridor_policy`, `selection`, `execution`,
   `outputs`, `compatibility`, and `package`.
2. `ResolvedTomeBuildConfig` is a validated intent plus resolution provenance
   (`source`, optional semantic-size preset name, and explicit overrides).
   Resolution is pure: it does no filesystem probing, runtime loading, or
   artifact construction.
3. `TomeExecutionPlan` derives existing report-path defaults and operational
   batch/shard controls from resolved config.  It is not canonical Tome
   identity and is not yet the production execution input.

The legacy adapter copies all 67 current `ProductionBuildConfig` fields into
those sections.  The old approximate count of 61 is superseded by the
dataclass-derived 67-field inventory in `M5_CONTRACT_OWNERSHIP.md` and
`test_m5a_contract_characterization.py`; this is a repository-derived count,
not an invented `61 + 6` explanation.

`PackageIntent.profile` may be `unpacked`, `student`, or
`full_debug_provenance`; its `transport` may be `directory`, `rtome`, or
`tgz`.  These express package materialization only.  They are intentionally
outside both selection authority and canonical Tome identity.

## Preserved selection-authority projection

`selection_authority_payload_v1` has exactly the existing 25 keys, the same
path-to-string conversion, four fixed C2--C5 schema literals, compact sorted
JSON encoding, and SHA-256 recipe as
`builder.production._selection_integration_hash`:

```text
selection_integration_policy, teacher_model, tokenizer_id, dataset_path,
corpus_manifest_path, target_policy, sequence_length, vocab_size, top_k,
num_buckets, dynamic_top_k_min, dynamic_top_k_max,
dynamic_mass_threshold, selected_rerun_batch_size,
total_selected_exemplar_budget, fingerprint_corridor_budget_fraction,
fingerprint_corridor_budget_max, fingerprint_corridor_mode_cap,
fingerprint_corridor_candidate_pool_cap, require_full_selected_budget,
c2_schema, c3_schema, c4_schema, c5_schema, delivery_path
```

M5B does not replace the production implementation with this helper.  The
characterization requires both functions to produce the existing fixed hash.
Authority-hash contract v1/v2 and its Golden projection are likewise
unchanged; this is a configuration-boundary preservation, not another
authority-hash migration.

## Canonical Tome v3 contract

`radjax_tome_cover_v3` is a nested public descriptor, not a 67-field flat
cover.  It has exactly these top-level sections:

```text
identity, training, package, manifests, authority, provenance, validation
```

`identity` uses `radjax_tome_semantic_identity_v1` and hashes this exact
canonical projection:

```json
{
  "schema_version": "radjax_tome_semantic_identity_v1",
  "training_payload": [
    {"logical_id": "<stable training role>", "semantic_digest": "sha256:<digest>"}
  ],
  "training_contract": {"<schema-defined training settings>": "<value>"},
  "authority": {"<training authority binding>": "<value>"}
}
```

Entries are sorted by `logical_id`; the enclosing projection is compact,
sorted-key JSON hashed with SHA-256.  `training_contract` and `authority` are
schema-defined semantic maps.  The constructor rejects runtime-only
`created_at`/`generated_at`, package profile, transport/archive, manifest
digest, and raw-artifact-digest keys rather than silently dropping them.  Thus
the canonical Tome identity derives only from training-authoritative semantic
payload; it contains no profile inventory or transport wrapping.

`tome_content_manifest_v2` is a separate, profile-specific inventory:

```text
schema_version, profile, semantic_identity_digest, inventory, manifest_digest
```

Each inventory entry retains its raw-byte `sha256`, size, classification, and
whether it is training-authoritative.  A full-debug package may therefore have
a different manifest digest from a student package with the same identity.
The content manifest must exclude `cover_page.json`, allowing the v3 cover to
reference its digest with no circular hashing.  The package section may name a
transport, but transport and tar/gzip wrapping remain nonsemantic.

### Closed-shape validation

M5B exposes three standalone validators:

- `validate_canonical_tome_semantic_identity` validates the complete
  `radjax_tome_semantic_identity_v1` structure and recomputes its digest.
- `validate_canonical_content_manifest` validates the complete
  `tome_content_manifest_v2` profile inventory and recomputes its digest.
- `validate_canonical_tome_cover` validates the closed v3 cover shape and its
  nested identity/manifest references.

There is no extension mechanism in these v1/v2/v3 core contracts. The identity
has exactly `schema_version`, `training_payload`, `training_contract`,
`authority`, and `semantic_digest`; each payload entry has exactly
`logical_id` and `semantic_digest`. Payload IDs must be nonempty, unique, and
strictly lexical-order sorted. Every semantic digest is exactly
`sha256:` plus 64 lowercase hexadecimal characters.

The content manifest has exactly `schema_version`, `profile`,
`semantic_identity_digest`, `inventory`, and `manifest_digest`; each inventory
entry has exactly `path`, `sha256`, `size_bytes`, `classification`, and
`training_authoritative`. Inventory paths are nonempty, normalized relative
POSIX paths, sorted and unique, and cannot be `cover_page.json`. Raw digests
use the same exact SHA-256 form; sizes are nonnegative integers (not
booleans); classifications are one of `training_critical`,
`integrity_or_provenance`, `diagnostic`, `human_readable`, or `operational`;
and `training_authoritative` is a boolean.

The v3 cover has exactly `schema_version`, `identity`, `training`, `package`,
`manifests`, `authority`, `provenance`, and `validation`. `package` has exactly
`profile` and `transport`; `manifests` has exactly `content`. `training` and
`authority` must exactly equal their identity counterparts. `provenance` and
`validation` are JSON objects whose leaf vocabularies remain governed by their
referenced producer/validator schemas; they do not add cover sections or
identity fields. Any future core extension requires a new versioned contract,
not an ignored extra field.

## M5D source-derived identity seam

`tome.canonical_artifact` derives a semantic identity from the source artifact
before any student/full-debug profile materialization. It hashes canonical JSON
projections of the core training sidecars, selected payload/curriculum JSON,
and the training authority binding after excluding runtime timestamps. It also
binds the authoritative shard NPZ payloads, corridor assignment NPY arrays,
and assignment example metadata. NPZ semantic digests use sorted uncompressed
member bytes so zip-container metadata is not identity. The physical inventory
excludes `cover_page.json` to avoid a circular cover/manifest hash, while
recording independent raw-byte digests for metadata, mode assignments,
corridor modes, and the production global selector when present. The profile
writer receives this source identity unchanged, so extra provenance receipts
and transport wrapping cannot change the Tome identity.
M5D package writers now emit `radjax_tome_cover_v3`; the former package-v1
cover is retained only as `provenance.historical_package_cover_v1` for explicit
compatibility diagnostics. The live package validator validates the v3 cover
and its raw inventory first, then reuses the existing manifest-level checks
through explicit internal references rather than treating the legacy receipt
as public authority.

The unpacked artifact writer now follows the same path and retains its former
cover-page v2 document only as `provenance.historical_cover_page_v2`. The
canonical bundle writer consumes the same v3 inventory for uncompressed
`.rtome` and deterministic gzip transport. New `tgz` profile packages use that
bundle writer at archive root rather than a second package-specific tar path.
The archive wrapper is transport-only: it may alter v3 package transport
metadata and raw archive bytes, but it cannot alter the embedded semantic
identity. Archive validation and directory validation both begin with the same
closed v3 cover/manifest validation; archive validation also proves the exact
member inventory and recorded raw digests.

## Compatibility and migration behavior

Historical cover-page v2 and `radjax_tome_package_cover_v1` continue to be
validated by their native validators.  M5B does not reinterpret either as v3,
does not write a v3 cover, and does not infer absent historical information.
Later M5 readers may map known historical fields into this nested descriptor
only with explicit provenance of the source contract and without asserting
unknown facts.  Equality helpers require the same identity schema version;
cross-version comparisons fail closed rather than treating unequal contracts
as interchangeable.

The required compatibility mapping is deliberately source-specific:

| Historical source | Known fields | Canonical concept | Mapping limit |
|---|---|---|---|
| cover-page v2 | `teacher`, `tokenizer`, `targets`, `behavioral_surfaces`, `recommended_training_plan` | `identity` and `training` descriptors | Retain only fields present in v2; do not invent package inventory or selection authority. |
| cover-page v2 | `contents` entries and their digests/classifications | Candidate `manifests.content.inventory` entries | Preserve v2's stated file facts; a later adapter must label the source contract and cannot claim a v2 inventory is package-profile complete. |
| cover-page v2 | `created_at`, `created_by`, corpus/model provenance, claims | `provenance` / validation claims | Runtime creation time is provenance only and is excluded from semantic identity. |
| cover-page v2 | `validation`, `claims_not_made` | `validation` | Preserve stated claims and non-claims without upgrading their proof. |
| package-cover v1 | `package_profile`, package schema, archive/layout facts | `package` and source-contract provenance | Profile/transport are package metadata only; never copy them into semantic identity. |
| package-cover v1 | content, shard, assignment, and selected-payload manifest references/digests | `manifests.content` and profile-specific inventory | Preserve raw integrity data and profile constraints; do not infer missing teacher/training-plan fields. |
| package-cover v1 | package audit and validation facts | `validation` and provenance | Preserve the historical validator's scope; do not imply v3 validation. |

An M5E adapter must emit the source schema and every unavailable v3 section as
unknown or absent, rather than filling it from a sibling file or a default.

M5E implements that boundary in `tome.compatibility`. Its
`adapt_historical_tome_cover` and `read_historical_tome_descriptor` APIs accept
only cover-page v2 or `radjax_tome_package_cover_v1`; they return a
`HistoricalTomeDescriptor`, not a v3 cover. The descriptor uses the v3 section
vocabulary only for facts proven by the source: v2 can provide target settings
and a non-profile-complete inventory claim, while package v1 can provide its
declared profile, explicit `unpacked_directory` to `directory` mapping, and
manifest references. Unknown identity, authority, training, profile, or
transport facts remain absent as applicable. The path-based reader first
validates a directory under its native v2 or v1 validator; for a legacy `tgz`
outer directory, it safely materializes the one cover-root into a temporary
directory and applies that same native validator before mapping. It never
rewrites the source, validates historical data as v3, or infers sibling facts.
A standalone JSON cover has no artifact context and must use the pure adapter
directly; it cannot claim native validity. Unsupported layouts/profiles,
malformed inventories, and unknown schemas fail closed with a migration
diagnostic. New artifact and package writers always emit v3.

## Rollback and acceptance

Because M5B does not route production or emit v3 artifacts, rollback is the
single M5B commit.  Existing artifacts, exact selection authority, CLI
compatibility, M3/M4 Path-B ordering, and frozen v1 Golden evidence retain
their pre-M5B behavior.  The final M5 acceptance condition is no uncommitted
M5-owned changes and a clean tracked worktree, except for the explicitly
documented pre-existing unrelated `.DS_Store`; it does not mean M5 makes no
tracked changes.
