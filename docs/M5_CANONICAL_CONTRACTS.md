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

## Rollback and acceptance

Because M5B does not route production or emit v3 artifacts, rollback is the
single M5B commit.  Existing artifacts, exact selection authority, CLI
compatibility, M3/M4 Path-B ordering, and frozen v1 Golden evidence retain
their pre-M5B behavior.  The final M5 acceptance condition is no uncommitted
M5-owned changes and a clean tracked worktree, except for the explicitly
documented pre-existing unrelated `.DS_Store`; it does not mean M5 makes no
tracked changes.
