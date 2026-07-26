# Authority-Hash v2 Migration Contract

## Status and scope

This document defines the narrowly scoped migration from the historical C6
score-pass authority hash to a reproducible semantic authority hash. It does
not alter `tests/fixtures/golden_t4_1k`, whose v1 authority hash and semantic
root are immutable historical evidence.

The migration follows the accepted T4 diagnosis: authority-hash v1 includes
raw `metadata.json` and `c6/production_global_selector.json` bytes. Both files
contain a runtime `created_at` value, so semantically identical fresh runs can
produce different v1 authority hashes.

## Versioned authority contracts

`radjax.c6.score_pass_authority.v1` is the historical contract. Its compact,
sorted JSON payload contains raw SHA-256 values for:

- `metadata.json`;
- `corridors/mode_assignments.json`;
- `corridors/corridor_modes.json`;
- `c6/production_global_selector.json`; and
- the unchanged `selection_integration_config_hash`.

Existing artifacts with no explicit authority-contract field are v1. Their
recorded `score_pass_authority_hash` remains authoritative and verifiable
under v1; no reader may reinterpret it as v2.

`radjax.c6.score_pass_authority.v2` is the current contract. New authority
manifests explicitly record it in `score_pass_authority_contract_version` and
make the v2 digest the active `score_pass_authority_hash`. They also retain
`score_pass_authority_hash_v1` for raw-byte lineage.

## Exact v2 authority projection

The v2 preimage is compact, sorted canonical JSON with these keys:

```text
authority_hash_contract_version
metadata
mode_assignments
modes
selector
selection_integration_config_hash
```

`selection_integration_config_hash` is used unchanged. Its existing 25-field
projection is already the configuration binding for this migration.

The remaining projections are schema-defined and fail closed when their
declared schema is unsupported:

- `metadata` is a strict allowlist over validated `TargetStoreMetadata`:
  schema/store versions; model and tokenizer identity; vocabulary; target type;
  dtype; sequence length; example/shard counts; canonical corpus identity;
  declared model/backend identity; and declared score/corridor/selection policy
  parameters. Unknown metadata, backend telemetry, completion counters, and
  locator values are deliberately outside the projection rather than removed
  by a broad recursive filter. In particular, `created_at`, `created_by`,
  source `kind`, corpus/model provenance paths, provenance `phase`, artifact
  paths, progress paths, and device/workspace telemetry are excluded.
- `mode_assignments` is the v3 assignment manifest's schema, policy, storage,
  observation basis, retention/count fields, plus canonical little-endian
  semantic digests of each packed assignment sidecar and ordered
  `examples_metadata` rows. The sidecars bind the stored
  `(example_index, position, mode_id, fingerprint_index, weight)` data, and
  the ordered example mapping binds index to example ID. The projection
  therefore binds the actual assignment data rather than only its JSON
  manifest and array descriptors.
- `modes` is the `corridor_modes_v2` schema, policy, stat support, observation
  basis/count fields, and ordered mode records. Every mode key, bounds,
  representative coordinate, and count is authority-bearing.
- `selector` is the validated `exemplar_selection_manifest_v1` selection
  surface: selector policies, counts, budget/retention decisions, ordered
  boards and ranked candidates, and selected examples. Only `created_at` is
  excluded.

Changing an included field changes the v2 authority hash. Changing only the
excluded runtime fields does not.

## Raw integrity and migration behavior

Raw-byte integrity is not discarded. V2 authority manifests contain
`raw_artifact_digests` with exactly these file-keyed SHA-256 values:

```text
metadata.json
corridors/mode_assignments.json
corridors/corridor_modes.json
c6/production_global_selector.json
```

The legacy `target_store_metadata_sha256` and existing selector hash remain
compatible raw-integrity fields and must agree with that mapping. Raw digest
changes caused solely by timestamps are expected and are not a v2 semantic
authority change.

For an existing v1 artifact, an explicit v2 Golden capture computes the v2
projection read-only from the source artifact, verifies its recorded v1
authority when possible, and substitutes the v2 authority only in the
captured v2 projection. It never rewrites the artifact or a v1 fixture.

## Golden-contract versioning

`radjax_tome.golden_contract.v1` keeps its existing `:v1` semantic-digest
domain, root `sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`,
and historical authority
`sha256:39588f7bbc69285c9a86c2fb13a7ff34f8ad093e8093abfc56776866b355657a`.

`radjax_tome.golden_contract.v2` uses a distinct `:v2` semantic-digest
domain and records `score_pass_authority_contract_version` in semantic policy.
V1 and v2 contracts are intentionally incompatible: comparison reports an
explicit incompatible-version result and never compares their rows as though
they shared a root contract. A v2 fixture may compare a fresh v2 artifact or
a historical v1 artifact through the explicit read-only v2 projection.

The CLI exposes deliberate capture selection (`--contract-version v1` or
`v2`); automatic artifact comparison captures the observed artifact using the
fixture's contract version. No GPU rerun and no fixture regeneration are part
of this migration.
