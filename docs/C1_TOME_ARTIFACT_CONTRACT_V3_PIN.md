# Tome Artifact Contract v3 Adoption Pin

This branch adopts the released RADJAX Contract v3 artifact contract only for
explicit, opt-in Tome publication. The default Tome artifact path remains v2.

| Identity | Value |
| --- | --- |
| package | `radjax-contract==0.9.0` |
| release tag | `v0.9.0` |
| peeled release commit | `1fa43e1aea2e198511db86dafb0aeefa525d48c7` |
| reviewed semantic implementation | `63ea3cfa6c7ae91e6a42f4929d59d9cdd6748836` |
| artifact contract | `radjax_tome_artifact_contract@3.0.0` |
| semantic profile | `selected_exemplar_semantic_profile_v3` |
| Contract v3 asset-tree digest | `sha256:4f81cd901ad074cc24e279e37b7fbfbe25c22fb6b7fd77cbcc747b202995acf8` |

The dependency is pinned by commit, not by a mutable branch or tag. The
tracked `contracts/radjax_tome/v3` directory is an offline mirror of the
released wheel/source asset tree. Its `SHA256SUMS` file is the authority for
the mirror contents; the mirror digest above is the SHA-256 of that inventory.

The release receipt identity is
`sha256:78b3827b450b68ee7cbaa7852639aeeef4c87fb289ca388ab76c066df99b0806`.
This v3 identity is distinct from Student v5/v6 contracts and from the
existing M7/v4 transport. No released Contract schema is copied into the v2
path and no default-version migration is made here.

## Release-evidence parity

The sanitized release metadata committed beside the ordinary-CPU v3 fixture is

* `tests/fixtures/tome_artifact_v3_smoke/contract_release_receipt_v0.9.0.json`;
* `tests/fixtures/tome_artifact_v3_smoke/contract_release_SHA256SUMS_v0.9.0`;
* `tests/fixtures/tome_artifact_v3_smoke/contract_release_asset_hashes_v0.9.0.json`.

It records the published wheel and sdist names, sizes, SHA-256 values, the
release receipt and `SHA256SUMS` asset digests, the annotated tag/peeled commit,
and the reviewed semantic commit. It contains no local paths or network
locations. The C1 tests verify that the release receipt, release-asset metadata,
source/wheel/sdist v3 asset-tree identity, and the checked-in offline mirror all
agree. The committed receipt and `SHA256SUMS` bytes are hashed locally against
their published asset digests. The test suite is intentionally network-
independent: it verifies committed release identities and mirror bytes, while
a release download or signature service remains an external release-pipeline
concern.
