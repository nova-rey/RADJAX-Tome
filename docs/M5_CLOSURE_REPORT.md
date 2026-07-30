# M5 Closure Report — Canonical Tome Contract

## Scope and status

M5 establishes one canonical path from build intent through profile- and
transport-independent Tome identity to a validated directory or archive. It
does not change model behavior, corpus content, authority-hash recipes,
canonical Path-B stage order, frozen Golden fixtures, GPU execution, or later
Student/runtime work.

The checkpoint commits before this closure record are:

| Checkpoint | Commits | Result |
|---|---|---|
| M5A | `d675095` | 67-field ownership ledger and behavior characterization |
| M5B | `a9d23d0`, `dc424a4`, `4badd93` | typed contracts, historical mapping documentation, closed-shape hardening |
| M5C | `49172b4`, `430ea1f` | canonical normalization/presets and fail-closed validation |
| M5D | `372f7ea`, `c450eac`, `e642b5c` | source identity, v3 writers, complete inventory, unified deterministic transport |
| M5E | `ba5ae0e` | native historical validation plus non-inferential adapters |

The M5F closure commit is the Git commit containing this report and its final
validation evidence. No merge to `main` is part of M5 closure.

## Contract versions introduced or activated

- Configuration: `radjax_tome_build_intent_v1`,
  `radjax_tome_resolved_build_config_v1`,
  `radjax_tome_execution_plan_v1`, and
  `radjax_tome_normalized_production_request_v1`.
- Tome identity/metadata: `radjax_tome_semantic_identity_v1`,
  `tome_content_manifest_v2`, and `radjax_tome_cover_v3`.
- Historical compatibility retained: cover-page v2 and
  `radjax_tome_package_cover_v1`; neither is a new-writer output or a v3
  identity claim.
- Preserved authority/Golden contracts: the unchanged fixed 25-field
  selection-authority projection, historical
  `radjax.c6.score_pass_authority.v1`, and versioned
  `radjax.c6.score_pass_authority.v2` Golden projection behavior.

## Closure matrix

| Required proof | Evidence |
|---|---|
| Preset resolution, CLI/programmatic equivalence, explicit override order, early contradictions | `tests/test_m5c_configuration_normalization.py`, `tests/test_m5b_canonical_config_contract.py` |
| Exact canonical Path-B route and preserved native boundary | `tests/test_m3c_canonical_config_boundary.py`, `tests/test_native_path_b_api.py`, `tests/test_native_path_b_contracts.py`, `tests/test_native_path_b_resume.py` |
| Fixed 25-field projection and authority v1/v2 behavior | `tests/test_m5a_contract_characterization.py`, `tests/test_m5b_canonical_config_contract.py`, `tests/test_authority_hash_contract_v2.py` |
| Frozen Golden v1 and versioned v2 projections | `tests/test_golden_t4_1k_fixture.py`, `tests/test_golden_contract_compare.py`, `tests/test_golden_projection_truth_gate.py`, `tests/test_authority_hash_contract_v2.py` |
| Profile-neutral identity, profile-specific manifests, complete raw inventory | `tests/test_m5b_canonical_tome_contract.py`, `tests/test_m5d_canonical_artifact_contract.py`, `tests/test_tome_packaging_profiles.py` |
| Equivalent directory/archive validation and deterministic `.rtome`/gzip round trips | `tests/test_tome_cover_page.py`, `tests/test_tome_bundle.py` |
| Native historical v2/v1 validation, non-inferential mapping, ambiguity rejection | `tests/test_m5e_compatibility_adapters.py`, `tests/test_tome_packaging_profiles.py` |
| Canonical output only and profile constraints | `tests/test_tome_cover_page.py`, `tests/test_tome_packaging_profiles.py` |
| Artifact writer atomicity on completed streaming resume | `tests/test_streaming_backend_builder.py` |
| Dependency, import, and tracked-source disposition checks | `tests/test_m3a_import_isolation.py`, `tests/test_hydra_disposition.py` |

## Final validation commands

```bash
PYTHONPATH=src:. pytest -q --cache-clear
ruff check src tests
ruff format --check src tests
git diff --check
python3 -m json.tool docs/hydra_disposition.json >/dev/null
```

## Compatibility behavior and limits

Current writers emit only `radjax_tome_cover_v3`. The v3 directory cover and
archive cover validate the same closed identity/manifest contract; archive
validation additionally proves its exact member inventory and raw digests.
`student` and `full_debug_provenance` can have different physical inventories
and manifest digests while carrying the same source-derived Tome identity.

Historical cover-page v2 and package-cover v1 retain their native validators.
`tome.compatibility` validates a historical directory or safe archive under
that native contract before mapping known information into
`HistoricalTomeDescriptor`. It never creates a semantic identity, authority
binding, profile-complete inventory, or unrecorded training fact. A standalone
historical JSON cover has no artifact context and therefore cannot claim native
validity.

## Intentional deferrals and non-claims

- No GPU/T4 inference was run for M5, and no Golden fixture was regenerated.
- The immutable v1 Golden root remains
  `sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`.
- The conditional July 19/July 24 historical-artifact v2 comparison remains a
  read-only integration test when both externally mounted artifacts are
  available; their reviewed M4D proof is preserved, not rerun by M5.
- M5 does not begin M6, optimize performance, alter corpora/models, expand the
  CLI/TUI, add multi-GPU work, delete research code, or merge `main`.
