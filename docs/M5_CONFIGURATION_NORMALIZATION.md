# M5C Canonical Configuration Normalization

## Public configuration path

M5C makes `TomeBuildIntent` and `ResolvedTomeBuildConfig` the authoritative
production input. Every supported public entry point uses the same sequence:

1. create canonical defaults or explicitly adapt a historical
   `ProductionBuildConfig`;
2. apply one named preset, when requested;
3. apply only explicitly supplied advanced overrides;
4. validate the complete nested intent before runtime initialization;
5. resolve configuration provenance;
6. derive the execution plan; and
7. derive the unchanged 25-field selection-authority projection.

The final stage boundary is deliberately narrow. `production_build_config_from_resolved`
copies the complete resolved request into the historical flat execution shape
for the preserved production and canonical Path-B stages. It does not choose
defaults, interpret aliases, or apply policy. This adapter is a removable
compatibility seam, not a second configuration engine.

`build_production_gpu_tome()` accepts legacy flat input only through this
adapter path, and it also accepts canonical intent or resolved configuration.
The current `production-build` CLI passes resolved configuration directly to
that facade.

## CLI defaults and advanced overrides

The CLI deliberately suppresses `argparse` defaults for `production-build`.
This makes absence distinguishable from an explicit advanced override, so a
preset cannot be accidentally overwritten by parser-owned values. Canonical
defaults preserve the previous command defaults except that
`retain_unselected_exemplar_payloads` is now false for canonical CLI requests.
Keeping unselected payloads is an explicit advanced compatibility override.

`--print-resolved-config` prints stable JSON containing the resolved nested
config, derived execution plan, protected selection-authority payload, and its
hash, then exits before any preflight, model loading, or artifact write.

## Presets

| Preset | Purpose | Semantic configuration |
|---|---|---|
| `smoke` | Small local canonical Path-B proof | CPU reference runtime, 4 examples, exact two-pass C6 route, 4 selected exemplars, selected rerun batch 2 |
| `t4-1k` | Reviewed T4 1K semantic configuration | Canonical C6/PB route; 128 sequence length; 262144 vocabulary; top-k 32; dynamic range 32–262144 at 0.99 mass; 256 selected exemplars; selected rerun batch 8; 1,000 examples |
| `t4-10k` | Same reviewed semantics at a larger corpus limit | Identical to `t4-1k` except `max_examples=10,000` |
| `production-100k` | Same reviewed semantics at production corpus limit | Identical to `t4-1k` except `max_examples=100,000` |

No preset chooses a teacher model, tokenizer, corpus, provenance receipt,
output path, package profile, or external compatibility authority. Those are
required caller intent or explicit overrides. In particular, the two T4 growth
presets are semantic-size presets only; they do not conceal a resource,
teacher, corpus, or output change.

## Preserved authority and Path-B contracts

The selection integration projection remains exactly the documented 25-field
compact, sorted JSON SHA-256 projection. M5C computes it from resolved
configuration and tests equality against the historical
`_selection_integration_hash`; it neither changes the v1/v2 authority-hash
contracts nor changes frozen Golden fixtures.

The canonical Path-B gate remains the exact existing tuple:

- `target_policy=corridor_exemplar_v1`;
- `selection_integration_policy=corridor_first_global_backfill_v1`;
- `exemplar_selection_enabled=true`;
- `exemplar_delivery_path=two_pass_rerun_selected`; and
- non-null `total_selected_exemplar_budget`.

M5C does not extract, reorder, or otherwise alter the native Path-B stage
sequence. In particular, early provisional corridor materialization and late
selected-linked corridor finalization remain separately ordered operations.

## Compatibility and non-claims

Legacy `ProductionBuildConfig` remains accepted as an explicit input adapter.
M5C does not migrate artifact covers, package writers, manifests, archive
transport, historical cover readers, authority-hash recipes, or Golden
fixtures; those are M5D/M5E responsibilities. It makes no performance claim
and does not authorize GPU inference.
