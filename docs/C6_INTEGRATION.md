# C6 Corridor-First Production Integration

C6 is an opt-in production policy that makes the C5 multi-role coordinate set
authoritative for delivery and package-visible selection surfaces.

## Policies

`global_only_v1` remains the default and preserves the existing production
selection path. `corridor_first_global_backfill_v1` requires all of the
following production inputs:

- a production-grade ranked global-board supply artifact;
- a real source-passport index containing shard/row/position and payload lookup
  information.

C6.1 derives strict C2 candidate features itself from the current build's packed
corridor assignments, mode bounds, and shard statistics. The generated stream
records the assignment and mode hashes, normalization derivations, and feature
file hash in `c6/corridor-features/`. External C2 JSONL and C4/C5 checkpoints
are intentionally rejected in the integrated production policy because they
cannot prove linkage to the current score surface.

Compatibility-proxy C2 records, development selector manifests, and synthesized
passports are rejected by the production path. The global supply must carry the
selector policy/schema provenance emitted by the production exporter.

## Production invocation

The C6 flags are available on `production-build`:

```text
--selection-integration-policy corridor_first_global_backfill_v1
--total-selected-exemplar-budget N
--fingerprint-corridor-budget-fraction 0.50
--fingerprint-corridor-budget-max N
--fingerprint-corridor-mode-cap 10
--fingerprint-corridor-candidate-pool-cap 4
--require-full-selected-budget
```

The policy is also recorded in the plan, `emission_config.json`, streaming
resume hash, `production_build_report.json`, and
`reports/fingerprint_corridor_coverage.json`.

## Authority and validation

C5 rich records are the authoritative unique coordinate set. The integrated
validator compares that set with the legacy selected projection, delivered
payloads, source passports, the actual `curriculum/selected_routes.json` union,
audit, and package surfaces. Route count is distinct from unique coordinate
count, so legitimate multi-board routes do not inflate selected payload counts.
The coverage report distinguishes unique selected records from selection
obligations and multi-role counts.

The production selector exporter is
`export_production_global_board_supply`. It accepts only selector manifests
that explicitly declare `production_global_selector=true` and provide ranked
candidate supply; the development selector-manifest adapter is intentionally
not accepted.

## Delivery and packaging

Path A materializes one payload per C5 coordinate from the captured source
payload. Path B reruns exactly the unique C5 coordinate set. Multi-role records
never duplicate payload storage. Full-debug and student packages copy the C6
coverage and validation reports when present, retain the curriculum route
artifact, and block publication if package-local C5/legacy/payload/passport/
curriculum/audit parity fails.

## Scope boundary

The C6 producer remains PyTorch CPU/NVIDIA GPU only. C6 does not add TPU/JAX
execution, change global-only defaults, redesign C1-C5 selection math, or
claim T4 empirical success from CPU/fake tests.
