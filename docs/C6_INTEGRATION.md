# C6 Corridor-First Production Integration

C6 is an opt-in production policy that makes the C5 multi-role coordinate set
authoritative for delivery and package-visible selection surfaces.

## Policies

`global_only_v1` remains the default and preserves the existing production
selection path. `corridor_first_global_backfill_v1` requires all of the
following production inputs:

- a strict C2 candidate-feature JSONL artifact with explicit or reproducibly
  derived membership, core distance, mode support, and difficulty fields;
- a production-grade ranked global-board supply artifact;
- a C4 claims artifact or enough inputs to build C2 through C4;
- a C5 selection artifact or enough inputs to build C5;
- a real source-passport index containing shard/row/position and payload lookup
  information.

Compatibility-proxy C2 records, development selector manifests, and synthesized
passports are rejected by the production path.

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
payloads, source passports, curriculum projection, audit, and package surfaces.
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
coverage and validation reports when present and recompute package-local
selected summaries.

## Scope boundary

The C6 producer remains PyTorch CPU/NVIDIA GPU only. C6 does not add TPU/JAX
execution, change global-only defaults, redesign C1-C5 selection math, or
claim T4 empirical success from CPU/fake tests.
