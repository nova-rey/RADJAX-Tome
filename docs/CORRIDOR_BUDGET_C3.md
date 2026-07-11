# C3 Bounded Corridor Coverage Budget

C3 converts a validated C2 per-mode candidate leaderboard into a compact
coverage plan. It allocates representative *slots* only. It does not claim
candidate coordinates, deduplicate global roles, backfill global boards, route
curriculum records, materialize payloads, or change production selection. C4
owns those operations.

## Budget Contract

For total selected-exemplar budget `B`, corridor fraction `F`, optional hard
maximum `H`, and per-mode cap `C`:

```text
fractional_ceiling = floor(B * F)
corridor_budget_ceiling = fractional_ceiling if H is unset else min(fractional_ceiling, H)
q_m = min(C, retained_pool_count_m, eligible_candidate_count_m)
raw_mode_capacity = sum(q_m)
actual_corridor_budget = min(corridor_budget_ceiling, raw_mode_capacity)
global_budget = B - actual_corridor_budget
```

`F` is parsed as `Decimal`, so decimal fractions such as `10 * .3` floor to
three without binary floating-point drift. The default fraction is `0.50` and
the default mode cap is `10`.

## Allocation

When the budget can cover every capacity-positive mode, C3 performs deterministic
ascending-mode round-robin water filling. Every eligible mode receives one slot
before any mode receives a second slot, and rounds continue until capacity or
budget is exhausted.

When the first round is oversubscribed, recipients are ordered by descending
top-candidate utility, descending top-candidate membership, descending
top-candidate centrality, descending mode support, then ascending mode ID. The
plan stores only these scalar priority snapshots; it never stores or claims a
candidate coordinate.

Modes remain visible when they receive zero slots. Reasons are
`no_eligible_candidates`, `empty_candidate_pool`,
`corridor_budget_exhausted`, or `mode_capacity_zero`. A lower actual corridor
budget than the ceiling is reported as `unused_corridor_ceiling`.

## Input and Provenance

C3 loads C2 through the canonical validated loader. Production-grade C2 input
is required by default. A non-production C2 artifact requires both
`--allow-nonproduction-leaderboards` and a nonempty override reason; the plan
then remains non-production and records that override. C2 feature fidelity,
source identity, feature hash, C2 policy ID, leaderboard manifest hash, and
mode-leaderboard hash are retained in the plan provenance.

## Artifact

```text
coverage-plan-dir/
  coverage_plan.json
  validation_report.json
```

The plan contains the complete budget policy, exact ceilings, raw capacity,
actual corridor/global budgets, every observed mode allocation, zero reasons,
priority snapshots, source provenance, and deterministic summary counts. It
contains no candidate records or payloads. The validation report records the
SHA-256 of `coverage_plan.json`.

## API and CLI

```python
from radjax_tome.fingerprint.corridor_budget import (
    CorridorBudgetPolicy,
    allocate_corridor_coverage,
    validate_corridor_coverage_plan_object,
    write_corridor_coverage_plan,
)
```

```bash
radjax-tome allocate-fingerprint-corridor-coverage \
  --leaderboards ./c2-leaderboards \
  --total-selected-exemplar-budget 5000 \
  --corridor-budget-fraction 0.50 \
  --corridor-mode-cap 10 \
  --output ./corridor-coverage-plan \
  --overwrite
```

Validation is reusable for both in-memory plans and serialized plan
directories. C4 should combine the C3 slot allocation with the C2 ranked pools
when it begins corridor-first coordinate claims and global backfill.
