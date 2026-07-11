# Fingerprint API

The `radjax_tome.fingerprint` package front door exposes the recommended
happy-path API for producer-side behavioral fingerprint artifacts.

Artifact schemas and generated behavior are unchanged. This document only
clarifies which imports are public.

## Recommended Imports

Use the package root for stable, common fingerprint workflows:

```python
from radjax_tome.fingerprint import (
    build_minimal_fingerprint_artifact_from_target_store,
    inspect_fingerprint_artifact,
    validate_fingerprint_artifact,
)
```

The package root also exposes corridor and exemplar generation helpers used by
the public proof path.

## Recommended Workflow

```python
from radjax_tome.fingerprint import (
    build_minimal_fingerprint_artifact_from_target_store,
    inspect_fingerprint_artifact,
    validate_fingerprint_artifact,
)

artifact = build_minimal_fingerprint_artifact_from_target_store(...)
validation = validate_fingerprint_artifact(artifact)
summary = inspect_fingerprint_artifact(artifact)
```

## Advanced Module Imports

Deeper modules remain available for advanced use:

```python
from radjax_tome.fingerprint.artifacts import FingerprintManifest
from radjax_tome.fingerprint.corridor import CorridorMeasurementReport
from radjax_tome.fingerprint.exemplars import FingerprintExemplarRecord
from radjax_tome.fingerprint.generation import generate_exemplar_reservoir
from radjax_tome.fingerprint.loader import load_fingerprint_artifact
```

Use these explicit module paths when you need schema dataclasses, low-level
read/write helpers, validation internals, provenance helpers, or loader details.

## C1 Corridor Archetype Primitive

The C1 pure scoring seam is available from the existing fingerprint package:

```python
from radjax_tome.fingerprint.corridor_archetypes import (
    CorridorArchetypePolicy,
    CorridorCandidateFeatures,
    score_corridor_archetype_candidate,
)
```

It performs corridor-core eligibility before difficulty scoring and returns a
bounded deterministic utility only for eligible candidates. It does not build
micro-leaderboards or change production selection; those begin in C2.

## C2 Offline Corridor Micro-Leaderboards

C2 builds deterministic bounded candidate pools from explicit compact feature
records:

```python
from radjax_tome.fingerprint.corridor_leaderboards import (
    CorridorLeaderboardPolicy,
    build_corridor_candidate_leaderboards,
    validate_corridor_candidate_leaderboards,
    write_corridor_candidate_leaderboards,
)
```

The default pool cap is four candidates per observed corridor mode. Production
grade builds reject compatibility-proxy feature provenance; developer fixtures
must opt in with an explicit override reason. C2 is offline and does not alter
production selection or payloads. See `docs/CORRIDOR_LEADERBOARDS_C2.md` for
the input, ranking, artifact, validation, and CLI contracts.

## C3 Bounded Corridor Coverage

C3 consumes a validated C2 `CorridorLeaderboardArtifact` and allocates bounded
representative slots without selecting coordinates:

```python
from radjax_tome.fingerprint.corridor_budget import (
    CorridorBudgetPolicy,
    allocate_corridor_coverage,
    write_corridor_coverage_plan,
)
```

The allocator uses Decimal floor arithmetic, breadth-first round-robin filling,
and deterministic top-candidate priority only for first-round oversubscription.
See `docs/CORRIDOR_BUDGET_C3.md` for the plan schema and CLI.

## C4 Corridor-First Claims

C4 consumes validated C2 leaderboards, a validated C3 coverage plan, and an
explicit ranked global-board supply. It claims corridor coordinates first and
records global collisions and backfill lineage without producing payloads:

```python
from radjax_tome.fingerprint.corridor_claims import (
    CorridorGlobalClaimPolicy,
    claim_corridor_then_backfill_global,
    validate_corridor_global_claim_result,
    write_corridor_global_claim_result,
)
```

`load_global_board_input` accepts the stable
`radjax.c4_global_board_supply.v1` input contract. Existing selector manifests
are accepted only as explicitly non-production development adapters. C4 is
offline and does not alter production selection; the claim obligations are
reserved for the later multi-role curriculum schema.

## What Is Not Public API

The package root does not advertise every constant, dataclass, record type, or
private helper. Names that start with `_`, migration-only helpers, test-only
helpers, validator internals, and raw record details should not be imported from
`radjax_tome.fingerprint`.

Import advanced names from the module that defines them instead of relying on a
broad package-level re-export.
