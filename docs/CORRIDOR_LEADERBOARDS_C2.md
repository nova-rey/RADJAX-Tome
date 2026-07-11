# C2 Offline Corridor Candidate Micro-Leaderboards

C2 builds deterministic, bounded candidate pools for each already-assigned
corridor mode. It is an offline fingerprint-corridor seam and does not change
production exemplar selection, curriculum budgets, or payload materialization.

## Input Contract

The builder consumes explicit compact candidate features. Each JSONL record has
a `features` object accepted by
`radjax_tome.fingerprint.corridor_archetypes.CorridorCandidateFeatures` and a
feature provenance declaration. Real features must identify how membership,
core distance, and useful difficulty were measured. `compatibility_proxy`
features are rejected by default and require an explicit developer override and
reason.

For `explicit` and `derived` records, the JSONL loader requires numeric
`membership_strength`, `core_distance`, `mode_support`, and `difficulty_score`
fields. `derived` records must also provide nonempty derivation metadata. Only
an explicitly declared `compatibility_proxy` record may use the C1 mapping
defaults, and it must set `compatibility_proxy_used=true`.

The C2 CLI does not infer missing features from selected exemplars or corridor
filenames. Use `--candidate-jsonl` when the source artifact does not contain
`corridors/candidate_features.jsonl` or `candidate_features.jsonl`.

## Ranking and Bounds

Each mode retains at most `candidate_pool_cap` records, defaulting to four.
Candidates are ordered by:

1. descending deterministic corridor training utility
2. descending membership score
3. descending centrality score
4. descending useful difficulty score
5. ascending `(candidate_id, position)`

Duplicate coordinates collapse only when their complete records are identical.
Conflicting duplicates, conflicting mode support, invalid assignments, and
out-of-core candidates are rejected rather than silently merged. Observed
modes with no eligible candidates remain in the artifact as coverage holes.

## Artifact and Validation

The output directory contains:

```text
manifest.json
mode_leaderboards.jsonl
validation_report.json
```

The manifest hashes the JSONL content and records the policy, provenance,
bounded counts, and explicit production-grade status. Validation rechecks
ordering, pool caps, duplicate coordinates, score bounds, count arithmetic,
hashes, and feature provenance. The artifact is deterministic for equivalent
input records regardless of input ordering.

JSONL records are yielded incrementally. Duplicate and conflicting-coordinate
detection uses a temporary disk-backed SQLite index, so working memory remains
bounded by mode state, rejection counters, and `candidate_pool_cap` pools rather
than the corpus candidate count. The index is removed after the build.

## API and CLI

Library consumers can use the direct module API:

```python
from radjax_tome.fingerprint.corridor_leaderboards import (
    CorridorLeaderboardPolicy,
    build_corridor_candidate_leaderboards,
    validate_corridor_candidate_leaderboards,
    write_corridor_candidate_leaderboards,
)
```

The equivalent offline command is:

```bash
radjax-tome build-fingerprint-corridor-leaderboards \
  --artifact ./corridor_tome \
  --candidate-jsonl ./corridor_tome/corridors/candidate_features.jsonl \
  --candidate-pool-cap 4 \
  --output ./corridor_tome/fingerprint_leaderboards \
  --overwrite
```

Compatibility proxies are a developer-only escape hatch and require an
explicit reason. C2 intentionally stops before global budgets,
cross-mode deduplication, and final corridor target construction; C3 owns the
final corridor budget.
