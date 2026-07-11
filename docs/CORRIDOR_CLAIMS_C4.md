# C4 Corridor-First Claims

C4 is the offline coordinate-claim stage after C2 corridor candidate
leaderboards and the C3 coverage budget. It claims corridor representatives
first, then fills the remaining global budget from ranked board supply. A
coordinate is selected once, but every collision remains an explicit
obligation so later curriculum stages can preserve both reasons for selection.

C4 does not run teacher inference, materialize payloads, or modify production
selection. Its output is a compact claim artifact for a later stage:

```text
claim_manifest.json
corridor_claims.jsonl
global_claims.jsonl
collision_obligations.jsonl
backfill_lineage.jsonl
selected_coordinates.jsonl
validation_report.json
```

## CLI

```bash
radjax-tome claim-corridor-and-backfill-global \\
  --leaderboards PATH_TO_C2 \\
  --coverage-plan PATH_TO_C3 \\
  --global-leaderboards PATH_TO_GLOBAL_SUPPLY \\
  --output PATH_TO_C4 \\
  --overwrite
```

The default requires the full C3 budget to be fulfilled. Use
`--allow-underfill` only when the ranked supply is intentionally incomplete.
Non-production source artifacts are rejected by default. To use one, pass
`--allow-nonproduction-sources` together with
`--nonproduction-override-reason REASON`.

## Global supply contract

The preferred global input is a JSON document with schema
`radjax.c4_global_board_supply.v1`:

```json
{
  "schema_version": "radjax.c4_global_board_supply.v1",
  "source_provenance": {
    "source_artifact_id": "global-scoreboards",
    "production_grade": true
  },
  "boards": [
    {
      "board_id": "global_max_entropy",
      "priority": 0,
      "requested_slots": 128,
      "candidates": [
        {
          "example_id": "corpus_000001",
          "position": 17,
          "rank": 1,
          "score": 8.4,
          "eligible": true
        }
      ]
    }
  ]
}
```

Boards are processed by `(priority, board_id)`. Candidate ranks are retained
in claims and backfill lineage. An existing
`exemplar_selection_manifest_v1` can also be read as a development adapter;
it is explicitly marked non-production and therefore requires the override.

## Guarantees

- C2 and C3 provenance, hashes, and source coordinates are validated before
  claiming.
- Corridor obligations are processed in ascending corridor mode order.
- Global collisions never erase corridor or earlier global obligations.
- Backfill records the skipped rank, reason, and replacement coordinate.
- Claim artifacts contain no logits, probabilities, payload references, text,
  or input arrays.
- Artifact writes are atomic and include content hashes.

The selected coordinate list is deliberately not a training payload schema.
C5 owns durable multi-role training records; C4 only establishes deterministic
coordinate claims and their lineage.
