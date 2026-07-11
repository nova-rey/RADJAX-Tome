# C5 Multi-Role Selected Exemplars

C5 is the durable offline projection of a validated C4 claim artifact. It
does not rerun selection, emit teacher payloads, route curriculum boards, or
change `production-build`.

## Invariant

Each C4 coordinate becomes exactly one rich C5 record and exactly one future
payload identity. A record can retain multiple selection obligations, but
roles never duplicate the coordinate or promise multiple payloads.

The C4 `primary_claim` is ownership, not exclusive credit. Secondary corridor
or global-board obligations remain in `selection_obligations` and are never
discarded because another obligation is primary.

## Artifact

```text
multi-role-selection/
  manifest.json
  selected_exemplars.jsonl
  legacy_selected_exemplars.json
  validation_report.json
```

`selected_exemplars.jsonl` uses schema
`radjax.multi_role_selected_exemplar.v1`. Records contain the canonical
coordinate, C4 selection index, primary claim, ordered obligations, derived
corridor/global source lists, a source passport, and a coordinate-derived
`payload_identity` with status `not_materialized_in_c5`.

Obligations are ordered as the primary corridor obligation, remaining corridor
obligations by mode/rank, then global obligations by board priority, board ID,
and rank. The current C4 contract permits at most one corridor obligation;
C5 rejects more than one.

## API and CLI

```python
from radjax_tome.fingerprint.multi_role_selection import (
    build_multi_role_selected_exemplars,
    project_legacy_selected_exemplars,
    validate_multi_role_selection_artifact,
    write_multi_role_selection_artifact,
)
```

Build records from the public C4 loader with:

```bash
radjax-tome build-multi-role-selected-exemplars \
  --claims ./c4_claims \
  --output ./multi-role-selection \
  --overwrite
```

An optional source-passport JSON document may contain a `passports` list with
`example_id` and `position` identity fields. Supplying that index enables
strict passport verification. Non-production C4 sources require the explicit
`--allow-nonproduction-sources` override and a reason.

The artifact stores the canonical C4 result digest and inherited C2/C3/global
provenance. Validation checks hashes, record order, obligations, derived
fields, passports, payload keys, summary arithmetic, and flat-projection
parity. C6 owns coverage reports, audits, packaging, and eventual production
integration.
