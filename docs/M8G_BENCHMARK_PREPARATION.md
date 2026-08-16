# M8G benchmark preparation disposition

The exact historical M8 workload has now been recovered from the documented
retained external artifact and copied to the durable local artifact store
`/Users/Cooper/Documents/radjax-tome-artifacts/m8b-authority-recovered/`.
The replay root is the 43-file `verify-checkpoint` tree; the larger anchor and
model trees are retained beside it for teacher/tokenizer replay. Raw provider
metadata remains outside Git because it contains private path-bearing fields.
The committed bundle manifest records only stable artifact identifiers,
relative member paths, and digests.

The durable machine-readable disposition is
`docs/evidence/M8G_BENCHMARK_PREPARATION_DISPOSITION.json`, validated by:

```text
PYTHONPATH=src uv run --no-sync python scripts/validate_m8g_workload_bundle.py \
  docs/evidence/M8G_BENCHMARK_WORKLOAD_BUNDLE.json \
  --bundle-root /path/to/m8b-authority-recovered
```

The validator reports `HISTORICAL_M8_WORKLOAD_RECOVERED`, 213 selected sources,
and 256 selected coordinates. It is run against the committed bundle manifest
from a fresh local checkout.

Mode integration is not complete. The canonical selected-source path is
`run_selected_delivery_rerun -> _selected_payloads_from_backend ->
_selected_payload_from_emission -> assembly`. Its CPU/GPU backends currently
allocate rectangular selected evidence before extraction. The compact-K adapter
and immutable-body transaction are not called by this path. In addition, the
current Contract compact resource is not a canonical monolithic Tome package
profile. Wiring the three requested modes therefore requires a bounded
backend-emission and Contract/package-profile design change; no benchmark or
standalone-adapter timing is claimed here.

The exact blocker is recorded in
`docs/evidence/M8G_MODE_INTEGRATION_BLOCKER.json`. Per the authorized stop
condition, no production mode flag or padded-to-compact workaround was added:
the latter would make compact measurements invalid by retaining the current
vocabulary-width allocation and serializer path.
