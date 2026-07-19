# Golden 1K Capture

M2A/M2B status: **capture_pending**. No coordinate, passport, allocation, or
payload-semantic record is committed until it is exported read-only from the
terminal T4 artifact using the corrected sparse-payload capture exporter. Any
earlier dense payload capture is not committable and must be recaptured.
Capture projects selected payload shards one at a time and retains only compact
scalar metadata plus versioned semantic digests of active token IDs,
probabilities, log-probabilities, and ordered active entries. It never commits
payload arrays, so comparison against an artifact does not retain the full
source payload projection in memory. Active values use the v2 canonical binary
encoding: big-endian signed int64 token IDs, big-endian IEEE-754 float64 values,
an explicit active count, and normalized signed zero (`-0.0` hashes as `+0.0`).

```bash
cd /teamspace/studios/this_studio/radjax/RADJAX-Tome
OUT=/teamspace/studios/this_studio/radjax_t4_path_b_1k/c6_3_2_native_clean
CAPTURE=/teamspace/studios/this_studio/radjax_t4_path_b_1k/golden_contract_capture

radjax-tome validate --path "$OUT"
radjax-tome audit-selected-linkage --artifact "$OUT" --strict \
  --profile full_debug_provenance --output "$OUT/selected_linkage_audit.json"
radjax-tome golden capture --artifact "$OUT" --output "$CAPTURE"
radjax-tome golden validate --fixture "$CAPTURE"
radjax-tome golden compare --fixture "$CAPTURE" --artifact "$OUT"
```

Before committing, verify 256 unique obligations, 256 payload-semantic rows,
and no corpus text, prompt text, absolute rental paths, credentials, model
weights, raw target shards, padded backend payload bodies, dense vocabulary
arrays, or active payload arrays. The capture command requires passing
canonical production, validation, delivery, and strict linkage reports and
never modifies or reruns the source artifact. Artifact locators are portable:
local storage paths are excluded from semantic board summaries and rejected
from every committed fixture surface.
