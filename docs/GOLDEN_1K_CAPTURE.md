# Golden 1K Capture

M2A status: **capture_pending**. No coordinate, passport, allocation, or
payload-semantic record is committed until it is exported read-only from the
terminal T4 artifact.

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
weights, raw target shards, or selected payload bodies. The capture command
requires passing canonical production, validation, delivery, and strict linkage
reports and never modifies or reruns the source artifact.
