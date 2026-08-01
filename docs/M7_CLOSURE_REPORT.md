# M7 Corrective Implementation Record — Payload Sharding and Streaming Validation

> Historical status: the prior `f0b1e2e` closure claim was rejected by
> independent review. This document records the useful v4 foundation and the
> corrective branch work; M7 is **not** officially closed. Corrective
> implementation is in progress and independent closure review is pending.

The original M7 foundation added an additive v4 physical payload layout. The
corrective branch routes ordinary native Path B to a reported canonical v4
consumer artifact after the preserved M4 proof sequence, while retaining the
legacy tree only as historical/resume input. It does not alter M5 canonical-v3
behavior, M6 dependency direction, authority v1/v2, the fixed 25-field
selection projection, or immutable Golden fixtures.

## Released shared contract

- Published RADJAX-Contract baseline: `0.3.1`, tag `v0.3.1`, commit
  `f8ca8c0885d7c539a51d1594ba7a38c4d457b4d` (unchanged).
- Corrective RADJAX-Contract candidate: `0.3.2`, untagged commit
  `78ba36300f201d75b016b2fdcf5720e467310815`; release approval is pending.
- Tome corrective branch: `m7-corrective-production-streaming`.
- Tome's `contracts/radjax_tome/v2` is a checked-in offline mirror verified
  byte-for-byte against Contract source and installed wheel assets.

## Consumer handoff

An independent consumer uses Contract package resources only:

```python
from radjax_contract.tome import (
    tome_streaming_contract_root,
    tome_streaming_contract_asset_path,
)

root = tome_streaming_contract_root()
layout_schema = tome_streaming_contract_asset_path(
    "schemas/payload_layout_v1.json"
)
```

The v4 cover is the entry point. Validate the acyclic cover/header/inventory
graph, then stream `payload-shards.jsonl`, each shard JSONL, and
`payload-index.jsonl` in selection order. The semantic identity binds the
non-selected v3 semantic projection plus the selected sequence digest; physical
shard count and archive wrapping remain raw-integrity/transport facts.

## Corrective claims under review

- Count-only deterministic shards, stable coordinate-derived logical IDs, raw
  per-record/per-shard/index/inventory digests, and a streamed sequence digest.
- Contract's corrective reader processes canonical archives sequentially rather
  than extracting the package before iteration; it rejects unsafe members and
  transport declaration mismatch, and reports safe noncanonical metadata in
  permissive mode while strict mode rejects it.
- Student and full-debug profiles share identity while permitted provenance
  inventory differs. Legacy v3 remains historical and native; it is not
  silently promoted to v4.
- Native Path B now consumes the sealed-shard receipt transaction before v4
  materialization. Resume revalidates any completed directory/archive pair and
  rejects corrupt output; candidate directory and archive forms each pass the
  strict Contract validator before atomic promotion.
- The public v4 writer rejects Contract-invalid semantic types before its
  directory becomes visible, and archive creation uses a temporary sibling,
  validates it, then atomically replaces the final path.

## Remaining nonclaims and review obligations

M7 does not implement Student training, model inference, accelerator runs,
selected-pass batching, UX, corpus building, or an M8+ performance program.
Direct archive support is sequential streaming; selective random access is an
extracted/indexed-consumer responsibility. This record does not claim the
independent closure review. The adverse cases exercised locally include missing
semantic fields, invalid semantic types, transport mismatch, corrupt completed
resume output, interrupted shard sealing, interrupted materialization, failed
archive packing, and interruption before final promotion. The broader published
Contract corpus remains the source of portable error-category coverage.

## Reproduction

```bash
# Tome
RADJAX_CONTRACT_STREAMING_ROOT=/path/to/RADJAX-Contract/src/radjax_contract/contracts/radjax_tome/v2 \
  python3 -m pytest -q -p no:cacheprovider
python3 -m ruff check .
python3 -m ruff format --check .

# Contract
python3 -m pytest -q -p no:cacheprovider
python3 -m ruff check .
python3 -m ruff format --check .
```
