# M7 Closure Report — Payload Sharding and Streaming Validation

M7 adds an additive v4 physical payload layout without altering native M4
delivery, M5 canonical-v3 behavior, M6 dependency direction, authority v1/v2,
the fixed 25-field selection projection, or immutable Golden fixtures.

## Released shared contract

- RADJAX-Contract: `0.3.1`, tag `v0.3.1`, commit
  `f8ca8c0885d7c539a51d1594ba7a38c4d457b4d`.
- Tome branch: `m7-payload-sharding-streaming`, closure commit `a859a0c`.
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

## Claims

- Count-only deterministic shards, stable coordinate-derived logical IDs, raw
  per-record/per-shard/index/inventory digests, and a streamed sequence digest.
- Safe archive validation spools sequential streams to temporary disk, rejects
  traversal, duplicates, links/special members, corruption, and malformed
  package references without loading a payload collection into memory.
- Student and full-debug profiles share identity while permitted provenance
  inventory differs. Legacy v3 remains historical and native; it is not
  silently promoted to v4.

## Nonclaims and deferrals

M7 does not implement Student training, model inference, accelerator runs,
selected-pass batching, UX, corpus building, random archive seeks, or an M8+
performance program. Direct archive support is sequential streaming; selective
random access is an extracted/indexed-consumer responsibility.

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
