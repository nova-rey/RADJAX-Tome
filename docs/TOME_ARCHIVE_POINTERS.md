# RADJAX-Tome Archive Pointers

RADJAX-Tome keeps active source, tests, and compact project docs on `main`.
Historical migration evidence is preserved on archive branches so `main` does
not carry the full split excavation forever.

## Archive Branches

- `archive/tome-migration-audit` preserves quarantine references, extraction
  audits, closure audits, quarantine surgery ledgers, migration maps, and old
  proof artifacts from the `qrwkv-xla` split.
- `archive/tome-large-docs` preserves oversized generated JSON, full forensic
  reports, large manifests, and bulky historical documentation.

Active development continues on `main`.

## Mainline Policy

`quarantine/` was historical evidence, not runtime code. It preserved old
`qrwkv_xla` source references while each path was classified as promoted,
split-promoted, kept quarantined, Student-bound, Contract-bound, deprecated,
deferred, or waived.

The giant forensic docs and JSON ledgers were archived rather than deleted from
history. `main` now keeps compact pointers and the hygiene ledger at
`docs/TOME_MAINLINE_HYGIENE_LEDGER.json`.

## Inspecting Archives

```bash
git fetch origin archive/tome-migration-audit:archive/tome-migration-audit
git fetch origin archive/tome-large-docs:archive/tome-large-docs

git switch archive/tome-migration-audit
git switch archive/tome-large-docs
git switch main
```
