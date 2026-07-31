# M6 Closure Report — Tome Contract Publication and Conformance

## Published contract

The portable `radjax_tome_artifact_contract` publication version is `1.0.0`.
Its authority is RADJAX-Contract `0.2.0`, tag `v0.2.0`, commit
`147ca371c78a98dbc82de1ea93deb4f3ae27f399`. The release pin is recorded in
`pyproject.toml`; the contract `SHA256SUMS` digest is
`7cb2657c647bc0bbb696f6ac225a2c4b4a7d54fbe77b2ede7d0b41bfc652015c`.

Tome retains the same tree as a checked-in offline mirror. Its checksum and
source-equivalence test prevent independent drift. Contract source, Contract
wheel, and Tome mirror were verified byte-identical for all 12 contract assets.

## Claims proven

- v3 cover, identity, content manifest, profiles, raw integrity, and
  canonicalization recipes are published as implementation-neutral assets.
- Canonical writers are byte-deterministic for directory/`.rtome`/gzip
  transport. Consumer safety rejects unsafe or corrupt transport; safe
  noncanonical metadata is reported by default and rejected in strict mode.
- Native and portable validators agree for canonical outputs, profile identity,
  inventory integrity, and adversarial safety cases, except the documented
  producer-canonicality versus consumer-warning distinction.
- v2 and package-v1 remain native-first, incomplete compatibility descriptors.
- M5 identity/authority semantics, fixed 25-field selection projection,
  authority v1/v2, Path-B ordering, and frozen Golden fixtures remain intact.

## Downstream verification

```bash
python -m pip install 'radjax-contract @ git+https://github.com/nova-rey/RADJAX-Contract.git@v0.2.0'
python -c 'from radjax_contract.tome import tome_contract_root; print(tome_contract_root())'
python tools/validate_radjax_tome_contract.py /path/to/tome
python tools/validate_radjax_tome_contract.py --strict-canonicality /path/to/tome.rtome
```

The Contract release assets, integration guide, digest vectors, catalog, and
portable validator define the artifact boundary. They do not implement a
Student runtime or authorize model inference.

## Intentional deferrals

No RADJAX-Student code, consumer runtime, model inference, accelerator work,
Golden-fixture regeneration, repository merge, or release of either `main`
branch occurred. The historical mounted July artifact comparison remains an
external conditional test when those artifacts are available.
