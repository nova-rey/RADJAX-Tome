# Tome Bundle

`.rtome` and `.tgz` are portable single-file transports for a canonical
RADJAX-Tome directory. Transport wrapping is packaging only: the semantic
artifact is still the v3 `cover_page.json` identity and its manifest-described
contents inside the archive.

## Format

Canonical bundles are deterministic tar archives with their files at archive
root:

```text
cover_page.json
metadata.json
vocab_contract.json
teacher_manifest.json
emission_config.json
validation_report.json
shards/shard-00000.npz
```

The bundle does not wrap files in an outer artifact directory. New package
`tgz` output uses this same root layout. Historical `tgz` packages with an
outer directory remain compatibility inputs; they are not emitted by the
canonical writer.

## Cover Page

`cover_page.json` stays inside the bundle at archive root. Its closed
`radjax_tome_cover_v3` contract carries the profile-specific raw inventory and
references the profile- and transport-independent semantic identity. Quick
inspection reads that file directly from the archive without extracting every
member. Directory and archive validation share the v3 cover/manifest contract;
archive validation additionally proves that every member is listed and has its
recorded raw SHA-256 digest and byte size.

## Determinism

Packing normalizes tar metadata:

- sorted entry order
- `mtime=0`
- `uid=0`, `gid=0`
- empty `uname` and `gname`
- file mode `0o644`
- relative POSIX member paths only

The same input file bytes produce the same uncompressed `.rtome` bytes and the
same gzip-compressed `.tgz` bytes. Gzip is container compression only and never
changes semantic identity.

## Commands

```bash
python -m radjax_tome.cli.main pack \
  --input artifacts/fake_tome \
  --output artifacts/fake_tome.rtome \
  --overwrite

python -m radjax_tome.cli.main pack \
  --input artifacts/fake_tome \
  --output artifacts/fake_tome.tgz \
  --compression gz \
  --overwrite

python -m radjax_tome.cli.main inspect \
  --path artifacts/fake_tome.rtome

python -m radjax_tome.cli.main validate \
  --path artifacts/fake_tome.rtome

python -m radjax_tome.cli.main unpack \
  --input artifacts/fake_tome.rtome \
  --output artifacts/fake_tome_unpacked \
  --overwrite
```

## Compression Policy

The default is uncompressed stdlib tar. `--compression gz` uses deterministic
stdlib gzip wrapping. Compression is not semantic compression policy; no new
semantic compression or third-party dependency is introduced. Other container
formats remain unsupported.

## Validation Scope

RADJAX-Tome validates bundle safety, duplicate members, root `cover_page.json`,
the closed v3 cover contract, complete manifest membership, raw SHA-256 hashes,
byte sizes, deterministic tar metadata, and safe unpacking. Historical cover
forms remain native compatibility inputs and are not silently reinterpreted as
v3. RADJAX-Contract formal validation comes later.
