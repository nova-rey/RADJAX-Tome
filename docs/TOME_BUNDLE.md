# Tome Bundle

`.rtome` is the portable single-file form of an unpacked RADJAX-Tome
directory. Bundle v1 is packaging only: the semantic artifact is still the
`cover_page.json`-described Tome contents inside the archive.

## Format

RTOME v1 is a deterministic tar archive with the unpacked Tome files at archive
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

The bundle does not wrap files in an outer artifact directory.

## Cover Page

`cover_page.json` stays inside the bundle at archive root. Quick inspection
reads that file directly from the tar archive without extracting every member.
Spec 3.2 does not mutate the cover page while packing; its `layout` continues to
describe the inner unpacked layout.

## Determinism

Packing normalizes tar metadata:

- sorted entry order
- `mtime=0`
- `uid=0`, `gid=0`
- empty `uname` and `gname`
- file mode `0o644`
- relative POSIX member paths only

The same input file bytes produce the same uncompressed `.rtome` bytes.

## Commands

```bash
python -m radjax_tome.cli.main pack \
  --input artifacts/fake_tome \
  --output artifacts/fake_tome.rtome \
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

Default bundle v1 uses uncompressed stdlib tar. Compression is not semantic
compression policy, and no compression is required. Optional stdlib compression
may be added later; zstd is deferred because Spec 3.2 adds no new dependency.

## Validation Scope

RADJAX-Tome validates bundle safety, duplicate members, root `cover_page.json`,
cover-page-listed contents, SHA-256 hashes, byte sizes, and safe unpacking.
RADJAX-Contract formal validation comes later.
