# RADJAX-Tome

RADJAX-Tome produces teacher-side distillation artifacts: TeacherTomes, target
stores, behavioral fingerprint artifacts, exemplar reservoirs, and split
manifests.

It emits portable artifacts validated by RADJAX-Contract. It does not train
student models.

## Recommended CLI

Start with the public CLI:

```bash
python -m radjax_tome.cli.main build \
  --output artifacts/cli_happy_path_fake_tome \
  --teacher-mode fake \
  --max-examples 2 \
  --sequence-length 8 \
  --overwrite

python -m radjax_tome.cli.main validate \
  --path artifacts/cli_happy_path_fake_tome

python -m radjax_tome.cli.main inspect \
  --path artifacts/cli_happy_path_fake_tome
```

Installed console entry point:

```bash
radjax-tome build --output artifacts/cli_happy_path_fake_tome --teacher-mode fake --max-examples 2 --sequence-length 8 --overwrite
radjax-tome validate --path artifacts/cli_happy_path_fake_tome
radjax-tome inspect --path artifacts/cli_happy_path_fake_tome
```

For advanced/dev scripts, see `docs/CLI_GUIDE.md`.

RADJAX-Tome now owns the migrated legacy Tome Builder / TeacherTextbook builder
from the historical `qrwkv-xla` repo. The migrated builder preserves existing
TeacherTextbook sidecars and now adds an unpacked Tome cover page:

```text
cover_page.json
metadata.json
vocab_contract.json
teacher_manifest.json
emission_config.json
validation_report.json
shards/shard-00000.npz
```

`cover_page.json` is the unpacked Tome front door added in Spec 3.1. See
`docs/TOME_COVER_PAGE.md`.

The canonical production consumer semantics are versioned by RADJAX-Contract in
the [Tome/Student consumer handoff](https://github.com/nova-rey/RADJAX-Contract/blob/main/docs/reference/RADJAX_TOME_STUDENT_CONSUMER_HANDOFF.md).

Portable `.rtome` bundles are deterministic tar archives added in Spec 3.2. See
`docs/TOME_BUNDLE.md`.

Fake/offline smoke:

```bash
python scripts/build_teacher_textbook.py \
  --output artifacts/fake_teacher_textbook \
  --teacher-mode fake \
  --max-examples 2 \
  --sequence-length 8 \
  --vocab-size 16 \
  --overwrite

python scripts/validate_teacher_textbook.py \
  --path artifacts/fake_teacher_textbook \
  --write-report
```

PyTorch and Transformers are optional `teacher-hf` extras. They are not required
for default install or tests.

Historical migration/audit artifacts are archived on:
- `archive/tome-migration-audit`
- `archive/tome-large-docs`

See `docs/TOME_ARCHIVE_POINTERS.md`.
