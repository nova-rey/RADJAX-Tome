# RADJAX-Tome

RADJAX-Tome produces teacher-side distillation artifacts: TeacherTomes, target
stores, behavioral fingerprint artifacts, exemplar reservoirs, and split
manifests.

It emits portable artifacts validated by RADJAX-Contract. It does not train
student models.

## Recommended CLI

Start with the six-command public CLI. `build` consumes a complete canonical
M5 intent; see [the example](docs/examples/m9_tome_build_intent.yaml).

```bash
python -m radjax_tome.cli.main build --config docs/examples/m9_tome_build_intent.yaml --preflight-only
radjax-tome build --config docs/examples/m9_tome_build_intent.yaml --preflight-only

radjax-tome validate ./OUTPUT.v4.tgz

radjax-tome inspect ./OUTPUT.v4.tgz
```

Installed console entry point:

```bash
radjax-tome --help
radjax-tome doctor
radjax-tome research --help
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
