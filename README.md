# RADJAX-Tome

RADJAX-Tome produces teacher-side distillation artifacts: TeacherTomes, target
stores, behavioral fingerprint artifacts, exemplar reservoirs, and split
manifests.

It emits portable artifacts validated by RADJAX-Contract. It does not train
student models.

RADJAX-Tome now owns the migrated legacy Tome Builder / TeacherTextbook builder
from the historical `qrwkv-xla` repo. The migrated builder preserves the legacy
TeacherTextbook layout for now:

```text
metadata.json
vocab_contract.json
teacher_manifest.json
emission_config.json
validation_report.json
shards/shard-00000.npz
```

The new `cover_page.json` Tome contract is intentionally deferred to a later
phase.

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
