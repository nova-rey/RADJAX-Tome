# Legacy TeacherTextbook A/B Parity

`scripts/ab_compare_teacher_textbook.py` compares the migrated `RADJAX-Tome`
legacy TeacherTextbook builder against the archived `qrwkv-xla` builder.

The harness is migration-only. It verifies the legacy TeacherTextbook artifact
shape and does not add `cover_page.json` or emit the newer Contract-valid Tome
format.

## Usage

```bash
python scripts/ab_compare_teacher_textbook.py \
  --old-repo ../qrwkv-xla \
  --work-dir artifacts/ab_teacher_textbook \
  --overwrite
```

`--old-repo` may also be supplied with `QRWKV_XLA_OLD_REPO`. The old repo is
treated as read-only; all case outputs are written under `--work-dir`.

The harness writes:

```text
work_dir/
  cases/<case_id>/
    old/
    new/
    report.json
  ab_summary.json
  ab_summary.md
```

## Default Cases

The `fake-default` case set covers:

- dense logits with `float32` and `float16`
- built-in examples and explicit JSONL records
- single-shard and multi-shard batches
- `topk_with_tail_v0` with `top_k` values `1`, `4`, and a vocab-safe full value
- `cascaded_soft_labels_v1` with default and custom bucket edges
- compressed dtype variants for top log-probs, bucket mass, and bucket mean logp

Full old-vs-new tests are skipped in CI unless `QRWKV_XLA_OLD_REPO` points to a
local archived repo clone.

## Comparison Rules

The comparison is strict for artifact meaning:

- sidecar file presence
- directory structure
- JSON fields and values
- shard count and names
- shard array keys, shapes, dtypes, and values
- target type, sequence length, example count, vocab size, top-k metadata, and
  cascaded bucket metadata

Integer and boolean arrays must match exactly. Floating arrays are compared
exactly first; if exact equality fails, a tiny dtype-appropriate tolerance is
allowed only with a warning in the report.

## Allowed Differences

Only these known volatile migration differences are normalized:

| file | JSON pointer | reason |
| --- | --- | --- |
| `metadata.json` | `/created_at` | volatile artifact build timestamp |
| `metadata.json` | `/created_by` | expected package/module provenance change |
| `metadata.json` | `/provenance/phase` | expected migration phase provenance change |
| `teacher_manifest.json` | `/created_at` | volatile artifact build timestamp |

Every allowed difference is listed in each case report. Any other sidecar
difference is a blocker.
