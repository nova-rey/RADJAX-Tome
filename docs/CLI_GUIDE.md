# RADJAX-Tome CLI Guide

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

If installed from the package, the console entry point is equivalent:

```bash
radjax-tome build --output artifacts/cli_happy_path_fake_tome --teacher-mode fake --max-examples 2 --sequence-length 8 --overwrite
radjax-tome validate --path artifacts/cli_happy_path_fake_tome
radjax-tome inspect --path artifacts/cli_happy_path_fake_tome
```

Fake mode is CPU-only, offline, and does not require PyTorch, Transformers, JAX,
or network access.

For advanced diagnostics:

```bash
python -m radjax_tome.cli.main prove-capabilities \
  --work-dir artifacts/cli_capabilities \
  --overwrite
```

`prove-capabilities` is a public diagnostic command backed by reusable library
code under `radjax_tome.capabilities`. The legacy-compatible
`scripts/prove_tome_generation_capabilities.py` wrapper remains available for
existing automation.

`scripts/` contains lower-level utilities used by tests, development, and
targeted inspection workflows. They remain available, but they are not all
equally user-facing.

| Script | Classification | Use when |
|---|---|---|
| `scripts/build_teacher_textbook.py` | recommended wrapper / legacy-compatible | You need the current TeacherTextbook builder directly. |
| `scripts/build_teacher_tome.py` | recommended wrapper / legacy-compatible | You need the older toy module CLI directly. |
| `scripts/validate_teacher_textbook.py` | recommended wrapper / legacy-compatible | You need direct TeacherTextbook validation. |
| `scripts/inspect_targets.py` | recommended wrapper / legacy-compatible | You need direct target-store inspection. |
| `scripts/prove_tome_generation_capabilities.py` | advanced diagnostic | You are validating the repo capability surface. |
| `scripts/export_teacher_targets.py` | advanced | You need raw synthetic target-store export. |
| `scripts/validate_fingerprint_artifact.py` | advanced | You need direct fingerprint artifact validation. |
| `scripts/inspect_fingerprint_artifact.py` | advanced | You need direct fingerprint artifact inspection. |
| `scripts/inspect_prompt_corpus.py` | advanced | You need prompt corpus inspection. |
| `scripts/resolve_qwen_policy.py` | advanced | You need to inspect a Qwen policy resolution. |
| `scripts/split_prompt_corpus.py` | advanced | You need to split prompt corpora outside the happy path. |
| `scripts/tokenize_corpus.py` | advanced | You need standalone corpus tokenization. |
| `scripts/validate_producer_pipeline.py` | internal/development | You are checking producer pipeline development state. |
| `scripts/ab_compare_teacher_textbook.py` | internal/development | You are doing builder parity or regression work. |
| `scripts/audit_tome_refactor_surface.py` | internal/development | You are working on cleanup/refactor audits. |

Historical migration and quarantine audit scripts are archive-only on:

- `archive/tome-migration-audit`
- `archive/tome-large-docs`

See `docs/TOME_ARCHIVE_POINTERS.md` for archive inspection commands.
