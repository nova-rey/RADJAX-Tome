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

Builds now write an unpacked Tome `cover_page.json`; `validate` checks it when
present, and `inspect` prints its summary fields. See `docs/TOME_COVER_PAGE.md`.

Use `pack` and `unpack` for deterministic `.rtome` bundle v1 archives. Bundle
validation and inspection work through the same `validate --path` and
`inspect --path` commands. See `docs/TOME_BUNDLE.md`.

Build deterministic local corpus artifacts before Tome generation:

```bash
radjax-tome corpus build \
  --input ./sources \
  --output ./corpus_out \
  --include "**/*.md" \
  --include "**/*.txt" \
  --overwrite

radjax-tome corpus inspect --path ./corpus_out
radjax-tome corpus validate --path ./corpus_out

radjax-tome build \
  --dataset ./corpus_out/corpus.jsonl \
  --corpus-manifest ./corpus_out/corpus_manifest.json \
  --output artifacts/from_corpus \
  --teacher-mode fake \
  --overwrite
```

The corpus builder is local-only. It writes `corpus_hash` and
`manifest_hash` provenance that generated Tomes can cite. See
`docs/CORPUS_BUILDER.md`.

Corpus source formats are intentionally narrow: `.txt`, `.md`, `.markdown`,
`.py`, and `.jsonl` rows with `text`. Structured `.json` import is not
supported yet.

Inspect and validate local teacher model provenance before Tome generation:

```bash
radjax-tome model inspect \
  --model-path ./local_teacher \
  --output ./teacher_model_provenance.json

radjax-tome model validate \
  --provenance ./teacher_model_provenance.json

radjax-tome build \
  --teacher-model ./local_teacher \
  --teacher-model-provenance ./teacher_model_provenance.json \
  --output artifacts/from_teacher_model_provenance \
  --teacher-mode fake \
  --overwrite
```

`model inspect` is local-only and does not download teacher models. It records
verified file hashes, inferred or declared friendly identity, and
`network_used=false`. See `docs/TEACHER_MODEL_PROVENANCE.md`.

Compare two generated Tome artifact directories after they exist:

```bash
radjax-tome parity \
  --left ./artifact_cpu \
  --right ./artifact_gpu \
  --left-label cpu_reference \
  --right-label gpu_torch \
  --output ./parity_report.json
```

`parity` writes `tome_parity_report_v1` and checks sidecars, target-store
metadata, shard arrays, finite values, numeric tolerances, selector manifests,
corpus provenance, teacher model provenance, and metadata truth. See
`docs/PARITY_HARNESS.md`.

For runtime/backend preflight and artifact metadata sanity checks:

```bash
radjax-tome doctor

radjax-tome doctor \
  --teacher-backend gpu_torch \
  --runtime-mode cpu_gpu \
  --target-policy corridor_exemplar_v1

radjax-tome inspect \
  --path artifacts/backend_tome \
  --metadata-sanity

radjax-tome validate \
  --path artifacts/backend_tome \
  --metadata-sanity \
  --write-report
```

`doctor` writes a `runtime_doctor_report_v1` preflight summary when
`--write-report PATH` is provided. Metadata sanity writes
`metadata_sanity_report.json` during `validate --metadata-sanity
--write-report`. These commands report backend availability, remediation
hints, selector metadata sanity, and batch-size metadata sanity; they do not
add reducer math, selector policy, real auto batch probing, production global
selection, multidevice scheduling, or TPU/JAX support.

For GPU teacher setup on a fresh machine, install the GPU/HF optional extra and
run doctor before building:

```bash
pip install -e ".[gpu-teacher]"

radjax-tome doctor \
  --teacher-backend gpu_torch \
  --runtime-mode cpu_gpu \
  --target-policy corridor_exemplar_v1 \
  --write-report runtime_doctor_report.json
```

`gpu-teacher` currently aliases the `torch` and `transformers` dependencies
used by `teacher-hf`. PyTorch CUDA wheels are platform-specific; follow
PyTorch's install selector if the default wheel does not expose CUDA. See
`docs/GPU_INSTALL.md`.

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

For fingerprint API imports, see `docs/FINGERPRINT_API.md`.

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
