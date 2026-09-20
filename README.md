# RADJAX-Tome

RADJAX-Tome produces teacher-side distillation artifacts: corpus artifacts,
Tomes, target stores, and selected-exemplar packages. It emits portable
artifacts validated by RADJAX-Contract. It does not train Student models.

## Recommended CLI

## Start here

The current reader journey is [Start here](docs/START_HERE.md). It separates
the small offline CPU demonstration from the real teacher-backed production
path and links the complete configuration and artifact references.

Install from a built wheel (the repository is the distribution authority):

```bash
python -m pip install ./dist/radjax_tome-*.whl
radjax-tome --version
radjax-tome doctor
```

The supported command family is `corpus`, `build`, `validate`, `inspect`,
`package`, `doctor`, `research`, and optional `tui`. Build consumes a
complete canonical M5 intent; see the [v1 example](docs/examples/m9_tome_build_intent.yaml)
and the [configuration reference](docs/CONFIGURATION_REFERENCE.md).

```bash
python -m radjax_tome.cli.main build --config ./tome-intent.yaml --preflight-only
radjax-tome corpus build --config ./corpus-intent.json
radjax-tome corpus validate ./corpus-artifact
radjax-tome build --config ./tome-intent.json --preflight-only
radjax-tome validate ./producer-workspace
radjax-tome inspect ./producer-workspace
```

For the full walkthrough, package profiles, TUI, and research boundaries see
[CLI guide](docs/CLI_GUIDE.md), [artifacts and packages](docs/ARTIFACTS_AND_PACKAGES.md),
and [research status map](docs/RESEARCH_STATUS_MAP.md). For advanced/dev
scripts, see `docs/CLI_GUIDE.md`.

`--json` is a global option and must precede the command, for example
`radjax-tome --json validate ./producer-workspace`.

The canonical production consumer semantics are versioned by
RADJAX-Contract; Tome supplies teacher-side evidence and Student consumes a
validated package. See [the artifact boundary](docs/ARTIFACTS_AND_PACKAGES.md).

The historical fake/offline smoke remains useful for compatibility fixtures,
but is not evidence of real teacher inference:

```bash
python scripts/build_teacher_textbook.py \
  --output artifacts/fake_teacher_textbook \
  --teacher-mode fake \
  --max-examples 2 \
  --sequence-length 8 \
  --vocab-size 16 \
  --overwrite
```

PyTorch and Transformers are optional `teacher-hf` extras. They are not
required for default install or tests. Historical migration and audit
artifacts remain in the archive; see `docs/TOME_ARCHIVE_POINTERS.md`.

For advanced/dev scripts, see `docs/CLI_GUIDE.md`.
