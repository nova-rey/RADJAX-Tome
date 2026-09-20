# RADJAX-Tome CLI Guide

## Recommended CLI

The installed mainline advertises `corpus`, `build`, `validate`, `inspect`,
`package`, `doctor`, `research`, and optional `tui`. Build consumes a
complete canonical M5 intent; it does not accept the historical flag bag.

```bash
radjax-tome corpus build --config ./corpus-intent.json
radjax-tome corpus validate ./corpus-artifact
radjax-tome corpus inspect ./corpus-artifact

radjax-tome build --config ./tome-intent.json --preflight-only
radjax-tome validate ./producer-workspace
radjax-tome inspect ./producer-workspace
radjax-tome package ./producer-workspace \
  --output ./student.tgz --profile student --transport tgz
radjax-tome doctor --config ./tome-intent.json
```

Machine-readable output is selected before the command:

```bash
radjax-tome --json validate ./producer-workspace
```

`--output PATH`, `--resume`, and `--overwrite` are narrow operational
overrides applied by the canonical M5/M4 lifecycle. They do not introduce a
second destination model. Preflight validates ownership and destination
identity before any mutation. `package` is a separately confirmed
operation; it does not implicitly package a build.

`validate` and `inspect` route through the canonical artifact dispatcher and
accept supported Contract, producer-workspace, and unpacked package directories.
The current mainline does not validate a .tgz path directly; extract an archive
and validate its unpacked directory. `--json`
results use the stable `radjax_tome_cli_result_v1` envelope; human output is
intended for terminals. Errors name a phase, code, and repair when available.

### Command safety

| Command | Reads | Writes/side effects | Expensive work |
| --- | --- | --- | --- |
| `doctor` | config and local capability metadata | optional report only | no model execution |
| `corpus build` | declared local sources | corpus artifact and journal | CPU corpus work |
| `corpus validate/inspect` | corpus artifact | no mutation | bounded verification |
| `build --preflight-only` | intent, corpus, provenance | no output mutation | no model/backend allocation |
| `build` | verified corpus and teacher | producer workspace | teacher/selection work |
| `validate/inspect` | artifact/package | optional report only | artifact verification |
| `package` | validated producer workspace | new package only | archive/copy work |
| `tui` | config or wizard inputs | Save As only after confirmation | same canonical commands |

Resume is only at the durable lifecycle boundaries recorded by the producer;
it is not a promise of arbitrary mid-shard recovery. Overwrite requires an
explicit flag and positive ownership. Cancellation leaves a recoverable or
quarantined state rather than silently replacing an unrelated destination.

## Retained script classification

| Script | Classification | Use when |
| --- | --- | --- |
| mainline CLI | recommended wrapper / legacy-compatible | normal production and corpus work |
| research commands | advanced diagnostic | reproducing accepted experiments |
| internal helpers | internal/development | contributor tests only |
| archived reports | archive-only | historical evidence and provenance |

## Research and compatibility

Legacy and research commands remain available for reproducibility but are not
the normal happy path. Use `radjax-tome research --help` or the existing
scripts when reproducing an archived result. The human-readable status map is
`docs/RESEARCH_STATUS_MAP.md`; its source of truth is
`docs/hydra_disposition.json`.

The historical names below are retained as compatibility/research interfaces:
`build-fingerprint-corridor-leaderboards`,
`allocate-fingerprint-corridor-coverage`,
`claim-corridor-and-backfill-global`,
`build-multi-role-selected-exemplars`, `pack`, and `unpack`. They are not
a second production pipeline and should not be copied into new user configs.

## Corpus workflow (current grammar)

```bash
radjax-tome corpus build --config ./corpus-intent.json
radjax-tome corpus inspect ./corpus-artifact
radjax-tome corpus validate ./corpus-artifact
radjax-tome build --config ./tome-intent-v2.json --preflight-only
```

The corpus builder is local-only. It writes a verified corpus-v2 artifact and
semantic identity that a production intent references. See
`docs/CORPUS_BUILDER.md` and `docs/CONFIGURATION_REFERENCE.md`.

## Optional TUI

Install the optional extra only when a terminal wizard is useful:

```bash
python -m pip install './dist/radjax_tome-*.whl[tui]'
radjax-tome tui corpus --config ./corpus-intent.json
radjax-tome tui production --config ./tome-intent-v2.json
```

The TUI is a controller over the same canonical loader, preflight, state
machine, validators, and package lifecycle. Save As is no-clobber and requires
confirmation. On a non-TTY or a terminal below the supported size, use the
headless commands; the TUI does not provide a separate semantics system.


## Supporting utilities and historical references

The supported parity utility remains documented at docs/PARITY_HARNESS.md and
invoked as `radjax-tome parity` when comparing already-built artifacts. Teacher
model provenance is prepared with `radjax-tome model inspect` and the legacy
flag `--teacher-model-provenance` is retained only in the compatibility
interfaces. Fingerprint API details remain at docs/FINGERPRINT_API.md; they are
not a second public build configuration.
