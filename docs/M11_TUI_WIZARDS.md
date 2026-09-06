# M11 TUI and corpus wizards

M11 adds a lazy optional Textual 8.2.8 interface for the two canonical
workflows (using the versioned Textual testing APIs documented at
https://textual.textualize.io/guide/testing/):
corpus-v2 construction and production Tome construction. The interface is an
editor and workflow shell; it does not define a second configuration or
execution model.

## Canonical boundaries

- `builder.config_io` owns production intent normalization, serialization, and
  reload. Complete v2 corpus references are exported as exactly
  `artifact_path`, `expected_semantic_identity`, and `max_examples`; v1
  selection authority remains the existing fixed projection.
- `corpora.config` owns corpus intent normalization, serialization, and reload.
- `io.config_export.save_as_config` is an exclusive, fsynced, no-clobber Save As
  operation.
- `corpora.feasibility` performs read-only source, JSONL, and tokenizer checks.
- `builder.status` reads journal and production progress without mutating a
  build.
- `tui.controller` and `tui.process` invoke the public canonical CLI and keep
  packaging as an explicit post-build action.
- The ordinary package import does not import Textual. The optional `tui`
  dependency is installed only for the interactive surface.

The CLI remains the owner of configuration, validation, execution, verified
corpus reading, semantic identity, resume, destination safety, and packaging.
The TUI calls those owners rather than reimplementing them.

## User-visible behavior

`radjax-tome tui corpus` and `radjax-tome tui production` open the optional
wizard. The root parser advertises `tui` and `corpus`; `--json` remains global
and must precede the command. Draft edits are saved with explicit no-clobber
Save As, preflight is read-only, execution requires confirmation, and package
publication is a separate action. Noninteractive invocation retains one JSON
document on stdout, diagnostics on stderr, stable exits, and the existing
interrupt/broken-pipe conventions.

## Validation summary

The canonical corpus and production paths were exercised from a clean Python
3.12 wheel installation pinned to Contract commit
`373e3d17060d4ce1c4a0db6065c9289da714bde7`. A v2 corpus was built, validated,
and inspected; its semantic identity was preserved after relocation. A complete
CPU production v2 build performed the score pass, selection, selected rerun,
and final Tome validation.

The 50,000-record proof used explicit `deduplication.enabled=false`, so it
exercised the bounded direct-stream path rather than claiming a DuckDB spill.
It emitted 50,000 examples in 98 shards with identity
`sha256:8f1bc7682cbb0f3bab11eb94e3279d241a23402027738dd0acca20c0e1ed828d`;
the measured child maximum RSS was 55,780 KiB and elapsed time was 8.691 s.
The DuckDB-enabled path remains covered separately by the multi-batch and
large-duplicate-group regressions.

Packaging remains a separate confirmed action. The clean CPU smoke package
passed with `full_debug_provenance` and directory transport. The `student`
profile rejected that synthetic artifact because tokenizer-binding capture was
not present; this is the inherited tokenizer limitation reported by preflight.
