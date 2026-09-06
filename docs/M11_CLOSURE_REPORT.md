# M11 closure report

Implementation branch: `m11/tui-production-corpus-wizards-implementation`.

Audited implementation tip: `20cbbdeb732f4717d668791c7690967f6ae10690`.
Contract: `373e3d17060d4ce1c4a0db6065c9289da714bde7`.
M10 accepted base: `6b1e3c8563af748b872f68b8f4efd604fcbd4288`.

## Delivered

M11 provides canonical serializable production and corpus intents, strict v2
corpus reference projection, no-clobber Save As, read-only feasibility and
status boundaries, a lazy optional Textual corpus/production wizard, and public
CLI routing. The wizard invokes the canonical CLI and leaves package publication
as a separate confirmed action. v1 configuration and the fixed v1 selection
authority projection remain unchanged.

The v2 production path resolves one verified corpus input and carries it through
planning, streaming, teacher validation, score/selection, selected rerun,
resume/status reporting, and final validation. Semantic identity and tokenizer
binding are checked before model/backend creation. Artifact validation remains
builder-independent.

## Gates

- Full pytest: `1254 passed, 9 skipped, 0 failed`.
- `python -m ruff check .`: pass.
- `python -m ruff format --check .`: pass.
- `python -m compileall -q src scripts tests`: pass.
- `git diff --check`: pass.
- `python -m build`: pass; wheel and sdist produced.
- Clean Python 3.12 install with the exact Contract commit: `pip check` pass.

## Workflow evidence

The clean-install public corpus flow passed build, validate, and inspect. A
relocated copy preserved the corpus semantic identity. Resume, interruption
boundaries, corruption rejection, and deterministic rebuild regressions pass.
The complete CPU production v2 smoke passed score pass, selection, selected
rerun, and final Tome validation.

The 50K synthetic corpus proof passed with 50,000 examples and 98 shards. The
direct bounded path measured 55,780 KiB child maximum RSS over 8.691 seconds;
the run used explicit deduplication disabled and therefore makes no DuckDB
spill claim. Enabled deduplication was exercised by cross-batch, collision, and
large duplicate-group tests.

## Review and limitations

No independent review was performed during this run. This branch is ready for
one external audit, not for a claim of independent acceptance. Filesystem
publication guarantees remain the implemented recoverable journal/quarantine
semantics; universal atomic replacement is not claimed. The original dirty M9
worktree was preserved separately and was not copied or modified.

Machine-readable evidence is in `evidence/m11_tui_wizards/`.
