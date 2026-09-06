# M11 closure report

Implementation branch: `m11/tui-production-corpus-wizards-implementation`.

Approved plan: `Approved Plans/M11 TUI Production and Corpus Wizards/PLAN.md`
at `009f36042d7aad2bc976c3ea546187032cad16c6`.
Audited implementation tip after the bounded correction:
`07367a7fc56ac285768857a9440ac95e1353f276`.
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

- Full pytest: `1257 passed, 9 skipped, 0 failed`.
- `python -m ruff check .`: pass.
- `python -m ruff format --check .`: pass.
- `python -m compileall -q src scripts tests`: pass.
- `git diff --check`: pass.
- `python -m build`: pass; wheel and sdist produced.
- Clean Python 3.12 install with the exact Contract commit: `pip check` pass.
- Wheel SHA-256: `3d55be62c6304a6a3d9b501c1dd91068ba1e7588d1321aa46574a6b3ba27b293`.

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

The independent read-only auditor initially returned `FAIL` on the prior
implementation. One bounded correction pass fixed the four blocking findings;
the targeted recheck and final gates pass. No second reviewer was invoked.
The original review and correction record are in
`evidence/m11_tui_wizards/independent_review.md`. Filesystem
publication guarantees remain the implemented recoverable journal/quarantine
semantics; universal atomic replacement is not claimed. The original dirty M9
worktree was preserved separately and was not copied or modified.

Machine-readable evidence is in `evidence/m11_tui_wizards/`.
