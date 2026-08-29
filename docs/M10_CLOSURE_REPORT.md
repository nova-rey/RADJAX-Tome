# M10 closure report

Audited implementation commit: `2190756`

M10 adds the strict local corpus builder v2, deterministic normalization and
exact byte-defended deduplication, canonical shards and offset indexes,
verified readers, path-independent semantic identity, and restart-safe staged
publication. Historical v1 corpus behavior and M9 CLI behavior remain
covered.

Validation performed:

- full pytest: 1,225 passed, 9 skipped, after Hydra inventory reconciliation;
- `ruff check .`: pass;
- `ruff format --check .`: pass;
- `compileall`, `git diff --check`, and wheel build: pass;
- clean Python 3.12 wheel install and `pip check`: pass;
- clean-install `corpus build`, `corpus validate`, and `corpus inspect`: pass;
- focused M10 tests: 12 passed;
- deterministic rebuild: semantic identity and shard/index bytes matched;
- operational `normalized_intent.json` retains destination paths and therefore
  is intentionally the only path-dependent diagnostic member.

Contract pin: `373e3d17060d4ce1c4a0db6065c9289da714bde7`.

The original dirty M9 worktree remains preserved separately and was not copied
or committed. No Contract, Golden evidence, or M11/M14 work was changed.

Independent review: not obtained; the bounded auditor did not return a verdict.
