# M9 Closure Audit

Status: **closed**.

The audited implementation is commit `7550ea9bcae23917fdaaee3d7506efaef849c6bf`.
The continuation branch is `m9/opinionated-mainline-cli-ci-reconciliation`.
The original dirty worktree remains separate and preserved at
`/home/nyx/radjax-tome`.

## Authority

- accepted M8: `88d6f418e0f39ab2ec61d9047f974f04657e5214`
- live origin/main: `deed286a8725cc98f4a5d6a25aa80582d692e108` (documentation-only descendants)
- Contract pin: `373e3d17060d4ce1c4a0db6065c9289da714bde7`
- public commands: `build`, `validate`, `inspect`, `package`, `doctor`, `research`

The branch descends from accepted M8 and integrates the two harmless approved-
plan documentation commits. No Contract, Golden, or historical evidence was
rewritten. The original dirty `hf_torch.py` correction was not copied because
it was a pre-existing compatibility correction, not an M9 regression.

## Gates

- `python -m pytest -q -p no:cacheprovider`: **1,212 passed, 9 skipped**
- `python -m ruff check .`: **pass**
- `python -m ruff format --check .`: **pass**
- `python -m compileall src scripts tests`: **pass**
- `git diff --check`: **pass**
- `python -m build`: **pass**
- clean-install wheel smoke, package → validate → inspect: **pass**
- selected canonical preset reruns: effective batch size **1**

The complete failure mapping is in
`docs/M9_CI_RECONCILIATION_REPORT.md`. The 20 original failures were inherited
or stale-boundary expectations except for demonstrated bounded production
corrections; all are now covered by green tests. The preserved P6 historical
receipt remains unchanged.

## Review and evidence

Focused independent read-only review found no remaining mandatory failure or
concrete finding requiring repair. The evidence receipt and checksums in
`evidence/m9_mainline_cli/` are bound to the audited implementation commit.
