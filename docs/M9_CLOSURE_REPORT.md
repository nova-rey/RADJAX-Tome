# M9 Closure Audit

Status: **implementation partial; not closed**.

This report records the final audited state of the reconciled branch. It does
not promote the branch to `main` and does not treat the rejected historical M9
branch as evidence.

## Authority

- accepted M8 base: `88d6f418e0f39ab2ec61d9047f974f04657e5214`
- live `origin/m8/sparse-selected-logits-final-closure`: the accepted M8 base
- live `origin/main`: `deed286a8725cc98f4a5d6a25aa80582d692e108`
- reconciled branch: `m9/opinionated-mainline-cli-reconciled`
- audited branch tip: `f9bf9eb03d1d7469547e516ecbc1770c3ec31a7a`
- rejected branch `8a2dc2794ec32227f6187fa22837ba1f8dae0ccb`: untouched and not used
- Contract pin: `373e3d17060d4ce1c4a0db6065c9289da714bde7`

The plan required `origin/main` to equal `6a6c653` and to be fast-forwarded to
M8. Live main has since advanced to approved-plan descendants, so that exact
prerequisite is no longer available without an authority decision.

## Implemented surface

The branch provides the six-command public parser (`build`, `validate`,
`inspect`, `package`, `doctor`, `research`), strict canonical M5 JSON/YAML
loading, canonical output overrides, pure destination safety checks, Contract
v3 mode dispatch, M7/package dispatch, machine results, receipts, compatibility
routing, help snapshots, and selected-rerun batch size 1 in accepted presets.

Focused M9 plus Hydra disposition checks passed: **19 passed**, including the
final inventory record. A Python 3.12
wheel install passed version, canonical help, module help, machine doctor,
package, validate, and inspect checks. The installed Contract resolved to the
exact pinned commit above. Wheel SHA-256 for that smoke was recorded as:

`2fa80fdaf31e72f1cd0bfd4c3f276a6433ada712511c0fc73d8b297913f9d091`

## Blocking evidence

The full repository run stopped after 20 failures with **581 passed and 25
skipped**. The failures include stale accepted-base characterization tests,
legacy/public command-surface assertions, M4 resume terminal-semantics
assertions, and existing backend/Contract compatibility cases. Full Ruff also
reports pre-existing malformed archival benchmark code. These results do not
prove an M9 closure gate.

An independent reviewer disposition was not obtained in this environment;
therefore the review gate is also incomplete. No semantic identity, Contract,
Student, Golden, GPU performance, or M10–M14 claim is made.

## Successful bounded checks

- focused M9/Hydra tests: 19 passed
- `compileall src scripts tests`: passed
- `git diff --check`: passed
- Python 3.12 clean wheel install: passed
- exact Contract direct-url commit: passed
- public package → validate → inspect smoke: passed

The branch is intentionally left as a pushed, reviewable partial
implementation pending authority resolution and correction of the remaining
correctness gates.
