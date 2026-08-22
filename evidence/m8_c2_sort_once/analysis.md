# M8 C2 sort-once correction

## Scope

Recorded v10 C1-output-through-C5 input: 128,000 candidate records, 106,617 eligible candidates, 46 nonempty corridor modes. Contract is unchanged. The accepted starting-authority baseline is reused from SELECTION_PIPELINE_PROFILE (full wall 267.216111s; instrumented C2 312.584874s).

## Correction

Complete candidate pools now append in canonical arrival order and are sorted once after the complete stream. The bounded legacy path (retain_complete_candidate_pool=false) retains its prior per-candidate sort/truncate behavior. C3-C5 are unchanged.

## Results

| metric | accepted baseline | patched samples | patched median | change vs baseline |
|---|---:|---:|---:|---:|
| complete C1-output-through-C5 wall (s) | 267.216111 | 10.922040, 10.560748, 10.547046 | 10.560748 | -256.655363 s (96.05%) |
| complete process CPU (s) | 266.903717 | 10.911546, 10.524734, 10.480520 | 10.524734 | — |
| C2 leaderboard wall (s) | 312.584874 instrumented | 6.046327, 5.926389, 5.811267 | 5.926389 | -306.658485 s (98.10%) |

Patched peak RSS was 206.2 MiB. C2 diagnostics were identical in every run: 128,000 seen, 106,617 eligible/appended, 46 nonempty pools, 46 sort calls, and 106,617 cumulative sort-input items.

## Equivalence

The patched selected count remained 253, budget shortfall remained 3 with reason global_ranked_supply_exhaustion, and selected-coordinate ordering matched the accepted baseline and all three patched runs. C2 mode leaderboard sequences and C3/C4/C5 coordinate/claim/selection JSONL projections were identical. Raw output digests differ only because output paths and path-derived manifest digests are run-local nonsemantic metadata. See equivalence.json.

## Validation

Focused corridor leaderboard, allocator, global claims, and multi-role selection tests: 82 passed. Ruff, format, compileall, and diff checks passed.
