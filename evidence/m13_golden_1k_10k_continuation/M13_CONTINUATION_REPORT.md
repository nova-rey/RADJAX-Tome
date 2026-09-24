# M13 continuation report

## Scope

This continuation starts from `1de3bcd87a9f7012bf374aede5936f29d534a295` and preserves the prior M13 evidence. It fixes only the compact Golden payload reader and Hydra inventory omission; it does not change Golden, Contract, tolerances, selection policy, M8 behavior, or the accepted 10K measurement.

## Comparator correction

The original comparator rejection was a reader gap: compact `compact_k_monolithic` bodies contain the three active arrays and omit `top_selection_mask`. The reader now validates and hashes those arrays as the active payload. Legacy dense masked payloads remain validated. It still validates all governed fields and returns `fail` for semantic differences; no unsupported fields are dropped and no failure is converted to pass. Execution-only `temp_directory` is excluded from portable board summaries.

## Earliest divergence

The frozen and current runs use identical corpus, manifest, model, weights, tokenizer, example count, and sequence length. They are not the same experiment: frozen C2 candidate-pool cap is 4 versus current 10000; frozen C3 corridor-mode cap is 10 versus current 64; frozen selected rerun batch is 8 versus current 1; and the recorded runtime/reduction differs (Torch 2.13/bfloat16 historical versus Torch 2.14/current float32). Golden and current score/selection configuration hashes differ before C5. The selected sets contain 126 common coordinates, 130 missing, and 130 extra in the original current run. This is an authority/policy mismatch, not a coordinate mapping defect; no coordinates were forced and no tolerance was widened.

## Policy-aligned decisive T4 runs

A first policy-aligned attempt used teacher batch 8 and OOMed before completion; the raw app/log are preserved. A feasible run with frozen C2/C3 caps, teacher batch 1, and selected rerun batch 8 completed score/selection/validation but failed selected delivery at `corpus_000000118` (score top token 708 versus rerun top token 184; entropy delta 3.438110828399658 versus allowed 0.00390625).

The final discriminator used frozen C2/C3 caps, teacher batch 1, and selected rerun batch 1. It completed the full production build, selected delivery, validation, and publication successfully with 256 coordinates. Comparing its selected coordinates to the frozen Golden still gives 128 common, 128 missing, and 128 extra. This proves the remaining mismatch persists after policy alignment and feasible rerun batching: the unresolved cause is historical/current teacher and score-reduction authority, not coordinate mapping or delivery. No Golden or tolerance change is justified.

The temporary harness returned stale metadata naming `b5ce585`, but its source mount was the checked-out `d659d15` branch; no selected-pass production code changed between them.

## Package and resume

The retained Golden output was packaged with the canonical `full_debug_provenance` profile, transferred, and revalidated after extraction. The Student profile correctly rejected the retained output because it lacks tokenizer-binding capture; no binding was fabricated. A controlled three-record/two-shard interruption at `after_v4_shard_sealed` resumed from the sealed prefix and converged to valid directory/archive output. The receipt reports `staging_exists_after=true`; this is receipt-backed completed state, so this report does not claim staging cleanup.

## Hydra and repository gates

The tracked inventory now includes the M13 closure report and T4 smoke helper. The focused Hydra/Golden suite passes 45 tests. Ruff, formatting, compileall, and diff checks pass. A correctly rooted full suite reports 1243 passed, 27 skipped, and one unrelated P6/U1 reduced-burn failure (`numpy.ndarray` lacks `first_valid_index`); it is preserved as a limitation. The wheel was built and an offline clean install using the exact local Contract wheel plus the Tome wheel succeeded for CLI help.

## Gate status

The accepted 10K measurement was not rerun. Compact comparison, package/transfer/resume, Hydra, and the policy-aligned 1K discriminator are complete. The historical Golden and current accepted production teacher/reducer authorities remain irreconcilable under the current rules. Passing M13 now requires an owner decision about which authority is the parity oracle or authorization to reproduce the historical runtime; neither is authorized in this checkpoint.

## Disposition

M13 remains blocked by the unresolved historical-versus-current Golden authority conflict. All other newly actionable gates are preserved with receipts.
