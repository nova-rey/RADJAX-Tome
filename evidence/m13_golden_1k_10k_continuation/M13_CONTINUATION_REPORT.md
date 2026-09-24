# M13 continuation report

## Scope

This continuation starts from `1de3bcd87a9f7012bf374aede5936f29d534a295` and preserves the prior M13 evidence. It fixes only the compact Golden payload reader and Hydra inventory omission; it does not change Golden, Contract, tolerances, selection policy, M8 behavior, or the accepted 10K measurement.

## Comparator correction

The original comparator rejection was a reader gap: compact `compact_k_monolithic` bodies contain the three active arrays and omit `top_selection_mask`. The reader now validates and hashes those arrays as the active payload. Legacy dense masked payloads remain validated. It still validates all governed fields and returns `fail` for semantic differences; no unsupported fields are dropped and no failure is converted to pass. Execution-only `temp_directory` is excluded from portable board summaries.

## Earliest divergence

The frozen and current runs use identical corpus, manifest, model, weights, tokenizer, example count, and sequence length. They are not the same experiment: frozen C2 candidate-pool cap is 4 versus current 10000; frozen C3 corridor-mode cap is 10 versus current 64; frozen selected rerun batch is 8 versus current 1; and the recorded runtime/reduction differs (Torch 2.13/bfloat16 historical versus Torch 2.14/current float32). Golden and current score/selection configuration hashes differ before C5. The selected sets contain 126 common coordinates, 130 missing, and 130 extra. This is an authority/policy mismatch, not a coordinate mapping defect; no coordinates were forced and no tolerance was widened.

## Policy-aligned decisive T4 run

One additional 1K T4 run used the frozen C2/C3 caps and selected-rerun batch 8 with the feasible current teacher batch size 1. It completed score, selection, and validation with 256 selected coordinates, but selected delivery failed at `corpus_000000118`: score top token 708 versus rerun top token 184 and entropy delta 3.438110828399658 against the existing 0.00390625 tolerance. A first attempt using teacher batch 8 OOMed before completion; its failure is preserved. This confirms current teacher/reducer runtime drift under the frozen policy. It does not justify changing Golden or tolerances.

## Package and resume

The retained Golden output was packaged with the canonical `full_debug_provenance` profile, transferred, and revalidated after extraction. The Student profile correctly rejected the retained output because it lacks tokenizer-binding capture; no binding was fabricated. A controlled three-record/two-shard interruption at `after_v4_shard_sealed` resumed from the sealed prefix and converged to valid directory/archive output. The receipt reports `staging_exists_after=true`; this is receipt-backed completed state, so this report does not claim staging cleanup.

## Hydra and repository gates

The tracked inventory now includes the M13 closure report and T4 smoke helper. The focused Hydra/Golden suite passes 45 tests. Ruff, formatting, compileall, and diff checks pass. A correctly rooted full suite reports 1243 passed, 27 skipped, and one unrelated P6/U1 reduced-burn failure (`numpy.ndarray` lacks `first_valid_index`); it is preserved as a limitation. The wheel was built and an offline clean install using the exact local Contract wheel plus the Tome wheel succeeded for CLI help.

## Gate status

The accepted 10K measurement was not rerun. The compact reader and portability fixes are complete, package/transfer/resume are complete, and the policy-aligned T4 run confirms the remaining Golden failure is a frozen/current teacher/reducer authority conflict. A parity pass would require an owner decision to change the parity oracle or reproduce the historical runtime; neither is authorized in this checkpoint.

## Disposition

M13 remains blocked by the unresolved historical-versus-current Golden authority conflict. All other newly actionable gates are preserved with receipts.
