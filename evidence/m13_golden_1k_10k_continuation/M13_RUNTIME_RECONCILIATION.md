# M13 runtime-authority reconciliation

This is an evidence addendum. It does not change the frozen Golden fixture,
the accepted 10K result, Contract, tolerances, selection policy, or any raw
run report.

## What is authoritative

The checked-in Golden contract binds the corpus, manifest, model directory,
weights, tokenizer, sequence length, policy hashes, and semantic root. Its
semantic_policy records selected_rerun_batch_size=8, C2 candidate-pool cap
4, and C3 corridor-mode cap 10. It does not contain a Torch version or a
model dtype field. Those are therefore not part of the frozen Golden semantic
authority.

The earlier T4 smoke receipt records Torch 2.13.0+cu130; that receipt was a
small model/logit smoke, not the Golden capture. The recovered production
output metadata and the policy-aligned discriminator both record Torch
2.14.0+cu130 and float32 output. This distinction is preserved rather than
used to infer a Golden runtime that the contract does not declare.

## Current production capability

At the audited implementation commit, the canonical gpu_torch backend loads
the model with the ordinary Transformers from_pretrained call and has no
production configuration field or code path that selects bfloat16. The
current policy-aligned T4 run therefore used the supported float32 path; no
harness monkeypatch or alternate inference implementation was introduced.

## Discriminating run

The final policy-aligned run used the Golden C2/C3 caps, teacher batch 1, and
selected rerun batch 1. It completed score, selection, selected delivery,
validation, and publication. The comparator still found 128 common, 128
missing, and 128 extra coordinates against the frozen Golden semantic root
sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba.
Thus feasible batching and policy alignment did not restore parity. The
remaining difference is upstream score/reducer authority (or an undeclared
historical runtime detail), not the compact reader, coordinate remapping, or
selected publication path.

Reproducing an undeclared historical runtime is not a safe in-checkpoint
change: it would require an owner decision that the Golden's non-contract
runtime is a hard oracle, or an explicit update of the parity authority. No
such decision was supplied, and no Golden artifact or tolerance was changed.

## Disposition

M13 remains blocked only on that authority decision. The prior 10K result,
package/transfer receipts, controlled resume receipt, focused review, and all
raw T4 reports remain valid and preserved.
