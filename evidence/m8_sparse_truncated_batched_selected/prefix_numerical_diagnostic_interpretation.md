# Prefix numerical diagnostic interpretation

The direct T4 diagnostic repeated the same selected coordinate five times in one runtime. Full-length and causally truncated prefixes had identical token IDs and dynamic K (top token 529, K=9) on every iteration. The observed entropy values were 0.8519009351730347 and 0.8538008332252502, a repeatable delta of 0.0018998980502155, below the existing governed entropy tolerance of 0.00390625.

This is supporting evidence that causal truncation can produce ordinary low-order GPU numerical variation without changing the selected token or K. It does not establish equivalence for the complete 71-coordinate production authority: the production selected-pass smoke still failed at corpus_000000662, position 6, with entropy 0.85546875 against the frozen authority 0.84765625 (delta 0.0078125, two governed tolerance units).

The diagnostic was run in a separate T4 image with Torch 2.13.0+cu130 and CUDA 13.0. The production benchmark image/runtime was not identical, so this result is not a substitute for an all-coordinate production comparison. No broader tolerance is justified from this evidence, and no output sample was accepted.

The exploratory prefix-padded-to-128 comparison produced materially different logits (mean absolute delta 5.547763347625732; top token 179241; K=462). It is not used as acceptance evidence because direct padding/logits-to-keep semantics in this diagnostic runtime do not establish the production model's governed position behavior.

Disposition remains SELECTED_PREFIX_OPTIMIZATION_EQUIVALENCE_FAILED.
