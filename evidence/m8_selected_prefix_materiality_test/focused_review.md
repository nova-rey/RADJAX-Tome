# Focused materiality review

Disposition: BLOCKED / REJECTED FOR ADOPTION.

The 71-coordinate comparison is internally coherent. Top-token identity remained unchanged, but retained-token ordering/set changed on 8/71 coordinates and dynamic K changed on 2/71. Entropy delta reached 0.035998, with the first difference at corpus_000000662 position 6. The FP32 comparison is mixed and does not restore categorical equivalence. Truncated BF16 was slower (median 2.861979 s versus 2.834895 s, +0.9554%), so timing is diagnostic only.

The earlier tokenize-at-original-policy-then-slice correction addresses special-token relocation. Prefix accounting and remapping are deterministic. The remaining retained-set and K changes violate the existing teaching-payload contract. No tolerance relaxation, T4 batch-8 test, or production adoption is justified.

Disposition: SELECTED_PREFIX_BEHAVIORALLY_DIFFERENT_REJECTED.
