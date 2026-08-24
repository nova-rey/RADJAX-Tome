# M8 selected-prefix materiality test

## Result

The three-watch comparison ran on one Tesla T4 using the fixed 64-source/71-coordinate authority. The test is diagnostic and non-authoritative.

Full-length BF16 versus truncated BF16:

- categorical top-token changes: 0/71;
- retained-token ordering/set changes: 8/71;
- dynamic K changes: 2/71;
- maximum entropy difference: 0.0359980464;
- median entropy difference: 0;
- mean entropy difference: 0.0017036197;
- p90 entropy difference: 0.0018998981;
- p95 entropy difference: 0.0126686394;
- p99 entropy difference: 0.0286337674;
- maximum tail-mass difference: 0.0002246046;
- first differing coordinate: corpus_000000662 position 6.

The changed retained-token ordering/set and K values violate the exact teaching-payload requirement. The absence of top-token changes is not sufficient for adoption because the compact target contains the full retained ordering and dynamic K.

The full BF16 and truncated BF16 paths were compared against a full FP32 diagnostic reference. Across the compared top arrays, truncated BF16 was closer to FP32 on 6/71 coordinates for probabilities and 5/71 for log probabilities; full BF16 was closer on 2/71 and 3/71 respectively, with ties on the remainder. For tail/top mass, truncated was closer on 6/71 and full on 2/71. This does not cure the categorical differences.

No Student-impact loss probe was run: no existing selected-target loss API was found in Tome, and no new training objective was invented.

## Timing

Three repeated teacher-forward measurements per path, batch 1:

| Path | Samples (s) | Median (s) |
|---|---:|---:|
| Full-length BF16 | 2.834895, 2.836579, 2.820165 | 2.834895 |
| Truncated BF16 | 2.858860, 2.871119, 2.861979 | 2.861979 |

The truncated path was 0.9554% slower in this measurement. These are controlled teacher-path timings, not accepted selected-pass publication timings, because the candidate failed logical equivalence. Batch 8 was not run.

## Decision

The numerical differences are not only insignificant rounding: eight coordinates changed retained-token order/set and two changed dynamic K. The truncated path is therefore behaviorally different under the existing Contract policy. No equivalence-rule relaxation, Contract change, or production adoption is justified.

Disposition: `SELECTED_PREFIX_BEHAVIORALLY_DIFFERENT_REJECTED`.
