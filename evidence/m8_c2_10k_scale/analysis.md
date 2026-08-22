# M8 C2 10K scaling measurement

Dataset: M8_DERIVED_10K_SELECTION_SCALING_FIXTURE_V1, a deterministic 10x expansion of recorded v10 candidate/passport evidence. This is non-authoritative scaling evidence; C6 was not run.

## Timings

| stage | 1K reference | 10K measured | multiplier |
|---|---:|---:|---:|
| C2 construction | 5.926s | 86.755s | 14.64x |
| complete C1-C5 | 10.561s | 177.470s | 16.80x |
| peak RSS | 203.1 MiB | 1590.1 MiB | 7.83x |
| C2 artifact | 47435094 B | 474219516 B | 10.00x |
| output bytes | 50141579 B | 475805641 B | 9.49x |
| backfill iterations | 5875 | 2477 | 0.42x |

The corrected sort-once structure is visible in the structural counters: 46 sort calls at both scales, while cumulative sort input grows from 106,617 to 1,066,170, exactly the final eligible-reserve total. Runtime growth is approximately linear over these two points; this is an observation, not a formal complexity claim.

The 10K run selected 256 coordinates with no shortfall (reason=None), unlike the 1K reference's authorized global-supply exhaustion. This difference is a consequence of the derived replicated global supply and is retained as fixture behavior. Full-width candidates considered from unique global coordinates: 9590; selected: 44; cap allowance: 85; observed cap rejections: 0.

At unchanged scaling, a 100K-source extrapolation would be approximately 1774.7s wall, 15.53 GiB peak RSS, and 4.43 GiB output. These are arithmetic extrapolations, not measurements.

Host available memory remained above 1.7 GiB during the run; swap was already in use before execution and did not show a new sustained-thrashing abort. The generated fixture and output are removed after evidence hashing.
