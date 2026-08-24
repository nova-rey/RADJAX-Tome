# M8 sparse-selected-logits final closure

## Scope

This final trial compared full-length dense logits with full-length sparse selected logits at batch size 1. Both used the original 128-token inputs, the same 64 sources and 71 coordinates, the same tokenizer, model, reducer, compact publication path, and no suffix truncation.

Dense projection produced 8,192 vocabulary rows; sparse projection produced 71 rows, avoiding 8,121 rows (99.13%). The prior rejected prefix/truncation path was disabled for this comparison.

## Equivalence

All 71 coordinates were persisted and compared field-by-field.

- body records: 71 dense, 71 sparse;
- canonical body digest differences: 4;
- logical root dense: `sha256:e2df0c33984c9836fb0cffa9e8b23f4b209c707da680c2f06e994d37f7d9d35a`;
- logical root sparse: `sha256:4d083dbad3986da0b5463477477b877a9d470134779bfb8f483fe6691c241f2e`;
- body root dense: `sha256:f63622774d09ba2de5f714982d6ffd37e2451037d4ccbe44032e6584bcdf72ca`;
- body root sparse: `sha256:4fcb35932e3e0c557b79ae5f3774df76c8860b37578aef46fa46449a46e70`.

The differing coordinates were:

- corpus_000000206, position 35: retained arrays differed;
- corpus_000000260, position 104: retained arrays differed;
- corpus_000000298, position 105: retained arrays and K differed (361 vs 421);
- corpus_000000978, position 118: retained arrays differed.

The top-token identity did not change at these coordinates, but exact retained-token ordering/content and dynamic K are authoritative teaching fields. No tolerance was widened. Sparse is therefore behaviorally different under the existing Contract policy.

## Timing

Three alternating samples per mode were completed:

| Mode | Samples (s) | Median wall | Median teacher |
|---|---:|---:|---:|
| Dense | 24.592561, 23.875768, 24.879150 | 24.592561 | 7.290268 |
| Sparse | 24.356270, 23.377428, 24.210041 | 24.210041 | 6.864035 |

Sparse changed total wall time by -1.5554% and teacher time by -5.8466%. Total improvement is below the 3% materiality threshold and is not actionable given the equivalence failure.

Peak allocated GPU memory was identical in all samples: 1,755,066,368 bytes. The row/traffic reduction did not produce a measured peak-allocation reduction. Host RSS was approximately 5.32–5.37 GiB. No fallback occurred.

The cumulative BF16 projection accounting remains 4.00 GiB dense versus approximately 35.5 MiB sparse, but this is cumulative traffic/row accounting, not a measured peak-memory saving.

## M8 closure

Batch sizes 2/4/8 previously failed the ≥20% selected-pass gate; batch 1 remains canonical. Suffix truncation was behaviorally different and 0.9554% slower, so it remains rejected. Sparse selected-logit projection is rejected because four coordinates changed canonical teaching payloads and total wall improvement was immaterial.

No further selected-pass optimization is required before M9. Existing evidence directories are preserved unchanged. M8 is closed under the roadmap clause allowing measured proof that no further safe material gain is available.

Disposition: `SPARSE_SELECTED_LOGITS_BEHAVIORALLY_DIFFERENT_REJECTED`

M8_CLOSED
