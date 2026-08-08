# Private journal transition recipe

The construction journal is producer-private. It is never public, inventoried,
archived, or semantic-root input. Contract validates only its typed states,
sealed receipts, contiguous committed ranges, and restart classification.

For the local POSIX filesystem profile, a producer writes and fsyncs a staged
shard, computes its raw receipt, fsyncs the append-only receipt, fsyncs the
contiguous-range commit, fsyncs `COMPLETE_INTENT`, fsyncs `PROMOTION_INTENT`,
uses same-filesystem atomic `rename(no_replace)`, then fsyncs `PROMOTED` after
standard validation. A producer lacking durable replacement/fsync or that
rename guarantee must reject the output transport before claiming transactional
publication.

States are `OPEN`, `SEALING`, `COMPLETE_INTENT`, `PROMOTING`, `PROMOTED`, and
`ABORTED`. A missing durable promoted marker leaves a tree nonconsumable;
restart validates private receipts and ranges before retry, rollback, or refusal.
