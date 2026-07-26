# Golden 1K Capture

M2A and M2B are **complete**. The canonical portable Golden 1K contract is
checked in at `tests/fixtures/golden_t4_1k` with semantic root
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`
and 256 selected coordinates. The fixture passed its recursive portability
gate: no local POSIX, Windows, UNC, file URI, or home-relative storage locator
is committed.

The source artifact was a completed native two-pass fingerprint-corridor Path
B Tesla T4 run with Gemma 3 270M, 1,000 corpus examples, sequence length 128,
vocabulary size 262144, a selected rerun batch size of 8, dynamic top-k range
32 through 262144, and dynamic mass threshold 0.99. The source Tome is not
committed. The fixture is a portable semantic contract containing only
reviewable selection, passport, board, and digest-level payload records.

Validate the fixture without model loading, corpus access, GPU access, or the
source Tome:

```bash
radjax-tome golden validate --fixture tests/fixtures/golden_t4_1k
```

Future canonical-pipeline changes must compare their produced artifact against
this fixture:

```bash
radjax-tome golden compare \
  --fixture tests/fixtures/golden_t4_1k \
  --artifact /path/to/canonical-artifact
```

Any semantic difference requires an explicit explanation and an intentional
fixture update. Payload bodies remain excluded: the contract stores versioned
binary semantic digests of ordered active token IDs, probabilities, and
log-probabilities rather than dense or raw payload arrays.

## Historical v1 and reproducible v2

This checked-in fixture is permanently a
`radjax_tome.golden_contract.v1` historical contract. Its root and recorded
authority hash must not be changed or reinterpreted. Its v1 authority binds
raw artifact bytes and therefore includes runtime timestamps from the source
artifact; a fresh artifact must not be expected to reproduce the v1 root.

New reproducibility work uses the explicit v2 authority and Golden contracts
described in [Authority-Hash v2 Migration](AUTHORITY_HASH_V2_MIGRATION.md).
Capture a v2 projection read-only from a terminal source artifact, then use
that new v2 fixture only with v2 comparisons:

```bash
radjax-tome golden capture \
  --contract-version radjax_tome.golden_contract.v2 \
  --artifact /path/to/terminal-artifact \
  --output /path/to/golden-v2-fixture

radjax-tome golden compare \
  --fixture /path/to/golden-v2-fixture \
  --artifact /path/to/another-terminal-artifact
```

V1 and v2 contracts are intentionally incompatible. `golden compare` reports
their differing schema versions rather than treating them as comparable roots.
