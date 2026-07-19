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
