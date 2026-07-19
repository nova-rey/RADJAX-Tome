# Canonical Golden T4 1K Fixture

This is the canonical portable regression contract for the native two-pass
fingerprint-corridor Path B pipeline. Its semantic root is
`sha256:4dcc4baa6bfc1c065d2f45268289db504a511891b875c40315c5748825e261ba`
and it contains 256 selected coordinates.

It was captured from a completed Tesla T4 run using Gemma 3 270M over 1,000
corpus examples at sequence length 128 and vocabulary size 262144. The run
used fingerprint-corridor-first/global-backfill selection, a selected rerun
batch size of 8, dynamic top-k range 32 through 262144, and dynamic mass
threshold 0.99.

Validate the committed contract without a model, corpus, GPU, or source Tome:

```bash
radjax-tome golden validate --fixture tests/fixtures/golden_t4_1k
```

Use `radjax-tome golden compare` to explain semantic differences between a
future canonical-pipeline artifact and this fixture. Payload arrays are not
committed: each selected payload is represented by deterministic binary
semantic digests of its ordered active token IDs, probabilities, and log
probabilities.

The fixture contains no corpus text, prompts, credentials, model weights,
rental paths, raw payload arrays, or dense vocabulary arrays.
