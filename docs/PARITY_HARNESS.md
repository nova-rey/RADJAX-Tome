# Parity Harness

Spec 4.3 adds a post-build A/B parity harness for generated Tome artifact
directories.

Parity does not mean byte-identical output. Parity means the artifacts are
compatible where they must be compatible: required sidecars exist, schemas and
target-store metadata line up, shard arrays have compatible fields, shapes, and
dtypes, floating arrays are finite, numerical differences stay inside declared
tolerances, metadata claims remain honest, and provenance links are comparable.

Run a comparison:

```bash
radjax-tome parity \
  --left ./artifact_cpu \
  --right ./artifact_gpu \
  --left-label cpu_reference \
  --right-label gpu_torch \
  --output ./parity_report.json
```

Useful options:

```bash
radjax-tome parity \
  --left ./artifact_a \
  --right ./artifact_b \
  --output ./parity_report.json \
  --rtol 1e-4 \
  --atol 1e-5 \
  --max-examples 8 \
  --no-compare-values
```

`--no-compare-values` still checks sidecars, target metadata, array fields,
shapes, dtypes, and finite floating values. It skips strict value/tolerance
checks for cases where exact value comparison is not meaningful.

## Strict Checks

The harness fails on required sidecar absence, schema mismatches, target type
mismatches, required array absence, array shape mismatches, dtype mismatches,
non-finite floating values, metadata sanity failures, forbidden truth claims,
corpus hash mismatches when both artifacts claim corpus provenance, and teacher
model hash mismatches when both artifacts claim teacher model provenance.

## Tolerant Checks

Floating-point values use `atol + rtol * abs(reference)` tolerance.

exact floating equality is not required because CPU and GPU paths may differ by
small rounding amounts while still producing compatible artifacts. Reports
include `max_abs_diff`, `mean_abs_diff`, `max_rel_diff`, and
`within_tolerance_fraction`.

Selector manifests compare strict policy/truth fields and measure selected
example/position overlap with Jaccard scores. Identical selection is not
required by default when comparing different capture modes or backends.

## Provenance

corpus/model provenance affects parity. If both artifacts claim corpus
provenance, `source_corpus_hash` and `source_corpus_manifest_hash` must match.
If only one artifact records corpus provenance, the report warns by default.

If both artifacts claim teacher model provenance, config, tokenizer, weight,
and directory hashes must match. If only one artifact records teacher model
provenance, the report warns by default.

Spec 4.3 does not download models, perform network verification, run GPU
planning, change backend reducer behavior, change selector behavior, or add
TPU/JAX work.
