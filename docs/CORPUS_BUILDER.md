# Corpus Builder

Spec 4.1 adds a deterministic local corpus builder for RADJAX-Tome.

A corpus artifact is a small provenance package:

```text
corpus.jsonl
corpus_manifest.json
corpus_build_report.json
```

Build one from local files:

```bash
radjax-tome corpus build \
  --input ./sources \
  --output ./corpus_out \
  --include "**/*.md" \
  --include "**/*.txt" \
  --min-chars 1 \
  --max-chars 12000 \
  --overwrite
```

Inspect and validate it:

```bash
radjax-tome corpus inspect --path ./corpus_out
radjax-tome corpus validate --path ./corpus_out
```

Each `corpus.jsonl` row records source identity, normalized text, chunk
position, source hash, and content hash. The corpus hash is computed from the
canonical `corpus.jsonl` bytes. The manifest hash is computed from canonical
manifest JSON with `manifest_hash` excluded, so the manifest can validate
without a self-reference loop.

Use a corpus manifest when building a Tome:

```bash
radjax-tome build \
  --dataset ./corpus_out/corpus.jsonl \
  --corpus-manifest ./corpus_out/corpus_manifest.json \
  --output ./teacher_textbook \
  --teacher-mode fake \
  --overwrite
```

The generated Tome records `source_corpus_hash`,
`source_corpus_manifest_hash`, corpus schema/version counts, normalization
policy, chunking policy, deduplication policy, and manifest path in metadata,
`teacher_manifest.json`, `emission_config.json`, and `cover_page.json`.

Spec 4.1 is local only. Do not scrape the internet, clone GitHub repositories,
download teacher models, make license/legal judgments, add semantic filtering,
plan GPU runs, or touch TPU/JAX.
