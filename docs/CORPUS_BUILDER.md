# Corpus Builder

Spec 4.1 adds a deterministic local corpus builder for RADJAX-Tome. Spec
4.1.1 clarifies corpus format truth and manifest hash policy.

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

Supported source formats are `.txt`, `.md`, `.markdown`, `.py`, and `.jsonl`
rows with a string `text` field. .json is not supported yet because
arbitrary structured JSON extraction is ambiguous; convert JSON externally to
JSONL rows with `text` before building a corpus.

Inspect and validate it:

```bash
radjax-tome corpus inspect --path ./corpus_out
radjax-tome corpus validate --path ./corpus_out
```

Each `corpus.jsonl` row records source identity, normalized text, chunk
position, source hash, and content hash. The corpus hash is computed from the
canonical `corpus.jsonl` bytes. The manifest records
`manifest_hash_policy=exclude_self_hash_and_created_at_v1`: `created_at` is a
real UTC build timestamp for humans, while the manifest hash is computed from
canonical manifest JSON excluding both `manifest_hash` and `created_at`. This
keeps manifest hashes stable across identical corpus content and build
configuration.

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
