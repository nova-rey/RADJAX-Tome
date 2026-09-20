# Corpus builder (corpus-v2)

The canonical corpus builder consumes a complete
`radjax_tome_corpus_build_intent_v1` document. It is local and deterministic;
the output is a rebuildable, verified input to the M5 production intent.

## Small local example

Create a source tree and a complete intent (the repository test fixture is a
copyable reference):

```json
{
  "schema_version": "radjax_tome_corpus_build_intent_v1",
  "artifact": {"schema_version": "radjax_tome_corpus_artifact_v2"},
  "sources": [{"source_id": "text", "adapter": "local_text_tree_v1", "path": "sources"}],
  "policy": {
    "normalization": "text_normalize_lf_strip_trailing_ws_v1",
    "filtering": {"min_chars": 1},
    "chunking": {"name": "char_window_v1", "max_chars": 100},
    "deduplication": {"enabled": true},
    "ordering": "declared_source_ordinal_logical_locator_chunk_index_v1",
    "tokenizer": "smoke"
  },
  "layout": {"shard_capacity": 2},
  "resources": {"memory_limit": "64MB", "worker_count": 1},
  "output": {"artifact_path": "artifact"},
  "execution": {"resume": false, "overwrite": false},
  "reporting": {"progress": false}
}
```

Run the public path:

```bash
radjax-tome corpus build --config ./corpus-intent.json
radjax-tome corpus validate ./artifact
radjax-tome corpus inspect ./artifact
```

This exact form was executed outside the checkout in the M12 CPU journey:
one local source produced one record, a verified semantic identity, and passing
JSON results. The example is an offline CPU demonstration, not teacher
inference and not a claim of Student readiness.

## Inputs and identity

Supported source adapters are `local_text_tree_v1` and
`local_jsonl_text_v1`. Text trees accept `.txt`, `.md`, `.markdown`, and
`.py`. JSONL rows require a string `text` field. Arbitrary structured JSON
is not an adapter.

The policy fixes LF normalization, filtering, `char_window_v1` chunking,
optional exact deduplication, and deterministic ordering. The artifact contains
shards and indexes, a cover/manifest, reports, and strict duplicate provenance.
`VerifiedCorpusReader` checks shard and index digests, ranges, offsets,
identities, and counts before yielding a record. Semantic identity is
path-independent and covers policy, tokenizer binding, source declarations, and
ordered record metadata. Physical shard layout may differ without changing
semantic identity.

## Tokenizer binding

The production tokenizer must match the binding captured in the corpus before
teacher/backend construction. A mismatch fails closed with
`CORPUS_TOKENIZER_BINDING_MISMATCH`.

The accepted schema is intentionally narrow. The `smoke` tokenizer is used by
offline fixtures; the `hf`/`qwen` adapter requires the optional
Transformers dependency and complete binding capture. Arbitrary HF IDs or
revisions are not supported unless the adapter can capture and match the full
canonical binding. Do not patch a corpus binding by editing a digest.

## Use it from a Tome intent

The v1 M5 example uses dataset/manifest paths for compatibility. A complete
`radjax_tome_build_intent_v2` replaces those two fields with one corpus
reference:

```json
{
  "schema_version": "radjax_tome_build_intent_v2",
  "corpus": {
    "artifact_path": "./artifact",
    "expected_semantic_identity": "sha256:...",
    "max_examples": null
  }
}
```

All other teacher, behavior, selection, output, package, and reporting
sections remain the canonical M5 sections. Run preflight before expensive work:

```bash
radjax-tome build --config ./tome-intent-v2.json --preflight-only
```

Resolution produces one `ResolvedCorpusInput` consumed by teacher,
streaming, score/selection, and selected-rerun stages. A wrong semantic
identity or tokenizer binding fails before model loading or accelerator
allocation.

Corpus resume recovers only the durable journal/publication boundaries that the
builder records. It does not promise arbitrary recovery inside an unfinished
source or shard. `--overwrite` is explicit and cannot replace an unrelated
destination.

## What this artifact is not

Corpus construction does not scrape the internet, clone repositories, or
download teacher models. Teacher provenance and accelerator prerequisites are
separate production concerns; see `docs/START_HERE.md`.


## Compatibility provenance names

Reports retain the canonical corpus_hash and manifest_hash fields for v1-compatible consumers. The v2 semantic identity is the authoritative path-independent identity; these fields are not interchangeable.

Do not scrape the internet or download teacher models during corpus construction.
 Structured .json is not supported yet; convert it to JSONL rows with a text field.

The manifest uses manifest_hash_policy=exclude_self_hash_and_created_at_v1: created_at is a human UTC timestamp, while the manifest hash excludes manifest_hash and created_at so identical content remains stable.
