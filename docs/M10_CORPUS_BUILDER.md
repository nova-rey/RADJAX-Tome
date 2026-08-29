# M10 corpus builder

M10 adds the strict, local-only `radjax_tome_corpus_artifact_v2` builder while
leaving the historical v1 corpus format and selection authority unchanged.

The public grammar is:

```text
radjax-tome --json corpus build --config CONFIG
radjax-tome --json corpus validate ARTIFACT
radjax-tome --json corpus inspect ARTIFACT
```

Build intents are strict JSON or YAML documents. Sources are UTF-8 text trees
or JSONL records. Normalization, deterministic chunking, exact SHA-256
deduplication, source provenance, canonical JSONL shards, and byte offset
indexes are recorded in the artifact. Readers verify each complete shard and
index before yielding rows.

Publication uses an fsynced staging journal. New destinations are promoted by
rename; overwrite publication explicitly reports non-atomic visibility and
keeps a recoverable quarantine until validation succeeds. `--resume` accepts
only a valid artifact or owned staging transaction.

The v2 semantic identity contains corpus semantics, ordered source
declarations, ordered example identities, policy, and tokenizer binding. It
does not contain absolute paths, timestamps, or physical shard layout.
