# Canonical language/tokenizer binding JSON and JSONL v5

Semantic JSON is UTF-8 without BOM, lexicographically sorted keys, compact
separators, finite numbers, and no negative zero. Duplicate object keys are
invalid. Vocabulary JSONL has exactly one canonical object per LF-terminated
line, with ordered `{"token_id", "token_utf8_b64"}` records. Blank lines,
CRLF, duplicate IDs, invalid UTF-8 base64 payloads, and a token ID sequence
other than exactly `[0, vocabulary_size)` are invalid.

`canonical_inventory_digest` is the SHA-256 of the canonical ordered list of
`resource_id`, `role`, and `content_digest`. `canonical_binding_digest` is the
SHA-256 of the complete semantic projection: tokenizer identity, canonical
inventory digest, vocabulary map identity/size/domain, and added, reserved,
and special-token declarations. Transport, inventory paths, raw sizes,
timestamps, archive metadata, and wrappers are excluded.
