# Canonical JSON and JSONL v4

Semantic JSON is UTF-8 without BOM, lexicographically sorted keys and compact
separators. Duplicate object keys, nonfinite numbers, and negative zero are
invalid. JSONL is one canonical object per LF-terminated line; CRLF and blank
lines are invalid.

The consumption semantic digest is `sha256:` plus SHA-256 of the canonical
semantic-identity object with its self-referential `semantic_digest` omitted.
Its resource projection is the ordered tuple `(resource_id, role, instance_id,
semantic_digest)`. Physical locators are excluded. The required v4 semantic
declarations are resources with consumption values exactly `{"kind":
"row_range_declaration"}`, `{"kind": "delivery_receipt"}`, and `{"kind":
"authority_reference"}`; the independently-delivered JSON resources carry
their corresponding closed bodies.
