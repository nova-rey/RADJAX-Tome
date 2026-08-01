# Canonical JSON and JSONL v2

Semantic JSON is UTF-8 without BOM; duplicate object keys, nonfinite numbers,
and noncanonical negative zero are rejected.  It uses lexicographically sorted
keys and compact JSON separators.  JSONL consists of one canonical JSON object
per LF-terminated line; CRLF and blank lines are invalid.

The consumption semantic digest is `sha256:` plus SHA-256 of the canonical
semantic-identity object with its `semantic_digest` member omitted.  Its
resource projection is exactly the ordered tuple `(resource_id, role,
instance_id, semantic_digest)`.  `inventory_binding`, manifest path, archive
member name, and every other physical locator are excluded from this digest.
