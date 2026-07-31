# Canonical JSON SHA-256 v1

For every v1 digest projection, serialize the JSON object as UTF-8 with keys
sorted lexically, separators `,` and `:`, no whitespace, and the host JSON
number representation used by the M5 Python implementation. Prefix the
lowercase hexadecimal SHA-256 digest with `sha256:`. This is not JCS.

Identity hashes `schema_version`, `training_payload`, `training_contract`, and
`authority`. Manifest hashes `schema_version`, `profile`,
`semantic_identity_digest`, and `inventory`. Runtime-only keys are rejected,
not normalized away, at the identity envelope boundary.
