# Native-v3 Student-Consumption v5 Packaging

New Student packages use the explicit `native_v3_student_v5` profile.  The
ordinary package transaction accepts only
`language_tokenizer_binding_v1.json` captured from the tokenizer instance that
encoded the source payload IDs.  It verifies the source capture with the
published Contract, checks all source `input_ids` against the captured exact
vocabulary domain, copies the binding and every inventory resource into the
ordinary package inventory, then emits the v5 manifest and cover declaration.

`smoke_tokenizer` is the explicit CPU-only v5 producer backend.  It uses an
instantiated `SmokeTokenizer` with `vocab_size >= 257`; the same instance
encodes source text and supplies the capture.  It is separate from
`cpu_reference`, whose historical synthetic encoder remains byte-for-byte
unchanged and therefore has no v5 capture.  Existing CPU/fake artifacts fail
v5 packaging rather than receiving inferred or reconstructed tokenizer facts.

Before public promotion, directory candidates are admitted by Contract v5.
For a `.tgz`, Tome first validates the deterministic archive inventory, safely
extracts its candidate bytes, then invokes the same strict Contract v5
admission.  V4 remains a historical Contract validation path; no v5-to-v4
fallback occurs.
