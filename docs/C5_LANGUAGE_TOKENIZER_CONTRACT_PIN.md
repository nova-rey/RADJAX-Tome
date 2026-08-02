# Language/Tokenizer Binding Contract Pin

Tome pins RADJAX-Contract `v0.7.0` at immutable commit
`cac3dd21e0d56df5a9e6fd50b20267e0b8960995`.  The release artifacts are
mirrored byte-for-byte under
`contracts/radjax_tome/student_consumption/v5`; `SHA256SUMS` and the T1 tests
make this an offline verification mirror, never a second normative source.

The binding is captured from the instantiated tokenizer that encoded payload
token IDs.  Tome supports its deterministic `SmokeTokenizer` and its explicit
fast Hugging Face adapter only.  The latter requires a complete contiguous
vocabulary, configuration and normalization evidence, token declarations, and
an immutable revision (content digest, full git commit, or immutable release).
Missing evidence is an error; metadata-only reconstruction and profile
fallback are forbidden.

V4 remains an explicit historical validation profile.  T1 does not alter a
v4 package, identity, cover, archive, or receipt.
