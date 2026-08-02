# T3 Native-v3 Student-Consumption v5 Fixture Receipt

This is T3 evidence, not a downstream training, tokenizer-runtime, accelerator,
or model-quality claim. The committed fixture at
`tests/fixtures/native_v3_student_v5_smoke` was produced through the ordinary
`smoke_tokenizer` production builder and ordinary `student` package transaction.

The machine-readable receipt is
`tests/fixtures/native_v3_student_v5_smoke/FIXTURE_RECEIPT.json`. It binds
Contract `v0.7.0` / `cac3dd21e0d56df5a9e6fd50b20267e0b8960995`, the Tome
commit at production time `3861c23`, profile `native_v3_student_v5`, the
generic binding digest, fixture semantic/raw/tree digests, binding and
vocabulary raw digests, and the exact Contract validator entry point.

Reproduce the fixture with the pinned Contract dependency installed:

```console
python3 scripts/build_v5_language_tokenizer_fixture.py \
  --output tests/fixtures/native_v3_student_v5_smoke
```

The fixed producer configuration is recorded in the receipt. T3's repeated
generation test proves byte-identical binding and vocabulary resources and
equal generic-binding and package semantic digests. Raw and tree digests are
physical evidence for the committed package tree, not semantic identity.
