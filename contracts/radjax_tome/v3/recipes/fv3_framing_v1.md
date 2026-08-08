# FV3-1 framing recipe

`FRAME(label, payload)` is exactly `RJTFE1\0 || U16BE(len(label)) || label ||
U64BE(len(payload)) || payload`. `RJTFE1\0` is hex `524a5446453100`; the
escaped `\0` is one NUL octet. Labels are raw bytes, never length-prefixed
values. Hashes are SHA-256 emitted as `sha256:` followed by lowercase hex.

`FV3(value)` tags: null `00`; false `01`; true `02`; signed i64 `10 || I64BE`;
finite binary64 `11 || IEEE754BE`; strict UTF-8 text `20 || U64BE(length) ||
bytes`; list `30 || U64BE(count) ||` each `U64BE(length) || FV3(item)`; closed
map `40 || U64BE(count) ||` each ascending UTF-8-byte key as
`U64BE(key_length) || key || U64BE(value_length) || FV3(value)`.

Integers are exactly `[-2^63, 2^63-1]`; their source lexeme is strict JSON
integer syntax. Binary64 source lexemes are exact decimal rationals rounded to
finite IEEE-754 binary64 with round-to-nearest ties-to-even. Negative zero,
nonfinite values, and overflow reject. The release vectors provide every exact
preimage and expected digest.

| purpose | escaped bytes | hex | bytes |
| --- | --- | --- | ---: |
| logical record ID | `b"radjax.tome.v3.logical-record-id.v1"` | `7261646a61782e746f6d652e76332e6c6f676963616c2d7265636f72642d69642e7631` | 35 |
| authority identity | `b"radjax.tome.v3.semantic-authority.v1"` | `7261646a61782e746f6d652e76332e73656d616e7469632d617574686f726974792e7631` | 36 |
| policy identity | `b"radjax.tome.v3.behavioral-policy.v1"` | `7261646a61782e746f6d652e76332e6265686176696f72616c2d706f6c6963792e7631` | 35 |
| record sequence | `b"radjax.tome.v3.record-sequence.v1"` | `7261646a61782e746f6d652e76332e7265636f72642d73657175656e63652e7631` | 33 |
| semantic root | `b"radjax.tome.v3.semantic-root.v1"` | `7261646a61782e746f6d652e76332e73656d616e7469632d726f6f742e7631` | 31 |
