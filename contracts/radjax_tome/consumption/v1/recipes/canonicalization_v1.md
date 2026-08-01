# Student-consumption canonicalization v1

UTF-8 JSON uses RFC 8259 values, sorted object keys, compact separators, no
NaN/Infinity, and a trailing LF only for stored JSON documents. JSONL applies
the same rule independently to each nonblank record. Semantic digests use the
existing Tome v3 recipe; this profile does not redefine Tome identity.

Logical IDs are stable role names. Fixed roles map exactly through the profile
table. Repeated families substitute a zero-based, five-digit decimal index in
both logical ID and inventory path. Ordering is fixed-role lexical order then
family index order. Physical placement, archive member order, package profile,
transport, timestamps, and Path A/Path B delivery provenance do not alter batch
semantics.

NPY/NPZ arrays are little-endian or native-endian NumPy encodings with
`allow_pickle=false`. Declared dtype and rank are exact; no implicit cast,
broadcast, transpose, padding, token shift, or loss-mask inference is allowed.
