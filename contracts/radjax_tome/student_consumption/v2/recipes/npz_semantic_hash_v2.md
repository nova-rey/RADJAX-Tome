# NPZ Semantic Hash v2

NPZ members sort by UTF-8 member name.  For each member, hash a length-framed
sequence of member name, declared dtype, rank, dimensions, declared axes, and
canonical little-endian C-order value bytes.  Frame each text or byte segment
with an unsigned 64-bit little-endian length.  Object arrays, duplicate members,
noncanonical dtype declarations, and undeclared members are invalid.

This yields the derived resource `semantic_digest`, independent of the NPZ
filename, archive member path, manifest location, transport, and raw inventory
SHA-256.  The raw inventory SHA-256 separately protects delivered bytes.
