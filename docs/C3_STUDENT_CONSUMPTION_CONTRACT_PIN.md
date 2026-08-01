# Native-v3 Student-Consumption Contract Pin

The authoritative new-production native-v3 Student-consumption v3 contract is
RADJAX-Contract `v0.5.1`, commit
`f9c9278b6a467a6ba7a3972e1644bfc3d13abd6b`. Tome pins its existing
RADJAX-Contract dependency to that immutable commit in `pyproject.toml`.

`contracts/radjax_tome/student_consumption/v3` is a checked-in,
offline-capable byte-for-byte mirror of the Contract release assets. It is
strictly a verification mirror: `SHA256SUMS` and parity tests reject any drift
from the pinned Contract source tree. The mirror is not a normative source and
does not create a runtime import dependency for production writers or the CLI.

Published v2 remains available only for historical validation. V3 replaces it
for new native-v3 Student-consumption artifacts because v2 did not close the
row-range, delivery-receipt, and authority-reference evidence bodies. V3 keeps
the native-v3 base semantic root intact, binds independently semantically-
digested derived resources, and uses Contract-owned validation and conformance
only. There is no v3-to-v2 fallback.
