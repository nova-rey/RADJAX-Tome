# Native-v3 Student-Consumption Contract Pin

The authoritative new-production native-v3 Student-consumption v4 contract is
RADJAX-Contract `v0.6.0`, commit
`b1209f21fef9405776a757f1a5749d3152bbc3c6`. Tome pins its existing
RADJAX-Contract dependency to that immutable commit in `pyproject.toml`.

`contracts/radjax_tome/student_consumption/v4` is a checked-in,
offline-capable byte-for-byte mirror of the Contract release assets. It is
strictly a verification mirror: `SHA256SUMS` and parity tests reject any drift
from the pinned Contract source tree. The mirror is not a normative source and
does not create a runtime import dependency for production writers or the CLI.

Published v2 and v3 remain available only for historical validation. V4
replaces them for new native-v3 Student-consumption artifacts. It keeps the
native-v3 base semantic root intact and keeps delivery-receipt and
authority-reference sidecars integrity-bound and Contract-validated without
including their body digests in the batch-semantic consumption identity. There
is no v4-to-v3 or v4-to-v2 fallback.
