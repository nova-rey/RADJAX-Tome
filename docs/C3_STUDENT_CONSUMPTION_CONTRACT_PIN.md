# Native-v3 Student-Consumption Contract Pin

The authoritative native-v3 Student-consumption v2 contract is
RADJAX-Contract `v0.4.1`, commit
`a6877178d5f07d68f5e0bc28419d0e8e1a58890e`. Tome pins its existing
RADJAX-Contract dependency to that immutable commit in `pyproject.toml`.

`contracts/radjax_tome/student_consumption/v2` is a checked-in,
offline-capable byte-for-byte mirror of the Contract release assets. It is
strictly a verification mirror: `SHA256SUMS` and parity tests reject any drift
from the pinned Contract source tree. The mirror is not a normative source and
does not create a runtime import dependency for production writers or the CLI.

The v2 profile replaces the unusable v1 sidecar binding rule for new
native-v3 Student-consumption artifacts. It keeps the native-v3 base semantic
root intact, binds independently semantically-digested derived resources, and
uses Contract-owned validation and conformance only. Historical Contract
assets, M6/M7 mirrors, production behavior, and archive behavior are unchanged
by this pin.
