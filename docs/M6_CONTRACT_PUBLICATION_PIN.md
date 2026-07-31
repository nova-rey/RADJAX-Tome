# M6 Contract Publication Pin

The normative portable contract is RADJAX-Contract release `0.2.0`, Git tag
`v0.2.0`, commit `147ca371c78a98dbc82de1ea93deb4f3ae27f399`.

`pyproject.toml` pins the development verification dependency to `v0.2.0`.
The checked-in `contracts/radjax_tome/v1` tree is an offline-capable verified
mirror, not an independently editable authority. Its `SHA256SUMS` inventory
is validated in Tome tests. Release verification additionally compares all
mirror file hashes with Contract source assets and installed wheel assets.

Production writer and CLI modules do not import RADJAX-Contract for M6. The
pin is limited to contract verification, conformance, packaging, and
development/testing.
