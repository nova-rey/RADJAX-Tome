# M8A T4 Baseline Receipt

This receipt commits content-addressed, sanitized evidence for the private M8A
T4 run. Raw reports remain outside the repository because they contain private
machine paths and detailed runtime observations. Their SHA-256 digests below
allow an authorized reviewer to verify the exact source bytes without treating
the prose summary as the evidence itself.

| Evidence | SHA-256 |
| --- | --- |
| Canonical production report | `43e4bdfe93012b21569d058939e51ca68b42dd08934112461c2640f0e410d77f` |
| Canonical validation report | `a737dcc7ae371946ca8083325fff2d449f865294f6295a84165063abd95f5733` |
| Selected-linkage audit | `34a1472dfb521b917192373919ba42c3a4e5934babc52a726bda4b388114ba8b` |
| Immutable post-C5 manifest | `0ea34090bf7b2e6331644d2eaa57b738c771ca48e62dc76527878a6b9b915a39` |
| M8A raw baseline report | `1330e9b19f328af4a07b755349b8d3aa6a95c2d9344c2901389295d967d21013` |

The canonical production report is `pass`; it records a 256-record Tome with
semantic identity
`sha256:9d5796a7ff2d4db5000b9a128502cc68d0933f13be4786c4cefc1982a615dea7`.
The post-C5 checkpoint digest is
`sha256:a850ad85f3a0ca95a000dda239ff7f46296de7b411fe5db20b844bfa78ad6e12`
and binds 43 files. The raw baseline report contains the three complete
cap-eight replays plus the 64-source cap matrix, its writer/checkpoint/accounting
assertions, and the exact cross-cap comparison.

This receipt does not promote a benchmark artifact, change production behavior,
or authorize M8B. It merely makes the external evidence auditable while
preserving the no-private-path repository boundary.
