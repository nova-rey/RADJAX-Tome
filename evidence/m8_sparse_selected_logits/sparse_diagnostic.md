# M8 sparse selected-coordinate diagnostic

The audit found dense vocabulary projection before selected-row gathering. The implementation now uses the official `logits_to_keep` interface, maps the returned union rows back to the frozen coordinates, and records sparse projection counts in backend metadata. Contract bytes and reducer code were not changed.

The T4 application `ap-BdSjE78YtTVlsLRomb1QYq` completed the dense/sparse production publication sequence. Dense sample timings captured were 23.8302 s, 24.7104 s, and 24.3984 s. Sparse timings captured were 24.0413 s and 23.8895 s; the third sparse result completed but its returned JSON was lost to console truncation. Both implementations passed selected-source parity/publication, but the harness did not preserve per-coordinate canonical bodies or a logical-evidence root, so exact 71-coordinate equivalence cannot be claimed.

The benchmark evidence is therefore insufficient for the adoption criteria. No sparse machinery is shipped as canonical on this checkpoint; no Contract change was made. The production change and focused tests are preserved for the next bounded evidence-capture attempt.
