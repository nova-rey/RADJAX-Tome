# M8F sub-checkpoint 2: coordinate deduplication

The legacy board pool now collapses repeated selected coordinates before
winner assignment, retaining the best governed rank. This closes the
within-board duplicate gap without changing cross-board reason preservation or
selected-source multiplicity. Focused tests cover duplicate replay and the
existing deterministic selector suite.

The broader C2-C6 complete reserve/backfill redesign remains separate work:
the current corridor and global supplies are still bounded and this commit
does not claim complete-pool exhaustion guarantees. No production fixture,
Contract, Student, or Golden evidence changed.
