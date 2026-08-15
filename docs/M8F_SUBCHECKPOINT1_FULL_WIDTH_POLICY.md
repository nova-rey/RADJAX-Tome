# M8F sub-checkpoint 1: exact full-width policy

This checkpoint adds a Tome-owned exact-rational policy primitive for ordinary
leaderboard composition. The authoritative representation is the reduced
integer pair `{numerator: 1, denominator: 3}` and the allowance is
`max(1, floor(N*numerator/denominator))`. Selection is performed from the
complete ranked pool, so arrival order cannot reserve capped slots; duplicate
coordinates are collapsed before ranking and a better full-width candidate can
replace a worse one.

Full-width exemplars are now treated as ordinary eligible exemplars by the
long-tail board classifier and Student-profile inclusion defaults. The
diagnostic class remains available for observability. The policy pair is bound
into the production selection integration hash. No Contract or Student source
was changed; the existing Contract authority remains 0.9.0 at
`1fa43e1aea2e198511db86dafb0aeefa525d48c7`.

This is only the policy/authority primitive. Wiring the capped composition into
the C2-C6 ranked candidate flow, complete reserves, and deterministic backfill
is the next sub-checkpoint. No production fixture or Golden evidence changed.
