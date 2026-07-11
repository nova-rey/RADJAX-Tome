# C1 Corridor Archetype Scoring

`radjax_tome.fingerprint.corridor_archetypes` is a pure CPU scoring primitive
for the fingerprint-corridor representative leaderboard arc. It answers whether
one already-assigned candidate position is sufficiently central to its corridor
core and, only after that eligibility decision, computes a bounded training
utility.

Eligibility always precedes difficulty. A weak membership, invalid assignment,
unsupported mode, invalid position, or out-of-core candidate receives
`corridor_training_utility = null`; a high difficulty value cannot rescue it.
Stable reason codes are emitted in this order where applicable:
`invalid_position`, `unassigned_corridor`, `invalid_assignment_status`,
`nonfinite_feature`, `feature_out_of_range`, `membership_below_minimum`,
`outside_corridor_core`, and `mode_support_below_minimum`.

The default policy uses membership/core thresholds of `0.5` and `0.5`, minimum
mode support `1`, and raw weights `0.45` membership, `0.40` centrality,
`0.15` useful difficulty, and `0.0` optional quality. Weights are normalized
before scoring. Membership, centrality, difficulty, quality, and the final
utility are all bounded in `[0, 1]`; core centrality is `1 - distance / maximum`
and therefore decreases monotonically as distance grows.

`CorridorCandidateFeatures.from_mapping()` accepts current selected/corridor
field names. Current artifacts do not yet expose explicit membership or core
distance, so a linked assignment maps to full membership, zero core distance,
and support one as a documented compatibility proxy. Future corridor phases can
provide explicit compact statistics without changing the scoring API.

C1 is intentionally not connected to production selection or artifact emission.
Offline per-corridor micro-leaderboards begin in C2; budgets, global deduplication,
curriculum routing, and payload materialization remain later milestones.
