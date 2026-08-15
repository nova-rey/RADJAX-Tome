# M8E Selection-System Audit

This is a read-only audit branch from Tome `8bafe2ec123d7462ff43a44a54b5420d8c21c665`. It does not change selection, payloads, Contract authority, Student, Golden evidence, or defaults. Terminology follows the supplied official TOME Generation Glossary: the score pass is the corpus-wide compact-measurement pass; a corridor is one numbered behavioral lane; a corridor leaderboard is a ranked candidate list within one corridor; a global leaderboard is the corpus-wide ranked candidate list; and an exemplar is the high-resolution output at one frozen selected coordinate.

## Authority and evidence

Contract authority remains `radjax-contract 0.9.0` at commit `1fa43e1aea2e198511db86dafb0aeefa525d48c7`. The retained M8D report is the source for the 256-coordinate dynamic-K anatomy. The retained v6 smoke fixture at `tests/fixtures/native_v3_student_v6_smoke/producer_artifact.v4` is the available committed C2-C6 selection artifact. No committed production score-pass artifact for the 256-coordinate M8 workload contains a governed full-width flag, so a real-workload 1/3 composition was not asserted or fabricated.

The deterministic audit analyzer is `scripts/audit_selection_system.py`; its generated report is `docs/evidence/M8E_SELECTION_AUDIT.json` (SHA-256 `sha256:4c95936cd359bd91cd18abc35480c5b77cf1aa435a5009bc80bff1c21fee2fa6`). It reads retained artifacts and creates only in-memory synthetic policy fixtures. It never invokes a producer or writes an artifact.

## Canonical call graph and path inventory

The active Path-B route is:

```text
production
  -> C1 score pass
  -> authorities.export_c6_fingerprint_selection_authority
  -> build_exemplar_selection_manifest(production_global_selector=True,
       canonical_score_fields_only=True, use_score_pass_fields=True)
  -> export_production_global_board_supply
  -> C2 build_corridor_candidate_leaderboards
  -> C3 allocate_corridor_coverage
  -> C4 claim_corridor_then_backfill_global
  -> C5 build_multi_role_selected_exemplars
  -> selected-source rerun / delivery / payload staging
```

The canonical production implementation is in `builder/production_stages/selection.py`, with selector mechanics in `builder/exemplar_selection.py`, C2 in `fingerprint/corridor_leaderboards.py`, C3 in `fingerprint/corridor_budget.py`, C4 in `fingerprint/corridor_claims.py`, and C5 in `fingerprint/multi_role_selection.py`. The CLI exposes the C2/C3/C4 seams as offline commands. The preserved C2-C5 modules and tests are compatibility/research seams consumed by the canonical path, not alternate producers. Golden and fixture builders are characterization/fixture paths; they do not replace the canonical C6 route.

No literal `Phil Connor` implementation, schema, or test exists. However, full-width candidates have a semantic post-selection route: `artifact_validation/long_tail.py:selected_board_for_long_tail()` maps `FULL_VOCAB_OR_NEAR_FULL_VOCAB` and `SUSPICIOUS_FLAT` to `perverse_tail_diagnostic` unless `include_perverse_tail_in_primary=True`. Defaults are `include_perverse_tail_in_primary=False`, `include_perverse_tail_in_student=False`, `perverse_tail_side_board_cap=32`, and `reject_perverse_exemplars=False`. Thus full-width exemplars are not rejected by default, but they are routed to a diagnostic side board and may be excluded from the Student profile. M8D field observations independently show K=38 samples with `selected_board=primary` and K=262,144 samples with `selected_board=perverse_tail_diagnostic`. This is the concrete conflict with the authorized target of ordinary leaderboard competition.

## Current leaderboard semantics

### Legacy/global selector

`builder/exemplar_selection.py:_Board` ranks descending by score, then `example_id`, selected position, and board ID. Each board has configured capacity and retains only a bounded `capacity * 4` runner-up pool. Candidates are sorted on every arrival, so arrival order does not change the retained pool while the pool is large enough; candidates discarded beyond the bounded pool cannot be used for later backfill. The active board families are `global_max_entropy`, `global_mean_entropy`, `low_confidence`, `high_tail_mass`, `high_effective_top_k`, position/length bucket entropy, and shard coverage. There is no full-width ratio and no separate named Phil Connor board here.

### C2/C3/C4 selector

C2 uses `CorridorLeaderboardPolicy(candidate_pool_cap=4)` by default and ranks corridor candidates by corridor-training utility, membership, centrality, useful difficulty, quality, candidate ID, and position. It deduplicates exact coordinates in SQLite and rejects conflicting duplicates. It rejects a coordinate assigned to more than one corridor. C3 allocates corridor slots breadth-first with an exact Decimal corridor fraction, but that fraction is a corridor-budget fraction, not a full-width-composition fraction. C4 claims corridor coordinates first, then processes global boards in `(priority, board_id)` order. Global candidates are ranked by explicit contiguous rank and backfilled around collisions/ineligible entries. C4 retains collision obligations and backfill lineage.

Both systems therefore have deterministic ranking and cross-board deduplication, but neither has the authorized final-composition full-width cap. C2 takes `candidates[:allocated_slots]` and has no post-dedup corridor backfill. C4 backfills only from bounded exported global supplies, not from a provably complete eligible pool.

### Deduplication and reasons

The uniqueness key is `(example_id, position)`, i.e. selected coordinate, not selected source. C4 appends global obligations to an existing corridor obligation on a collision; C5 emits one durable record and one payload key per coordinate while retaining all obligations, roles, corridor IDs, global board IDs, and rankings. C2 deliberately rejects multiple corridor assignments for one coordinate rather than merging them.

The smoke fixture contains 3 unique selected coordinates, 3 C5 records, and 2 selected sources; `corpus_000000003` contributes two distinct positions. It contains no corridor/global collision, so no multiply-reason coordinate is present in that fixture. Its C4 budget is 1 corridor plus 2 global slots, with no shortfall. The exported global supply has seven boards, requested capacity 16 each, and only 2–4 candidates per board. The legacy selector reports 7 boards, total capacity 112, runner-up multiplier 4, 4 duplicate candidates, and 0 backfill successes. The C4 diagnostic reports 20 within-role duplicate entries in the exported supplies, but accepted cross-role overlap is 0.

The legacy selector has a robustness gap: duplicate copies of the same coordinate supplied twice to one board can remain as two assigned entries before later example-level aggregation collapses positions. Canonical score extraction normally emits one coordinate per source row, and C2 rejects conflicting duplicate coordinates, but the future invariant needs explicit within-board duplicate tests.

## Dynamic top-K physical representation

M8D reports 256 coordinates, K minimum 32, median/p90/p95/p99/max 262,144, 145 full-width coordinates (56.640625%), and 38,129,241 retained entries. The reducer metadata explicitly declares `padding_policy=pad_to_dynamic_top_k_max_effective` and masked-slot semantics. In the M8 vocabulary-wide configuration, bounded field observations show that even K=38 stores vocabulary-length (262,144) arrays for `top_token_ids`, `top_probs`, `top_log_probs`, and `top_selection_mask`; full-width samples store the same lengths with all slots active. This is a physically padded selected-exemplar representation, not dense score-pass-logit retention. The score pass retains compact fields; the selected-source pass constructs these high-resolution arrays.

The current production score extractor binds `diagnostic_effective_top_k` in the canonical score-only projection, while the `high_effective_top_k` legacy board requests `effective_top_k`. That width-binding mismatch must be resolved before a future cap can classify real score-pass candidates without guessing. Compacting physical arrays would be a payload/schema decision and must preserve masked-slot semantics, reconstruction, raw identity, and Contract admission; it is not changed here.

## Exact 1/3 simulation

The exact policy formula used by the audit analyzer is `max(1, floor(N*numerator/denominator))`. The synthetic complete eligible pool has capacity 6, five full-width candidates ranked above five narrow candidates, deterministic coordinate tie-breaks, and reversed arrival order. Results are:

| policy | exact allowance | full-width retained | filled |
| --- | ---: | ---: | ---: |
| 1/4 | 1 | 1 | 6 |
| 1/3 | 2 | 2 | 6 |
| 1/2 | 3 | 3 | 6 |
| uncapped | — | 5 | 6 |

The 1/3 result is `{f1,f2,n1,n2,n3,n4}`. Reversing candidate arrival order produces the same coordinate set. A later higher-ranked full-width candidate replaces a worse retained full-width candidate because final composition is selected from the complete sorted pool rather than reserving slots on arrival. These are audit fixtures, not changed production outputs. No real M8 composition simulation was claimed because retained score-pass records lack governed full-width metadata.

The exact rational must be represented as `{numerator: 1, denominator: 3}` (or an equivalent integer pair), validated positive and reduced, and applied per final leaderboard. `max(1, floor(N/3))` gives one allowance for N=1–5, two for N=6–8, and so on. The cap is not a global union cap and must not be an arrival-order reservation.

## Configuration and authority implications

The logical home is the selection-policy configuration alongside `selected_exemplar_budget` and board policies, but the current fixed 25-field `selection_integration_hash()` omits long-tail routing and any full-width ratio. A ratio changes frozen selected coordinates, so it must enter selection authority, checkpoint compatibility, resume validation, reports, and source/selection provenance. It cannot safely be an unbound CLI/config field. A versioned Tome authority/schema addition may be possible, but Contract impact must be reviewed because the selection authority and public records are Contract-facing. Student profile behavior must explicitly state whether the ratio is inherited, projected, or rejected. Older checkpoints should reject a changed ratio rather than silently resume.

Future field proposal (not implemented): `full_width_composition_cap: {numerator: 1, denominator: 3}`, default enabled only under a new selection-policy version; exact integer validation; `max(1, floor(N*numerator/denominator))`; emitted in selector manifest, board reports, C6 authority, checkpoint compatibility, and receipts; changing it changes selection authority and semantic identity. Full-width candidates remain ordinary candidates and only final composition is constrained.

## M8 relationship and future plan

The frozen 256-coordinate M8 workload remains the engineering-performance comparison. A future policy simulation must not be compared as an implementation speedup because it changes selected coordinates and payload composition. The current M8 evidence shows full-width payloads can dominate bytes, but that is a workload-composition observation, not permission to cap or compact them.

Minimal independently reviewable future checkpoints:

1. **Authority/schema checkpoint:** bind exact rational cap and width classification to score-pass candidate authority; decide Contract/schema impact and checkpoint rejection rules.
2. **Pure leaderboard checkpoint:** implement complete-pool final composition, live full-width ranking/displacement, coordinate deduplication, and deterministic backfill; add adversarial order, duplicate, invalidation, and exhaustion tests.
3. **Integration checkpoint:** preserve all selection reasons and one coordinate/one exemplar semantics through C2-C6, source passports, resume, and reports; prove old artifacts remain unchanged when policy is absent.
4. **Packaging/measurement checkpoint:** verify ordinary and full-debug profiles, M8 fixed-coordinate performance separately, and any payload-size composition effect without changing dynamic-K meaning.

No production change, cap, Phil Connor path removal, dynamic-K change, payload truncation, Contract change, Student change, or optimization was made in this audit.
