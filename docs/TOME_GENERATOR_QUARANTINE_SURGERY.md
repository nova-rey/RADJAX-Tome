# Tome Generator Quarantine Surgery

Spec 2.9 processes the Spec 2.8 quarantine as migration input, not runtime API.
Producer-side artifact schemas, JSON readers/writers, validators, provenance
records, inspection helpers, and report structures were promoted into active
`RADJAX-Tome` modules. Student runtime, training, evaluation, checkpointing, TPU,
and full-burn orchestration remain quarantined or out of scope.

The source of truth is
`docs/TOME_GENERATOR_QUARANTINE_SURGERY_LEDGER.json`.

## Ledger Counts

| Decision | Count |
| --- | ---: |
| `promoted` | 3 |
| `split_promoted` | 78 |
| `kept_quarantined` | 5 |
| `belongs_student` | 32 |
| `belongs_contract` | 2 |
| `deprecated` | 1 |
| `deferred` | 151 |
| `waived` | 1 |

Total quarantined paths processed: 273.

## Active Surfaces Promoted

- Fingerprint artifact manifests, byte accounting, summaries, loaders,
  validators, and inspection helpers under `src/radjax_tome/fingerprint/`.
- Exemplar reservoir records, corridor measurement reports, and aggressiveness
  calibration schemas.
- Source provenance, source/example-id joins, file hashes, and stable hashes.
- Real-teacher capture summaries with teacher identity, local-files-only, and
  source corpus references.
- Quality-per-byte, baseline comparison, and arc report dataclasses, JSON
  round-trips, and summary renderers under `src/radjax_tome/reports/`.
- HF specimen/export metadata boundaries under `src/radjax_tome/backends/`, with
  no default `torch` or `transformers` imports.

## Mixed Files Split

Mixed archived fingerprint, report, HF, and capture paths were classified as
`split_promoted` when producer-safe data structures moved into active Tome while
student/runtime execution stayed out. This includes old exemplar/corridor
passes, baseline/quality report runners, real-teacher capture and rehearsal
paths, and HF specimen/export smokes.

## Student, Contract, Deprecated, Deferred

Student-owned paths are recorded as `belongs_student` when the durable behavior
is training, student forward/eval, checkpointing, optimizer, or burn
orchestration. Contract-owned schema/compatibility paths are recorded as
`belongs_contract`. Historical archive-only material is `deprecated`, and
supporting docs/configs/fixtures not needed for the Spec 2.9 producer surface
are `deferred`.

## Remaining Quarantine Blockers

The ledger has no remaining quarantine entry marked as a Spec 3 blocker after
this surgery pass. That is not a claim that Spec 3 is globally unblocked:
Contract-valid Tome emission, `cover_page.json`, audit closure interpretation,
and explicit final gate checks remain later work.

## Spec 3 Gate Status

Spec 3 is allowed after the Spec 2.10.1 adversarial closure audit rerun. The
surgery ledger has no remaining quarantine entry marked as a Spec 3 blocker, and
`docs/TOME_GENERATOR_CLOSURE_AUDIT.md` resolves the prior
`src/qrwkv_xla/artifacts/fingerprint.py` active-equivalence concern by exact
symbol mappings into `src/radjax_tome/fingerprint/artifacts.py`. The committed
`docs/TOME_GENERATOR_CLOSURE_AUDIT.json` is a compact summary; the full detailed
closure JSON is generated under `artifacts/tome_generator_closure_audit/` and is
intentionally not committed.

## Next Step Recommendation

The closure gate is clear. Keep `cover_page.json` work scoped to the next Spec 3
phase; this surgery document does not implement it.
