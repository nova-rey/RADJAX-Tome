# M6 Production Boundary Ownership

Status: **M6A characterization baseline**
Implementation base: `a7109c25b78780dd64eae6a923a33aac2a2a86b7`

This record governs the roadmap's production-module-boundary M6.  It is
distinct from the completed inserted contract-publication work, whose v3
assets remain pinned to RADJAX-Contract `v0.2.0` and mirrored offline in this
repository.

## Canonical ownership

| Responsibility | Canonical owner at M6A | Canonical handoff | M6 disposition |
| --- | --- | --- | --- |
| Public production request | `builder.production` and `builder.config` | normalized request, resolved config, execution plan | retain façade and M5 boundary |
| State machine | `builder.native_path_b` | existing `StageResult`, evidence, slice executions, terminal result | retain as the sole orchestrator |
| Preflight and score pass | `builder.production` with corpus, provenance, backend, and report services | typed M4 stage evidence | extract callback implementations only |
| Corridor, authority, and selection | fingerprint modules, `builder.c6_integration`, and production C6 glue | existing C2--C5 artifacts and authority evidence | retain semantic owners; extract production glue |
| Selected rerun and payload delivery | `builder.exemplar_delivery` | `SelectedRerunHandoff`, selected-linked corridor evidence, delivery report | split behind a stable forwarding module |
| Assembly, validation, reconciliation | production, teacher validation, and selected-linkage audit | existing assembly and validation handoffs | extract stage callbacks without changing reports or paths |
| Package/profile/archive emission | `tome.packaging` and `tome.bundle` | validated canonical Tome artifact | make packaging consume validated Tome-side descriptor, never builder internals |
| Validation and inspection | `tome.contracts`, cover/package/bundle validators, audit | reusable validation report/descriptor | consolidate Tome-side façade; no Student import |
| Reporting | `reports` plus production progress/report functions | report payloads and sidecars | move producer reporting helpers out of façade |

## Dependency policy

The only permitted direction is:

```text
CLI, research, compatibility, tests -> public facades -> native Path-B state machine
    -> stage services -> shared domain primitives and IO
```

`native_path_b` may depend only on its own contracts and injected callbacks;
it must not import `builder.production`, `cli`, research modules, or optional
ML frameworks.  Stage services must not import CLI or research entry points.
Tome packaging and inspection may consume Tome validation/descriptors but must
not import `builder`.  Research and compatibility modules point inward through
stable facades; they never become a production dependency.

The package initializers identified by the Hydra record remain intentionally
tracked violations until M6E: `backends.__init__`, `builder.__init__`, and
`reports.__init__`.  Their existing lazy compatibility exports remain public
until replacement/removal conditions are recorded; M6E will narrow them
without dropping supported symbols.

Production callers may use the `radjax_tome.tome` façade for `write_cover_page`;
they must not import Tome implementation leaves such as `contracts`,
`canonical_artifact`, `cover_page`, `packaging`, `bundle`, or `compatibility`.
The currently direct CLI `cover_page` import is an explicitly recorded M6E
remediation, not permission for new leaf imports.  Historical v2/v1 adapters
remain Tome-side diagnostics and never enter a writer path.

## Protected behavior

M6 preserves the M5 intent -> resolved configuration -> execution-plan
boundary; the M4 five-slice order (including separate provisional early and
selected-linked late corridor operations); the fixed 25-field selection
authority projection; authority v1/v2; all v3 identity/profile/transport
semantics; historical non-inferential adapters; Contract `v0.2.0` pin and
offline mirror; Golden fixtures; artifact paths/schemas; CLI compatibility;
and progress/report behavior.

The architecture tests introduced with this record characterize public façades
and native-boundary isolation.  Later M6 checkpoints strengthen them only
after the corresponding dependency edge has actually been eliminated, so a
test never asserts an architecture that the checked-in code has not reached.

M6C owns the first delivery split: `builder.exemplar_delivery_contracts` is the
single definition of delivery configuration, delivery failures, and the
`PreparedSelectedDelivery` handoff.  `builder.exemplar_delivery` re-exports
those exact objects while retaining the established public functions and
implementation body.  This is intentionally a type extraction, not a second
delivery algorithm or altered transactional behavior.

## Explicit deferrals

M6 does not redesign v3 contracts, authority recipes, production policy, CLI,
corpus building, payload sharding/streaming validation, selected-pass
performance, TUI, model behavior, Student, or RADJAX-Contract.  M7 owns
sharding/streaming work; M8 performance; M9 public CLI; M10 corpus building;
M11 TUI; and M12 broad documentation work.
