# M6 Production Module Boundary Closure

Status: **corrective closure pending final commit**

Base: `a7109c25b78780dd64eae6a923a33aac2a2a86b7`

## Corrected claims

- `builder.production` is the stable public façade for compatibility,
  configuration normalization, canonical Path-B routing, and legacy entry
  points. Its substantive preflight, score-pass, authority, selection,
  delivery, assembly, verification, and reporting operations are owned by the
  corresponding `builder.production_stages` modules.
- `builder.native_path_b` remains the sole state machine. The stage package
  only supplies injected operations; it creates no alternate ordering, resume
  state, failure normalization, or evidence representation. In particular,
  provisional early corridors and selected-linked late corridors remain
  separate ordered operations.
- `builder.exemplar_delivery` is an import-compatible façade (49 lines).
  Selected rerun/staging, payloads, assembly, linkage validation, parity, and
  report rendering have concrete delivery-owned modules. The façade keeps
  private compatibility names while behavioral tests cover selected-only rerun,
  OOM reduction, transactional staging, exact-coordinate linkage, entropy
  tolerance, Path A/Path B distinctions, and late corridor ordering.
- `artifact_validation` is the one Builder-independent runtime owner for
  reusable TeacherTextbook, C6 selection, corridor, long-tail, and selected
  delivery/linkage validation. Builder compatibility paths forward to it where
  the primitive is shared. Tome packaging, inspection, archive validation, and
  producer validation cannot reach Builder, directly or transitively.
- Package materialization first validates the source adapter, then consumes a
  complete `ValidatedTomeArtifact` for its canonical output: root, cover,
  semantic identity, content manifest, profile/inventory, validation evidence,
  and authority references are all represented and validated.
- The v3 Contract `v0.2.0` pin and checksum-enforced offline mirror remain
  unchanged; no new production import of Contract publication assets exists.

## Ownership movement

| Surface | Before corrective work | Corrected ownership | Compatibility retained |
| --- | --- | --- | --- |
| `builder/production.py` | ~4,300 lines mixing stage implementations with façade routing | 3,073-line façade plus `production_stages/{preflight,score_pass,authorities,selection,delivery,assembly,verification,reporting}.py` | public build/configuration and legacy helpers |
| `builder/exemplar_delivery.py` | 3,441-line mixed delivery owner | 49-line façade plus focused `builder/delivery/*` owners | historic imports and private helper forwarding |
| package validation | packaging reached a lazy adapter that could reach Builder | `artifact_validation/*` shared leaves, `tome.producer_validation`, and typed descriptors | Builder validation imports forward where required |

The remaining large production façade intentionally retains public request
normalization, report compatibility assembly, finalization/resume compatibility,
and state-machine callback binding. Its removal is not an M6 goal.

## Mechanical enforcement and evidence

| Gate | Result |
| --- | --- |
| Production/delivery characterization | native Path-B, live execution, production, delivery, and adversarial linkage passed (95 targeted tests) |
| Validation/descriptor boundary | architecture graph, primitive, package, bundle, C6, TeacherTextbook, long-tail, delivery, and production coverage passed (targeted 148 tests) |
| Transitive graph policy | static graph reports complete offending paths; all Tome packaging/validation-to-Builder paths are absent; production-stage, delivery, and validation subgraphs are acyclic |
| Forwarding regression | a synthetic packaging -> forwarding module -> Builder path is detected and asserted |
| Public compatibility | historic `exemplar_delivery` imports and native Path-B characterization pass after owner-targeted test migration |
| Cache-free full suite | passed after the corrective graph and Hydra-inventory fixes |

The graph policy is deliberately stronger than a direct-import scan: it walks
the internal AST import graph, reports the complete route, and rejects a
forwarding module that would conceal a forbidden edge. Lightweight import and
CLI-help subprocess isolation remain separately covered by the focused M6
suite.

## Compatibility and deferrals

No v3 identity/manifest/profile/canonicalization semantics, fixed 25-field
selection-authority projection, authority v1/v2 recipe, Golden fixture, M4
ordering/resume behavior, artifact path/schema, public CLI command, Contract
pin, or Student boundary changed. Contract, Student, Golden fixtures, and
published v3 assets were untouched. No GPU/TPU execution, model inference,
fixture regeneration, Contract mutation, or Student work occurred.

Remaining forwarding façades are deliberate: `builder.exemplar_delivery`,
`builder.delivery._legacy`, and `builder.delivery.validation` retain public or
private compatibility imports until a public-surface review (no earlier than
M9). Package initializer compatibility registries retain existing exports.
M7 may add payload sharding and streaming validation, but must consume these
boundaries rather than reintroducing directory scraping, Builder-dependent Tome
validation, or a coequal production orchestrator.
