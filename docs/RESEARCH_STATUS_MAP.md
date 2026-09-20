# Research and production status map

This human-readable map is derived from docs/hydra_disposition.json, which
remains the authoritative machine inventory. This page resolves the ledger
into reader guidance; it is not a competing registry.

| Area / command family | Status | Public path? | Successor or evidence | Limitation |
| --- | --- | --- | --- | --- |
| Mainline build/validate/inspect/package/doctor | canonical/supporting | yes | CLI_GUIDE.md, M9 closure | legacy parser remains beneath executable |
| Corpus-v2 build/validate/inspect | canonical | yes | CORPUS_BUILDER.md, M10 closure | v1 compatibility remains; tokenizer binding is strict |
| Corpus split/tokenization utilities | research-frozen | no | Hydra records and historical scripts | not equivalent to corpus-v2 production |
| Canonical C1-C5 corridor path | canonical | internal production | corridor docs and M4/M5 evidence | expert stage commands are not normal user path |
| Expert C2/C3/C4/C5 commands | remove-after-parity | research/compatibility only | Hydra command records | complete reserves remain production-owned |
| Golden tools | supporting/canonical | engineering only | cli:golden and GOLDEN_1K_CAPTURE | offline fixtures, not teacher production |
| M8 selected-pass diagnostics | research-frozen | no | M8A-M8G evidence and closure | historical measurements; batch 1 remains canonical |
| M8 compact delivery | canonical/supporting | yes through build | compact delivery and C6 docs | rejected optimization candidates are preserved |
| M8 replay/finalization utilities | research-frozen | compatibility only | workload-finalization evidence | not a new workload source |
| Generalized fingerprint artifacts | research-frozen | no | fingerprint docs/scripts | not canonical Path-B artifact interface |
| HF specimen/export/Qwen policy experiments | research-frozen | no | historical experiment docs | no public support claim |
| Multi-GPU Path B | research-frozen | no | MULTI_GPU_PATH_B.md | experimental scheduling harness |
| Capability proofs/model utilities | supporting/research-frozen | engineering only | doctor/parity/model docs | capability proof is not a production build |
| Legacy TeacherTextbook/fake routes | compatibility-only | research only | legacy scripts and Hydra records | not canonical teacher production |
| Historical migration/roadmap inventories | research-frozen/supporting | reference only | ROADMAP.md and archive pointers | do not rewrite historical reports |
| M10 corpus lifecycle/storage/identity | canonical | yes through corpus | M10 closure and corpus docs | scratch/lifecycle guarantees are bounded |

## Status meanings

Canonical is the normal supported path. Supporting is a shared validator or
utility used by that path. Compatibility-only is retained for fixtures or
migration. Research-frozen is preserved evidence or an experiment, not a
current support claim. Remove-after-parity has a retirement condition; it is
not permission to delete the command today.

Historical reports are intentionally not rewritten as current manuals. M8
accepted evidence records that selected batch 1 remains canonical; rejected
batching, suffix-truncation, and sparse-projection experiments are research
evidence only.

## Current command grammar

The installed entry point is radjax_tome.cli.main:main. The normal public
surface is build --config, corpus build/validate/inspect, validate, inspect,
package, doctor, and the optional tui corpus/production surface. research is
an explicit compatibility namespace, not a second producer. The old parser
remains available for historical forms; new documentation must use the
unambiguous canonical grammar.

The canonical production owner production-build, linkage audit, corridor
stages, and package/transport helpers are contributor-facing internals or
supporting engineering surfaces unless listed as normal commands above.
Expert C2/C3/C4/C5 commands are remove-after-parity and must not bypass the
canonical lifecycle. Golden capture/compare/validate is supporting offline
contract tooling, not a teacher run.

## Research evidence boundaries

M8 evidence is workload- and hardware-scoped evidence, not a capacity promise.
The accepted DuckDB C2 scale evidence belongs to the production C2
implementation, while its scratch database is rebuildable and is not a
Contract artifact. Selected-pass batching, suffix truncation, and sparse
selected-logit projection remain rejected or noncanonical experiments; the
full-length batch-1 path is the current authority.
