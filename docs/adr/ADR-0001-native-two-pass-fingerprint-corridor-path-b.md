# ADR-0001: Native Two-Pass Fingerprint-Corridor Path B

**Status:** Accepted for M1 canonization
**Date:** 2026-07-12

## Decision

RADJAX-Tome's product mainline is the native, single-GPU, two-pass
fingerprint-corridor Path B pipeline. It scores the corpus once, establishes
fingerprint-corridor selection authority, and reruns only the selected source
examples to produce compressed student-facing payloads.

Fingerprint and corridor are one product-domain concept. They are not peer
product stages and future user-facing architecture, schemas, and commands must
use the combined term consistently.

Research is preserved as valuable history, but is not a peer supported
mainline. Its disposition and preservation conditions are recorded in
`docs/hydra_disposition.json` and `docs/RESEARCH_STATUS.md`.

## Golden Behavioral Lock

Structural refactoring is governed by the verified T4 golden 1K contract:

- 1,000 examples at sequence length 128 and vocabulary 262,144;
- 47 fingerprint-corridor modes and 34,220 assigned positions;
- 128 corridor and 128 global claims yielding 256 unique coordinates;
- 2 cross-role overlaps and approximately 0.001776 Jaccard;
- 214 source examples rerun in 27 batches of 8;
- one full pass, one native selected rerun, and zero legacy reruns;
- 256 indexed selected payloads; and
- passing delivery, validation, strict linkage audit, reconciliation, and
  final production status.

## Consequences

M1 makes this direction explicit without changing runtime behavior. M2 freezes
the behavioral fixture; M3 preserves research history; and M6 removes known
package-initializer dependency-boundary violations. The current initializer
imports are baseline facts recorded in the disposition ledger, not evidence
that research is a supported peer path.
