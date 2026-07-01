# Spec 3 Roadmap

Spec 3 resumes RADJAX-Tome planning after the 2.14-2.18 cleanup arc. These
are roadmap arcs, not necessarily single-shot implementation specs.

## Arcs

| Arc | Title | Status |
|---|---|---|
| 3.0 | Optimization Handoff Inventory and Roadmap Lock | complete when this document and inventory land |
| 3.1 | Cover Page v1 for unpacked Tome directory | complete once cover-page generation and validation land |
| 3.2 | Tome Bundle Container v1 | complete once deterministic .rtome pack/inspect/validate/unpack land |
| 3.3 | Teacher Backend Runtime Modes: CPU, CPU+GPU, CPU+TPU | planned |
| 3.4 | Dynamic Top-K Compression Policy | planned |
| 3.5 | Final CLI Polish / Optional TUI | planned |

## Recommended Ordering

1. Bookmark optimization handoff.
2. Implement cover page for unpacked Tome.
3. Implement bundle container.
4. Implement backend runtime modes.
5. Implement dynamic top-k.
6. Polish CLI / optional TUI.

## Ordering Rationale

Cover page defines the artifact contract before optimized generator machinery
targets it. Backend runtime abstraction should exist before porting
CUDA-specific optimizations. Dynamic top-k should wait until artifact metadata
and backend reduction policy surfaces exist. TUI is optional polish after
functional readiness.

Spec 3.1 adds `cover_page.json` for unpacked Tome directories so optimized
generation can target a contract-shaped artifact instead of forcing the contract
to follow an optimization-specific layout later.

Spec 3.2 adds `.rtome` bundle v1 as a deterministic tar packaging layer around
the cover-page-described files. It is packaging, not a new semantic compression
policy.

## Spec 3.3 Mini-Roadmap

Spec 3.3 is split into runtime/backend roadmap units:

| Unit | Title |
|---|---|
| 3.3A | Runtime Mode Capability Model |
| 3.3B | Backend Contract + Registry Skeleton |
| 3.3C | CPU Reference Backend |
| 3.3D | CPU Orchestration Modes: auto / serial / staged |
| 3.3E | HF Torch Backend Behind the Contract |
| 3.3F | GPU Compact Reduction Migration |
| 3.3G | TPU/JAX Backend Skeleton |
| 3.3H | Runtime Metadata + CLI/Doctor Polish |

3.3A defines vocabulary.

3.3B defines the internal backend contract wall.

3.3C establishes the CPU correctness baseline.

3.3D adds CPU orchestration and staged execution.

3.3E moves current HF Torch behavior behind the contract.

3.3F ports GPU compact/chunked reduction after the contract exists.

3.3G adds TPU/JAX shape without CUDA assumptions.

3.3H exposes backend status through CLI/doctor polish.
