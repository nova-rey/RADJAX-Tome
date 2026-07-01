# RADJAX-Tome Project Ledger

Earlier history in this root ledger was reconstructed from current repository
state because no root `bible.md` existed when Spec 3.1 landed. Existing
historical notes remain in `docs/BIBLE.md`; future spec commits should append
here unless a spec explicitly says not to.

## 2026-07-01 — Cleanup Arc Catch-Up And Spec 3.1 Cover Page

The cleanup arc from 2.14 through 2.18 is complete on `main`: archive/mainline
hygiene, public CLI happy path, shared report rendering and thin capability
script, narrowed fingerprint API boundary, and shared test fixture helpers are
all represented in the current repository state.

Spec 3.0 locked the post-cleanup roadmap and preserved the historical
optimization handoff in repository-local docs and deterministic inventory JSON.

Spec 3.1 implements the first unpacked Tome directory front door:
`cover_page.json`. Fake/offline builds now write the cover page beside existing
TeacherTextbook sidecars, public validation checks it when present, and inspect
prints cover-page summary fields. This does not implement the Spec 3.2 bundle
container, compression layer, dynamic top-k, or CPU/GPU/TPU runtime modes.

## 2026-07-01 — Spec 3.2 Tome Bundle Container

Spec 3.2 adds `.rtome` as a deterministic tar bundle for moving and storing an
unpacked Tome directory as one file. The public CLI now supports `pack`,
bundle-aware `inspect`, bundle-aware `validate`, and safe `unpack`.

The bundle is packaging only: it keeps `cover_page.json` at archive root, packs
the cover-page-listed files, validates hashes and sizes without extraction, and
does not impose a compression requirement. Dynamic top-k and backend runtime
optimization remain future Spec 3 arcs.
