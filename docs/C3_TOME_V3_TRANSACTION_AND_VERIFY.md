# Program C: v3 verification and transactional publication

This document records the Tome-side boundary for the released
`radjax_tome_artifact_contract@3.0.0` adoption. It does not change the v2
default, the released Contract, or Student.

## Verification modes

`radjax-tome verify-artifact` is a thin adapter over the Contract v3 APIs. Its
default mode is `standard`; it validates the artifact's own cover, manifest,
indexes, shards, semantic sequence, and semantic root. `governed` additionally
requires an expectation file supplied outside the artifact. `external-attestation`
requires an independently supplied attestation, an explicit availability policy,
and an evaluation time. No mode discovers comparison evidence from the artifact
under test. Contract phase and issue codes remain authoritative; Tome maps the
requested mode to exit codes 0, 2, 3, 4, 5, and 6.

## Private journal boundary

The publisher uses same-filesystem private staging and a private journal. It
records only the transaction identity, v3 configuration identity,
semantic-authority identity, sealed shard receipts, contiguous committed
selection range, completion intent, and promotion marker. Shard bytes are
fsynced before the receipt state is written; receipt state is fsynced before the
committed range and completion states. Journal files and staging are removed
only after both independent transactions reach `PROMOTED`. They are never
inventoried, archived, included in a semantic identity, or required by
ordinary consumers or Student.

Directory and archive states are stored in separate private journal records;
the archive transaction never overwrites the directory transaction's durable
`PROMOTED` marker.

The Contract state machine is used without reinterpretation:

`OPEN` → `SEALING` → `OPEN` (committed sealed prefix) → `COMPLETE_INTENT`
→ `PROMOTING` → `PROMOTED`.

The publisher exposes test-only fault boundaries PC39–PC47 at each durable
transition. An injected interruption preserves the private state exactly as a
real process crash would; the conformance tests inspect the state and then
quarantine it. A new run refuses to start while matching private staging or a
journal remains, rather than silently mixing runs or deleting unreceipted
bytes. Recovery of a directory-promoted/ archive-incomplete result is explicit
through the archive-resume adapter and validates the directory before creating
or promoting the archive.

An incomplete or stale state is not publicly visible. A fresh directory
publication refuses any surviving private transaction; the explicit archive
resume adapter first validates an already visible directory before producing
and promoting the archive. Cross-authority, mixed-run, unreceipted, or otherwise
invalid journal state fails closed through Contract validation. Directory and archive
publication are separate transactions; each is validated and promoted
independently. A report may bind both to the same semantic root, but does not
claim atomic visibility of the pair.

The publisher validates a complete staged package before directory promotion,
fsyncs the parent directory after promotion, then constructs and validates the
archive in its own transaction *before* archive promotion. Archive bytes are
fsynced and promoted with an atomic no-replace link; directory promotion uses
same-filesystem rename under a per-target exclusive lock and target check. The
lock serializes cooperating Tome publishers; this adapter does not claim to
exclude an unrelated writer that ignores the lock, so callers must provide
exclusive ownership of the output base when the filesystem lacks a native
directory no-replace primitive. Existing targets, stale private state, and
partial/unreceipted state are rejected before a new publication. If directory
promotion succeeds but archive
promotion fails, the directory remains explicitly visible as a partial result
and the caller may invoke archive resume; no report may call the pair atomic.
Filesystem transports that cannot provide the required local staging, fsync,
and no-replace assumptions are not represented as successfully promoted by this
adapter.

Archive tar and `.rtome` transport metadata is normalized (member order,
mtime, uid/gid, names, and mode; gzip mtime and filename) so repeated builds
from identical logical handoffs reproduce identical transport bytes and raw
receipts.

## Scope

The v3 projection consumes the finalized typed selected-record handoff after
late corridor linkage. It does not reread emitted v4/staging files, repeat
score or selection logic, alter fixed 25-field authority projections, change
historical v2/v4/v5/v6 paths, or make a performance claim.

For explicit v3 production, the handoff retains the already-materialized closed
selected payload values long enough for the v3 projection. This is a typed
handoff extension of the same selected-rerun owner, enabled only for v3; it is
not a second selection or assembly path and it does not reread the staged JSON
files. The v2 path retains its prior summary-only behavior.
