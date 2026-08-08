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

The publisher uses same-filesystem private staging and a private journal. The
Contract-owned journal state is wrapped by the Tome-private
`radjax_tome_private_publication_binding_v2` object. That binding records the
transaction relationship, directory/archive names, staging and journal-root
ownership, configuration/layout identity, authority and policy identities,
semantic root, ordered-sequence identity, and expected sealed-shard shape. It
also records the explicit private topology (`canonical` or `archive_only`). It
is checked against the visible promoted directory and (when present) archive;
filenames, directory counts, and public-output presence alone never select a
resume candidate. Contract validates both journal state machines, receipt
shape/ranges, and transition legality; Tome owns filesystem discovery,
cross-object comparison, refusal/quarantine, and execution.

The private identity chain is deterministic and does not take its authority
from a journal field being compared with itself. The directory transaction ID
is `directory-v3-` plus the SHA-256 of a framed Tome-private identity object
covering the Contract/profile, output names, configuration identity, authority
identity, behavioral-policy identity, record count, and shard capacity. The
canonical archive transaction ID is derived from that directory ID by the
`canonical_archive` identity object. A fresh archive-only transaction instead
uses `archive-only-v3-` plus an identity object that additionally covers the
validated public semantic root and the `fresh_archive_only` kind. Journal-root
names and staging names are deterministic functions of these transaction IDs
and their role (`directory` or `archive`). Recovery recomputes these values
from the validated public directory and intended output before accepting any
journal claim. A journal-supplied transaction ID, topology, or staging basename
must equal the independently derived value.

Contract 0.9.0 remains authoritative for journal schema, receipt shape,
transition legality, restart disposition, and declared capabilities. It treats
transaction IDs as opaque; the deterministic derivation and filesystem-object
ownership policy are Tome rules and are documented as such. Tome validates the
exact derived private paths with no-follow operations, rejects foreign sibling
staging, and authorizes cleanup only for those exact paths. Missing directory
staging is required to fail for directory-only recovery; missing archive
staging is legal only before archive staging begins or after a public archive
has made it consumable. A matching archive with residual private state may
therefore complete cleanup without rewriting bytes.

Shard bytes are fsynced before receipt state is written; receipt state is
fsynced before committed range and completion states. Journal files and all
bound private staging are removed only after both independent transactions
reach `PROMOTED` and the promoted outputs pass Contract validation. They are
never inventoried, archived, included in a semantic identity, or required by
ordinary consumers or Student.

Directory and archive states are stored in separate private journal records;
the archive transaction never overwrites the directory transaction's durable
`PROMOTED` marker.

The reachable private/public topologies are deliberately finite:

* Before directory visibility (PC39--PC44), the canonical journal contains
  `directory-journal.json` only; no public directory is consumable. Recovery
  resumes or refuses the directory transaction according to the Contract
  restart disposition.
* After PC45 or PC46, the directory is visible and the canonical root contains
  only `directory-journal.json` while that journal is `PROMOTING`. After PC47,
  it contains the same single directory journal in `PROMOTED`. Recovery first
  validates the directory, completes the legal `PROMOTING` to `PROMOTED`
  transition when needed, then creates the archive journal. It never fabricates
  a missing historical archive journal or assumes that both journals must
  already exist.
* Once archive publication starts, a canonical root contains both journals.
  Archive construction, promotion, completion marking, and cleanup resume from
  the independently validated archive state while the directory remains
  independently `PROMOTED`.
* An independently validated promoted directory with no private state may be
  explicitly repacked. This creates an `archive_only` root containing only
  `archive-journal.json`, with the independently derived `archive-only-v3-`
  transaction ID and matching derived root/staging names. It is a new,
  explicitly bound transaction, not a stripped canonical run. A canonical root
  with its directory journal removed cannot be relabeled `archive_only` because
  its root, transaction, and staging identities do not match the fresh
  derivation. Every archive fault boundary and interrupted cleanup resumes
  through this topology; a malformed or ambiguous root is refused.
* After both independent transactions are durably `PROMOTED` and validated,
  cleanup removes the bound staging and journals. A repeated call with valid
  public outputs and no private state is a completed no-op.

Every private root, journal file, receipt, marker, staging path, and nested
private component is inspected with no-follow `lstat` checks before parsing,
adoption, mutation, promotion, or cleanup. Symlinks, aliases, path escapes,
non-regular journal files, and private roots containing undeclared members fail
closed without following or deleting the target. This ownership check belongs
to Tome; Contract remains authoritative for journal schemas, state transitions,
restart dispositions, and receipt validation.

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
resume adapter first validates every relevant private journal object and the
already visible directory before producing or promoting the archive. It accepts
only the three topologies above (canonical directory-only, canonical
directory-plus-archive, or explicit archive-only); a journal-less staging
remnant, a canonical root missing its directory journal, or extra undeclared
private members is not inferred or repaired.
Cross-authority, cross-configuration, mixed-run, unreceipted, malformed, or
otherwise invalid state fails closed before any journal overwrite. Directory
and archive publication are separate transactions; each is validated and
promoted independently. A report may bind both to the same semantic root, but
does not claim atomic visibility of the pair.

Archive recovery is idempotent. If a matching archive is already visible,
Tome validates its raw and semantic identity, marks a validated interrupted
promotion complete when necessary, and performs cleanup without rewriting the
archive. A conflicting archive is refused without modifying or demoting the
promoted directory. Cleanup removes bound staging before the journal root and
fsyncs the parent after each removal; interruption therefore leaves a
deterministically discoverable journal for a later no-op cleanup. A matching
validated public directory with no private state may be explicitly repacked as
a new archive transaction; this is not inferred resume of an unknown prior
run.

The publisher validates a complete staged package before directory promotion,
fsyncs the parent directory after promotion, and validates the promoted
directory before its completion marker. It then constructs and validates the
archive in its own transaction *before* archive promotion; after the no-replace
promotion and parent-directory fsync it validates the promoted archive before
the archive completion marker. Archive bytes are promoted with an atomic
no-replace link; directory promotion uses
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

The private-path checks establish ownership against the cooperative publisher
model: deterministic names, no-follow type checks, and exclusive output-base
ownership prevent accidental or cooperating-run substitution. They do not claim
to authenticate against an arbitrary same-privilege process that can replace
private directory entries between checks, rewrite every journal and receipt, or
alter the running process. Such a process is outside this unsigned local
transaction integrity claim.

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
