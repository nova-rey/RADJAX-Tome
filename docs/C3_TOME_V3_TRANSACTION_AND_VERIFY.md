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

The publisher uses a same-filesystem private temporary journal. It records only
the transaction identity, v3 configuration identity, semantic-authority
identity, sealed shard receipts, contiguous committed selection range,
completion intent, and promotion marker. Journal files are fsynced and removed
after the corresponding public transaction reaches `PROMOTED`. They are never
inventoried, archived, included in a semantic identity, or required by
ordinary consumers or Student.

The Contract state machine is used without reinterpretation:

`OPEN`/`SEALING` → `COMPLETE_INTENT` → `PROMOTING` → `PROMOTED`.

An incomplete or stale state is not publicly visible. A `PROMOTING` restart
validates an already visible target before marking it promoted, otherwise it
retries promotion. Cross-authority, mixed-run, unreceipted, or otherwise invalid
journal state fails closed through Contract validation. Directory and archive
publication are separate transactions; each is validated and promoted
independently. A report may bind both to the same semantic root, but does not
claim atomic visibility of the pair.

The publisher validates a complete staged package before directory promotion,
fsyncs the parent directory after promotion, then constructs and validates the
archive in its own transaction. Existing targets are rejected before promotion;
the output path is never silently replaced. Filesystem transports that cannot
provide the required local staging and durable publication assumptions are not
represented as successfully promoted by this adapter.

## Scope

The v3 projection consumes the finalized typed selected-record handoff after
late corridor linkage. It does not reread emitted v4/staging files, repeat
score or selection logic, alter fixed 25-field authority projections, change
historical v2/v4/v5/v6 paths, or make a performance claim.
