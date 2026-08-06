# Private Provenance-Shape Bake-Off

Status: **private audit-branch experiment; not a Tome, Contract, or Student format**.
The original self-consistent-substitution interpretation below is superseded by
the corrected threat model in `PROVENANCE_SHAPE_THREAT_MODEL.md`.

Base: `b78821c6aec17335125df2e7f5823dce285735cf`. Inputs are the three
already-selected records from the committed v6 M7 fixture; no teacher,
score-pass, selector, canonical artifact, or fixture was modified or rerun.

## Frozen protocol

Three measurements per shape use median as the sorted middle observation and
spread as `max - min`. A result is material only when the reduction is both at
least 20 percent of the current median and greater than twice the combined
spread. Candidate peak RSS must be no greater than current maximum plus current
spread. These definitions preceded measurement.

The private current model includes a per-coordinate embedded temporary hash,
its reread/reparse/rehash/rewrite, duplicate final record digest fields, and a
repeated shard hash in every index row. The candidate has one framed ordered
record-sequence digest, one raw hash/range/count/size/path per sealed shard,
an index with stable locations and one record digest, an acyclic public
cover/header/inventory graph, an archive digest, and a private
authority-bound journal with sealed contiguous ranges and a completion marker.

## Historical three-record exploratory measurement

| Metric, median of 3 | Current model | Candidate | Result |
| --- | ---: | ---: | --- |
| Wall seconds | 0.013701 | 0.007940 | Not material after noise rule |
| Canonical serialization calls | 38 | 26 | Reduced |
| Canonical serialization bytes | 65,807 | 25,779 | Reduced |
| Bytes written | 36,801 | 15,903 | Material reduction |
| Bytes reread | 96,001 | 53,783 | Reduced |
| Bytes rewritten | 12,684 | 2,851 | Reduced |
| Parse calls / bytes | 3 / 9,833 | 0 / 0 | Reduced |
| Hash calls / input bytes | 32 / 124,779 | 23 / 73,264 | Reduced |
| Journal operations / shard seals | 5 / 3 | 5 / 3 | Preserved |

Peak RSS varied from 16,703,488 to 16,748,544 bytes in the current projection
and 16,748,544 to 16,793,600 bytes in the candidate, exactly at the frozen
current-maximum-plus-spread tolerance. This small fixture is not
representative performance evidence and does not isolate SHA-256 arithmetic;
the reductions arise from avoided canonical-byte production, parsing, and I/O.

## Corrected gate disposition

The private candidate validates each completed shard's raw hash before parsing
or yielding rows; rejects stale/cross-authority/incomplete journal state and
cover/header/inventory incoherence; and preserves semantic identity across
capacity-one resharding. Existing M7/v6 validators remain unchanged and pass.

The 38-case private mutation suite ran each stale public-proof content, index,
cover, raw-shard, and archive mutation against both models. Payload values,
token IDs, corridor/linkage values, deletion, duplication, reorder, declared
index tampering, index pointers, raw shard flip/append/truncation/replacement/
deletion, and archive truncation/append all failed before any corrupted shard
row was yielded. Candidate-specific open/stale, cross-authority,
configuration-mismatched, unreceipted, partial, and incomplete-promotion
journals failed before public-row yield. The public inventory catches stale
shard and index bytes before shard parsing; the journal is bound exactly to the
sealed final-shard index. A capacity-one versus capacity-three reshard preserves
the logical sequence digest while changing raw archive identity.

A fully recomputed candidate with a changed token is internally valid, as
expected. That is not an ordinary internal-integrity failure: an actor able to
replace all package bytes and receipts has no externally supplied expected
identity to contradict. The mutation is retained and now proves the intended
classification boundary:

1. Standard public validation accepts the self-consistent replacement.
2. Immutable expected-identity comparison rejects it as an accidental
   behavioral regression during Golden/Contract development.
3. Optional external-attestation comparison rejects it when a distinct-trust
   original semantic root is supplied.
4. A malicious producer, compromised source, compromised validator, and truthful
   teacher-origin claim remain explicit nonclaims without independent anchors.

The corrected experimental result is that the standard candidate preserves the
required operational checks while removing duplicate staging work. The later
five-run 256/2,048-record synthetic benchmark in
`PROVENANCE_SHAPE_THREAT_MODEL.md` meets its frozen materiality and RSS gates.
This supports a future **Contract-owned next-version design discussion**, not a
format approval or implementation. Contract v0.8.0 correctly cannot admit this
experimental schema, and all released verification paths remain unchanged.

Consequently this branch makes no production, Contract, Student, fixture,
Golden, default-profile, M8, or M9 change. The historical raw path-bearing
three-record measurement evidence is retained outside the repository as
`sha256:c6458db30a764a97569cf40777ffed17c4befa17a1ed0fc31e986e55505f547f`;
the corrected synthetic benchmark receipt is recorded in the threat-model
document. These documents contain sanitized evidence only.
