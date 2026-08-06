# Provenance-Shape Threat Model

Status: private audit-branch design evidence. This document specifies an
experimental next-version projection; it changes neither Tome production nor a
released Contract format.

## Claim boundaries

Standard Tome validation establishes operational integrity of a package relative
to its declared authority and, when supplied by the caller, an expected authority
configuration. It does not establish that a fully self-consistent package came
from an honest producer. Golden/Contract development comparison detects a change
against an already immutable expected behavioral identity. External attestation
compares that same identity with an expectation obtained from a distinct trust
domain. A checksum inside the mutable package cannot itself provide that trust
domain.

| Threat or failure | Standard validation | Golden/Contract development comparison | External attestation | Boundary and remaining nonclaim |
| --- | --- | --- | --- | --- |
| Random byte corruption | Required: inventory, shard raw hash, and archive digest reject it. | Not needed after standard rejection. | May also reject the changed semantic root. | Localizes a changed public member or shard; it does not identify a culprit. |
| Truncated or incomplete transfer | Required: inventory/member size, shard hash/count, and archive identity reject it. | Not needed. | May reject a root mismatch. | Does not recover omitted bytes. |
| Missing, duplicated, or reordered records | Required: contiguous ranges, payload index, count, and framed sequence reject stale evidence. | Required when compared to a fixed expected root. | Required when the external expected root is supplied. | A replacement that recomputes all local evidence is self-consistent, not locally distinguishable. |
| Stale resume state | Producer required: authority/configuration-bound journal and committed ranges reject it. | Not applicable. | Not applicable. | Ordinary consumers never read private journal state. |
| Mixed authority or cross-run state | Required at producer boundary; standard validation checks caller-supplied authority identity. | Required when the expected root is bound to authority. | Required when attestation binds authority. | A caller that deliberately accepts a different authority changes its own policy. |
| Incomplete shard sealing or promotion | Producer required: sealed-log receipts, contiguous ranges, completion marker, and promotion marker reject it. | Not applicable. | Not applicable. | A completed public artifact does not need hidden producer state. |
| Accidental producer regression | Detects stale receipts or incoherent output only. | Required: compare semantic root with immutable Golden/Contract expectation. | Can reject release substitution after independent publication. | No checksum proves the implementation was correct before the expected identity existed. |
| Accidental production under wrong governed configuration | Required when consumer/producer supplies expected authority; cover/layout bind authority, contract, and policy. | Required against the intended expected identity. | Required when attestation binds those fields. | A package may honestly declare a different configuration; policy decides whether to accept it. |
| Deliberate payload alteration without recomputing evidence | Required: member/shard/index/sequence checks reject it before corrupted shard rows yield. | Also rejects. | Also rejects. | This is integrity, not attribution. |
| Deliberate alteration with all internal evidence recomputed | Accept by design when no expected external identity is supplied. | Required: immutable expected root rejects the drift. | Required: independently obtained attested root rejects post-attestation substitution. | Standard validation cannot solve an attacker who controls all package bytes. |
| Compromised producer code | Outside standard validation. | A trusted independent expected identity may expose divergent output. | A distinct attester, independent reproduction, or trusted execution record may qualify a claim. | Still no proof when the producer and expected-identity source share the compromise. |
| Compromised validator code | Outside package evidence. | Outside unless validation is independently reproduced. | Outside unless verification is performed in an independently trusted verifier. | The package cannot attest to correctness of the program interpreting it. |
| Dishonest model-origin claim | Outside standard validation. | Behavioral comparison can expose a mismatch with known expected output, not prove origin. | A trusted execution record or independent reproduction may support a qualified origin claim. | Hashes, signatures, and receipts bind statements; they do not prove that a teacher truthfully produced content. |

## Standard candidate: operational proof surface

The public experimental projection has one semantic authority identity, stable
logical record IDs, a framed ordered full-record sequence digest, and a semantic
root that binds sequence, authority identity, Contract version, and behavioral
policy identity. Each final shard contributes its raw SHA-256, byte size,
logical range, count, and path exactly once to the shard index. The payload index
contains stable logical location and one record digest, not a repeated shard
hash. The public graph is acyclic:

```text
cover -> manifest-header -> inventory -> layout, indexes, final shards
```

An optional transport archive is protected by its outer raw digest. The standard
surface intentionally omits native embedded `payload_hash`, its linkage-time
reread/rehash/rewrite, the duplicate per-record digest, and the repeated
per-row shard hash. It also omits nested self-attestation added solely to resist
an actor who can rebuild every internal receipt.

The private transaction surface is excluded from the final archive and contains
authority/configuration identity, an append-only sealed-shard receipt log,
contiguous committed range, open/complete state, and completion/promotion
marker. It is necessary only while producing or resuming; standard consumers and
Student do not require it.

## Optional external-attestation interface

The optional layer does not add a second package format. An attestation obtained
outside the Tome contains:

```json
{
  "schema_version": "radjax_tome_provenance_bakeoff_experimental_vnext.external_attestation.v1",
  "semantic_root": "sha256:...",
  "semantic_authority_identity": "sha256:...",
  "contract_version": "...",
  "behavioral_policy_identity": "...",
  "reference": "external receipt, release digest, signature, or log entry"
}
```

The attested `semantic_root` is the exact binding of the sequence identity to
semantic authority, Contract version, and behavioral policy. The expected
attestation must reside in a separately governed release receipt, signature
service, transparency log, or independently retained digest. A consumer elects
standard mode by running only public validation, or attested mode by additionally
requiring a matching externally obtained record. The experimental local function
compares identities only; it deliberately implements no signing, key management,
signature verification, transparency service, or network transport.

This layer detects package substitution after a genuinely separate attestation
has been created. It does not prove model origin, honesty of the attester,
uncompromised producer/validator source, key safety, trusted execution, or
independence where the purported attester is controlled by the same adversary.

## Corrected acceptance gates

A fully recomputed altered artifact has three intentional outcomes: standard
validation accepts it without an external expected identity; immutable
Golden/Contract-style comparison rejects it against the original semantic root;
and optional external-attestation comparison rejects it against a separately
supplied original attestation. This is a classification boundary, not a
contradiction and not a reason to retain redundant internal hashes.

Standard acceptance still requires stale-evidence detection for payload/token/
probability/corridor/linkage drift, deletion/duplication/omission/reorder,
index/cover/range/count/pointer incoherence, shard corruption before row yield,
safe transaction refusal, nonsemantic resharding identity, historical-path
stability, no private-state consumer dependency, and an acyclic graph.

## Frozen synthetic benchmark protocol

No retained post-C5 selected payload corpus is available locally; the retained
receipt names its immutable checkpoint but does not commit raw payloads. The
benchmark therefore expands the committed three-record v6 selected shard
deterministically, changes only stable synthetic record identities/positions,
and invokes no teacher, scoring, selector, or production writer. It is a
synthetic construction benchmark, not production or model performance evidence.

- Sizes: 256 and 2,048 records; shard capacity 64.
- Comparison: current-model and candidate projections from identical generator
  records, each in a fresh process.
- Warm-up: one unreported run per shape and size.
- Measurements: five reported runs per shape and size.
- Primary statistic: median construction wall time; spread is max minus min.
- Materiality: candidate median must improve by at least 15 percent of current
  median and exceed twice the combined current/candidate spreads at 2,048
  records. Structural I/O reduction is reported separately and cannot replace
  this gate.
- Memory tolerance: candidate maximum peak RSS must be no more than 10 percent
  above the current maximum for each size.
- Additional observations: construction, archive, validation wall time,
  throughput, all counter bytes/calls, and raw per-run distributions.

Results are appended only after this protocol is executed. Raw path-bearing
machine evidence remains outside the repository and is represented here by its
digest only.

## Benchmark results

The benchmark used one warm-up and five fresh-process reported runs per shape
and size. It retained the same synthetic semantic root across current and
candidate shapes for each record count. Raw evidence is retained outside the
repository as
`sha256:fb2b4d6893b9381e396357e3fbe08bdd2d83592384358f01837360e6ca5fc503`.

| Records | Shape | Construction median (range) | Archive median | Validation median | Throughput median | Peak RSS range |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 256 | Current | 0.368551 s (0.360352–0.438495) | 0.011581 s | 0.052551 s | 694.61 rec/s | 20,164,608–20,201,472 B |
| 256 | Candidate | 0.098735 s (0.097013–0.100614) | 0.006052 s | 0.046376 s | 2,592.81 rec/s | 19,963,904–20,357,120 B |
| 2,048 | Current | 2.859818 s (2.857431–2.945507) | 0.077038 s | 0.413928 s | 716.13 rec/s | 29,741,056–29,818,880 B |
| 2,048 | Candidate | 0.723086 s (0.719906–0.807105) | 0.029727 s | 0.358802 s | 2,832.30 rec/s | 28,192,768–28,254,208 B |

At 2,048 records, the candidate improved median construction time by 74.72
percent (2.136732 s). That exceeds both the 15-percent threshold (0.428973 s)
and twice combined spread (0.247879 s). Candidate peak RSS was lower than the
current maximum at 2,048 records and 0.77 percent above it at 256 records,
within the 10-percent tolerance. The materiality and memory gates pass for this
synthetic construction workload.

| 2,048-record structural counter, per run | Current | Candidate | Change |
| --- | ---: | ---: | ---: |
| Serialization calls / bytes | 14,409 / 34,285,553 | 6,217 / 7,149,240 | -56.85% / -79.15% |
| Temporary bytes written | 13,482,031 | 18,321 | -99.86% |
| Final bytes written | 15,222,617 | 14,075,738 | -7.53% |
| Total bytes written | 28,704,648 | 14,094,059 | -50.90% |
| Bytes reread | 29,520,859 | 21,068,685 | -28.63% |
| Bytes rewritten | 6,744,180 | 12,325 | -99.82% |
| Parse calls / bytes | 2,048 / 6,731,855 | 0 / 0 | eliminated |
| Hash calls / input bytes | 8,268 / 49,099,482 | 4,172 / 34,279,999 | -49.54% / -30.18% |
| Journal operations / shard seals | 34 / 32 | 34 / 32 | preserved |

The evidence supports serialization, parsing, and temporary filesystem churn as
the dominant modeled avoided work. It does not isolate or attribute the gain to
SHA-256 arithmetic. Because the payload set is deterministically expanded from
the small committed fixture, these numbers do not establish full production
throughput, accelerator behavior, model execution cost, or unbounded-memory
behavior under a retained real corpus.

The final full-suite raw log is also retained outside the repository as
`sha256:00a18cafb4e02fbd2cc90a742f446c000d7b6593ee17438efb8f877df22a3c85`.
