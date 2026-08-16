# M8G Contract amendment design (design only)

This document is a buildable amendment design. It does not modify
RADJAX-Contract, Tome production, Student, fixtures, or releases. Authority is
radjax-contract 0.9.0 at `1fa43e1aea2e198511db86dafb0aeefa525d48c7`.

## Existing constraints

The current Contract v3 codec hashes closed semantic records: `codec.py`
`record_sequence_digest` covers the complete encoded record, while
`logical_record_id` covers selected example and position. `schema.py` uses a
closed `RECORD_FIELDS` map, and the v3 resource inventory has fixed roles.
Current Tome backend and delivery code deliberately emit padded rectangular
dynamic-head arrays with an authoritative `top_selection_mask`; Golden
validation requires equal array lengths and `active_count == effective_top_k`.
Current v4 staging receipts do not carry the Contract-v3 journal fields or
promotion state machine. Therefore neither compact storage nor body/linkage
splitting can be a silent Tome-only reinterpretation.

## B1 compact dynamic-top-K storage flavor

Proposed opt-in resource type: `selected_exemplar_payload_compact_v1`, under a
new Contract semantic profile version (the exact release number is a Contract
maintainer decision). The existing padded resource remains valid and is not
rewritten.

Each compact shard contains a fixed record table plus flattened active arrays:

```text
record_count: u32
position_count: u32
top_offsets: u64[position_count + 1]
top_lengths: u32[position_count]
top_token_ids: i32[sum(top_lengths)]
top_probs: f32[sum(top_lengths)]
top_log_probs: f32[sum(top_lengths)]
top_selection_mask: omitted (or all-true active mask in an explicitly named compatibility field)
effective_top_k: u32[position_count]
top_mass: f32[position_count]
tail_mass: f32[position_count]
bucket_masses: f32[position_count, num_buckets]
```

For every position, `top_offsets[i+1]-top_offsets[i] == top_lengths[i] ==
effective_top_k[i]`. IDs are descending-probability order with the existing
stable tie rule. Probabilities and log-probabilities retain current float32
bits; the Contract must specify whether `log(p)` is checked numerically or
treated as governed supplied evidence. `top_mass`, `tail_mass`, and bucket
masses retain current meanings and sum checks. Full-width K remains exactly
vocabulary width. No score-pass dense logits are admitted.

The compact profile gets a distinct schema/resource flavor and canonical
projection. A compact record may be semantically equivalent to a padded record
when active entries, masses, bucket evidence, coordinate, and authority fields
match; raw digests remain different. The Contract must explicitly define
semantic equivalence rather than relying on byte equality. Padded records remain
accepted for legacy profiles. Compact records reject offset overflow, negative
or non-monotonic offsets, length mismatches, truncation, extra trailing bytes,
non-finite values, token-order violations, and mass inconsistencies.

Migration is read-compatible only: a validator may project padded active slots
to the compact semantic projection, but no writer rewrites old artifacts. A
checkpoint records the storage flavor and rejects a resume that changes it.
Student profile support requires a compatibility fixture and consumer review;
it is not part of this Tome checkpoint.

## B2 immutable body and linkage manifest

Proposed opt-in resources:

* `selected_exemplar_body_v1`: immutable high-resolution arrays, CSL/tail
  evidence, dynamic-K metadata, body schema, and body semantic identity.
* `selected_exemplar_manifest_v1`: selected coordinate, source-passport
  reference, corridor assignment, all selection reasons and leaderboard
  provenance, body identity/digest, authority/profile versions, and package
  metadata.

The body moves only payload-bearing fields: compact IDs/probabilities/log
probabilities, CSL buckets, retained mass, tail mass, and immutable teacher
evidence. The manifest retains selected/source coordinates, source passport,
corridor and mode linkage, selection obligations, board ranks/scores, profile,
schema, and authority declarations. Derived body byte length and digest are not
mutable claims; they are checked against the body resource.

One body may be content-addressed and referenced by multiple manifests only if
the Contract distinguishes body identity from exemplar identity. Coordinate
identity remains unique at the manifest level; content deduplication must not
collapse two selected coordinates or their selection reasons.

## B3 identity and authority

`body_semantic_id` is the canonical projection of body fields plus body schema,
numeric/dtype rules, and Contract profile. `body_raw_digest` covers exact body
bytes. `manifest_semantic_id` covers coordinate, source passport, linkage,
selection reasons, body semantic ID, authority, and profile. `exemplar_id` is a
coordinate-scoped identity that includes both manifest semantic ID and body
semantic ID. A final package root covers ordered manifest inventory and body
references. No mutable manifest field authenticates itself; every digest is
recomputed from the referenced bytes and canonical projection.

Validation order is: body framing and bounds; body raw digest; body semantic
projection; manifest closed-schema validation; body-reference equality;
selection/source/corridor authority; package inventory; final root. A body can
be shared by manifests, but a manifest cannot point to a body with a mismatched
schema, profile, authority, or digest.

## B4 transaction state machine

The new producer-private transaction uses Contract-style journal semantics:

`BODY_GENERATING -> BODY_WRITTEN -> BODY_VALIDATED -> BODY_DIGESTED ->
BODY_PROMOTED -> MANIFEST_PROVISIONAL -> LINKAGE_FINALIZED ->
MANIFEST_VALIDATED -> MANIFEST_PROMOTED -> INVENTORY_COMMITTED ->
PACKAGE_VALIDATED -> COMMITTED`.

Each state transition is receipt-covered and fsync/atomic-rename rules are
specified by Contract. Temporary body or manifest files are never consumable.
Resume may reuse only a promoted body whose receipt, digest, schema, authority,
and complete byte length validate. An unverified or partial body is discarded
or quarantined. A stale or wrong-body manifest is rejected before package
mutation. Missing body, corrupt body, duplicate writer, concurrent resume,
orphan temporary, and orphan promoted body each have explicit refusal,
quarantine, or cleanup-only dispositions. Archive construction validates the
directory inventory and body references before promotion; repeated recovery is
idempotent.

## B5 required Contract amendments

### Proposed closed registries and framing

The amendment uses canonical FV3 framing for every field: a UTF-8 field-name
length/value-length tuple in schema order, little-endian unsigned integers for
counts and offsets, IEEE-754 binary32 for probabilities, and length-delimited
arrays. The exact resource header is:

`magic="RDXC" | version=u16(1) | profile=u16(1) | record_count=u32 |
position_count=u32 | vocab_size=u32 | num_buckets=u32 | body_bytes=u64 |
manifest_bytes=u64 | header_crc32=u32 | payload_crc32=u32`.

The compact closed field registry is exactly:
`schema_version:string`, `profile:string`, `record_count:u32`,
`position_count:u32`, `vocab_size:u32`, `num_buckets:u32`,
`top_offsets:u64[]`, `top_lengths:u32[]`,
`top_token_ids:i32[]`, `top_probs:f32[]`, `top_log_probs:f32[]`,
`effective_top_k:u32[]`, `top_mass:f32[]`, `tail_mass:f32[]`,
`bucket_masses:f32[position_count,num_buckets]`.
Offsets and flattened arrays are ordered by selected-coordinate manifest order;
arrays are little-endian and finite checks are mandatory. The compact manifest
registry is exactly: `schema_version:string`, `profile:string`,
`selected_example_id:string`, `selected_position:u32`,
`source_passport_id:string`, `corridor_mode_id:i32|null`,
`corridor_fingerprint_id:string|null`, `selection_obligation_count:u32`,
`selection_obligations:selection_obligation[]`, `body_semantic_id:bytes32`,
`body_raw_digest:bytes32`, `authority_id:bytes32`,
`selection_authority_id:bytes32`,
`package_role:string`, and `manifest_semantic_id:string`. Unknown fields are
rejected.

`selection_obligation` is a closed, non-recursive tuple in manifest order:
`role:u8` (`1=corridor`, `2=global`), `source_id:string`, `rank:u32`,
`score:f32|null`, and `collision_kind:u8` (`0=none`, `1=corridor`,
`2=global`). `selection_obligation_count` must equal the number of tuples;
no arbitrary metadata map or nested object is admitted.

Identity preimages are domain separated and length-delimited. Every domain
label is encoded as `u16_le(label_byte_count) || ASCII(label_bytes)`; the
following bytes are the canonical FV3 sequence for the named fields, never
JSON or a platform-native struct:

* `body_semantic_id = sha256("RDX-BODY-SEM-1" || FV3(body registry including
  vocab_size, num_buckets, and the two-dimensional bucket shape))`;
* `body_raw_digest = sha256("RDX-BODY-RAW-1" || exact body bytes)`;
* `manifest_semantic_id = sha256("RDX-MANIFEST-SEM-1" || FV3(manifest registry excluding manifest_semantic_id))`;
* `exemplar_id = sha256("RDX-EXEMPLAR-1" || FV3(coordinate, manifest_semantic_id, body_semantic_id))`;
* package root = ordered binary-tree SHA-256 over leaf frames
  `RDX-INVENTORY-LEAF-1 || u16(label_len) || ASCII(label) ||
  u32(role_len) || role || u32(path_len) || path ||
  exemplar_id || raw_digest`; parent frames are
  `RDX-INVENTORY-NODE-1 || left || right`, duplicating the final leaf at odd
  levels. Paths are UTF-8 NFC, slash-separated, relative, and reject `..`.

The package inventory has one leaf for every promoted body and manifest, plus
one authority leaf and one profile leaf. Role bytes are respectively `body`,
`manifest`, `authority`, and `profile`; each leaf is
`u16_le(label_len) || ASCII(label) || u32_le(role_len) || role ||
u32_le(path_len) || path ||
  u32_le(id_len) || identity || u32_le(raw_digest_len) || raw_digest`. All
identity and digest fields typed `bytes32` are exactly 32 raw digest bytes
(never lowercase hexadecimal text); textual IDs retain the string framing.
Leaves
are sorted by `(role, path)` UTF-8 bytes. The authority leaf binds
`selection_authority_id` and the profile leaf binds the header profile code,
so either change updates the package root without changing body semantics.

Padded and compact records may share a semantic identity only after both are
projected to the compact active-entry registry and all equivalence invariants
pass; their raw digests necessarily remain distinct.

The projection is normative: for each position `i`, read the padded mask from
left to right and copy exactly the entries whose mask bit is `true` into the
flattened arrays, preserving index order. Require
`active_count == effective_top_k[i]`, `top_lengths[i] == active_count`, and
`top_offsets[position_count] == len(top_token_ids) == len(top_probs) ==
len(top_log_probs)`. The compact record has no mask field: every flattened
entry is active. `top_mass`, `tail_mass`, and bucket values are copied
bit-for-bit and retain the padded validator's mass and bucket checks. A mask
with interspersed inactive entries is compacted in mask order, never by
slicing a prefix.

Header profile codes are normative: `1` is
`selected_exemplar_payload_compact_v1/student`, `2` is
`selected_exemplar_payload_compact_v1/full_debug`, and `3` is
`selected_exemplar_payload_compact_v1/producer_evidence`. The registry
`profile` string must equal that mapping; header version, profile code,
vocabulary, bucket, record, and position counts must agree with the registry.

### Journal receipt matrix

Each transition receipt has this closed ordered tuple: `transaction_id:string`
(required), `schema_version:string`, `profile_code:u16`, `state:u16`,
`parent_transaction_id:string|null`,
`body_path:string|null`, `manifest_path:string|null`,
`body_raw_digest:bytes32|null`, `body_size_bytes:u64|null`,
`manifest_raw_digest:bytes32|null`, `committed_next_state:u16|null`,
`configuration_identity:bytes32`, `semantic_authority_identity:bytes32`, and
`receipt_digest:bytes32`. Null is encoded as `0x00`; a present value is
encoded as `0x01` followed by its framed value. State codes are
`1=BODY_GENERATING`, `2=BODY_WRITTEN`, `3=BODY_VALIDATED`,
`4=BODY_DIGESTED`, `5=BODY_PROMOTED`, `6=MANIFEST_PROVISIONAL`,
`7=LINKAGE_FINALIZED`, `8=MANIFEST_VALIDATED`, `9=MANIFEST_PROMOTED`,
`10=INVENTORY_COMMITTED`, `11=PACKAGE_VALIDATED`, and `12=COMMITTED`.
Receipt digest is `sha256("RDX-RECEIPT-1" || FV3(all fields except receipt_digest))`.
The writer fsyncs file contents before atomic rename, fsyncs the containing
directory after rename, and writes the next receipt before exposing the next
state. Recovery maps missing/partial receipts to refusal or quarantine; only
`BODY_PROMOTED` with a validated body digest may be reused. `MANIFEST_PROMOTED`
requires a validated body reference. `INVENTORY_COMMITTED` requires a complete
inventory; `PACKAGE_VALIDATED` is the only state eligible for `COMMITTED`.
Duplicate writers must fail closed on transaction-id mismatch. `profile` is a
u16 registry code (`1=student`, `2=full_debug`, `3=producer_evidence`) with a
normative presentation mapping. Strings use `u32_le byte_length || UTF-8 NFC
bytes`; arrays use `u64_le element_count` and the declared scalar encoding.
The header carries `header_crc32`, CRC-32/ISO-HDLC over header bytes preceding
that field, and `payload_crc32`, CRC-32/ISO-HDLC over the complete body and
manifest payload bytes. The payload CRC is not a header-prefix checksum.
Body and receipt digest preimages exclude their own digest fields.

The restart matrix is normative: interruption before a receipt leaves the
prior state reusable; `BODY_WRITTEN` temporary files are never reusable;
`BODY_PROMOTED` is reusable only after digest, schema, authority, and length
validation; `MANIFEST_PROVISIONAL` is never public; `MANIFEST_PROMOTED`
requires a validated body reference; `INVENTORY_COMMITTED` requires a complete
inventory; and only `PACKAGE_VALIDATED` may advance to `COMMITTED`. Missing,
malformed, or non-contiguous receipts quarantine newer temporary resources and
resume from the last contiguous receipt.

The only legal next-state edges are
`1->2->3->4->5->6->7->8->9->10->11->12`; a receipt may repeat its current
state idempotently, but may not skip, regress, or change profile/schema.

The required receipt/restart cases are explicit for each state:

| State | Required durable evidence | Restart disposition |
|---|---|---|
| BODY_GENERATING | transaction and configuration identities | discard temp body |
| BODY_WRITTEN | temp path and byte count | revalidate or quarantine temp |
| BODY_VALIDATED | validated schema/profile/shape | recompute digest |
| BODY_DIGESTED | body digest and length | atomically promote body |
| BODY_PROMOTED | promoted path plus receipt digest | reuse only after full revalidation |
| MANIFEST_PROVISIONAL | body reference and provisional digest | never publish; resume linkage |
| LINKAGE_FINALIZED | finalized authority/linkage fields | validate manifest |
| MANIFEST_VALIDATED | manifest digest and body binding | atomically promote manifest |
| MANIFEST_PROMOTED | manifest/body pair | rebuild inventory |
| INVENTORY_COMMITTED | complete role/path inventory | validate archive/package |
| PACKAGE_VALIDATED | package root and Contract result | commit transaction |
| COMMITTED | final receipt | idempotent no-op |

The next Contract branch must add compact resource framing, compact semantic
projection, body/manifest resource roles, closed manifest fields, digest-domain
definitions, profile compatibility rules, package inventory entries, checkpoint
storage-flavor binding, and conformance fixtures/tests. This should be an
opt-in backward-compatible Contract release only if the existing semantic root
can honestly define padded/compact equivalence; otherwise it requires a new
semantic-contract version. No 0.9.0 release claim is made here.

## B6 rollout

1. Contract schemas, codecs, validators, and fixtures.
2. Contract equivalence, corruption, truncation, inventory, and journal tests.
3. Tome writer behind explicit compact/body-manifest opt-in.
4. Tome reader, inspector, and resume support.
5. Padded-vs-compact semantic-equivalence fixtures.
6. Body/manifest crash and concurrent-resume tests.
7. Student compatibility fixture and separate Student authorization.
8. Frozen 256-coordinate M8 measurement.
9. Default transition only after review of identity, compatibility, and
   practical performance evidence.

## B7 requirements traceability

| Rule | Current authority | Amendment | Affected components | Required test |
|---|---|---|---|---|
| Compact arrays have physical length K | Tome backend padded arrays; Golden contract | compact resource codec/validator | Contract codec, Tome writer/reader | K=32/38/medium/full-width |
| CSL and mass preserved | backend dynamic reducer and Golden checks | compact projection invariants | Contract validator, Tome delivery | mass/tail/bucket equivalence |
| Padded compatibility | existing v2/v3 resource flavor | profile matrix and adapter | Contract validator, Tome inspector | legacy padded acceptance |
| Body digest is independently checked | current raw resource digests | body digest domain and inventory role | Contract package validator | wrong-body/truncation |
| Linkage changes do not rewrite body | current monolithic records | manifest/body split resources | Contract schema, Tome transaction | linkage-only update |
| Resume is crash-safe | Contract journal restart rules; v4 Tome staging is insufficient | body/manifest journal states | Contract journal, Tome resume | every interruption state |
| Coordinate identity remains unique | current logical record ID | manifest identity rule | Contract semantic root, Tome C5 | shared body, distinct coordinates |

Contract review remains required only for release/version assignment and final
approval of the proposed choices: compact and padded forms are semantically
equivalent but raw-distinct; the compact mask is omitted; archive members use
the ordered inventory entries above; and body content deduplication is allowed
across manifests but never collapses coordinate-scoped exemplar identities.
