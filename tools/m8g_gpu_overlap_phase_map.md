# M8G GPU/CPU overlap phase map

## Before correction

`assembly.py::assemble_selected_delivery_artifacts` called the synchronous compact writer. The prior `simple_compact_body.py::write_compact_body_store_pipelined_from_compact` admitted `(payload, encoded_bytes, digest)` tuples, so `compact_body_from_logical_payload`, `encode_compact_body_packed`, and `body_raw_digest` all ran before queue admission. Queue accounting was encoded-byte based.

## After correction

Canonical compact C6 calls `write_compact_body_store_pipelined_from_compact`. The producer admits `RawCompactDescriptor` objects containing exact-K arrays, identity metadata, optional CUDA readiness events, and ownership release callbacks. Workers wait on the item event, build the Contract body, encode, hash, validate through the existing Contract encoder boundary, fsync, and atomically publish private body bytes. Metadata is assembled in descriptor order after worker completion; completion order cannot affect identity.

`gpu_overlap.py::descriptor_stream_from_cuda` creates deterministic exact-K CUDA tensors, issues nonblocking device-to-host copies, records a per-transfer CUDA event, and yields descriptors without building encoded bytes or hashing before admission. CPU-only callers use the same descriptor type with no event.

## Explicit nonclaims

The existing Contract v2 encoder still receives governed Python sequence values at its public body-construction boundary; this checkpoint does not change Contract bytes or schema. The synthetic T4 measurement therefore reports the actual current encoder cost and does not claim a new native packed-array codec.
