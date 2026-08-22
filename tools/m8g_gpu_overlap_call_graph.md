# M8G buffer-native compact C6 call graph

`rerun._materialize_selected_payloads` -> `staging._selected_payloads_from_backend` -> backend `emit_batch`; compact arrays are extracted by `payloads._payload_buffer_slice` into contiguous governed buffers. Native C6 retains the authoritative full payload handoff in `rerun.py` for compact modes.

`assembly.assemble_selected_delivery_artifacts` dispatches native C6 compact mode to `write_compact_body_store_pipelined_from_compact`. That function admits `RawCompactDescriptor` objects to `_ByteBoundedQueue` before worker-side Contract construction, validation, packed encoding, hashing, private write/fsync, and deterministic metadata publication.

Synthetic CUDA production uses `gpu_overlap.descriptor_stream_from_cuda`: exact-K CUDA tensors -> reusable pinned CPU buffers -> nonblocking copies -> CUDA events -> raw descriptors. Workers call the public Contract `compact_body_from_buffers` and `encode_compact_body_packed_from_buffers`.

The native C6 integration correction is the compact publication handoff: compact native execution no longer bypasses the canonical compact writer. The retained handoff is authority-preserving; it does not rerun teacher, selection, or replay.
