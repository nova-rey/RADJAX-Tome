# M8 sparse truncated selected-pass call graph

`_selected_payloads_from_backend` groups selected source records, derives max(selected_position)+1 prefix lengths, deterministically orders execution batches by prefix length, and restores record order by record index. `GPUTorchTeacherEmissionBackend.emit_batch` tokenizes each text under the original sequence-length policy, slices each token/mask prefix, pads to the batch maximum, transfers input tensors, calls the official `logits_to_keep` interface with the sorted selected-position union, remaps rows in `_gpu_selected_position_logits`, and passes the unchanged reducer to native compact publication.

The first corrected smoke did not reach publication acceptance: corpus_000000662 position 6 changed entropy from 0.84765625 to 0.85546875, beyond the existing 0.00390625 governed tolerance. Therefore no T4 timing sample is accepted and no sparse-truncated output is promoted.
