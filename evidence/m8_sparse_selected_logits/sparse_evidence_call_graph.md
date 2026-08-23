# Sparse selected-logits call graph

`builder/delivery/staging.py::_selected_payloads_from_backend` supplies frozen selected positions to `backends/gpu_torch.py::GPUTorchTeacherEmissionBackend.emit_batch`.

The backend tokenizes and collates with the existing attention mask, transfers inputs to CUDA, calls `_torch_model_forward`, then applies the unchanged selected-row mapping and dynamic reducer before the existing compact publication path.

Before this checkpoint, model output was `[batch, sequence, vocab]` and `_gpu_selected_position_logits` gathered after `lm_head`.

After this checkpoint, selected compact delivery builds the sorted union of requested sequence positions and passes it as the official Hugging Face `logits_to_keep` argument. The returned sparse rows are mapped back to source-specific coordinates, then the existing reducer and C6 publication run unchanged. Unsupported model signatures raise `TeacherBackendUnsupportedPolicyError`; there is no silent dense fallback.

For the 64-source/71-coordinate batch-1 workload, dense projection would produce 8,192 vocabulary rows (64 sources x 128 positions across the selected pass). Sparse projection requests 71 rows in aggregate, avoiding 8,121 rows. At BF16, this is approximately 4.00 GiB versus 35.5 MiB of vocabulary-logit bytes before reducer work.
