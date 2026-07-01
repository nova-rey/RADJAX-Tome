# Tome Optimization Backend Handoff

Historical code is reference material. Do not copy it wholesale into RADJAX-Tome.

## Historical Scope

The historical optimization work came from `nova-rey/qrwkv-xla`, with the most
relevant commit:

```text
6c21171bf76d341b476128d929d58469d4d06f18
Add chunked compact GPU reduction / P156.5.6.1
```

That work explored real-teacher behavioral fingerprint and TOME generation
around CPU staging, GPU compact reduction, and chunked GPU vocabulary
reduction. It is useful as proof that the ideas are viable, but the active
RADJAX-Tome architecture should be backend-neutral before any runtime-specific
optimization is ported.

## Historical Components

The old optimization code lived mainly in:

```text
scripts/build_real_teacher_fingerprint_artifact.py
src/qrwkv_xla/fingerprint/real_teacher.py
src/qrwkv_xla/fingerprint/topology.py
src/qrwkv_xla/teachers/hf.py
src/qrwkv_xla/fingerprint/capture.py
src/qrwkv_xla/artifacts/cascaded_soft_labels.py
```

## Threaded / Staged CPU Pipeline

The CPU pipeline split generation into source corpus loading, prompt batching,
prefetch/tokenization, inference plus compact reduction, and ordered commits
into the artifact writer. This structure is still desirable because it separates
IO, tokenization, teacher execution, reduction, and artifact writing.

The old configuration had a `reducer_workers` setting, but true reducer-worker concurrency was deferred.
RADJAX-Tome should not claim that behavior until a new implementation proves it.

## GPU Compact Reduction

The GPU compact path ran a Hugging Face teacher on CUDA/MPS-like devices and
reduced full-vocabulary teacher logits into compact fingerprint statistics on
device before sending compact arrays back to the host. This was a practical
memory and transfer optimization, not a generic backend design.

## Chunked GPU Compact Reduction

The chunked GPU reducer fixed a memory trap by reducing over the vocabulary axis
in chunks instead of materializing full-vocabulary temporary workspaces. This
was primarily a memory-scalability fix, not an automatic throughput guarantee.
Throughput still depends on model size, device characteristics, chunk sizing,
and transfer overhead.

## Metadata And Provenance

Future optimized generation should record which runtime mode actually ran, the
teacher backend identity, reduction policy, chunking settings when used, local
files/download policy, source corpus identity, and artifact writer version. The
metadata should distinguish planned capability from observed execution.

## Desired Runtime Modes

RADJAX-Tome should move toward these runtime modes:

```text
CPU-only
CPU + GPU
CPU + TPU
```

TPU support requires a non-CUDA-shaped abstraction. CUDA/MPS paths assume
PyTorch-style device tensors and vocabulary reduction patterns, while TPU paths
are more likely to require JAX/XLA-style compilation boundaries, static shapes,
and different host/device transfer rules. The backend interface should describe
teacher execution and reduction policy concepts without baking in CUDA-specific
control flow.

## Known Limitations

- The optimized QRWKV-XLA stack was built for the old monorepo layout.
- The GPU reducer assumed CUDA/MPS-style execution and should not define the
  default architecture.
- TPU execution was not implemented by the historical work.
- `reducer_workers` existed as configuration, but true reducer-worker
  concurrency was deferred.
- Chunked GPU reduction addressed memory scalability; it did not guarantee
  higher throughput.

## Recommended Migration Plan

1. Land the unpacked Tome cover page so optimized writers have an artifact
   contract to target.
2. Land the bundle container after the unpacked contract is stable.
3. Define teacher runtime modes and metadata before porting GPU-specific logic.
4. Recreate CPU-only generation first as the reference behavior.
5. Add CPU + GPU compact reduction with chunked vocabulary reduction as an
   optional runtime implementation.
6. Add CPU + TPU through a backend interface that does not assume CUDA-shaped
   tensor control flow.
7. Add dynamic top-k only after metadata and reduction policy surfaces exist.
