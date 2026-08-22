"""CUDA-to-CPU exact-K descriptor production for compact C6."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from typing import Any

from .simple_compact_body import RawCompactDescriptor


def descriptor_stream_from_cuda(
    ks: Iterable[int],
    *,
    vocab_size: int,
    batch_size: int = 8,
    cadence: str = "maximum_pressure",
    torch_module: Any | None = None,
) -> Iterator[RawCompactDescriptor]:
    """Yield raw exact-K CPU buffers while CUDA produces the next batch.

    The only producer-side work is deterministic tensor construction, one
    nonblocking D2H copy per array, event recording, and queue ownership.
    """
    torch = torch_module
    if torch is None:
        import torch as torch
    values = [int(value) for value in ks]
    pool: dict[tuple[str, int], list[Any]] = {}

    def acquire(dtype: Any, k: int) -> Any:
        key = (str(dtype), k)
        if pool.get(key):
            return pool[key].pop()
        return torch.empty((k,), dtype=dtype, device="cpu", pin_memory=True)

    def release(dtype: Any, k: int, buffer: Any) -> None:
        pool.setdefault((str(dtype), k), []).append(buffer)

    def release_triplet(
        k: int, ids_buffer: Any, probs_buffer: Any, logs_buffer: Any
    ) -> None:
        release(torch.int64, k, ids_buffer)
        release(torch.float32, k, probs_buffer)
        release(torch.float32, k, logs_buffer)

    for batch_start in range(0, len(values), batch_size):
        batch = values[batch_start : batch_start + batch_size]
        for offset, k in enumerate(batch):
            index = batch_start + offset
            ids = (
                torch.arange(k, device="cuda", dtype=torch.int64)
                .add_(index * 17)
                .remainder(vocab_size)
            )
            probs = torch.full((k,), 0.5 / k, device="cuda", dtype=torch.float32)
            logs = torch.log(probs)
            if cadence == "production_shaped":
                work = torch.ones((256, 256), device="cuda", dtype=torch.float32)
                for _ in range(3):
                    work = work @ work.t() * 0.0001
            cpu_ids = acquire(torch.uint32, k)
            cpu_probs = acquire(torch.float32, k)
            cpu_logs = acquire(torch.float32, k)
            cpu_ids.copy_(ids, non_blocking=True)
            cpu_probs.copy_(probs, non_blocking=True)
            cpu_logs.copy_(logs, non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            payload = {
                "selected_example_id": f"synthetic_{index:04d}",
                "selected_position": index,
                "vocab_size": vocab_size,
                "num_buckets": 3,
                "effective_top_k": k,
                "top_token_ids": cpu_ids,
                "top_probs": cpu_probs,
                "top_log_probs": cpu_logs,
                "top_mass": 0.5,
                "tail_mass": 0.5,
                "bucket_masses": __import__("numpy").asarray((0.2, 0.3, 0.5), dtype="<f4"),
            }

            def release_current(
                k=k,
                ids_buffer=cpu_ids,
                probs_buffer=cpu_probs,
                logs_buffer=cpu_logs,
            ) -> None:
                release_triplet(k, ids_buffer, probs_buffer, logs_buffer)

            yield RawCompactDescriptor(
                payload=payload,
                estimated_bytes=int(
                    cpu_ids.numel() * 8
                    + cpu_probs.numel() * 4
                    + cpu_logs.numel() * 4
                    + 1024
                ),
                ordinal=index,
                ready_event=event,
                release=release_current,
            )


def producer_lower_bound(
    ks: Iterable[int],
    *,
    vocab_size: int,
    batch_size: int = 8,
    torch_module: Any | None = None,
) -> float:
    torch = torch_module
    if torch is None:
        import torch as torch
    started = time.perf_counter()
    values = [int(value) for value in ks]
    pool: dict[tuple[str, int], list[Any]] = {}

    def acquire(dtype: Any, k: int) -> Any:
        key = (str(dtype), k)
        if pool.get(key):
            return pool[key].pop()
        return torch.empty((k,), dtype=dtype, device="cpu", pin_memory=True)

    def release(dtype: Any, k: int, buffer: Any) -> None:
        pool.setdefault((str(dtype), k), []).append(buffer)

    def release_triplet(
        k: int, ids_buffer: Any, probs_buffer: Any, logs_buffer: Any
    ) -> None:
        release(torch.int64, k, ids_buffer)
        release(torch.float32, k, probs_buffer)
        release(torch.float32, k, logs_buffer)

    for batch_start in range(0, len(values), batch_size):
        for offset, k in enumerate(values[batch_start : batch_start + batch_size]):
            index = batch_start + offset
            ids = (
                torch.arange(k, device="cuda", dtype=torch.int64)
                .add_(index * 17)
                .remainder(vocab_size)
            )
            probs = torch.full((k,), 0.5 / k, device="cuda", dtype=torch.float32)
            _ = torch.log(probs) + ids.to(torch.float32) * 0.0
    torch.cuda.synchronize()
    return time.perf_counter() - started
