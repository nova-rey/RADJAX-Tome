"""Bounded subprocess transport for canonical CLI execution."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    result: dict[str, object] | None
    stderr: str
    transport_error: str | None = None


async def _read_bounded(
    stream: asyncio.StreamReader,
    limit: int,
) -> tuple[bytes, bool]:
    """Drain a child stream while retaining at most ``limit`` bytes."""

    retained = bytearray()
    exceeded = False
    total = 0
    while True:
        chunk = await stream.read(64 * 1024)
        if not chunk:
            return bytes(retained), exceeded
        total += len(chunk)
        if len(retained) < limit:
            retained.extend(chunk[: limit - len(retained)])
        if total > limit:
            exceeded = True


async def _send_interrupt(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "posix" and process.pid:
        os.killpg(process.pid, signal.SIGINT)
    else:
        process.send_signal(signal.SIGINT)


async def _run_json_process(
    argv: list[str],
    *,
    stderr_limit: int,
    stdout_limit: int,
    cancel_event: asyncio.Event | None = None,
    force_event: asyncio.Event | None = None,
    process_holder: Callable[[asyncio.subprocess.Process | None], None] | None = None,
) -> ProcessResult:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=(os.name == "posix"),
    )
    assert process.stdout is not None
    assert process.stderr is not None
    if process_holder is not None:
        process_holder(process)

    stdout_task = asyncio.create_task(_read_bounded(process.stdout, stdout_limit))
    stderr_task = asyncio.create_task(_read_bounded(process.stderr, stderr_limit))
    wait_task = asyncio.create_task(process.wait())
    cancel_task: asyncio.Task[bool] | None = None
    force_task: asyncio.Task[bool] | None = None
    interrupted = False
    try:
        waiters: set[asyncio.Task[Any]] = {wait_task}
        if cancel_event is not None:
            cancel_task = asyncio.create_task(cancel_event.wait())
            waiters.add(cancel_task)
        done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        if cancel_task is not None and cancel_task in done and not wait_task.done():
            interrupted = True
            await _send_interrupt(process)
            force_task = asyncio.create_task(
                force_event.wait() if force_event is not None else asyncio.sleep(10**6)
            )
            done, _ = await asyncio.wait(
                {wait_task, force_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if force_task in done and not wait_task.done():
                process.kill()
                await wait_task
        await asyncio.gather(wait_task, stdout_task, stderr_task)
        stdout, stdout_exceeded = await stdout_task
        stderr, stderr_exceeded = await stderr_task
    finally:
        if process_holder is not None:
            process_holder(None)
        for task in (cancel_task, force_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (cancel_task, force_task) if task is not None),
            return_exceptions=True,
        )

    transport_error: str | None = None
    if stdout_exceeded:
        transport_error = f"CLI stdout exceeded {stdout_limit} byte limit"
    elif stderr_exceeded:
        transport_error = f"CLI stderr exceeded {stderr_limit} byte limit"
    if interrupted:
        return ProcessResult(
            130,
            None,
            stderr.decode("utf-8", errors="replace"),
            transport_error or "canonical CLI interrupted",
        )
    try:
        result = json.loads(stdout.decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError("CLI result is not an object")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return ProcessResult(
            process.returncode or 1,
            None,
            stderr.decode("utf-8", errors="replace"),
            transport_error or f"invalid CLI JSON result: {exc}",
        )
    return ProcessResult(
        process.returncode or 0,
        result,
        stderr.decode("utf-8", errors="replace"),
        transport_error,
    )


async def run_canonical_cli(
    workflow: str,
    config: str | Path,
    *,
    stderr_limit: int = 256 * 1024,
    stdout_limit: int = 1024 * 1024,
    cancel_event: asyncio.Event | None = None,
    force_event: asyncio.Event | None = None,
    process_holder: Callable[[asyncio.subprocess.Process | None], None] | None = None,
) -> ProcessResult:
    command = "corpus" if workflow == "corpus" else "build"
    argv = [
        sys.executable,
        "-m",
        "radjax_tome",
        "--json",
        command,
        "build" if command == "corpus" else "--config",
        *(["--config", str(config)] if command == "corpus" else [str(config)]),
    ]
    return await _run_json_process(
        argv,
        stderr_limit=stderr_limit,
        stdout_limit=stdout_limit,
        cancel_event=cancel_event,
        force_event=force_event,
        process_holder=process_holder,
    )


async def run_package_cli(
    workspace: str | Path,
    output: str | Path,
    *,
    profile: str = "student",
    transport: str = "directory",
    stderr_limit: int = 256 * 1024,
    stdout_limit: int = 1024 * 1024,
) -> ProcessResult:
    return await _run_json_process(
        [
            sys.executable,
            "-m",
            "radjax_tome",
            "--json",
            "package",
            str(workspace),
            "--output",
            str(output),
            "--profile",
            profile,
            "--transport",
            transport,
        ],
        stderr_limit=stderr_limit,
        stdout_limit=stdout_limit,
    )


async def interrupt_process(process: asyncio.subprocess.Process) -> None:
    await _send_interrupt(process)
    await process.wait()


__all__ = [
    "ProcessResult",
    "interrupt_process",
    "run_canonical_cli",
    "run_package_cli",
]
