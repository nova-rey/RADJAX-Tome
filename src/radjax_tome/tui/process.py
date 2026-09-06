"""Bounded subprocess transport for canonical CLI execution."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    result: dict[str, object] | None
    stderr: str
    transport_error: str | None = None


async def run_canonical_cli(
    workflow: str,
    config: str | Path,
    *,
    stderr_limit: int = 256 * 1024,
) -> ProcessResult:
    command = "corpus" if workflow == "corpus" else "build"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "radjax_tome",
        "--json",
        command,
        "build" if command == "corpus" else "--config",
        *(["--config", str(config)] if command == "corpus" else [str(config)]),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=(os.name == "posix"),
    )
    stdout, stderr = await process.communicate()
    clipped = stderr[:stderr_limit]
    try:
        result = json.loads(stdout.decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError("CLI result is not an object")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return ProcessResult(
            process.returncode or 1,
            None,
            clipped.decode("utf-8", errors="replace"),
            f"invalid CLI JSON result: {exc}",
        )
    return ProcessResult(
        process.returncode or 0,
        result,
        clipped.decode("utf-8", errors="replace"),
    )


async def run_package_cli(
    workspace: str | Path,
    output: str | Path,
    *,
    profile: str = "student",
    transport: str = "directory",
    stderr_limit: int = 256 * 1024,
) -> ProcessResult:
    process = await asyncio.create_subprocess_exec(
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
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=(os.name == "posix"),
    )
    stdout, stderr = await process.communicate()
    clipped = stderr[:stderr_limit]
    try:
        result = json.loads(stdout.decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError("CLI result is not an object")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return ProcessResult(
            process.returncode or 1,
            None,
            clipped.decode("utf-8", errors="replace"),
            f"invalid CLI JSON result: {exc}",
        )
    return ProcessResult(
        process.returncode or 0,
        result,
        clipped.decode("utf-8", errors="replace"),
    )


async def interrupt_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    if os.name == "posix" and process.pid:
        os.killpg(process.pid, signal.SIGINT)
    else:
        process.send_signal(signal.SIGINT)
    await process.wait()


__all__ = [
    "ProcessResult",
    "interrupt_process",
    "run_canonical_cli",
    "run_package_cli",
]
