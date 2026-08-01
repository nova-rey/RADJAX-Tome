from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def repo_python_env(root: Path) -> dict[str, str]:
    source = str(root / "src")
    inherited = os.environ.get("PYTHONPATH")
    pythonpath = source if not inherited else os.pathsep.join((source, inherited))
    return {
        **os.environ,
        # Keep the caller's dependency/source overlays.  Contract parity
        # runners intentionally point at a pinned Contract source tree; a
        # subprocess must not silently fall back to a globally installed,
        # older Contract while testing Tome's public CLI path.
        "PYTHONPATH": pythonpath,
    }


def run_repo_python(
    root: Path,
    *args: str,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=root,
        env=repo_python_env(root),
        text=True,
        capture_output=True,
        check=check,
    )


def run_cli(
    root: Path,
    *args: str,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return run_repo_python(
        root,
        "-m",
        "radjax_tome.cli.main",
        *args,
        check=check,
    )


def run_script(
    root: Path,
    script: str,
    *args: str,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return run_repo_python(root, script, *args, check=check)
