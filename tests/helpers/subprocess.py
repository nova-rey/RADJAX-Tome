from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def repo_python_env(root: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
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
