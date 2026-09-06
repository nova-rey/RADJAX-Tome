from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sys

import pytest

from radjax_tome.cli.main import main
from radjax_tome.tui.process import _run_json_process


def test_json_tui_is_rejected_and_headless_fallback_is_actionable() -> None:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exit_code = main(["--json", "tui", "corpus"])
    payload = json.loads(output.getvalue())
    assert exit_code == 2
    assert payload["error"]["code"] == "INVALID_CONFIGURATION"


def test_textual_workflows_mount_and_resize() -> None:
    textual = pytest.importorskip("textual")
    del textual
    from radjax_tome.tui.app import WizardApp

    async def exercise(workflow: str) -> None:
        async with WizardApp(workflow).run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert pilot.app.query_one("#title").display
            await pilot.resize_terminal(60, 18)
            await pilot.pause()
            notice = str(pilot.app.query_one("#notice").render())
            assert "Small terminal" in notice

    asyncio.run(exercise("corpus"))
    asyncio.run(exercise("production"))


def test_textual_save_button_requires_a_new_filename() -> None:
    pytest.importorskip("textual")
    from radjax_tome.tui.app import WizardApp

    async def exercise() -> None:
        async with WizardApp("corpus").run_test(size=(100, 30)) as pilot:
            await pilot.press("ctrl+s")
            await pilot.pause()
            assert "new config filename" in str(pilot.app.query_one("#status").render())

    asyncio.run(exercise())


def test_subprocess_transport_bounds_stderr() -> None:
    async def exercise() -> None:
        result = await _run_json_process(
            [
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('x' * 4096); print('{}')",
            ],
            stderr_limit=128,
            stdout_limit=1024,
        )
        assert result.result == {}
        assert result.transport_error == "CLI stderr exceeded 128 byte limit"
        assert len(result.stderr.encode()) == 128

    asyncio.run(exercise())


def test_subprocess_transport_interrupts_and_preserves_state() -> None:
    async def exercise() -> None:
        cancel = asyncio.Event()
        force = asyncio.Event()
        task = asyncio.create_task(
            _run_json_process(
                [
                    sys.executable,
                    "-c",
                    "import time; print('{}', flush=True); time.sleep(30)",
                ],
                stderr_limit=128,
                stdout_limit=1024,
                cancel_event=cancel,
                force_event=force,
            )
        )
        await asyncio.sleep(0.1)
        cancel.set()
        result = await asyncio.wait_for(task, timeout=5)
        assert result.returncode == 130
        assert result.transport_error == "canonical CLI interrupted"

    asyncio.run(exercise())
