from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_project_ledger_bible_exists_and_mentions_current_arc() -> None:
    text = (ROOT / "bible.md").read_text(encoding="utf-8")

    assert "2.14" in text
    assert "2.18" in text
    assert "3.0" in text
    assert "3.1" in text
    assert "cover_page.json" in text
    assert "3.2" in text
    assert ".rtome" in text
    assert "deterministic tar" in text
    assert "3.3A" in text
    assert "runtime mode" in text
    assert "capability matrix" in text
    assert "3.3B" in text
    assert "backend contract" in text
    assert "fake_numpy" in text
    assert "3.3C" in text
    assert "CPU reference backend" in text
    assert "cpu_reference" in text
