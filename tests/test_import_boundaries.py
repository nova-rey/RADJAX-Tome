from pathlib import Path


def test_tome_does_not_import_student() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "radjax_tome"
    offenders: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "import radjax_student" in text or "from radjax_student" in text:
            offenders.append(str(path.relative_to(root)))

    assert offenders == []
