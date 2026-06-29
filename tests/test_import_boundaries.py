from pathlib import Path


def test_tome_does_not_import_student() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "radjax_tome"
    offenders: list[str] = []
    forbidden = (
        "import radjax_student",
        "from radjax_student",
        "qrwkv_xla.students",
        "qrwkv_xla.training",
        "qrwkv_xla.checkpointing",
        "import jax",
        "from jax",
        "pallas",
    )
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(pattern in text for pattern in forbidden):
            offenders.append(str(path.relative_to(root)))

    assert offenders == []
