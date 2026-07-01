from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROADMAP_ARCS = ("3.0", "3.1", "3.2", "3.3", "3.4", "3.5")


def test_spec3_roadmap_exists_and_locks_ordering() -> None:
    path = ROOT / "docs" / "SPEC3_ROADMAP.md"
    text = path.read_text(encoding="utf-8")

    assert path.is_file()
    for arc in ROADMAP_ARCS:
        assert arc in text
    assert "not necessarily single-shot implementation specs" in text
    assert "Implement cover page for unpacked Tome" in text
    assert "Implement backend runtime modes" in text
    assert text.index("Implement cover page for unpacked Tome") < text.index(
        "Implement backend runtime modes"
    )
    assert "Cover page defines the artifact contract" in text
    assert "Backend runtime abstraction should exist before porting" in text
    assert "Dynamic top-k should wait" in text
    assert "3.3A" in text
    assert "Runtime Mode Capability Model" in text
    assert "3.3H" in text
    assert "Runtime Metadata + CLI/Doctor Polish" in text
    assert "3.3B | Backend Contract + Registry Skeleton | complete" in text
    assert "3.3A defines vocabulary" in text
    assert "3.3B defines the internal backend contract wall" in text
    assert "does not complete CPU/GPU/TPU runtime implementation" in text
    assert "3.3C establishes the CPU correctness baseline" in text
    assert "3.3D adds CPU orchestration and staged execution" in text
    assert "3.3E moves current HF Torch behavior behind the contract" in text
    assert "3.3F ports GPU compact/chunked reduction" in text
    assert "3.3G adds TPU/JAX shape without CUDA assumptions" in text
    assert "3.3H exposes backend status through CLI/doctor polish" in text


def test_optimization_handoff_doc_preserves_runtime_context() -> None:
    path = ROOT / "docs" / "TOME_OPTIMIZATION_BACKEND_HANDOFF.md"
    text = path.read_text(encoding="utf-8")

    assert path.is_file()
    assert "Historical code is reference material" in text
    assert "Do not copy it wholesale into RADJAX-Tome" in text
    assert "CPU-only" in text
    assert "CPU + GPU" in text
    assert "CPU + TPU" in text
    assert "non-CUDA-shaped abstraction" in text
    assert "reducer_workers" in text
    assert "true reducer-worker concurrency was deferred" in text
    assert "memory-scalability fix" in text
    assert "not an automatic throughput guarantee" in text


def test_optimization_inventory_json_is_deterministic_and_complete() -> None:
    path = ROOT / "docs" / "TOME_OPTIMIZATION_MIGRATION_INVENTORY.json"
    raw = path.read_text(encoding="utf-8")
    inventory = json.loads(raw)

    assert path.is_file()
    assert raw.endswith("\n")
    assert raw == json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    assert inventory["kind"] == "radjax_tome_optimization_migration_inventory"
    assert inventory["version"] == 1
    assert inventory["historical_repo"] == "nova-rey/qrwkv-xla"
    assert inventory["active_repo"] == "nova-rey/RADJAX-Tome"
    assert inventory["historical_commit"] == "6c21171bf76d341b476128d929d58469d4d06f18"

    runtime_modes = {item["id"]: item for item in inventory["desired_runtime_modes"]}
    assert {"cpu_only", "cpu_gpu", "cpu_tpu"} <= set(runtime_modes)
    assert runtime_modes["cpu_tpu"]["status"] == "new_design_required"
    assert "no TPU implementation exists" in runtime_modes["cpu_tpu"]["notes"]

    roadmap = {item["id"]: item for item in inventory["roadmap_arcs"]}
    assert set(ROADMAP_ARCS) <= set(roadmap)

    components = {item["id"]: item for item in inventory["historical_components"]}
    assert "chunked_gpu_vocab_reducer" in components
    assert "memory-scalability fix" in components["chunked_gpu_vocab_reducer"]["notes"]
    assert "real_teacher_capture_orchestrator" in components
    assert (
        "staged CPU pipeline"
        in components["real_teacher_capture_orchestrator"]["notes"]
    )

    serialized = json.dumps(inventory, sort_keys=True)
    assert "Do not claim TPU implementation already exists." in serialized
    assert "tpu_implemented" not in serialized
