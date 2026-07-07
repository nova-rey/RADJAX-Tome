from __future__ import annotations

import importlib
import json
import tomllib
import types
from pathlib import Path

import radjax_tome.backends.gpu_torch as gpu_torch
import radjax_tome.reports.runtime_doctor as runtime_doctor
from radjax_tome.backends import TeacherBackendConfig
from radjax_tome.reports import (
    build_runtime_doctor_report,
    render_runtime_doctor_summary,
    write_runtime_doctor_report,
)

ROOT = Path(__file__).resolve().parents[1]


def _gpu_config(**overrides: object) -> TeacherBackendConfig:
    payload = {
        "backend_id": "gpu_torch",
        "runtime_mode": "cpu_gpu",
        "target_policy": "corridor_exemplar_v1",
        "model_id": "local/model",
        "tokenizer_id": "local/model",
        "local_files_only": True,
        "allow_downloads": False,
    }
    payload.update(overrides)
    return TeacherBackendConfig(**payload)


def _patch_imports(monkeypatch, mapping: dict[str, object]) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, package: str | None = None) -> object:
        if name in mapping:
            value = mapping[name]
            if isinstance(value, BaseException):
                raise value
            return value
        return real_import(name, package)

    monkeypatch.setattr(runtime_doctor, "import_module", fake_import)
    monkeypatch.setattr(gpu_torch, "import_module", fake_import)


def _fake_torch(*, cuda_available: bool) -> object:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return cuda_available

        @staticmethod
        def device_count() -> int:
            return 2 if cuda_available else 0

        @staticmethod
        def get_device_name(index: int) -> str:
            return ("NVIDIA L4", "NVIDIA A10")[index]

    class FakeMps:
        @staticmethod
        def is_available() -> bool:
            return False

    return types.SimpleNamespace(
        __version__="2.4.0+cu121",
        backends=types.SimpleNamespace(mps=FakeMps()),
        cuda=FakeCuda(),
        version=types.SimpleNamespace(cuda="12.1"),
    )


def _fake_transformers() -> object:
    loader = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: object())
    return types.SimpleNamespace(
        __version__="4.45.0",
        AutoModelForCausalLM=loader,
        AutoTokenizer=loader,
    )


def test_doctor_handles_missing_torch_without_crashing(monkeypatch) -> None:
    _patch_imports(
        monkeypatch,
        {
            "torch": ImportError("no torch"),
            "transformers": ImportError("no transformers"),
            "jax": ImportError("no jax"),
        },
    )

    report = build_runtime_doctor_report(_gpu_config())

    assert report["can_emit"] is False
    assert report["dependency_status"] == "missing_torch"
    assert report["gpu_install_diagnostics"]["torch_available"] is False
    assert 'pip install -e ".[gpu-teacher]"' in report["recommended_commands"]
    assert "gpu-teacher" in report["remediation_hint"]


def test_doctor_handles_missing_transformers_without_crashing(monkeypatch) -> None:
    _patch_imports(
        monkeypatch,
        {
            "torch": _fake_torch(cuda_available=True),
            "transformers": ImportError("no transformers"),
            "jax": ImportError("no jax"),
        },
    )

    report = build_runtime_doctor_report(_gpu_config())

    assert report["can_emit"] is False
    assert report["dependency_status"] == "missing_transformers"
    assert report["gpu_install_diagnostics"]["transformers_available"] is False
    assert "gpu-teacher" in report["remediation_hint"]


def test_doctor_reports_fake_torch_cuda_devices(monkeypatch, tmp_path: Path) -> None:
    _patch_imports(
        monkeypatch,
        {
            "torch": _fake_torch(cuda_available=True),
            "transformers": _fake_transformers(),
            "jax": ImportError("no jax"),
        },
    )

    report = build_runtime_doctor_report(_gpu_config())
    output = tmp_path / "doctor.json"
    write_runtime_doctor_report(report, output)
    rendered = "\n".join(render_runtime_doctor_summary(report))
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["can_emit"] is True
    assert report["torch_available"] is True
    assert report["torch_version"] == "2.4.0+cu121"
    assert report["transformers_version"] == "4.45.0"
    assert report["cuda_available"] is True
    assert report["cuda_device_count"] == 2
    assert report["cuda_device_names"] == ["NVIDIA L4", "NVIDIA A10"]
    assert report["torch_cuda_version"] == "12.1"
    assert "torch.cuda.is_available=true" in rendered
    assert "cuda_device_0_name=NVIDIA L4" in rendered
    assert "gpu_install_diagnostics" in written


def test_doctor_reports_cuda_unavailable_with_remediation(monkeypatch) -> None:
    _patch_imports(
        monkeypatch,
        {
            "torch": _fake_torch(cuda_available=False),
            "transformers": _fake_transformers(),
            "jax": ImportError("no jax"),
        },
    )

    report = build_runtime_doctor_report(_gpu_config())
    rendered = "\n".join(render_runtime_doctor_summary(report))

    assert report["can_emit"] is False
    assert report["failure_stage"] == "no_accelerator"
    assert report["cuda_available"] is False
    assert report["cuda_device_count"] == 0
    assert "CUDA-enabled PyTorch build" in report["remediation_hint"]
    assert "allow_downloads" not in report["remediation_hint"]
    assert "torch.cuda.is_available=false" in rendered


def test_gpu_teacher_extra_and_teacher_hf_extra_are_declared() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    assert "teacher-hf" in extras
    assert "gpu-teacher" in extras
    assert set(extras["teacher-hf"]) == set(extras["gpu-teacher"])
    assert any(requirement.startswith("torch") for requirement in extras["gpu-teacher"])
    assert any(
        requirement.startswith("transformers") for requirement in extras["gpu-teacher"]
    )


def test_gpu_install_docs_and_bible_cover_spec_4_4() -> None:
    docs = (ROOT / "docs" / "GPU_INSTALL.md").read_text(encoding="utf-8")
    matrix = (ROOT / "docs" / "TOME_GENERATION_CAPABILITY_MATRIX.md").read_text(
        encoding="utf-8"
    )
    bible = (ROOT / "bible.md").read_text(encoding="utf-8")

    assert "python3 -m venv .venv" in docs
    assert 'pip install -e ".[gpu-teacher]"' in docs
    assert 'pip install -e ".[teacher-hf]"' in docs
    assert "PyTorch CUDA wheels" in docs
    assert "RADJAX-Tome does not install NVIDIA drivers" in docs
    assert "does not silently download teacher models" in docs
    assert "radjax-tome doctor" in docs
    assert "radjax-tome model inspect" in docs
    assert "radjax-tome corpus build" in docs
    assert "radjax-tome parity" in docs
    assert "Spec 4.4" in matrix
    assert "Spec 4.4" in bible
