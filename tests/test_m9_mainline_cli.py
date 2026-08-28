from pathlib import Path
import json
from dataclasses import asdict

from radjax_tome.builder.config import canonical_production_build_intent
from radjax_tome.cli.main import main
from radjax_tome.cli.mainline import parser, run


def test_public_help_has_only_six_commands() -> None:
    commands = parser()._subparsers._group_actions[0].choices
    assert tuple(commands) == (
        "build",
        "validate",
        "inspect",
        "package",
        "doctor",
        "research",
    )


def test_preflight_only_uses_canonical_output_override(tmp_path: Path) -> None:
    source = canonical_production_build_intent(
        teacher_model="model",
        dataset_path=Path("data.jsonl"),
        corpus_manifest_path=Path("manifest.json"),
        teacher_model_provenance_path=Path("provenance.json"),
        output_dir=Path("config-output"),
    )
    payload = asdict(source)

    def encode(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        return value

    config = tmp_path / "intent.json"
    config.write_text(json.dumps(encode(payload)))
    args = parser().parse_args(
        [
            "build",
            "--config",
            str(config),
            "--output",
            str(tmp_path / "override"),
            "--preflight-only",
        ]
    )
    result = run(args)
    assert result.exit_code == 0
    assert result.artifact["workspace"] == str(tmp_path / "override")


def test_public_research_help_is_routed_without_legacy_required_command() -> None:
    assert main(["research", "--help"]) == 0
