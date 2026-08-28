from pathlib import Path

from radjax_tome.cli.mainline import parser, run


def test_public_help_has_only_six_commands() -> None:
    commands = parser()._subparsers._group_actions[0].choices
    assert tuple(commands) == ("build", "validate", "inspect", "package", "doctor", "research")


def test_preflight_only_uses_canonical_output_override(tmp_path: Path) -> None:
    # Invalid invocation is classified before any production import or write.
    args = parser().parse_args(["build", "--config", str(tmp_path / "missing.json")])
    result = run(args)
    assert result.exit_code == 2
    assert result.error is not None
