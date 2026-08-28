# M9 Opinionated Mainline CLI Closure

Status: implementation checkpoint complete on m9/opinionated-mainline-cli.

The verified mainline was fast-forwarded from 6a6c65378cfd86a190e44e861ed9323927c2acc8; accepted M8 closure 88d6f418e0f39ab2ec61d9047f974f04657e5214 remains historical and unchanged. Contract remains pinned to 373e3d17060d4ce1c4a0db6065c9289da714bde7.

## Surface

radjax-tome run --intent INTENT.json|yaml loads exact M5 radjax_tome_build_intent_v1 sections, rejects unknown/missing fields, performs destination preflight without filesystem mutation, and delegates to the existing normalized production builder/state machine. status, package, and explicit research containment are available alongside established validation/inspection commands.

Operational overrides are limited to --resume and --overwrite; destination classification is missing, empty_directory, nonempty_directory, or file. Nonempty destinations are preserved unless an explicit override is supplied.

## Acceptance receipts

- Contract dependency pin regression and buffer-native import: tests/test_m9_dependency_pin.py.
- Strict loader and mutation-free destination preflight: tests/test_m9_cli.py.
- M5 normalization and production routing suites continue to pass.
- Package build, compile, Ruff, formatting, and diff checks are recorded with this closure.
- No GPU, Modal, Student, M10+, or historical evidence was changed.
