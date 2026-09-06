# M11 independent read-only review

Reviewer: registered read-only auditor `Receipts`.
Review target: `20cbbdeb732f4717d668791c7690967f6ae10690`.
Review scope: approved M11 plan, implementation, tests, and closure evidence.

## Initial verdict

`FAIL`

The auditor reported these material findings:

1. The TUI was a generic editor/button row rather than workflow-specific
   wizard forms and navigation.
2. Cancellation was not wired and subprocess transport used unbounded
   `communicate()` buffering.
3. Production preflight did not consistently use effective resume/overwrite
   values, and the TUI did not invoke the read-only native resume resolver.
4. Execution did not byte-check the saved configuration against the edited
   document.

The auditor also recorded two reservations: closure evidence lacked durable
workflow receipts/help/keyboard/cancellation snapshots, and its own environment
could not import the pinned Contract package. It recorded one suggestion to
include `tui` in the recommended-command text.

## Bounded correction

The single permitted correction pass implemented workflow-specific Inputs,
Behavior, Resources/Destination, and Review/Advanced tabs with field controls;
bounded concurrent stdout/stderr draining; confirmed Ctrl-C interruption and
second-confirmation force-stop; effective production resume/overwrite
preflight; native resume projection; exact saved-byte checks; and durable help,
command, and review evidence. Focused regressions cover bounded stderr,
interruption, saved-byte mismatch, optional dependency isolation, round trips,
resize, and Save As.

## Targeted recheck

The corrected implementation passed the focused M11/TUI/config/CLI tests and
the existing mandatory repository gates were rerun after the correction. No
second reviewer was invoked; this report preserves the original FAIL and the
bounded correction result rather than fabricating a fresh independent verdict.
