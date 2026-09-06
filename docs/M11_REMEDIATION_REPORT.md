# M11 bounded remediation report

This report records the finite M11 implementation corrections and the single
independent read-only review. It does not claim a second independent review.

## Configuration and workflow

Canonical production and corpus intent documents now round-trip without losing
advanced fields or v2 corpus identity. Lossy v2 exports are rejected, v1
normalization and selection authority are preserved, and Save As is exclusive
and fsynced. The public TUI uses canonical preflight, build, validate, and
package commands. Textual is optional and lazy.

## Production integration

The v2 artifact reference is resolved once and carried through production
planning, streaming, teacher-textbook validation, score/selection, selected
rerun, progress reporting, and final validation. Corpus semantic identity and
tokenizer binding are checked before model/backend execution. The public CPU
smoke proves the complete v2 route, including selected rerun and final Tome
validation.

## Bounded corpus path

The explicit dedup-disabled path now streams records directly with bounded
counters instead of opening an unnecessary DuckDB pipeline. The enabled path
retains cursor-safe winner selection and streamed duplicate provenance. The
50K proof and large duplicate-group regressions preserve counts, identity, and
provenance without materializing the corpus.

## Architecture and inventory

Artifact validation consumes the lower-level verified corpus reader rather than
creating a builder-to-validation import cycle. The Hydra disposition inventory
contains every new M11 module. No Contract, Student, Golden evidence, or M8
behavior was modified.

## External review status

The registered read-only auditor returned `FAIL` on implementation
`20cbbdeb`. One bounded correction pass fixed each blocking finding: workflow
tabs and field controls, bounded subprocess streaming with confirmed interrupt
and force-stop, effective resume/overwrite preflight plus native resume
projection, and exact saved-byte checks. The targeted M11/config/CLI tests,
full suite, quality gates, and clean-install smoke passed afterward. No second
reviewer was invoked; the preserved initial report and correction evidence are
the auditable record.
