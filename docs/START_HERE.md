# Start here

## What Tome is

RADJAX-Tome is the teacher-side half of RADJAX. It turns a verified corpus and
teacher run into portable evidence: target stores, compact selected exemplars,
reports, manifests, provenance, and validation receipts. Contract defines byte
and validation rules; Student consumes a compatible package. Tome does not train
a Student and a corpus artifact is not a Student package.

The current public path is:

    corpus-v2 artifact
      -> canonical M5 intent
      -> mutation-free preflight
      -> M4 production lifecycle
      -> score/selection/selected rerun
      -> producer workspace
      -> validate / inspect
      -> separately confirmed package

## Install and check the environment

Build or obtain a wheel from the repository, then install it in a clean
environment. pyproject.toml pins the exact Contract commit
373e3d17060d4ce1c4a0db6065c9289da714bde7.

    python -m pip install ./dist/radjax_tome-*.whl
    radjax-tome --version
    radjax-tome doctor

The default install is CPU-safe and does not import optional Torch,
Transformers, or Textual just to show help. Install the teacher-hf extra for
the supported HF teacher backend and the tui extra for the optional wizard. A
real teacher run additionally requires model files, a matching tokenizer, disk
capacity, and an appropriate accelerator/runtime.

## Small offline CPU journey (executed)

This is the honest, cheap demonstration. It uses a local text source and the
smoke tokenizer; it is not teacher inference and does not establish Student
readiness.

1. Write the complete corpus intent from CORPUS_BUILDER.md to
   corpus-intent.json and put one txt file under sources/.
2. Build and verify the corpus:

       radjax-tome corpus build --config ./corpus-intent.json
       radjax-tome corpus validate ./artifact
       radjax-tome corpus inspect ./artifact

   The M12 clean-install journey executed this outside the checkout with one
   source and one record. JSON results reported status=pass, a sha256 semantic
   identity, and zero validation issues.
3. Use a complete M5 v1 or v2 intent to run preflight without loading a model:

       radjax-tome build --config ./tome-intent-v2.json --preflight-only

4. The accepted CPU reference smoke completed score, selection, selected rerun,
   publication, validation, and inspection for one example. Keep this output
   separate from production evidence and do not call it a real-teacher result.
5. Package only a corridor-selected producer workspace:

       radjax-tome package ./producer-workspace \
         --output ./student-package --profile student --transport directory
       radjax-tome --json validate ./student-package

   The synthetic dynamic CPU fixture intentionally lacks corridor artifacts and
   cannot produce a Student-compatible package. That is a supported
   precondition, not something to work around by fabricating files. A
   full-debug package of the accepted corridor-selected smoke workspace was
   exercised separately and validated.

## Real teacher template (not executed here)

Use a local model and tokenizer whose provenance can be captured. Keep
tokenizer_id aligned with the corpus binding; mismatch is rejected before
backend construction. Replace placeholders and run preflight first:

    radjax-tome build --config ./tome-intent-v2.json --preflight-only
    radjax-tome build --config ./tome-intent-v2.json

The v2 configuration above is canonical-loader-validated by the documentation checks. These are production templates, not a claim that this documentation run
downloaded a model or ran an accelerator. See CONFIGURATION_REFERENCE.md and
TEACHER_BACKENDS.md.

## Boundaries and safety

Read-only commands are doctor, corpus validate/inspect, build
--preflight-only, validate, and inspect. Corpus build, production build, and
package create outputs; package is a separate confirmed operation. Resume and
overwrite are explicit. An unrelated destination is preserved by default.
Cancellation and recovery are limited to durable lifecycle boundaries; no
universal filesystem atomicity is claimed.
