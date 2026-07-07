# Teacher Model Provenance

Spec 4.2 adds local teacher model provenance and setup utilities.

RADJAX-Tome is not a teacher-model acquisition frontend. RADJAX-Tome does not silently download teacher models.

It does not verify upstream Hugging Face state over the network and does not
choose a model for you. The user owns model acquisition. RADJAX-Tome owns local
inspection, local file hashing, honest identity metadata, validation, and Tome
provenance linkage.

## Inspect A Local Model

Inspect a local model directory or local Hugging Face cache snapshot:

```bash
radjax-tome model inspect \
  --model-path /models/gpt2 \
  --output teacher_model_provenance.json
```

The default check is `metadata_only`. It hashes local files and writes a
`teacher_model_provenance_v1` sidecar without importing Transformers, loading a
model, using GPU, or touching the network.

Validate the sidecar:

```bash
radjax-tome model validate --provenance teacher_model_provenance.json
```

Discover only reports candidate directories; it does not auto-select one:

```bash
radjax-tome model discover --search-path /models
```

## What Gets Hashed

The inspector recognizes config files such as `config.json` and
`generation_config.json`, tokenizer files such as `tokenizer.json`,
`tokenizer_config.json`, `vocab.json`, `merges.txt`, and `tokenizer.model`, and
weight files such as `model.safetensors`, `*.safetensors`,
`pytorch_model.bin`, and `pytorch_model-*.bin`.

Each record includes `relative_path`, `size_bytes`, and `sha256`. Aggregate
hashes are written as `config_hash`, `tokenizer_hash`, `weights_hash`, and
`model_directory_hash`.

## Identity Confidence

The sidecar separates friendly identity from file proof:

- `verified`: identity came directly from inspected local files such as
  `config.json`.
- `inferred`: identity came from local path conventions, such as a Hugging Face
  cache snapshot path.
- `declared`: identity was supplied by the user with `--model-name` or
  `--model-revision`.
- `unknown`: no friendly identity could be determined or supplied.

In short: verified vs inferred vs declared identity is explicit, while file
hashes remain the hard local proof.

For a local Hugging Face cache snapshot shaped like:

```text
.../models--org--model-name/snapshots/<revision>/
```

RADJAX-Tome infers `hf_repo_id=org/model-name` and `hf_revision=<revision>` from
the local path only. This is not network verification.

For ordinary local directories, declare identity when useful:

```bash
radjax-tome model inspect \
  --model-path ./local_teacher \
  --model-name "my-custom-model" \
  --model-revision "manual-2026-07-07" \
  --output teacher_model_provenance.json
```

## Build With Provenance

Pass the validated sidecar into Tome generation:

```bash
radjax-tome build \
  --teacher-model ./local_teacher \
  --teacher-model-provenance ./teacher_model_provenance.json \
  --output artifacts/from_local_teacher \
  --teacher-mode fake \
  --overwrite
```

Generated artifacts record a compact teacher model provenance summary in
`metadata.json` target params, `teacher_manifest.json`, `emission_config.json`,
and `cover_page.json`. The full file inventory stays in the sidecar.

Spec 4.2 records `network_used=false`, `local_files_only=true`,
`allow_downloads=false`, and `downloaded_by_radjax_tome=false`. Any future
download behavior must be explicit, opt-in, and recorded separately.
