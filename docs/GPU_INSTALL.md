# GPU Install

Spec 4.4 improves GPU teacher setup diagnostics for fresh machines. This page
is for local Linux/NVIDIA or Apple Silicon/MPS setup before production Tome
generation.

RADJAX-Tome does not install NVIDIA drivers. RADJAX-Tome does not silently download teacher models.

RADJAX-Tome does not choose one CUDA wheel that works for every platform.

## 1. Create A Fresh Venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. Install RADJAX-Tome

For editable development:

```bash
pip install -e .
```

For CPU/HF local teacher emission support:

```bash
pip install -e ".[teacher-hf]"
```

For HF/PyTorch teacher emission with GPU diagnostics/path support:

```bash
pip install -e ".[gpu-teacher]"
```

`gpu-teacher` currently installs the same `torch` and `transformers`
dependencies as `teacher-hf`, but names the intended GPU teacher workflow.

## 3. PyTorch CUDA Caveat

PyTorch CUDA wheels are platform-specific. If `pip install -e ".[gpu-teacher]"`
installs a CPU-only Torch build, follow PyTorch's official install selector for
your NVIDIA driver/CUDA platform and then rerun `radjax-tome doctor`.

Confirm Python can see CUDA:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## 4. Run Doctor

```bash
radjax-tome doctor \
  --teacher-backend gpu_torch \
  --runtime-mode cpu_gpu \
  --target-policy corridor_exemplar_v1 \
  --write-report runtime_doctor_report.json
```

The doctor report includes `gpu_install_diagnostics` with Python/platform,
Torch, Transformers, CUDA, MPS, JAX, and remediation fields. Missing `torch` or
`transformers` should recommend:

```bash
pip install -e ".[gpu-teacher]"
```

If Torch is installed but CUDA is unavailable, check the NVIDIA driver and
install a CUDA-enabled PyTorch build for your platform.

## 5. Prepare Local Inputs

Build local corpus provenance:

```bash
radjax-tome corpus build \
  --input ./sources \
  --output ./corpus_out \
  --overwrite
```

Inspect local teacher model provenance:

```bash
radjax-tome model inspect \
  --model-path /models/MODEL \
  --output teacher_model_provenance.json
```

The model must already be local or cached. RADJAX-Tome does not silently
download teacher models and does not perform network model verification.

## 6. Plan The GPU Run

Before a larger GPU build, write a run plan:

```bash
radjax-tome plan \
  --teacher-backend gpu_torch \
  --runtime-mode cpu_gpu \
  --target-policy corridor_exemplar_v1 \
  --teacher-model /models/MODEL \
  --tokenizer-id /models/MODEL \
  --dataset ./corpus_out/corpus.jsonl \
  --corpus-manifest ./corpus_out/corpus_manifest.json \
  --teacher-model-provenance teacher_model_provenance.json \
  --gpu-batch-size-mode auto \
  --gpu-batch-size-auto-min 1 \
  --gpu-batch-size-auto-max 64 \
  --output run_plan.json
```

The planner writes `gpu_run_plan_v1`, reuses doctor diagnostics, validates
supplied provenance, performs only bounded tiny auto-batch probes, and marks
memory/artifact estimates as rough. It does not download models or perform
network verification.

## 7. Tiny Local Smokes

Build a tiny fake artifact to confirm the package and artifact validators:

```bash
radjax-tome build \
  --teacher-mode fake \
  --output artifacts/fake_smoke \
  --max-examples 2 \
  --sequence-length 8 \
  --overwrite

radjax-tome validate --path artifacts/fake_smoke
```

When the model is already local and `doctor` says `can_emit=true`, try a tiny
GPU-routed local run:

```bash
radjax-tome build \
  --teacher-backend gpu_torch \
  --runtime-mode cpu_gpu \
  --teacher-model /models/MODEL \
  --teacher-model-provenance teacher_model_provenance.json \
  --dataset ./corpus_out/corpus.jsonl \
  --corpus-manifest ./corpus_out/corpus_manifest.json \
  --output artifacts/gpu_smoke \
  --max-examples 1 \
  --sequence-length 8 \
  --overwrite
```

Compare CPU/GPU or old/new artifacts after they exist:

```bash
radjax-tome parity \
  --left ./artifact_cpu \
  --right ./artifacts/gpu_smoke \
  --output parity_report.json
```

Spec 4.5 adds GPU run planning and bounded auto batch probing. It does not add
backend reducer changes, selector changes, production build orchestration,
streaming/resume, model downloading, network verification, Docker requirements,
multidevice scheduling, or TPU/JAX work.
