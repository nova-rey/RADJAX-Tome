from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

app = modal.App("radjax-m13-golden-smoke")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers",
    "safetensors",
)
workload = modal.Volume.from_name("m8g-current-1k-workload")


@app.function(image=image, gpu="T4", timeout=900, volumes={"/inputs": workload})
def smoke(expected_tome: str, expected_contract: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    corpus = Path("/inputs/current/corpus/corpus.jsonl")
    model_root = "/inputs/current/model"
    first = json.loads(corpus.open(encoding="utf-8").readline())
    text = first.get("text") or first.get("content") or first.get("prompt")
    if not isinstance(text, str) or not text:
        raise RuntimeError("first corpus record has no usable text")
    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_root,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        .to("cuda")
        .eval()
    )
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = encoded["input_ids"].to("cuda")
    attention_mask = encoded["attention_mask"].to("cuda")
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
    torch.cuda.synchronize()
    return json.dumps(
        {
            "status": "pass",
            "tome_commit": expected_tome,
            "contract_commit": expected_contract,
            "model_config_sha256": hashlib.sha256(
                Path(model_root, "config.json").read_bytes()
            ).hexdigest(),
            "tokenizer_config_sha256": hashlib.sha256(
                Path(model_root, "tokenizer_config.json").read_bytes()
            ).hexdigest(),
            "tokens": int(input_ids.shape[-1]),
            "logit_shape": list(out.logits.shape),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        sort_keys=True,
    )


@app.local_entrypoint()
def main() -> None:
    result = smoke.remote(
        "97907b3681e49bfc107866d480596981c2eccaa6",
        "373e3d17060d4ce1c4a0db6065c9289da714bde7",
    )
    print(result)
