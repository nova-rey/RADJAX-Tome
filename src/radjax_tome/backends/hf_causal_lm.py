from __future__ import annotations

from importlib import import_module
from typing import Any


def load_hf_causal_lm_classes() -> tuple[Any, Any, Any]:
    """Load optional HF teacher dependencies without making them mandatory."""
    try:
        torch = import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "HF causal-LM teacher mode requires optional dependency torch. "
            "Install the teacher-hf extra or use fake teacher mode."
        ) from exc
    try:
        transformers = import_module("transformers")
    except ImportError as exc:
        raise RuntimeError(
            "HF causal-LM teacher mode requires optional dependency transformers. "
            "Install the teacher-hf extra or use fake teacher mode."
        ) from exc
    return torch, transformers.AutoTokenizer, transformers.AutoModelForCausalLM
