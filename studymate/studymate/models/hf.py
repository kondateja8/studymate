from __future__ import annotations

import os
from typing import Optional

from huggingface_hub import InferenceClient


def _model_id() -> str:
    return os.getenv("HF_TEXT_GEN_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")


def _client() -> Optional[InferenceClient]:
    token = os.getenv("HF_API_TOKEN")
    if not token:
        return None
    return InferenceClient(model=_model_id(), token=token, timeout=60)


def generate_answer(prompt: str) -> str:
    client = _client()
    if client is None:
        return "[LLM unavailable] Provide HF_API_TOKEN in your environment to enable answers."

    max_new_tokens = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))
    temperature = float(os.getenv("HF_TEMPERATURE", "0.2"))
    top_p = float(os.getenv("HF_TOP_P", "0.9"))

    try:
        text = client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            stream=False,
            return_full_text=False,
        )
        return text
    except Exception as e:
        return f"[LLM error] {e}"
