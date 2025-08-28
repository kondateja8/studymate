from __future__ import annotations

import os
from typing import Optional

from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams


def _get_watsonx_model_id() -> str:
    # Public Mixtral model id in watsonx catalog
    # Reference ids can change; expose via env override when needed
    return os.getenv("IBM_WATSONX_MODEL_ID", "mistralai/mixtral-8x7b-instruct-v01")


def _resolve_project_or_space() -> dict:
    project_id = os.getenv("IBM_WATSONX_PROJECT_ID")
    space_id = os.getenv("IBM_WATSONX_SPACE_ID")
    if project_id:
        return {"project_id": project_id}
    if space_id:
        return {"space_id": space_id}
    return {}


def get_watsonx_model() -> Optional[Model]:
    api_key = os.getenv("IBM_WATSONX_API_KEY")
    url = os.getenv("IBM_WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    identifiers = _resolve_project_or_space()

    if not api_key or not url or not identifiers:
        return None

    params = {
        GenParams.MAX_NEW_TOKENS: int(os.getenv("WATSONX_MAX_NEW_TOKENS", "512")),
        GenParams.TEMPERATURE: float(os.getenv("WATSONX_TEMPERATURE", "0.2")),
        GenParams.TOP_P: float(os.getenv("WATSONX_TOP_P", "0.9")),
        GenParams.REPETITION_PENALTY: float(os.getenv("WATSONX_REPETITION_PENALTY", "1.1")),
        GenParams.DECODING_METHOD: "greedy",
    }

    return Model(
        model_id=_get_watsonx_model_id(),
        params=params,
        credentials={"apikey": api_key, "url": url},
        **identifiers,
    )


def generate_answer(prompt: str) -> str:
    model = get_watsonx_model()
    if model is None:
        return "[LLM unavailable] Please configure IBM watsonx credentials to enable answer generation."
    response = model.generate_text(prompt=prompt)
    if isinstance(response, dict):
        return response.get("results", [{}])[0].get("generated_text", "")
    return str(response)
