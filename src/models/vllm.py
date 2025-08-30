"""
vLLM-backed local model implementation.
"""

import time
import asyncio
from typing import Any, Dict, Optional

from .base import BaseModel, ModelConfig, ModelResponse

try:
    from vllm import LLM, SamplingParams  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    LLM = None
    SamplingParams = None


PROPRIETARY_PREFIXES = (
    "gemini",
)


class VLLMModel(BaseModel):
    """Local model loaded via vLLM for open-source weights.

    Expects an HF model id or local path in config.additional_params['hf_model'].
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        if LLM is None:
            raise ImportError(
                "vllm is not installed. Please install it (e.g., `pip install vllm`)."
            )

        # Determine which model to load locally via vLLM
        hf_model: Optional[str] = (
            config.additional_params.get("hf_model")
            or config.additional_params.get("model")
            or config.model_name
        )

        if not hf_model:
            raise ValueError(
                "No Hugging Face model id provided for local vLLM. Set additional_params.hf_model or pass via config file."
            )

        # Basic guard against obviously proprietary endpoints
        lower_name = (hf_model or "").lower()
        if lower_name.startswith(PROPRIETARY_PREFIXES):
            raise ValueError(
                f"Model '{hf_model}' cannot be loaded locally via vLLM. Provide an open-source HF model id (e.g., 'google/gemma-3-4b-it')."
            )

        engine_kwargs: Dict[str, Any] = {}
        for key in (
            "tensor_parallel_size",
            "dtype",
            "max_model_len",
            "trust_remote_code",
            "download_dir",
            "gpu_memory_utilization",
        ):
            if key in config.additional_params:
                engine_kwargs[key] = config.additional_params[key]

        self._hf_model = hf_model
        self._engine = LLM(model=hf_model, **engine_kwargs)

    def _make_sampling_params(self, **overrides) -> "SamplingParams":
        temperature = overrides.get("temperature", self.config.temperature)
        max_tokens = overrides.get("max_tokens", self.config.max_tokens or 512)
        top_p = overrides.get("top_p", self.config.top_p)
        top_k = overrides.get("top_k", self.config.top_k)
        stop = overrides.get("stop_sequences", self.config.stop_sequences)
        return SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )

    def generate_sync(self, prompt: str, **kwargs) -> ModelResponse:
        sampling_params = self._make_sampling_params(**kwargs)
        start = time.time()
        outputs = self._engine.generate([prompt], sampling_params)
        latency = time.time() - start

        output = outputs[0]
        text = output.outputs[0].text if output.outputs else ""

        prompt_tokens = None
        completion_tokens = None
        try:
            prompt_tokens = len(getattr(output, "prompt_token_ids", []) or [])
            if output.outputs and hasattr(output.outputs[0], "token_ids"):
                completion_tokens = len(output.outputs[0].token_ids)
        except Exception:
            pass

        return ModelResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency=latency,
            metadata={"backend": "vllm", "hf_model": self._hf_model},
        )

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.generate_sync(prompt, **kwargs))


