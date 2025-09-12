"""
Gemini model implementation for AgenticEvals.
"""

from __future__ import annotations

import time
import asyncio
from typing import List, Dict, Any, Optional

try:
    from google import genai
    from google.genai.types import GenerateContentConfig
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from .base import BaseModel, ModelConfig, ModelResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Optional: google.api_core retry for older environments or HTTP-level retries
try:
    from google.api_core import retry as gapic_retry  # type: ignore
    GAPIC_RETRY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    gapic_retry = None
    GAPIC_RETRY_AVAILABLE = False

class GeminiModel(BaseModel):
    """Gemini model implementation using Google's Gen AI SDK."""
    
    def __init__(self, config: ModelConfig):
        """Initialize Gemini model."""
        super().__init__(config)
        
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai is not installed. Please install it with `pip install google-genai`")
        
        if not config.api_key:
            raise ValueError("API key is required for Gemini model")
        
        # Create client instead of configuring globally, with optional retry configuration
        retry_config = self._build_retry_config()
        try:
            if retry_config is not None:
                self.client = genai.Client(
                    api_key=config.api_key,
                    retry_config=retry_config,
                )
            else:
                self.client = genai.Client(api_key=config.api_key)
        except TypeError:
            # Older SDK versions may not accept retry_config argument
            logger.warning("Installed google-genai Client does not support retry_config; creating client without it.")
            self.client = genai.Client(api_key=config.api_key)
        
        # Store generation config for later use
        self.generation_config = self._build_generation_config()
        
        logger.info(f"Initialized Gemini model: {config.model_name}")
    
    def _build_generation_config(self) -> GenerateContentConfig:
        """Build generation configuration from model config."""
        config_dict = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "stop_sequences": self.config.stop_sequences,
        }
        
        # Filter out None values
        filtered_config = {k: v for k, v in config_dict.items() if v is not None}
        
        return GenerateContentConfig(
            **filtered_config,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

    def _build_retry_config(self) -> Optional[Any]:
        """Build client-level retry configuration for the Gen AI SDK.

        Uses exponential backoff for transient errors like RESOURCE_EXHAUSTED and UNAVAILABLE.
        """
        # Resolve RetryConfig class at runtime for compatibility across SDK versions
        RetryConfigClass = None
        try:
            RetryConfigClass = getattr(getattr(genai, "retry", None), "RetryConfig", None)
        except Exception:
            RetryConfigClass = None
        if RetryConfigClass is None:
            # Fallback to types.RetryConfig if available in this SDK version
            RetryConfigClass = getattr(types, "RetryConfig", None)

        # If no RetryConfig is available, proceed without retries
        if RetryConfigClass is None:
            logger.warning("RetryConfig not available in installed google-genai SDK; proceeding without client-level retries.")
            return None

        # Allow overrides via additional_params.retry_config if provided
        retry_overrides = {}
        if isinstance(self.config.additional_params, dict):
            retry_overrides = self.config.additional_params.get("retry_config", {}) or {}

        # Default retry configuration aligned with SDK guidance
        default_retry = {
            "initial_delay": 1.0,  # seconds
            "max_delay": 60.0,     # seconds
            "max_retries": 8,      # total attempts
            "retry_on_status_codes": [
                types.Code.RESOURCE_EXHAUSTED.value,
                types.Code.UNAVAILABLE.value,
            ],
        }

        # Merge overrides if present
        retry_kwargs = {**default_retry, **retry_overrides}

        return RetryConfigClass(**retry_kwargs)

    def _get_retry_policy_values(self) -> Dict[str, Any]:
        """Return retry policy values (used for manual or GAPIC retry).

        Reads overrides from additional_params.retry_config when present.
        """
        # Defaults aligned with SDK examples
        defaults = {
            "initial_delay": 1.0,
            "max_delay": 60.0,
            "max_retries": 8,
            "multiplier": 2.0,
            "deadline": 120.0,
        }
        overrides = {}
        if isinstance(self.config.additional_params, dict):
            overrides = self.config.additional_params.get("retry_config", {}) or {}
        # Accept either keys: initial/max vs initial_delay/max_delay
        normalized: Dict[str, Any] = {}
        normalized["initial_delay"] = overrides.get("initial_delay", overrides.get("initial", defaults["initial_delay"]))
        normalized["max_delay"] = overrides.get("max_delay", overrides.get("maximum", defaults["max_delay"]))
        normalized["max_retries"] = overrides.get("max_retries", overrides.get("retries", defaults["max_retries"]))
        normalized["multiplier"] = overrides.get("multiplier", defaults["multiplier"]) 
        normalized["deadline"] = overrides.get("deadline", overrides.get("timeout", defaults["deadline"]))
        # Ensure types
        return {
            "initial_delay": float(normalized["initial_delay"]),
            "max_delay": float(normalized["max_delay"]),
            "max_retries": int(normalized["max_retries"]),
            "multiplier": float(normalized["multiplier"]),
            "deadline": float(normalized["deadline"]),
        }

    def _is_transient_error(self, error: Exception) -> bool:
        """Heuristically determine if an error is transient and should be retried."""
        try:
            # Check numeric HTTP-like codes
            for attr in ("status", "status_code", "http_status", "code"):
                value = getattr(error, attr, None)
                if isinstance(value, int) and value in {408, 409, 429, 500, 502, 503, 504}:
                    return True
                if isinstance(value, str) and value.upper() in {"RESOURCE_EXHAUSTED", "UNAVAILABLE", "ABORTED", "DEADLINE_EXCEEDED"}:
                    return True
            # String-based fallback
            text = str(error).upper()
            if any(token in text for token in [
                "RESOURCE_EXHAUSTED", "UNAVAILABLE", "RATE LIMIT", "429", "503", "TIMEOUT", "DEADLINE EXCEEDED"
            ]):
                return True
        except Exception:
            pass
        return False

    def _retry_after_seconds(self, error: Exception) -> Optional[float]:
        """Extract retry-after hint (seconds) from an exception if present."""
        # Common locations: response headers, attributes
        try:
            response = getattr(error, "response", None)
            if response is not None:
                headers = getattr(response, "headers", None)
                if headers and "Retry-After" in headers:
                    try:
                        return float(headers.get("Retry-After"))
                    except Exception:
                        return None
            for attr in ("retry_after", "retry_delay"):
                value = getattr(error, attr, None)
                if isinstance(value, (int, float)):
                    return float(value)
        except Exception:
            return None
        return None

    def _call_with_retries_sync(self, func) -> tuple[Any, float]:
        """Call a function with retries and return (result, success_duration_seconds).

        Uses GAPIC Retry if available; otherwise manual truncated exponential backoff.
        Measures only the successful attempt duration, excluding backoff waits and failed attempts.
        """
        policy = self._get_retry_policy_values()
        if GAPIC_RETRY_AVAILABLE and gapic_retry is not None:
            retry = gapic_retry.Retry(
                predicate=gapic_retry.if_transient_error,
                initial=policy["initial_delay"],
                maximum=policy["max_delay"],
                multiplier=policy["multiplier"],
                deadline=policy["deadline"],
            )
            def timed_call():
                start = time.time()
                result = func()
                return result, time.time() - start
            return retry(timed_call)()

        # Manual truncated exponential backoff
        delay = policy["initial_delay"]
        max_delay = policy["max_delay"]
        max_retries = policy["max_retries"]
        multiplier = policy["multiplier"]
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                start = time.time()
                result = func()
                return result, time.time() - start
            except Exception as error:  # noqa: BLE001
                last_error = error
                if not self._is_transient_error(error) or attempt == max_retries:
                    raise
                hint = self._retry_after_seconds(error)
                sleep_for = hint if hint is not None else delay
                time.sleep(sleep_for)
                delay = min(max_delay, delay * multiplier)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Retry loop exited unexpectedly")

    async def _call_with_retries_async(self, func_coro) -> tuple[Any, float]:
        """Call an async function with manual truncated exponential backoff.

        Returns (result, success_duration_seconds). Only the successful attempt duration is measured.
        """
        policy = self._get_retry_policy_values()
        delay = policy["initial_delay"]
        max_delay = policy["max_delay"]
        max_retries = policy["max_retries"]
        multiplier = policy["multiplier"]
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                start = time.time()
                result = await func_coro()
                return result, time.time() - start
            except Exception as error:  # noqa: BLE001
                last_error = error
                if not self._is_transient_error(error) or attempt == max_retries:
                    raise
                hint = self._retry_after_seconds(error)
                sleep_for = hint if hint is not None else delay
                await asyncio.sleep(sleep_for)
                delay = min(max_delay, delay * multiplier)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Async retry loop exited unexpectedly")
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response asynchronously."""
        start_time = time.time()
        
        response = None
        try:
            async def _do_call():
                return await self.client.aio.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=self.generation_config
            )

            response, success_latency = await self._call_with_retries_async(_do_call)
            latency = success_latency
            
            # Fallback token counts if usage metadata is missing or zero
            prompt_tokens_override: Optional[int] = None
            completion_tokens_override: Optional[int] = None
            if isinstance(prompt, str) and prompt:
                prompt_tokens_override = await self._count_tokens_async(prompt)
            raw_text = getattr(response, 'text', None)
            if isinstance(raw_text, str) and raw_text:
                completion_tokens_override = await self._count_tokens_async(raw_text)

            return self._create_model_response(
                response,
                latency,
                completion_tokens_override=completion_tokens_override,
                prompt_tokens_override=prompt_tokens_override,
            )
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            if response is not None:
                logger.debug(f"Response object that caused error: {response}")
            raise
    
    def generate_sync(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response synchronously."""
        start_time = time.time()
        
        response = None
        try:
            def _do_call():
                return self.client.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=self.generation_config
            )

            response, success_latency = self._call_with_retries_sync(_do_call)
            latency = success_latency

            # Fallback token counts if usage metadata is missing or zero
            prompt_tokens_override: Optional[int] = None
            completion_tokens_override: Optional[int] = None
            if isinstance(prompt, str) and prompt:
                prompt_tokens_override = self._count_tokens_sync(prompt)
            raw_text = getattr(response, 'text', None)
            if isinstance(raw_text, str) and raw_text:
                completion_tokens_override = self._count_tokens_sync(raw_text)

            return self._create_model_response(
                response,
                latency,
                completion_tokens_override=completion_tokens_override,
                prompt_tokens_override=prompt_tokens_override,
            )
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            if response is not None:
                logger.debug(f"Response object that caused error: {response}")
            raise

    def _get_token_counts(self, response: Any) -> tuple[Optional[int], Optional[int], Optional[int]]:
        """Extract token counts from the response metadata across SDK variants."""
        try:
            usage = getattr(response, 'usage_metadata', None)
            if usage is None:
                usage = getattr(response, 'usage', None)
            if usage is None:
                return None, None, None

            def _read(obj: Any, names: List[str]) -> Optional[int]:
                for name in names:
                    try:
                        if hasattr(obj, name):
                            value = getattr(obj, name)
                            if isinstance(value, (int, float)):
                                return int(value)
                        elif isinstance(obj, dict) and name in obj:
                            value = obj[name]
                            if isinstance(value, (int, float)):
                                return int(value)
                    except Exception:
                        continue
                return None

            prompt_tokens = _read(usage, ['prompt_token_count', 'prompt_tokens', 'input_token_count', 'input_tokens'])
            completion_tokens = _read(usage, ['candidates_token_count', 'completion_token_count', 'output_token_count', 'output_tokens'])
            total_tokens = _read(usage, ['total_token_count', 'total_tokens'])

            if completion_tokens is None and prompt_tokens is not None and total_tokens is not None:
                try:
                    completion_tokens = max(0, int(total_tokens) - int(prompt_tokens))
                except Exception:
                    completion_tokens = None
            
            return prompt_tokens, completion_tokens, total_tokens
        except Exception as e:
            logger.warning(f"Could not extract token counts from response: {e}")
            logger.debug(f"Response object that caused error: {response}")
        return None, None, None

    def _create_model_response(
        self,
        response,
        latency: float,
        completion_tokens_override: Optional[int] = None,
        prompt_tokens_override: Optional[int] = None,
    ) -> ModelResponse:
        """Create a ModelResponse object from the raw Gemini response."""
        logger.debug(f"Full response object for token count extraction: {response}")
        raw_text = getattr(response, 'text', None)
        # Ensure a valid string for ModelResponse.text even if SDK returns None or non-str types
        if raw_text is None:
            response_text = ""
        elif isinstance(raw_text, str):
            response_text = raw_text
        else:
            response_text = str(raw_text)
        
        # Extract token usage
        prompt_tokens, completion_tokens, total_tokens = self._get_token_counts(response)

        if completion_tokens_override is not None:
            completion_tokens = completion_tokens_override
            if prompt_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
        if prompt_tokens_override is not None:
            prompt_tokens = prompt_tokens_override
            if completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
        
        return ModelResponse(
            text=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency=latency,
            finish_reason=None,  # Extract if available in new SDK
            metadata={
                "model": self.config.model_name,
                "blocked": False,  # Update based on new SDK structure
                "truncated": False,  # Update based on new SDK structure
            }
        )

    async def generate_batch(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
        """Generate responses for multiple prompts concurrently."""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def supports_batch(self) -> bool:
        """Indicates that the model supports concurrent batch processing."""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Gemini model."""
        base_info = super().get_model_info()
        base_info.update({
            "provider": "Google",
            "model_type": "Gemini",
        })
        return base_info 

    # --- Token counting helpers -------------------------------------------------
    def _count_tokens_sync(self, text: str) -> Optional[int]:
        try:
            result = self.client.models.count_tokens(
                model=self.config.model_name,
                contents=text,
            )
            value = getattr(result, 'total_tokens', None)
            return int(value) if isinstance(value, (int, float)) else None
        except Exception:
            return None

    async def _count_tokens_async(self, text: str) -> Optional[int]:
        try:
            result = await self.client.aio.models.count_tokens(
                model=self.config.model_name,
                contents=text,
            )
            value = getattr(result, 'total_tokens', None)
            return int(value) if isinstance(value, (int, float)) else None
        except Exception:
            return None