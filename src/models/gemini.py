"""
Gemini model implementation for AgenticEvals.
"""

import time
import asyncio
from typing import List, Dict, Any, Optional

try:
    from google import genai
    from google.genai.types import GenerateContentConfig
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from .base import BaseModel, ModelConfig, ModelResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)

class GeminiModel(BaseModel):
    """Gemini model implementation using Google's Gen AI SDK."""
    
    def __init__(self, config: ModelConfig):
        """Initialize Gemini model."""
        super().__init__(config)
        
        if not GENAI_AVAILABLE:
            raise ImportError("google-genai is not installed. Please install it with `pip install google-genai`")
        
        if not config.api_key:
            raise ValueError("API key is required for Gemini model")
        
        # Create client instead of configuring globally
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
        
        return GenerateContentConfig(**filtered_config)
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response asynchronously."""
        start_time = time.time()
        
        try:
            response = await self.client.aio.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=self.generation_config
            )
            latency = time.time() - start_time
            
            # Manually count completion tokens
            completion_tokens = None
            if hasattr(response, 'text'):
                completion_tokens = (await self.client.aio.models.count_tokens(
                    model=self.config.model_name,
                    contents=response.text
                )).total_tokens

            return self._create_model_response(response, latency, completion_tokens_override=completion_tokens)
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            logger.debug(f"Response object that caused error: {response}")
            raise
    
    def generate_sync(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response synchronously."""
        start_time = time.time()
        
        try:
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=prompt,
                config=self.generation_config
            )
            latency = time.time() - start_time

            # Manually count completion tokens
            completion_tokens = None
            if hasattr(response, 'text'):
                completion_tokens = self.client.models.count_tokens(
                    model=self.config.model_name,
                    contents=response.text
                ).total_tokens

            return self._create_model_response(response, latency, completion_tokens_override=completion_tokens)
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            raise

    def _get_token_counts(self, response: Any) -> tuple[Optional[int], Optional[int], Optional[int]]:
        """Extract token counts from the response metadata."""
        try:
            if hasattr(response, 'usage_metadata'):
                logger.debug(f"Full response object for token count extraction: {response}")
                prompt_tokens = response.usage_metadata.prompt_token_count
                total_tokens = response.usage_metadata.total_token_count
                
                if hasattr(response.usage_metadata, 'candidates_token_count'):
                    completion_tokens = response.usage_metadata.candidates_token_count
                else:
                    # Calculate completion tokens if not directly available
                    completion_tokens = total_tokens - prompt_tokens
                    
                return prompt_tokens, completion_tokens, total_tokens
        except (AttributeError, TypeError, KeyError) as e:
            logger.warning(f"Could not extract token counts from response: {e}")
            logger.debug(f"Response object that caused error: {response}")
        return None, None, None

    def _create_model_response(self, response, latency: float, completion_tokens_override: Optional[int] = None) -> ModelResponse:
        """Create a ModelResponse object from the raw Gemini response."""
        logger.debug(f"Full response object for token count extraction: {response}")
        response_text = response.text if hasattr(response, 'text') else "RESPONSE_TEXT_UNAVAILABLE"
        
        # Extract token usage
        prompt_tokens, completion_tokens, total_tokens = self._get_token_counts(response)

        if completion_tokens_override is not None:
            completion_tokens = completion_tokens_override
            if prompt_tokens is not None:
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