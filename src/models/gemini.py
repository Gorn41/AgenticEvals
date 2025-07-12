"""
Gemini model implementation for AgenticEvals.
"""

import time
import asyncio
from typing import List, Dict, Any, Optional

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from .base import BaseModel, ModelConfig, ModelResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


class GeminiModel(BaseModel):
    """Gemini model implementation using Google's Generative AI API."""
    
    def __init__(self, config: ModelConfig):
        """Initialize Gemini model."""
        super().__init__(config)
        
        if not config.api_key:
            raise ValueError("API key is required for Gemini model")
        
        # Configure the API
        genai.configure(api_key=config.api_key)
        
        # Initialize the model
        generation_config = self._build_generation_config()
        safety_settings = self._build_safety_settings()
        
        self.model = genai.GenerativeModel(
            model_name=config.model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        logger.info(f"Initialized Gemini model: {config.model_name}")
    
    def _build_generation_config(self) -> genai.GenerationConfig:
        """Build generation configuration from model config."""
        config_dict = {
            "temperature": self.config.temperature,
        }
        
        if self.config.max_tokens:
            config_dict["max_output_tokens"] = self.config.max_tokens
        if self.config.top_p:
            config_dict["top_p"] = self.config.top_p
        if self.config.top_k:
            config_dict["top_k"] = self.config.top_k
        if self.config.stop_sequences:
            config_dict["stop_sequences"] = self.config.stop_sequences
        
        # Add any additional parameters
        config_dict.update(self.config.additional_params)
        
        return genai.GenerationConfig(**config_dict)
    
    def _build_safety_settings(self) -> Dict[HarmCategory, HarmBlockThreshold]:
        """Build safety settings for Gemini."""
        # Use permissive settings for benchmarking
        return {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response asynchronously."""
        start_time = time.time()
        
        try:
            # Gemini doesn't have native async support, so we use asyncio.to_thread
            response = await asyncio.to_thread(self._generate_sync_internal, prompt)
            
            latency = time.time() - start_time
            
            # Check if response was blocked or has no text
            response_text = ""
            finish_reason = None
            
            # Extract finish_reason properly from candidates
            try:
                if hasattr(response, 'candidates') and response.candidates:
                    finish_reason = response.candidates[0].finish_reason
                else:
                    finish_reason = getattr(response, "finish_reason", None)
            except (AttributeError, IndexError):
                finish_reason = None
            
            try:
                response_text = response.text
            except (AttributeError, ValueError) as e:
                # Handle cases where response.text is not available
                logger.warning(f"Response text not available, finish_reason: {finish_reason}, error: {e}")
                if finish_reason == 2:  # MAX_TOKENS
                    response_text = "RESPONSE_TRUNCATED_MAX_TOKENS"
                elif finish_reason == 3:  # SAFETY
                    response_text = "RESPONSE_BLOCKED_BY_SAFETY_FILTER"
                elif finish_reason == 4:  # RECITATION
                    response_text = "RESPONSE_BLOCKED_BY_RECITATION_FILTER"
                elif finish_reason == 5:  # OTHER
                    response_text = "RESPONSE_BLOCKED_OTHER_REASON"
                else:
                    response_text = "RESPONSE_TEXT_UNAVAILABLE"
            
            return ModelResponse(
                text=response_text,
                tokens_used=self._get_token_count(response),
                latency=latency,
                metadata={
                    "model": self.model_name,
                    "finish_reason": finish_reason,
                    "safety_ratings": getattr(response, "safety_ratings", []),
                    "blocked": response_text.startswith("RESPONSE_BLOCKED"),
                    "truncated": response_text.startswith("RESPONSE_TRUNCATED"),
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            raise
    
    def generate_sync(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response synchronously."""
        start_time = time.time()
        
        try:
            response = self._generate_sync_internal(prompt)
            latency = time.time() - start_time
            
            # Check if response was blocked or has no text
            response_text = ""
            finish_reason = None
            
            # Extract finish_reason properly from candidates
            try:
                if hasattr(response, 'candidates') and response.candidates:
                    finish_reason = response.candidates[0].finish_reason
                else:
                    finish_reason = getattr(response, "finish_reason", None)
            except (AttributeError, IndexError):
                finish_reason = None
            
            try:
                response_text = response.text
            except (AttributeError, ValueError) as e:
                # Handle cases where response.text is not available
                logger.warning(f"Response text not available, finish_reason: {finish_reason}, error: {e}")
                if finish_reason == 2:  # MAX_TOKENS
                    response_text = "RESPONSE_TRUNCATED_MAX_TOKENS"
                elif finish_reason == 3:  # SAFETY
                    response_text = "RESPONSE_BLOCKED_BY_SAFETY_FILTER"
                elif finish_reason == 4:  # RECITATION
                    response_text = "RESPONSE_BLOCKED_BY_RECITATION_FILTER"
                elif finish_reason == 5:  # OTHER
                    response_text = "RESPONSE_BLOCKED_OTHER_REASON"
                else:
                    response_text = "RESPONSE_TEXT_UNAVAILABLE"
            
            return ModelResponse(
                text=response_text,
                tokens_used=self._get_token_count(response),
                latency=latency,
                metadata={
                    "model": self.model_name,
                    "finish_reason": finish_reason,
                    "safety_ratings": getattr(response, "safety_ratings", []),
                    "blocked": response_text.startswith("RESPONSE_BLOCKED"),
                    "truncated": response_text.startswith("RESPONSE_TRUNCATED"),
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}")
            raise
    
    def _generate_sync_internal(self, prompt: str):
        """Internal synchronous generation method."""
        return self.model.generate_content(prompt)
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
        """Generate responses for multiple prompts."""
        # Gemini doesn't support native batch processing, so we process sequentially
        # with some delay to respect rate limits
        responses = []
        
        for i, prompt in enumerate(prompts):
            if i > 0:
                # Add small delay between requests to respect rate limits
                await asyncio.sleep(0.1)
            
            response = await self.generate(prompt, **kwargs)
            responses.append(response)
        
        return responses
    
    def _get_token_count(self, response) -> Optional[int]:
        """Extract token count from response if available."""
        try:
            if hasattr(response, "usage_metadata"):
                return getattr(response.usage_metadata, "total_token_count", None)
            return None
        except:
            return None
    
    def supports_batch(self) -> bool:
        """Gemini supports batch through sequential processing."""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Gemini model."""
        base_info = super().get_model_info()
        base_info.update({
            "provider": "Google",
            "model_type": "Gemini",
            "supports_streaming": False,
            "supports_function_calling": True,
        })
        return base_info 