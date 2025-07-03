"""
Base model interface for AgenticEvals.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel as PydanticBaseModel, Field, validator

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ModelResponse(PydanticBaseModel):
    """Response from a model generation call."""
    
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    latency: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator('total_tokens', always=True)
    def calculate_total_tokens(cls, v, values):
        """Calculate total tokens if not provided."""
        if v is None and values.get('prompt_tokens') and values.get('completion_tokens'):
            return values['prompt_tokens'] + values['completion_tokens']
        return v


class ModelConfig(PydanticBaseModel):
    """Configuration for model instantiation."""
    
    model_name: str
    api_key: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, gt=0)
    stop_sequences: Optional[List[str]] = None
    additional_params: Dict[str, Any] = Field(default_factory=dict)


class BaseModel(ABC):
    """Abstract base class for all language models."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the model with given configuration.
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.model_name = config.model_name
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up model-specific logging."""
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response to a prompt asynchronously.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResponse containing the generated text and metadata
        """
        pass
    
    @abstractmethod
    def generate_sync(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response to a prompt synchronously.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResponse containing the generated text and metadata
        """
        pass
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
        """Generate responses to multiple prompts.
        
        Default implementation calls generate() for each prompt.
        Override for models that support native batch processing.
        
        Args:
            prompts: List of prompt texts
            **kwargs: Additional generation parameters
            
        Returns:
            List of ModelResponse objects
        """
        responses = []
        for prompt in prompts:
            response = await self.generate(prompt, **kwargs)
            responses.append(response)
        return responses
    
    def supports_batch(self) -> bool:
        """Check if the model supports batch processing natively.
        
        Returns:
            True if batch processing is supported natively
        """
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "supports_batch": self.supports_batch(),
            "config": self.config.dict()
        } 