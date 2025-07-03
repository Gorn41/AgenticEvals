"""
Base model interface for LLM-AgentTypeEval.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel as PydanticBaseModel


class ModelResponse(PydanticBaseModel):
    """Response from a model call."""
    text: str
    tokens_used: Optional[int] = None
    latency: Optional[float] = None
    metadata: Dict[str, Any] = {}


class ModelConfig(PydanticBaseModel):
    """Configuration for model initialization."""
    model_name: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    additional_params: Dict[str, Any] = {}


class BaseModel(ABC):
    """Abstract base class for all model implementations."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the model with given configuration."""
        self.config = config
        self.model_name = config.model_name
        
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate a response for the given prompt."""
        pass
    
    @abstractmethod
    def generate_sync(self, prompt: str, **kwargs) -> ModelResponse:
        """Synchronous version of generate."""
        pass
    
    @abstractmethod
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
        """Generate responses for multiple prompts."""
        pass
    
    def supports_batch(self) -> bool:
        """Whether this model supports batch generation."""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "config": self.config.model_dump()
        } 