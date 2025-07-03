"""
Model loader for AgenticEvals.
"""

import os
from typing import Dict, Optional, Type, Any
from pathlib import Path
from dotenv import load_dotenv

from .base import BaseModel, ModelConfig
from .gemini import GeminiModel
from ..utils.logging import get_logger
from ..utils.config import get_config_manager

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


class ModelLoader:
    """Factory class for loading different model types."""
    
    # Registry of available model classes
    _model_registry: Dict[str, Type[BaseModel]] = {
        "gemini": GeminiModel,
            "gemini-pro": GeminiModel,
    "gemini-2.5-pro": GeminiModel,
    "gemini-2.5-flash": GeminiModel,
    }
    
    @classmethod
    def register_model(cls, model_name: str, model_class: Type[BaseModel]):
        """Register a new model class."""
        cls._model_registry[model_name] = model_class
        logger.info(f"Registered model: {model_name}")
    
    @classmethod
    def get_available_models(cls) -> list[str]:
        """Get list of available model names."""
        return list(cls._model_registry.keys())
    
    @classmethod
    def load_model(cls, model_name: str, **config_kwargs) -> BaseModel:
        """Load a model by name with given configuration."""
        if model_name not in cls._model_registry:
            raise ValueError(f"Unknown model: {model_name}. Available models: {cls.get_available_models()}")
        
        # Build model configuration
        config = cls._build_model_config(model_name, **config_kwargs)
        
        # Get model class and instantiate
        model_class = cls._model_registry[model_name]
        model = model_class(config)
        
        logger.info(f"Loaded model: {model_name}")
        return model
    
    @classmethod
    def _build_model_config(cls, model_name: str, **kwargs) -> ModelConfig:
        """Build model configuration with defaults and environment variables."""
        config_dict = {
            "model_name": model_name,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens"),
            "top_p": kwargs.get("top_p"),
            "top_k": kwargs.get("top_k"),
            "stop_sequences": kwargs.get("stop_sequences"),
            "additional_params": kwargs.get("additional_params", {}),
        }
        
        # Handle API key
        api_key = kwargs.get("api_key")
        if not api_key:
            # Try to get from environment variables
            if model_name.startswith("gemini"):
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        config_dict["api_key"] = api_key
        
        return ModelConfig(**config_dict)
    
    @classmethod
    def load_from_config_file(cls, config_path: str) -> BaseModel:
        """Load model from a configuration file."""
        import yaml
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        model_name = config_data.get("model_name")
        if not model_name:
            raise ValueError("model_name is required in config file")
        
        # Remove model_name from kwargs since it's passed separately
        config_kwargs = {k: v for k, v in config_data.items() if k != "model_name"}
        
        return cls.load_model(model_name, **config_kwargs)


# Convenience functions
def load_gemini(model_name: str = "gemini-2.5-pro", **kwargs) -> GeminiModel:
    """Convenience function to load a Gemini model."""
    return ModelLoader.load_model(model_name, **kwargs)


def load_model_from_name(model_name: str, **kwargs) -> BaseModel:
    """Convenience function to load any model by name."""
    return ModelLoader.load_model(model_name, **kwargs) 