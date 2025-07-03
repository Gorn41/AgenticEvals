"""
Tests for base model classes and interfaces.
"""

import pytest
from unittest.mock import MagicMock
from typing import List

from models.base import ModelResponse, ModelConfig, BaseModel


class TestModelResponse:
    """Test the ModelResponse class."""
    
    def test_model_response_creation(self):
        """Test creating a ModelResponse."""
        response = ModelResponse(
            text="Hello world",
            tokens_used=10,
            latency=1.5,
            metadata={"model": "test-model"}
        )
        
        assert response.text == "Hello world"
        assert response.tokens_used == 10
        assert response.latency == 1.5
        assert response.metadata == {"model": "test-model"}
    
    def test_model_response_minimal(self):
        """Test creating ModelResponse with only required fields."""
        response = ModelResponse(text="Hello")
        
        assert response.text == "Hello"
        assert response.tokens_used is None
        assert response.latency is None
        assert response.metadata == {}
    
    def test_model_response_serialization(self):
        """Test ModelResponse can be serialized to dict."""
        response = ModelResponse(
            text="Hello",
            tokens_used=5,
            latency=0.8,
            metadata={"test": "value"}
        )
        
        data = response.model_dump()
        assert isinstance(data, dict)
        assert data["text"] == "Hello"
        assert data["tokens_used"] == 5
        assert data["latency"] == 0.8
        assert data["metadata"] == {"test": "value"}


class TestModelConfig:
    """Test the ModelConfig class."""
    
    def test_model_config_creation(self):
        """Test creating a ModelConfig."""
        config = ModelConfig(
            model_name="test-model",
            api_key="test-key",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            top_k=40,
            stop_sequences=["END", "STOP"],
            additional_params={"custom": "value"}
        )
        
        assert config.model_name == "test-model"
        assert config.api_key == "test-key"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.stop_sequences == ["END", "STOP"]
        assert config.additional_params == {"custom": "value"}
    
    def test_model_config_minimal(self):
        """Test creating ModelConfig with only required fields."""
        config = ModelConfig(model_name="test-model")
        
        assert config.model_name == "test-model"
        assert config.api_key is None
        assert config.temperature == 0.7  # default
        assert config.max_tokens is None
        assert config.top_p is None
        assert config.top_k is None
        assert config.stop_sequences is None
        assert config.additional_params == {}
    
    def test_model_config_serialization(self):
        """Test ModelConfig can be serialized to dict."""
        config = ModelConfig(
            model_name="test-model",
            temperature=0.5,
            max_tokens=500
        )
        
        data = config.model_dump()
        assert isinstance(data, dict)
        assert data["model_name"] == "test-model"
        assert data["temperature"] == 0.5
        assert data["max_tokens"] == 500


class TestBaseModel:
    """Test the BaseModel abstract class."""
    
    def test_base_model_is_abstract(self):
        """Test that BaseModel cannot be instantiated directly."""
        config = ModelConfig(model_name="test")
        
        with pytest.raises(TypeError):
            BaseModel(config)
    
    def test_concrete_implementation(self):
        """Test a concrete implementation of BaseModel."""
        
        class ConcreteModel(BaseModel):
            async def generate(self, prompt: str, **kwargs) -> ModelResponse:
                return ModelResponse(text=f"Response to: {prompt}")
            
            def generate_sync(self, prompt: str, **kwargs) -> ModelResponse:
                return ModelResponse(text=f"Sync response to: {prompt}")
            
            async def generate_batch(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
                return [ModelResponse(text=f"Response to: {p}") for p in prompts]
        
        config = ModelConfig(model_name="concrete-model")
        model = ConcreteModel(config)
        
        assert model.model_name == "concrete-model"
        assert model.config == config
        assert model.supports_batch() is True  # default implementation
    
    def test_base_model_get_model_info(self):
        """Test the default get_model_info implementation."""
        
        class ConcreteModel(BaseModel):
            async def generate(self, prompt: str, **kwargs) -> ModelResponse:
                return ModelResponse(text="test")
            
            def generate_sync(self, prompt: str, **kwargs) -> ModelResponse:
                return ModelResponse(text="test")
            
            async def generate_batch(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
                return []
        
        config = ModelConfig(
            model_name="test-model",
            temperature=0.8,
            max_tokens=100
        )
        model = ConcreteModel(config)
        
        info = model.get_model_info()
        assert isinstance(info, dict)
        assert info["model_name"] == "test-model"
        assert "config" in info
        assert info["config"]["temperature"] == 0.8
    
    def test_supports_batch_override(self):
        """Test overriding the supports_batch method."""
        
        class NoBatchModel(BaseModel):
            async def generate(self, prompt: str, **kwargs) -> ModelResponse:
                return ModelResponse(text="test")
            
            def generate_sync(self, prompt: str, **kwargs) -> ModelResponse:
                return ModelResponse(text="test")
            
            async def generate_batch(self, prompts: List[str], **kwargs) -> List[ModelResponse]:
                raise NotImplementedError("Batch not supported")
            
            def supports_batch(self) -> bool:
                return False
        
        config = ModelConfig(model_name="no-batch-model")
        model = NoBatchModel(config)
        
        assert model.supports_batch() is False


class TestModelValidation:
    """Test validation and error handling in model classes."""
    
    def test_model_response_validation(self):
        """Test ModelResponse validation."""
        # Valid response
        response = ModelResponse(text="Hello")
        assert response.text == "Hello"
        
        # Test with invalid types - pydantic should handle this
        with pytest.raises((ValueError, TypeError)):
            ModelResponse(text=123)  # text should be string
    
    def test_model_config_validation(self):
        """Test ModelConfig validation."""
        # Valid config
        config = ModelConfig(model_name="test")
        assert config.model_name == "test"
        
        # Test with invalid temperature
        with pytest.raises((ValueError, TypeError)):
            ModelConfig(model_name="test", temperature="invalid")
        
        # Test with negative temperature (if validation is added)
        config = ModelConfig(model_name="test", temperature=-1.0)
        # Currently no validation, but this is where it would go
        assert config.temperature == -1.0 