"""
Tests for model base classes and interfaces.
"""

import pytest

from src.models.base import BaseModel, ModelConfig, ModelResponse


class TestModelResponse:
    """Test the ModelResponse class."""
    
    def test_model_response_creation(self):
        """Test creating a model response."""
        response = ModelResponse(
            text="Test response",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            finish_reason="completed",
            metadata={"model": "test-model"}
        )
        
        assert response.text == "Test response"
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5
        assert response.total_tokens == 15
        assert response.finish_reason == "completed"
        assert response.metadata == {"model": "test-model"}
    
    def test_model_response_minimal(self):
        """Test creating minimal model response."""
        response = ModelResponse(text="Simple response")
        
        assert response.text == "Simple response"
        assert response.prompt_tokens is None
        assert response.completion_tokens is None
        assert response.total_tokens is None
        assert response.finish_reason is None
        assert response.metadata == {}
    
    def test_model_response_token_calculation(self):
        """Test automatic token calculation if total not provided."""
        response = ModelResponse(
            text="Test",
            prompt_tokens=10,
            completion_tokens=5
        )
        
        # Should calculate total automatically
        assert response.total_tokens == 15


class TestModelConfig:
    """Test the ModelConfig class."""
    
    def test_model_config_creation(self):
        """Test creating a model configuration."""
        config = ModelConfig(
            model_name="test-model",
            api_key="test-key",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            top_k=40,
            additional_params={"custom": "value"}
        )
        
        assert config.model_name == "test-model"
        assert config.api_key == "test-key"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.additional_params == {"custom": "value"}
    
    def test_model_config_minimal(self):
        """Test creating minimal model configuration."""
        config = ModelConfig(model_name="minimal-model")
        
        assert config.model_name == "minimal-model"
        assert config.api_key is None
        assert config.temperature == 0.7  # Default value
        assert config.max_tokens is None
        assert config.additional_params == {}
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        # Test invalid temperature
        with pytest.raises(ValueError):
            ModelConfig(model_name="test", temperature=-1.0)
        
        with pytest.raises(ValueError):
            ModelConfig(model_name="test", temperature=2.0)
        
        # Test invalid top_p
        with pytest.raises(ValueError):
            ModelConfig(model_name="test", top_p=-0.1)
        
        with pytest.raises(ValueError):
            ModelConfig(model_name="test", top_p=1.1)
        
        # Test invalid max_tokens
        with pytest.raises(ValueError):
            ModelConfig(model_name="test", max_tokens=0)
        
        with pytest.raises(ValueError):
            ModelConfig(model_name="test", max_tokens=-100)


class TestBaseModel:
    """Test the BaseModel abstract class."""
    
    def test_abstract_methods_exist(self):
        """Test that BaseModel has the required abstract methods."""
        abstract_methods = BaseModel.__abstractmethods__
        
        expected_methods = {'generate', 'generate_sync'}
        assert expected_methods.issubset(abstract_methods)
    
    def test_cannot_instantiate_base_model(self):
        """Test that BaseModel cannot be instantiated directly."""
        config = ModelConfig(model_name="test")
        
        with pytest.raises(TypeError):
            # This should fail because BaseModel is abstract
            model = object.__new__(BaseModel)
            BaseModel.__init__(model, config)


class TestModelConfigValidation:
    """Test detailed model configuration validation."""
    
    def test_temperature_validation(self):
        """Test temperature parameter validation."""
        # Valid temperatures
        config = ModelConfig(model_name="test", temperature=0.0)
        assert config.temperature == 0.0
        
        config = ModelConfig(model_name="test", temperature=1.0)
        assert config.temperature == 1.0
        
        config = ModelConfig(model_name="test", temperature=0.5)
        assert config.temperature == 0.5
    
    def test_max_tokens_validation(self):
        """Test max_tokens parameter validation."""
        # Valid max_tokens
        config = ModelConfig(model_name="test", max_tokens=100)
        assert config.max_tokens == 100
        
        config = ModelConfig(model_name="test", max_tokens=8192)
        assert config.max_tokens == 8192
        
        # None should be allowed
        config = ModelConfig(model_name="test", max_tokens=None)
        assert config.max_tokens is None
    
    def test_top_p_validation(self):
        """Test top_p parameter validation."""
        # Valid top_p values
        config = ModelConfig(model_name="test", top_p=0.0)
        assert config.top_p == 0.0
        
        config = ModelConfig(model_name="test", top_p=1.0)
        assert config.top_p == 1.0
        
        config = ModelConfig(model_name="test", top_p=0.95)
        assert config.top_p == 0.95
    
    def test_model_name_validation(self):
        """Test model name validation."""
        # Test valid config creation
        config = ModelConfig(model_name="test-model")
        assert config.model_name == "test-model"
        
        # Test empty model name should work (no validation error)
        config = ModelConfig(model_name="")
        assert config.model_name == ""
    
    def test_additional_params_handling(self):
        """Test additional parameters handling."""
        params = {"custom_param": "value", "another": 123}
        config = ModelConfig(model_name="test", additional_params=params)
        
        assert config.additional_params == params
        
        # Should handle empty dict
        config = ModelConfig(model_name="test", additional_params={})
        assert config.additional_params == {}
        
        # Should default to empty dict if not provided
        config = ModelConfig(model_name="test")
        assert config.additional_params == {} 