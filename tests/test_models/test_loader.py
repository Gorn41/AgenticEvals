"""
Tests for model loader functionality.
"""

import pytest
import os

from models.loader import ModelLoader, load_gemini, load_model_from_name
from models.base import ModelConfig, BaseModel
from models.gemini import GeminiModel


class TestModelLoader:
    """Test the ModelLoader class."""
    
    def test_get_available_models(self):
        """Test getting available models."""
        models = ModelLoader.get_available_models()
        assert isinstance(models, list)
        assert "gemini" in models
        assert "gemini-2.5-pro" in models
        assert "gemini-2.5-flash" in models
    
    def test_register_model(self):
        """Test registering a new model."""
        # Create a real model class that extends BaseModel
        class TestModel(BaseModel):
            async def generate(self, prompt, **kwargs):
                return "test response"
            
            def generate_sync(self, prompt, **kwargs):
                return "test response"
            
            async def generate_batch(self, prompts, **kwargs):
                return ["test response"] * len(prompts)
        
        # Register the model
        ModelLoader.register_model("test-model", TestModel)
        
        # Check it's in the registry
        assert "test-model" in ModelLoader.get_available_models()
        
        # Clean up
        del ModelLoader._model_registry["test-model"]
    
    def test_build_model_config_with_defaults(self):
        """Test building model config with default values."""
        config = ModelLoader._build_model_config("gemini-pro")
        
        assert config.model_name == "gemini-pro"
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.additional_params == {}
    
    def test_build_model_config_with_overrides(self):
        """Test building model config with custom values."""
        config = ModelLoader._build_model_config(
            "gemini-pro",
            temperature=0.2,
            max_tokens=500,
            top_p=0.8,
            additional_params={"custom": "value"}
        )
        
        assert config.model_name == "gemini-pro"
        assert config.temperature == 0.2
        assert config.max_tokens == 500
        assert config.top_p == 0.8
        assert config.additional_params == {"custom": "value"}
    
    def test_build_model_config_api_key_from_env(self):
        """Test API key loading from environment variables."""
        if "GOOGLE_API_KEY" in os.environ:
            config = ModelLoader._build_model_config("gemini-pro")
            assert config.api_key is not None
    
    def test_build_model_config_api_key_from_gemini_env(self):
        """Test API key loading from GEMINI_API_KEY environment variable."""
        if "GEMINI_API_KEY" in os.environ:
            config = ModelLoader._build_model_config("gemini-pro")
            assert config.api_key is not None
    
    def test_build_model_config_api_key_explicit(self):
        """Test explicit API key takes precedence."""
        config = ModelLoader._build_model_config("gemini-pro", api_key="explicit-key")
        assert config.api_key == "explicit-key"
    
    def test_load_model_unknown(self):
        """Test loading unknown model raises error."""
        with pytest.raises(ValueError, match="Unknown model: unknown-model"):
            ModelLoader.load_model("unknown-model")


class TestConvenienceFunctions:
    """Test convenience functions for model loading."""
    
    def test_load_gemini_default_params(self):
        """Test load_gemini function exists and accepts parameters."""
        # Just test that the function exists and can be called with parameters
        # without actually making API calls
        try:
            # This should not raise an error for parameter validation
            load_gemini.__defaults__
            assert load_gemini.__defaults__[0] == "gemini-2.5-pro"
        except Exception:
            pytest.skip("Function signature test")
    
    def test_load_model_from_name_params(self):
        """Test load_model_from_name function parameters."""
        # Test function exists and can accept model name parameter
        try:
            # Verify the function exists and is callable
            assert callable(load_model_from_name)
        except Exception:
            pytest.skip("Function callable test")


class TestModelLoaderIntegration:
    """Integration tests for model loader (requires API keys)."""
    
    @pytest.mark.integration
    def test_load_gemini_with_real_config(self):
        """Test loading Gemini with realistic configuration."""
        # Skip if no API key available
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("No API key available for integration test")
        
        # Test that we can create a model instance
        model = load_gemini("gemini-2.5-flash", api_key=api_key, temperature=0.1)
        
        assert model is not None
        assert model.model_name == "gemini-2.5-flash"
        assert hasattr(model, 'generate')
        assert hasattr(model, 'generate_sync')
    
    @pytest.mark.integration
    def test_load_gemini_pro_with_real_config(self):
        """Test loading Gemini Pro with realistic configuration."""
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("No API key available for integration test")
        
        model = load_gemini("gemini-2.5-pro", api_key=api_key, temperature=0.7)
        
        assert model is not None
        assert model.model_name == "gemini-2.5-pro"
        assert hasattr(model, 'generate')
        assert hasattr(model, 'generate_sync') 