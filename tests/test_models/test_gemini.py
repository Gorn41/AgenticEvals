"""
Tests for Gemini model implementation.
"""

import pytest
import os

from src.models.gemini import GeminiModel
from src.models.base import ModelConfig, ModelResponse


def test_gemini_model_init():
    """Test GeminiModel initialization."""
    config = ModelConfig(model_name="gemini-2.5-pro", api_key="test-key")
    model = GeminiModel(config)
    
    assert model.config == config


class TestGeminiModelConfig:
    """Test Gemini model configuration."""
    
    def test_model_config_creation(self):
        """Test creating model configuration."""
        config = ModelConfig(
            model_name="gemini-2.5-pro",
            api_key="test-key",
            temperature=0.5
        )
        
        assert config.model_name == "gemini-2.5-pro"
        assert config.api_key == "test-key"
        assert config.temperature == 0.5
    
    def test_model_config_with_all_params(self):
        """Test model config with all parameters."""
        config = ModelConfig(
            model_name="gemini-2.5-flash",
            api_key="test-key",
            temperature=0.2,
            max_tokens=1000,
            top_p=0.9,
            top_k=40
        )
        
        assert config.model_name == "gemini-2.5-flash"
        assert config.temperature == 0.2
        assert config.max_tokens == 1000
        assert config.top_p == 0.9
        assert config.top_k == 40


class TestGeminiModelMethods:
    """Test Gemini model methods."""
    
    def test_has_required_methods(self):
        """Test that GeminiModel has all required methods."""
        config = ModelConfig(model_name="gemini-2.5-pro", api_key="test-key")
        model = GeminiModel(config)
        
        # Test that methods exist
        assert hasattr(model, 'generate')
        assert hasattr(model, 'generate_sync')
        assert hasattr(model, 'generate_batch')
        assert hasattr(model, 'get_model_info')
        
        # Test that methods are callable
        assert callable(model.generate)
        assert callable(model.generate_sync)
        assert callable(model.generate_batch)
        assert callable(model.get_model_info)
    
    def test_get_model_info(self):
        """Test getting model information."""
        config = ModelConfig(model_name="gemini-2.5-pro", api_key="test-key")
        model = GeminiModel(config)
        
        info = model.get_model_info()
        
        assert isinstance(info, dict)
        assert info["model_name"] == "gemini-2.5-pro"
        assert "provider" in info
        assert "model_type" in info


class TestGeminiModelIntegration:
    """Integration tests for Gemini model (requires API key)."""
    
    @pytest.mark.integration
    def test_generate_sync_integration(self):
        """Test synchronous generation with real API."""
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("No API key available for integration test")
        
        config = ModelConfig(model_name="gemini-2.5-flash", api_key=api_key)
        model = GeminiModel(config)
        
        response = model.generate_sync("Say hello in one word.")
        
        assert isinstance(response, ModelResponse)
        assert response.text is not None
        assert len(response.text) > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_generate_async_integration(self):
        """Test asynchronous generation with real API."""
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("No API key available for integration test")
        
        config = ModelConfig(model_name="gemini-2.5-flash", api_key=api_key)
        model = GeminiModel(config)
        
        response = await model.generate("Say hello in one word.")
        
        assert isinstance(response, ModelResponse)
        assert response.text is not None
        assert len(response.text) > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_generate_batch_integration(self):
        """Test batch generation with real API."""
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("No API key available for integration test")
        
        config = ModelConfig(model_name="gemini-2.5-flash", api_key=api_key)
        model = GeminiModel(config)
        
        prompts = ["Count to 3", "Say goodbye"]
        responses = await model.generate_batch(prompts)
        
        assert len(responses) == 2
        for response in responses:
            if response is not None:  # Some might fail
                assert isinstance(response, ModelResponse)
                assert response.text is not None


class TestGeminiModelValidation:
    """Test Gemini model validation and error handling."""
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        # Test with minimal config
        config = ModelConfig(model_name="gemini-2.5-pro", api_key="test-key")
        model = GeminiModel(config)
        assert model.config.model_name == "gemini-2.5-pro"
        
        # Test with full config
        config = ModelConfig(
            model_name="gemini-2.5-flash",
            api_key="test-key",
            temperature=0.8,
            max_tokens=2000
        )
        model = GeminiModel(config)
        assert model.config.temperature == 0.8
        assert model.config.max_tokens == 2000 