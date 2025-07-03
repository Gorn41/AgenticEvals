"""
Tests for model loader functionality.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

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
        assert "gemini-1.5-pro" in models
        assert "gemini-1.5-flash" in models
    
    def test_register_model(self):
        """Test registering a new model."""
        # Create a mock model class
        class MockModel(BaseModel):
            async def generate(self, prompt, **kwargs):
                pass
            
            def generate_sync(self, prompt, **kwargs):
                pass
            
            async def generate_batch(self, prompts, **kwargs):
                pass
        
        # Register the model
        ModelLoader.register_model("mock-model", MockModel)
        
        # Check it's in the registry
        assert "mock-model" in ModelLoader.get_available_models()
        
        # Clean up
        del ModelLoader._model_registry["mock-model"]
    
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
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key-123"}):
            config = ModelLoader._build_model_config("gemini-pro")
            assert config.api_key == "test-key-123"
    
    def test_build_model_config_api_key_from_gemini_env(self):
        """Test API key loading from GEMINI_API_KEY environment variable."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key"}):
            config = ModelLoader._build_model_config("gemini-pro")
            assert config.api_key == "test-gemini-key"
    
    def test_build_model_config_api_key_explicit(self):
        """Test explicit API key takes precedence."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key"}):
            config = ModelLoader._build_model_config("gemini-pro", api_key="explicit-key")
            assert config.api_key == "explicit-key"
    
    @patch('models.gemini.GeminiModel')
    def test_load_model_success(self, mock_gemini_class):
        """Test successful model loading."""
        mock_instance = MagicMock()
        mock_gemini_class.return_value = mock_instance
        
        model = ModelLoader.load_model("gemini", api_key="test-key")
        
        assert model == mock_instance
        mock_gemini_class.assert_called_once()
    
    def test_load_model_unknown(self):
        """Test loading unknown model raises error."""
        with pytest.raises(ValueError, match="Unknown model: unknown-model"):
            ModelLoader.load_model("unknown-model")
    
    @patch('models.loader.yaml.safe_load')
    @patch('builtins.open')
    @patch('models.gemini.GeminiModel')
    def test_load_from_config_file(self, mock_gemini, mock_open, mock_yaml):
        """Test loading model from config file."""
        # Mock file content
        mock_yaml.return_value = {
            "model_name": "gemini-pro",
            "temperature": 0.5,
            "api_key": "config-key"
        }
        mock_gemini.return_value = MagicMock()
        
        model = ModelLoader.load_from_config_file("test_config.yaml")
        
        mock_open.assert_called_once_with("test_config.yaml", 'r')
        mock_gemini.assert_called_once()
    
    def test_load_from_config_file_missing_model_name(self, tmp_path):
        """Test config file without model_name raises error."""
        config_file = tmp_path / "bad_config.yaml"
        config_file.write_text("temperature: 0.5")
        
        with pytest.raises(ValueError, match="model_name is required"):
            ModelLoader.load_from_config_file(str(config_file))


class TestConvenienceFunctions:
    """Test convenience functions for model loading."""
    
    @patch('models.loader.ModelLoader.load_model')
    def test_load_gemini_default(self, mock_load):
        """Test load_gemini with default parameters."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        result = load_gemini()
        
        mock_load.assert_called_once_with("gemini-1.5-pro")
        assert result == mock_model
    
    @patch('models.loader.ModelLoader.load_model')
    def test_load_gemini_custom(self, mock_load):
        """Test load_gemini with custom parameters."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        result = load_gemini("gemini-1.5-flash", temperature=0.2, api_key="test")
        
        mock_load.assert_called_once_with(
            "gemini-1.5-flash", 
            temperature=0.2, 
            api_key="test"
        )
        assert result == mock_model
    
    @patch('models.loader.ModelLoader.load_model')
    def test_load_model_from_name(self, mock_load):
        """Test load_model_from_name function."""
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        result = load_model_from_name("gemini-pro", temperature=0.8)
        
        mock_load.assert_called_once_with("gemini-pro", temperature=0.8)
        assert result == mock_model


class TestModelLoaderIntegration:
    """Integration tests for model loader (may require API keys)."""
    
    @pytest.mark.integration
    def test_load_gemini_with_real_config(self):
        """Test loading Gemini with realistic configuration."""
        # Skip if no API key available
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("No API key available for integration test")
        
        # This should not raise an error
        model = load_gemini("gemini-1.5-flash", api_key=api_key, temperature=0.1)
        
        assert model is not None
        assert model.model_name == "gemini-1.5-flash"
        assert hasattr(model, 'generate')
        assert hasattr(model, 'generate_sync') 