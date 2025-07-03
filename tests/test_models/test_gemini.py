"""
Tests for Gemini model implementation.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

from models.base import ModelConfig, ModelResponse
from models.gemini import GeminiModel


class TestGeminiModel:
    """Test the GeminiModel class."""
    
    def test_gemini_model_init_without_api_key(self):
        """Test that GeminiModel raises error without API key."""
        config = ModelConfig(model_name="gemini-1.5-pro")
        
        with pytest.raises(ValueError, match="API key is required"):
            GeminiModel(config)
    
    @patch('models.gemini.genai.configure')
    @patch('models.gemini.genai.GenerativeModel')
    def test_gemini_model_init_success(self, mock_generative_model, mock_configure):
        """Test successful GeminiModel initialization."""
        config = ModelConfig(
            model_name="gemini-1.5-pro",
            api_key="test-api-key",
            temperature=0.7,
            max_tokens=1000
        )
        
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model
        
        gemini_model = GeminiModel(config)
        
        # Check API was configured
        mock_configure.assert_called_once_with(api_key="test-api-key")
        
        # Check model was initialized
        mock_generative_model.assert_called_once()
        
        # Check instance properties
        assert gemini_model.model_name == "gemini-1.5-pro"
        assert gemini_model.config == config
        assert gemini_model.model == mock_model
    
    def test_build_generation_config_basic(self):
        """Test building generation config with basic parameters."""
        config = ModelConfig(
            model_name="gemini-1.5-pro",
            api_key="test-key",
            temperature=0.8
        )
        
        with patch('models.gemini.genai.configure'), \
             patch('models.gemini.genai.GenerativeModel'):
            
            gemini_model = GeminiModel(config)
            gen_config = gemini_model._build_generation_config()
            
            # The actual implementation returns a genai.GenerationConfig
            # We can't easily test the exact values without mocking genai
            # But we can test that it's called
            assert gen_config is not None
    
    def test_build_generation_config_full(self):
        """Test building generation config with all parameters."""
        config = ModelConfig(
            model_name="gemini-1.5-pro",
            api_key="test-key",
            temperature=0.5,
            max_tokens=500,
            top_p=0.9,
            top_k=40,
            stop_sequences=["END"],
            additional_params={"custom_param": "value"}
        )
        
        with patch('models.gemini.genai.configure'), \
             patch('models.gemini.genai.GenerativeModel'):
            
            gemini_model = GeminiModel(config)
            gen_config = gemini_model._build_generation_config()
            
            assert gen_config is not None
    
    def test_build_safety_settings(self):
        """Test building safety settings."""
        config = ModelConfig(model_name="gemini-1.5-pro", api_key="test-key")
        
        with patch('models.gemini.genai.configure'), \
             patch('models.gemini.genai.GenerativeModel'):
            
            gemini_model = GeminiModel(config)
            safety_settings = gemini_model._build_safety_settings()
            
            assert isinstance(safety_settings, dict)
            # Should have all harm categories set to BLOCK_NONE for benchmarking
            assert len(safety_settings) == 4
    
    @patch('models.gemini.genai.configure')
    @patch('models.gemini.genai.GenerativeModel')
    def test_generate_sync_success(self, mock_generative_model, mock_configure):
        """Test synchronous generation."""
        config = ModelConfig(model_name="gemini-1.5-pro", api_key="test-key")
        
        # Mock the response
        mock_response = MagicMock()
        mock_response.text = "stop"
        mock_response.finish_reason = "completed"
        mock_response.safety_ratings = []
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.total_token_count = 15
        
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_generative_model.return_value = mock_model
        
        gemini_model = GeminiModel(config)
        result = gemini_model.generate_sync("Test prompt")
        
        assert isinstance(result, ModelResponse)
        assert result.text == "stop"
        assert result.tokens_used == 15
        assert result.latency is not None and result.latency > 0
        assert "model" in result.metadata
        
        mock_model.generate_content.assert_called_once_with("Test prompt")
    
    @patch('models.gemini.genai.configure')
    @patch('models.gemini.genai.GenerativeModel')
    async def test_generate_async_success(self, mock_generative_model, mock_configure):
        """Test asynchronous generation."""
        config = ModelConfig(model_name="gemini-1.5-pro", api_key="test-key")
        
        # Mock the response
        mock_response = MagicMock()
        mock_response.text = "go"
        mock_response.finish_reason = "completed"
        mock_response.safety_ratings = []
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.total_token_count = 10
        
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_generative_model.return_value = mock_model
        
        gemini_model = GeminiModel(config)
        
        # Mock asyncio.to_thread to avoid actual threading
        with patch('asyncio.to_thread') as mock_to_thread:
            mock_to_thread.return_value = mock_response
            
            result = await gemini_model.generate("Test prompt")
            
            assert isinstance(result, ModelResponse)
            assert result.text == "go"
            assert result.tokens_used == 10
            assert result.latency is not None
            
            mock_to_thread.assert_called_once()
    
    @patch('models.gemini.genai.configure')
    @patch('models.gemini.genai.GenerativeModel')
    async def test_generate_batch(self, mock_generative_model, mock_configure):
        """Test batch generation."""
        config = ModelConfig(model_name="gemini-1.5-pro", api_key="test-key")
        
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model
        
        gemini_model = GeminiModel(config)
        
        # Mock the generate method
        mock_responses = [
            ModelResponse(text="stop", tokens_used=5, latency=0.5),
            ModelResponse(text="go", tokens_used=3, latency=0.3)
        ]
        
        with patch.object(gemini_model, 'generate', side_effect=mock_responses):
            with patch('asyncio.sleep'):  # Mock sleep to speed up test
                
                results = await gemini_model.generate_batch(["red light", "green light"])
                
                assert len(results) == 2
                assert results[0].text == "stop"
                assert results[1].text == "go"
    
    @patch('models.gemini.genai.configure')
    @patch('models.gemini.genai.GenerativeModel')
    def test_generate_sync_error_handling(self, mock_generative_model, mock_configure):
        """Test error handling in synchronous generation."""
        config = ModelConfig(model_name="gemini-1.5-pro", api_key="test-key")
        
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_generative_model.return_value = mock_model
        
        gemini_model = GeminiModel(config)
        
        with pytest.raises(Exception, match="API Error"):
            gemini_model.generate_sync("Test prompt")
    
    @patch('models.gemini.genai.configure')
    @patch('models.gemini.genai.GenerativeModel')
    async def test_generate_async_error_handling(self, mock_generative_model, mock_configure):
        """Test error handling in asynchronous generation."""
        config = ModelConfig(model_name="gemini-1.5-pro", api_key="test-key")
        
        mock_model = MagicMock()
        mock_generative_model.return_value = mock_model
        
        gemini_model = GeminiModel(config)
        
        with patch('asyncio.to_thread', side_effect=Exception("Async Error")):
            with pytest.raises(Exception, match="Async Error"):
                await gemini_model.generate("Test prompt")
    
    def test_get_token_count_with_usage_metadata(self):
        """Test token count extraction when usage metadata is available."""
        config = ModelConfig(model_name="gemini-1.5-pro", api_key="test-key")
        
        with patch('models.gemini.genai.configure'), \
             patch('models.gemini.genai.GenerativeModel'):
            
            gemini_model = GeminiModel(config)
            
            # Mock response with usage metadata
            mock_response = MagicMock()
            mock_response.usage_metadata = MagicMock()
            mock_response.usage_metadata.total_token_count = 25
            
            token_count = gemini_model._get_token_count(mock_response)
            assert token_count == 25
    
    def test_get_token_count_without_usage_metadata(self):
        """Test token count extraction when no usage metadata."""
        config = ModelConfig(model_name="gemini-1.5-pro", api_key="test-key")
        
        with patch('models.gemini.genai.configure'), \
             patch('models.gemini.genai.GenerativeModel'):
            
            gemini_model = GeminiModel(config)
            
            # Mock response without usage metadata
            mock_response = MagicMock()
            del mock_response.usage_metadata  # Remove the attribute
            
            token_count = gemini_model._get_token_count(mock_response)
            assert token_count is None
    
    @patch('models.gemini.genai.configure')
    @patch('models.gemini.genai.GenerativeModel')
    def test_supports_batch(self, mock_generative_model, mock_configure):
        """Test that Gemini supports batch processing."""
        config = ModelConfig(model_name="gemini-1.5-pro", api_key="test-key")
        
        gemini_model = GeminiModel(config)
        assert gemini_model.supports_batch() is True
    
    @patch('models.gemini.genai.configure')
    @patch('models.gemini.genai.GenerativeModel')
    def test_get_model_info(self, mock_generative_model, mock_configure):
        """Test getting model information."""
        config = ModelConfig(model_name="gemini-1.5-pro", api_key="test-key")
        
        gemini_model = GeminiModel(config)
        info = gemini_model.get_model_info()
        
        assert isinstance(info, dict)
        assert info["model_name"] == "gemini-1.5-pro"
        assert info["provider"] == "Google"
        assert info["model_type"] == "Gemini"
        assert info["supports_streaming"] is False
        assert info["supports_function_calling"] is True
        assert "config" in info 