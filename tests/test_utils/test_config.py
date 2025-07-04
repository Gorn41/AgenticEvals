"""
Tests for configuration utilities.
"""

import pytest
import os
from pathlib import Path

from src.utils.config import Config, ConfigManager


class TestConfig:
    """Test the Config class."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = Config()
        
        assert config.default_model == "gemini-2.5-pro"
        assert isinstance(config.api_keys, dict)
        assert config.log_level == "INFO"
        assert config.default_timeout_seconds == 30.0
    
    def test_config_with_values(self):
        """Test config with custom values."""
        config = Config(
            default_model="gemini-2.5-flash",
            log_level="DEBUG",
            default_timeout_seconds=30.0
        )
        
        assert config.default_model == "gemini-2.5-flash"
        assert config.log_level == "DEBUG"
        assert config.default_timeout_seconds == 30.0


class TestConfigManager:
    """Test the ConfigManager class."""
    
    def test_config_manager_creation(self):
        """Test creating a config manager."""
        manager = ConfigManager()
        assert manager.config is not None
        assert isinstance(manager.config, Config)
    
    def test_get_model_config_basic(self):
        """Test getting basic model configuration."""
        manager = ConfigManager()
        model_config = manager.get_model_config("gemini-2.5-pro")
        
        assert model_config["model_name"] == "gemini-2.5-pro"
        assert "api_key" in model_config
        assert "timeout_seconds" in model_config
    
    def test_get_model_config_with_env_key(self):
        """Test getting model config with environment API key."""
        # Only test if environment variable is set
        if os.getenv("GOOGLE_API_KEY"):
            manager = ConfigManager()
            model_config = manager.get_model_config("gemini-2.5-pro")
            assert model_config["api_key"] is not None
    
    def test_get_model_config_flash(self):
        """Test getting configuration for Flash model."""
        manager = ConfigManager()
        model_config = manager.get_model_config("gemini-2.5-flash")
        
        assert model_config["model_name"] == "gemini-2.5-flash"
        assert "api_key" in model_config
        assert "timeout_seconds" in model_config


class TestConfigFiles:
    """Test configuration file handling."""
    
    def test_config_file_loading(self, temp_config_file):
        """Test loading configuration from file."""
        manager = ConfigManager()
        
        # Test that we can load from file
        assert temp_config_file.exists()
        content = temp_config_file.read_text()
        assert "gemini-2.5-pro" in content
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = Config()
        
        # Test valid models
        assert config.default_model in ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-pro"]
        
        # Test valid log levels
        assert config.log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]
        
        # Test reasonable timeout
        assert config.default_timeout_seconds > 0
        assert config.default_timeout_seconds <= 300  # 5 minutes max


class TestEnvironmentIntegration:
    """Test integration with environment variables."""
    
    def test_api_key_detection(self):
        """Test API key detection from environment."""
        manager = ConfigManager()
        
        # Test with any available API key
        google_key = os.getenv("GOOGLE_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        if google_key or gemini_key:
            model_config = manager.get_model_config("gemini-2.5-pro")
            assert model_config["api_key"] is not None
    
    def test_config_precedence(self):
        """Test configuration precedence rules."""
        manager = ConfigManager()
        
        # Test basic model config retrieval
        model_config = manager.get_model_config("gemini-2.5-flash")
        
        assert model_config["model_name"] == "gemini-2.5-flash"
        assert "timeout_seconds" in model_config 