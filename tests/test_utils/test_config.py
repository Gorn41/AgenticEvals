"""
Tests for configuration management.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

from utils.config import Config, ConfigManager, get_config_manager, load_config, save_config


class TestConfig:
    """Test the Config dataclass."""
    
    def test_config_defaults(self):
        """Test Config with default values."""
        config = Config()
        
        assert config.default_model == "gemini-1.5-pro"
        assert config.api_keys == {}
        assert config.log_level == "INFO"
        assert config.timeout_seconds == 60.0
        assert config.max_retries == 3
        assert config.default_benchmark_config is not None
    
    def test_config_custom_values(self):
        """Test Config with custom values."""
        config = Config(
            default_model="custom-model",
            api_keys={"test": "key"},
            log_level="DEBUG",
            timeout_seconds=30.0
        )
        
        assert config.default_model == "custom-model"
        assert config.api_keys["test"] == "key"
        assert config.log_level == "DEBUG"
        assert config.timeout_seconds == 30.0


class TestConfigManager:
    """Test the ConfigManager class."""
    
    def test_init_without_existing_config(self, tmp_path):
        """Test ConfigManager initialization without existing config file."""
        config_path = tmp_path / "nonexistent.yaml"
        
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            manager = ConfigManager(config_path)
            
            assert manager.config.api_keys["google"] == "test-key"
    
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        manager = ConfigManager()
        
        with patch.dict(os.environ, {
            "GOOGLE_API_KEY": "google-key",
            "GEMINI_API_KEY": "gemini-key", 
            "LOG_LEVEL": "DEBUG",
            "TIMEOUT_SECONDS": "45.0"
        }):
            manager._load_from_env()
            
            assert manager.config.api_keys["google"] == "google-key"
            assert manager.config.api_keys["gemini"] == "gemini-key"
            assert manager.config.log_level == "DEBUG"
            assert manager.config.timeout_seconds == 45.0
    
    def test_load_config_yaml(self, tmp_path):
        """Test loading config from YAML file."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
default_model: "test-model"
api_keys:
  google: "yaml-key"
log_level: "WARNING"
""")
        
        manager = ConfigManager(config_file)
        config = manager.load_config()
        
        assert config.default_model == "test-model"
        assert config.api_keys["google"] == "yaml-key"
        assert config.log_level == "WARNING"
    
    def test_save_config_yaml(self, tmp_path):
        """Test saving config to YAML file."""
        config_file = tmp_path / "save_test.yaml"
        
        manager = ConfigManager(config_file)
        manager.config.default_model = "saved-model"
        manager.config.api_keys["test"] = "saved-key"
        
        manager.save_config()
        
        assert config_file.exists()
        content = config_file.read_text()
        assert "saved-model" in content
        assert "saved-key" in content
    
    def test_get_model_config(self):
        """Test getting model configuration."""
        manager = ConfigManager()
        manager.config.api_keys["google"] = "test-key"
        manager.config.timeout_seconds = 30.0
        
        model_config = manager.get_model_config("gemini-1.5-pro")
        
        assert model_config["model_name"] == "gemini-1.5-pro"
        assert model_config["api_key"] == "test-key"
        assert model_config["timeout_seconds"] == 30.0
    
    def test_get_benchmark_config(self):
        """Test getting benchmark configuration."""
        manager = ConfigManager()
        manager.config.default_benchmark_config = {
            "collect_detailed_metrics": True,
            "save_responses": False
        }
        
        benchmark_config = manager.get_benchmark_config(
            num_tasks=10,
            save_responses=True  # Should override default
        )
        
        assert benchmark_config["collect_detailed_metrics"] is True
        assert benchmark_config["save_responses"] is True  # Overridden
        assert benchmark_config["num_tasks"] == 10


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @patch('utils.config.ConfigManager')
    def test_load_config(self, mock_manager_class):
        """Test load_config function."""
        mock_manager = MagicMock()
        mock_config = Config()
        mock_manager.load_config.return_value = mock_config
        mock_manager_class.return_value = mock_manager
        
        result = load_config(Path("test.yaml"))
        
        mock_manager_class.assert_called_once_with(Path("test.yaml"))
        assert result == mock_config
    
    @patch('utils.config.get_config_manager')
    def test_save_config(self, mock_get_manager):
        """Test save_config function."""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        
        test_config = Config(default_model="test")
        save_config(test_config)
        
        assert mock_manager.config == test_config
        mock_manager.save_config.assert_called_once()
    
    @patch('utils.config._config_manager', None)
    def test_get_config_manager_creates_instance(self):
        """Test that get_config_manager creates new instance when needed."""
        manager = get_config_manager()
        assert manager is not None
        
        # Second call should return same instance
        manager2 = get_config_manager()
        assert manager is manager2 