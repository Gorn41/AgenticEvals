"""
Configuration utilities for AgenticEvals.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict, field
from .logging import get_logger

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class Config:
    """Main configuration class for AgenticEvals."""
    
    # API Keys
    api_keys: Dict[str, str] = field(default_factory=dict)
    
    # Model defaults
    default_model: str = "gemini-2.5-pro"
    default_temperature: float = 0.7
    default_max_tokens: Optional[int] = None
    
    # Benchmark defaults
    default_timeout_seconds: float = 30.0
    default_max_retries: int = 3
    default_collect_detailed_metrics: bool = True
    default_save_responses: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Results
    results_dir: str = "./results"
    
    # Additional custom parameters
    additional_params: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """Manager for loading and saving configuration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize config manager."""
        self.config_path = config_path or Path("config.yaml")
        self.config = Config()
        
        # Try to load existing config
        if self.config_path.exists():
            self.load_config()
        else:
            # Load from environment variables
            self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # API keys
        if google_key := os.getenv("GOOGLE_API_KEY"):
            self.config.api_keys["google"] = google_key
        if gemini_key := os.getenv("GEMINI_API_KEY"):
            self.config.api_keys["gemini"] = gemini_key
        
        # Other settings
        if log_level := os.getenv("LOG_LEVEL"):
            self.config.log_level = log_level
        if results_dir := os.getenv("RESULTS_DIR"):
            self.config.results_dir = results_dir
        if timeout := os.getenv("TIMEOUT_SECONDS"):
            try:
                self.config.timeout_seconds = float(timeout)
            except ValueError:
                logger.warning(f"Invalid timeout value: {timeout}")
    
    def load_config(self) -> Config:
        """Load configuration from file."""
        try:
            if self.config_path.suffix.lower() == '.json':
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
            else:
                with open(self.config_path, 'r') as f:
                    data = yaml.safe_load(f) or {}
            
            # Update config with loaded data
            for key, value in data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    logger.warning(f"Unknown config key: {key}")
            
            logger.info(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
        
        # Still load from environment to override file settings
        self._load_from_env()
        return self.config
    
    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            config_dict = asdict(self.config)
            
            # Create parent directory if it doesn't exist
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config_path.suffix.lower() == '.json':
                with open(self.config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                with open(self.config_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved configuration to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model configuration for a specific model."""
        config = {
            "model_name": model_name,
            "timeout_seconds": self.config.timeout_seconds,
        }
        
        # Add API key if available
        if model_name.startswith("gemini") or model_name.startswith("google"):
            if "google" in self.config.api_keys:
                config["api_key"] = self.config.api_keys["google"]
            elif "gemini" in self.config.api_keys:
                config["api_key"] = self.config.api_keys["gemini"]
        
        return config
    
    def get_benchmark_config(self, **overrides) -> Dict[str, Any]:
        """Get benchmark configuration with optional overrides."""
        config = self.config.default_benchmark_config.copy()
        config.update(overrides)
        return config


# Global config manager
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from file."""
    if config_path:
        manager = ConfigManager(config_path)
    else:
        manager = get_config_manager()
    return manager.load_config()


def save_config(config: Config, config_path: Optional[Path] = None) -> None:
    """Save configuration to file."""
    if config_path:
        manager = ConfigManager(config_path)
        manager.config = config
    else:
        manager = get_config_manager()
        manager.config = config
    manager.save_config()


def get_config() -> Config:
    """Get the current configuration."""
    return get_config_manager().config 