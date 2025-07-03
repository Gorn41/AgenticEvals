"""
Pytest configuration and common fixtures for LLM-AgentTypeEval tests.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any

# Add src to Python path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def sample_model_config():
    """Provide a sample model configuration."""
    from models.base import ModelConfig
    
    return ModelConfig(
        model_name="gemini-1.5-pro",
        api_key="test-api-key",
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9,
        stop_sequences=["END"]
    )


@pytest.fixture
def sample_model_response():
    """Provide a sample model response."""
    from models.base import ModelResponse
    
    return ModelResponse(
        text="stop",
        tokens_used=10,
        latency=1.5,
        metadata={"model": "gemini-1.5-pro", "finish_reason": "completed"}
    )


@pytest.fixture
def sample_benchmark_config():
    """Provide a sample benchmark configuration."""
    from benchmark.base import BenchmarkConfig, AgentType
    
    return BenchmarkConfig(
        benchmark_name="test_benchmark",
        agent_type=AgentType.SIMPLE_REFLEX,
        num_tasks=5,
        random_seed=42,
        timeout_seconds=30.0,
        max_retries=2,
        collect_detailed_metrics=True,
        save_responses=True
    )


@pytest.fixture
def sample_task():
    """Provide a sample task."""
    from benchmark.base import Task
    
    return Task(
        task_id="test_task_1",
        name="Test Traffic Light",
        description="Test red light response",
        prompt="You see a red traffic light. What do you do?",
        expected_output="stop",
        evaluation_criteria={"exact_match": True},
        metadata={"signal": "red", "difficulty": "basic"}
    )


@pytest.fixture
def sample_task_result():
    """Provide a sample task result."""
    from benchmark.base import TaskResult, AgentType
    from models.base import ModelResponse
    
    return TaskResult(
        task_id="test_task_1",
        task_name="Test Traffic Light",
        agent_type=AgentType.SIMPLE_REFLEX,
        success=True,
        score=1.0,
        metrics={"word_count": 1, "follows_instructions": True},
        model_response=ModelResponse(text="stop", tokens_used=5, latency=0.8),
        execution_time=0.8,
        metadata={"signal": "red"}
    )


@pytest.fixture
def mock_model():
    """Provide a mock model for testing."""
    mock = AsyncMock()
    mock.model_name = "test-model"
    mock.config = MagicMock()
    mock.generate.return_value = ModelResponse(
        text="stop",
        tokens_used=10,
        latency=1.0,
        metadata={}
    )
    mock.generate_sync.return_value = ModelResponse(
        text="stop", 
        tokens_used=10,
        latency=1.0,
        metadata={}
    )
    mock.supports_batch.return_value = True
    mock.get_model_info.return_value = {
        "model_name": "test-model",
        "provider": "test"
    }
    return mock


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    config_content = """
default_model: "gemini-1.5-pro"
api_keys:
  google: "test-key"
log_level: "DEBUG"
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean environment variables before each test."""
    # Store original values
    original_env = {}
    env_vars_to_clean = [
        "GOOGLE_API_KEY", "GEMINI_API_KEY", "LOG_LEVEL", 
        "RESULTS_DIR", "TIMEOUT_SECONDS"
    ]
    
    for var in env_vars_to_clean:
        original_env[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]
    
    yield
    
    # Restore original values
    for var, value in original_env.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]


@pytest.fixture
def mock_gemini_response():
    """Mock Gemini API response."""
    mock_response = MagicMock()
    mock_response.text = "stop"
    mock_response.finish_reason = "completed"
    mock_response.safety_ratings = []
    mock_response.usage_metadata = MagicMock()
    mock_response.usage_metadata.total_token_count = 15
    return mock_response


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", 
        "integration: marks tests as integration tests (may require API keys)"
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (may take longer to run)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests  
        if any(keyword in item.name.lower() for keyword in ["slow", "benchmark", "full"]):
            item.add_marker(pytest.mark.slow) 