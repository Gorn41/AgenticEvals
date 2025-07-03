"""
Pytest configuration and shared fixtures.
"""

import pytest
import os
from pathlib import Path

from models.base import ModelConfig, ModelResponse, BaseModel
from benchmark.base import Task, TaskResult, BenchmarkResult, AgentType


# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"


@pytest.fixture
def api_key():
    """Provide API key for integration tests."""
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")


@pytest.fixture
def sample_model_config():
    """Fixture providing a sample model configuration."""
    return ModelConfig(
        model_name="gemini-2.5-pro",
        api_key="test-api-key",
        temperature=0.7,
        max_tokens=1000
    )


@pytest.fixture
def sample_model_response():
    """Fixture providing a sample model response."""
    return ModelResponse(
        text="Test response",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        finish_reason="completed",
        metadata={"model": "gemini-2.5-pro", "finish_reason": "completed"}
    )


@pytest.fixture
def sample_task():
    """Fixture providing a sample task."""
    return Task(
        id="test_task_1",
        prompt="Test prompt",
        expected_output="Test expected output",
        metadata={"category": "test"}
    )


@pytest.fixture
def sample_task_result():
    """Fixture providing a sample task result."""
    return TaskResult(
        task_id="test_task_1",
        model_response=ModelResponse(
            text="Test model response",
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            finish_reason="completed"
        ),
        score=0.85,
        passed=True,
        execution_time=1.2,
        metadata={"evaluation_method": "exact_match"}
    )


@pytest.fixture
def sample_benchmark_result():
    """Fixture providing a sample benchmark result."""
    task_results = [
        TaskResult(
            task_id="task_1",
            model_response=ModelResponse(text="Response 1", prompt_tokens=5, completion_tokens=3, total_tokens=8),
            score=0.9,
            passed=True,
            execution_time=1.0
        ),
        TaskResult(
            task_id="task_2",
            model_response=ModelResponse(text="Response 2", prompt_tokens=6, completion_tokens=4, total_tokens=10),
            score=0.8,
            passed=True,
            execution_time=1.1
        )
    ]
    
    return BenchmarkResult(
        benchmark_name="test_benchmark",
        model_name="gemini-2.5-flash",
        agent_type=AgentType.SIMPLE_REFLEX,
        task_results=task_results,
        overall_score=0.85,
        metadata={"test_run": True}
    )


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    config_content = """
default_model: "gemini-2.5-pro"
api_keys:
  google: "test-key"
log_level: "DEBUG"
timeout_seconds: 30
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def sample_config():
    """Fixture providing a sample configuration object."""
    from utils.config import Config
    
    config = Config()
    config.default_model = "gemini-2.5-pro"
    config.api_keys = {"google": "test-key"}
    config.log_level = "DEBUG"
    config.timeout_seconds = 30
    
    return config


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test requiring API keys")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle integration tests."""
    for item in items:
        # Add integration marker to tests that need API keys
        if "integration" in item.keywords:
            if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
                item.add_marker(pytest.mark.skip(reason="No API key available"))


# Helper functions for tests
def create_test_model_response(text="test", tokens=10):
    """Helper to create test model responses."""
    return ModelResponse(
        text=text,
        prompt_tokens=tokens//2,
        completion_tokens=tokens//2,
        total_tokens=tokens,
        finish_reason="completed"
    ) 