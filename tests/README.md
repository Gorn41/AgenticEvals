"""
AgenticEvals Test Suite
============================

Comprehensive test suite for the AgenticEvals framework.

## Structure

```
tests/
├── conftest.py                     # Pytest configuration and fixtures
├── test_benchmark/                 # Benchmark framework tests
│   ├── test_base.py               # Core benchmark classes
│   └── test_loader.py             # Benchmark loading functionality
├── test_benchmarks/               # Specific benchmark tests
│   └── test_simple_reflex_example.py
├── test_models/                   # Model system tests
│   ├── test_base.py              # Base model classes
│   ├── test_gemini.py            # Gemini model implementation
│   └── test_loader.py            # Model loading functionality
├── test_utils/                    # Utility tests
│   ├── test_config.py            # Configuration management
│   └── test_logging.py           # Logging functionality
└── pytest.ini                    # Pytest configuration
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install pytest pytest-asyncio
```

### Basic Usage

```bash
# Run all tests
python3 run_tests.py

# Run specific test categories
python3 run_tests.py --unit          # Unit tests only
python3 run_tests.py --integration   # Integration tests only
python3 run_tests.py --fast          # Fast tests only

# Run with coverage
python3 run_tests.py --coverage

# Run specific test file
pytest tests/test_models/test_gemini.py

# Run specific test
pytest tests/test_models/test_gemini.py::TestGeminiModel::test_init
```

## Test Categories

### Test Markers

- **Unit tests** (default): Fast, isolated tests with minimal dependencies
- **Integration tests** (`@pytest.mark.integration`): Tests requiring API keys
- **Slow tests** (`@pytest.mark.slow`): Tests that take longer to execute

### Integration Tests

Integration tests require valid API keys:

```bash
# Set up API keys for integration tests
python3 setup_env.py

# Or set environment variables directly:
export GOOGLE_API_KEY="your-gemini-api-key"
# or
export GEMINI_API_KEY="your-gemini-api-key"
```

### Coverage Reporting

Generate coverage reports:

```bash
# Install coverage tool
pip install pytest-cov

# Run with coverage
python3 run_tests.py --coverage

# Generate HTML report
pytest --cov=src --cov-report=html
```

## Test Design Principles

### 1. Real Testing Over Mocking

Tests prioritize real functionality validation:
- Configuration and validation tests use real data structures
- Integration tests use real API calls when possible
- Environment-based conditional testing for external dependencies

### 2. Clear Test Categories

- **Unit tests**: Fast, isolated validation of individual components
- **Integration tests**: Real API interactions with proper error handling
- **Configuration tests**: Environment and setup validation

### Common Fixtures

Common fixtures are defined in `conftest.py`:

- `api_key`: Real API key for integration tests
- `sample_model_config`: Valid model configuration
- `sample_model_response`: Example response data
- `temp_config_file`: Temporary configuration for testing

## Writing Tests

### Guidelines

1. **Use descriptive test names** that explain what is being tested
2. **Test both success and failure cases**
3. **Use appropriate test markers** (`@pytest.mark.integration`, `@pytest.mark.slow`)
4. **Update fixtures** in `conftest.py` if needed
5. **Test configuration and validation** rather than implementation details

### Example Test Structure

```python
import pytest
import os

from models.loader import load_gemini


class TestModelIntegration:
    """Integration tests for model functionality."""
    
    @pytest.mark.integration
    def test_model_loading(self):
        """Test loading a real model."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            pytest.skip("API key not available")
        
        model = load_gemini("gemini-2.5-flash", api_key=api_key)
        assert model is not None
        assert model.model_name == "gemini-2.5-flash"


class TestConfiguration:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test creating valid configuration."""
        config = ModelConfig(model_name="gemini-2.5-pro")
        assert config.model_name == "gemini-2.5-pro"
```

## Continuous Integration

The test suite is designed to work in CI environments:

- Environment variables for API keys
- Conditional skipping of integration tests
- Fast unit tests for quick feedback
- Comprehensive coverage reporting

## Troubleshooting

### Common Issues

1. **"No API key available"**: Set `GOOGLE_API_KEY` or `GEMINI_API_KEY`
2. **Import errors**: Run tests from project root directory
3. **Async test issues**: Ensure `pytest-asyncio` is installed

### Running Specific Test Types

```bash
# Only unit tests (no API calls)
pytest -m "not integration"

# Only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
``` 