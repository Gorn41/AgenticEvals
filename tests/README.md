# LLM-AgentTypeEval Test Suite

This directory contains comprehensive tests for the LLM-AgentTypeEval project.

## Structure

```
tests/
├── conftest.py                     # Pytest configuration and fixtures
├── test_models/                    # Model-related tests
│   ├── test_base.py               # Base model interface tests
│   ├── test_gemini.py             # Gemini model implementation tests
│   └── test_loader.py             # Model loader tests
├── test_benchmark/                 # Benchmark framework tests
│   ├── test_base.py               # Base benchmark classes tests
│   ├── test_loader.py             # Benchmark loader tests
│   └── test_registry.py           # Benchmark registry tests
├── test_benchmarks/               # Specific benchmark tests
│   └── test_simple_reflex_example.py
├── test_utils/                    # Utility function tests
│   ├── test_config.py             # Configuration management tests
│   └── test_logging.py            # Logging utilities tests
└── pytest.ini                    # Pytest configuration
```

## Running Tests

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install test dependencies:
```bash
pip install pytest pytest-asyncio
```

### Basic Test Execution

Run all tests:
```bash
pytest
```

Run tests with verbose output:
```bash
pytest -v
```

Run specific test file:
```bash
pytest tests/test_models/test_loader.py
```

Run specific test class:
```bash
pytest tests/test_models/test_loader.py::TestModelLoader
```

Run specific test:
```bash
pytest tests/test_models/test_loader.py::TestModelLoader::test_load_model_success
```

### Test Categories

Tests are organized with markers:

- **Unit tests** (default): Fast, isolated tests with mocked dependencies
- **Integration tests**: Tests that may require API keys and external services
- **Slow tests**: Tests that take longer to run

Run only unit tests:
```bash
pytest -m "not integration and not slow"
```

Run integration tests (requires API keys):
```bash
pytest -m integration
```

Skip slow tests:
```bash
pytest -m "not slow"
```

### Environment Setup for Integration Tests

Integration tests require API keys. Set them before running:

```bash
# Option 1: Use .env file (recommended)
python3 setup_env.py

# Option 2: Set environment variables
export GOOGLE_API_KEY="your-gemini-api-key"
# or
export GEMINI_API_KEY="your-gemini-api-key"
```

Skip integration tests if no API key:
```bash
pytest -m "not integration"
```

## Test Organization

### Unit Tests
- Mock external dependencies (API calls, file I/O)
- Fast execution (< 1 second per test)
- Test individual components in isolation
- High coverage of edge cases and error conditions

### Integration Tests
- Test actual API calls and end-to-end workflows
- Require valid API keys
- May be slower due to network calls
- Marked with `@pytest.mark.integration`

### Fixtures

Common fixtures are defined in `conftest.py`:

- `mock_api_key`: Provides test API key
- `sample_model_config`: Pre-configured ModelConfig
- `sample_model_response`: Sample ModelResponse object
- `sample_benchmark_config`: Pre-configured BenchmarkConfig
- `sample_task`: Sample Task object
- `mock_model`: Mocked model for testing
- `clean_environment`: Cleans environment variables

### Mocking Strategy

Tests use `unittest.mock` for mocking:
- External API calls (Gemini API)
- File system operations
- Environment variables
- Network requests

## Coverage

Check test coverage:
```bash
pip install pytest-cov
pytest --cov=src --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html
```

## Adding New Tests

When adding new functionality:

1. **Create corresponding test file** in appropriate test directory
2. **Add unit tests** for all public methods and edge cases
3. **Mock external dependencies** to ensure fast, reliable tests
4. **Add integration tests** if component interacts with external services
5. **Use appropriate markers** (`@pytest.mark.integration`, `@pytest.mark.slow`)
6. **Update fixtures** in `conftest.py` if needed

### Example Test Structure

```python
"""
Tests for new functionality.
"""

import pytest
from unittest.mock import patch, MagicMock

from your_module import YourClass


class TestYourClass:
    """Test the YourClass class."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        instance = YourClass()
        
        # Act
        result = instance.method()
        
        # Assert
        assert result == expected_value
    
    @patch('your_module.external_dependency')
    def test_with_mocked_dependency(self, mock_dependency):
        """Test with mocked external dependency."""
        # Setup mock
        mock_dependency.return_value = "mocked_result"
        
        # Test
        instance = YourClass()
        result = instance.method_using_dependency()
        
        # Verify
        assert result == "expected_based_on_mock"
        mock_dependency.assert_called_once()
    
    @pytest.mark.integration
    def test_integration_scenario(self):
        """Test integration scenario (requires real dependencies)."""
        # Skip if prerequisites not met
        if not os.getenv("REQUIRED_API_KEY"):
            pytest.skip("API key not available")
        
        # Run integration test
        pass
```

## Continuous Integration

Tests are designed to run in CI environments:
- All unit tests should pass without external dependencies
- Integration tests can be skipped if API keys not available
- Fast execution (unit tests < 10 seconds total)
- Clear failure messages and debugging information 