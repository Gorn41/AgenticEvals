# Testing Guide

This document provides comprehensive testing guidelines for the AgenticEvals project.

## Test Structure

```
tests/
├── conftest.py                        # Central fixtures and configuration
├── pytest.ini                        # Pytest settings
├── test_benchmark/                    # Benchmark framework tests
├── test_benchmarks/                   # Specific benchmark implementations
├── test_models/                       # Model system tests
└── test_utils/                        # Utility function tests
```

## Test Categories

### Unit Tests
- **Purpose**: Fast, isolated testing of individual components
- **Scope**: Configuration validation, class initialization, method signatures
- **Dependencies**: Minimal external dependencies
- **Execution**: Always run in CI/CD

### Integration Tests
- **Purpose**: End-to-end testing with real APIs
- **Scope**: Real model calls, benchmark execution, API integration
- **Dependencies**: Requires valid API keys
- **Execution**: Conditional based on environment

### Configuration Tests
- **Purpose**: Validate setup and configuration
- **Scope**: Environment variables, file parsing, validation rules
- **Dependencies**: File system, environment variables
- **Execution**: Part of standard test suite

## Testing Strategy

### Real Data Over Simulation
- **Configuration objects** with real validation
- **Environment-based testing** for external dependencies
- **Integration testing** with actual APIs when available
- **Conditional test execution** based on resource availability

### Test Isolation
- **Independent test cases** that don't affect each other
- **Clean test environments** with proper setup/teardown
- **Focused test scope** testing one thing at a time

### Error Handling Coverage
- **Input validation** testing with invalid data
- **Exception handling** testing error conditions
- **Resource availability** testing missing dependencies
- **API failure scenarios** testing network/service issues

## Running Tests

### Standard Test Execution

```bash
# All tests
python3 run_tests.py

# Specific categories
python3 run_tests.py --unit          # Unit tests only
python3 run_tests.py --integration   # Integration tests (requires API keys)
python3 run_tests.py --fast          # Fast tests only
python3 run_tests.py --coverage      # With coverage reporting
```

### Environment Setup

```bash
# For integration tests
export GOOGLE_API_KEY="your_gemini_api_key"
# or
export GEMINI_API_KEY="your_gemini_api_key"

# Alternative: use setup script
python3 setup_env.py
```

### Pytest Commands

```bash
# Direct pytest usage
pytest tests/                         # All tests
pytest tests/test_models/             # Model tests only
pytest -m integration                 # Integration tests only
pytest -m "not integration"           # Exclude integration tests
pytest --cov=src                      # With coverage
```

## Test Development

### Writing New Tests

1. **Choose the right test type**:
   - Unit test for isolated functionality
   - Integration test for real API interactions
   - Configuration test for setup validation

2. **Use descriptive names**:
   ```python
   def test_model_config_validates_temperature_range():
       """Test that temperature validation rejects out-of-range values."""
   ```

3. **Test both success and failure**:
   ```python
   def test_valid_configuration_creation():
       # Test successful creation
       
   def test_invalid_configuration_raises_error():
       # Test error conditions
   ```

4. **Use appropriate markers**:
   ```python
   @pytest.mark.integration
   def test_real_api_call():
       # Test requiring external resources
   ```

### Test Fixtures

Use centralized fixtures from `conftest.py`:

```python
def test_with_fixtures(sample_model_config, api_key):
    """Test using common fixtures."""
    if not api_key:
        pytest.skip("No API key available")
    
    # Use fixtures in test
```

### Conditional Testing

Handle missing resources gracefully:

```python
@pytest.mark.integration
def test_requiring_api_key():
    """Test that requires API access."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.skip("No API key available for integration test")
    
    # Proceed with test
```

## Best Practices

### Test Design
- **One assertion per test concept**
- **Clear test documentation**
- **Predictable test data**
- **Isolated test execution**

### Performance
- **Fast unit tests** for quick feedback
- **Efficient resource usage** in integration tests
- **Parallel test execution** where possible
- **Appropriate test timeouts**

### Maintenance
- **Update tests with code changes**
- **Remove obsolete tests**
- **Keep test dependencies minimal**
- **Document complex test setups**

### Coverage Goals
- **High unit test coverage** (>90%)
- **Critical path integration testing**
- **Error condition coverage**
- **Configuration validation coverage**

## Continuous Integration

### CI Pipeline Testing
- **Fast unit tests** run on every commit
- **Integration tests** run on main branch
- **Coverage reporting** for quality metrics
- **Test result reporting** for visibility

### Environment Requirements
- **Python dependencies** installed
- **API keys** available (for integration tests)
- **Test data** accessible
- **Clean test environment** for each run

## Debugging Tests

### Common Issues
1. **Import errors**: Check Python path and working directory
2. **Missing dependencies**: Install test requirements
3. **API key issues**: Verify environment variables
4. **Async test problems**: Check pytest-asyncio installation

### Debugging Commands
```bash
# Verbose output
pytest -v -s tests/test_models/test_gemini.py

# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Run specific test with output
pytest -v -s tests/test_models/test_gemini.py::test_specific_function
```

## Quality Metrics

### Coverage Targets
- **Unit tests**: >90% line coverage
- **Integration tests**: Critical functionality covered
- **Overall**: >85% combined coverage

### Performance Targets
- **Unit tests**: <1 second per test
- **Integration tests**: <30 seconds per test
- **Full test suite**: <5 minutes total

### Reliability Targets
- **Test stability**: >99% pass rate in CI
- **Flaky test rate**: <1% of total tests
- **Test maintenance**: Regular review and updates 