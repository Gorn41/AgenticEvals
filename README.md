# AgenticEvals

A comprehensive benchmark for evaluating Large Language Models (LLMs) across five classic AI agent types for systematic agent capability assessment.

## Overview

AgenticEvals provides a structured framework to evaluate how well LLMs can embody different agent architectures:

- **Simple Reflex Agents**: Condition-action rules
- **Model-Based Reflex Agents**: Internal state tracking  
- **Goal-Based Agents**: Goal-directed reasoning
- **Utility-Based Agents**: Utility maximization
- **Learning Agents**: Adaptive behavior

## Quick Start

### 1. Installation

```bash
git clone https://github.com/Gorn41/AgenticEvals.git
cd AgenticEvals
pip install -r requirements.txt
```

### 2. API Key Setup

```bash
python3 setup_env.py
# Follow the prompts to set up your Gemini API key
```

Or set manually:
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

### 3. Run Example

```bash
python3 test_gemini_pro_full.py
```

## Usage

### Basic Example

```python
from models.loader import load_gemini
from benchmark.loader import load_benchmark

# Load model
model = load_gemini("gemini-2.5-pro")

# Load and run benchmark
benchmark = load_benchmark("traffic_light_simple")
results = await benchmark.run_benchmark(model)

print(f"Overall Score: {results.overall_score}")
print(f"Success Rate: {results.get_success_rate()}")
```

### Available Benchmarks

```python
from benchmark.loader import get_available_benchmarks

benchmarks = get_available_benchmarks()
for agent_type, benchmark_list in benchmarks.items():
    print(f"{agent_type}: {benchmark_list}")
```

### Custom Model Configuration

```python
model = load_gemini(
    "gemini-2.5-flash",
    temperature=0.1,
    max_tokens=1000,
    top_p=0.9
)
```

## Project Structure

```
AgenticEvals/
├── src/
│   ├── models/           # Model implementations
│   ├── benchmark/        # Benchmark framework
│   ├── benchmarks/       # Specific benchmarks
│   └── utils/           # Utilities
├── tests/               # Test suite
└── examples/           # Usage examples
```

## Supported Models

- **Gemini 2.5 Pro** - Most capable model
- **Gemini 2.5 Flash** - Fast and efficient

Easy to extend with new model providers.

## Benchmark Types

### Simple Reflex Agent
- **Traffic Light**: Basic stimulus-response scenarios
- **Security Guard**: Pattern recognition tasks

### Model-Based Reflex Agent  
- **Navigation**: State-aware pathfinding
- **Inventory Management**: Resource tracking

### Goal-Based Agent
- **Task Planning**: Multi-step goal achievement
- **Problem Solving**: Constraint satisfaction

### Utility-Based Agent
- **Resource Allocation**: Optimization under constraints
- **Decision Making**: Trade-off scenarios

### Learning Agent
- **Adaptation**: Performance improvement over time
- **Strategy Evolution**: Dynamic behavior modification

## Configuration

Create a `.env` file:
```bash
# Gemini API Configuration
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional: Alternative key name
GEMINI_API_KEY=your_gemini_api_key_here
```

Or use YAML configuration:
```yaml
# config.yaml
models:
  google: "your-gemini-api-key"

benchmarks:
  timeout_seconds: 30
  max_retries: 3
  collect_detailed_metrics: true
```

## Testing

```bash
# Run all tests
python3 run_tests.py

# Unit tests only
python3 run_tests.py --unit

# Integration tests (requires API key)  
python3 run_tests.py --integration

# With coverage
python3 run_tests.py --coverage
```

## Development

### Adding New Models

1. Implement the `BaseModel` interface
2. Register in `models/loader.py`
3. Add tests in `tests/test_models/`

### Adding New Benchmarks

1. Inherit from `BaseBenchmark`
2. Register with `@register_benchmark` decorator
3. Implement required methods
4. Add tests in `tests/test_benchmarks/`

Example:
```python
from benchmark.base import BaseBenchmark
from benchmark.registry import register_benchmark

@register_benchmark("my_benchmark", AgentType.SIMPLE_REFLEX)
class MyBenchmark(BaseBenchmark):
    def get_tasks(self):
        # Return list of tasks
        pass
    
    async def evaluate_task(self, task, model):
        # Evaluate single task
        pass
    
    def calculate_score(self, task, model_response):
        # Calculate task score
        pass
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use AgenticEvals in your research, please cite:

```bibtex
@software{agentic_evals,
  title={AgenticEvals: A Benchmark for LLM Agent Capabilities},
  author={Nattaput (Gorn) Namchittai},
  year={2025},
  url={https://github.com/Gorn41/AgenticEvals}
}
```

## Support

- Bug Reports: [GitHub Issues](https://github.com/Gorn41/AgenticEvals/issues)