# LLM-AgentTypeEval

A comprehensive benchmark for evaluating Large Language Models (LLMs) across the five classic AI agent architectures:

1. **Simple Reflex Agent** - Immediate rule-based responses
2. **Model-Based Reflex Agent** - Responses using internal state/memory  
3. **Goal-Based Agent** - Planning toward explicit goals
4. **Utility-Based Agent** - Utility maximization under trade-offs
5. **Learning Agent** - Adaptation over episodes

## Quick Start

### Prerequisites

- Python 3.8+
- Google API key for Gemini models

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd AgenticEvals
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up your API key:**
```bash
# Get your Gemini API key from: https://makersuite.google.com/app/apikey
export GOOGLE_API_KEY="your-gemini-api-key-here"

# Or create a .env file:
cp .env.example .env
# Edit .env and add your Gemini API key
```

4. **Test the installation:**
```bash
python quick_start.py
```

### Basic Usage

```python
import asyncio
from src.models.loader import load_gemini
from src.benchmark.loader import load_benchmark

async def main():
    # Load Gemini model
    model = load_gemini("gemini-1.5-flash", temperature=0.1)
    
    # Load a benchmark
    benchmark = load_benchmark("traffic_light_simple")
    
    # Run evaluation
    result = await benchmark.run_benchmark(model)
    
    # View results
    print(f"Score: {result.overall_score:.2f}")
    print(f"Success Rate: {result.get_success_rate():.1%}")

asyncio.run(main())
```

## Available Benchmarks

### Simple Reflex Agent
- **traffic_light_simple**: Traffic light response benchmark testing immediate rule-based responses

*More benchmarks coming soon for each agent type...*

## 🏗️ Architecture

### Core Components

- **`src/models/`**: Model loading and calling functionality
  - `base.py`: Abstract model interface
  - `gemini.py`: Gemini model implementation
  - `loader.py`: Model factory and loading utilities

- **`src/benchmark/`**: Benchmark framework
  - `base.py`: Abstract benchmark interface
  - `loader.py`: Benchmark loading utilities
  - `registry.py`: Benchmark registration system

- **`src/benchmarks/`**: Benchmark implementations
  - `simple_reflex_example.py`: Example traffic light benchmark

- **`src/utils/`**: Utilities
  - `logging.py`: Logging configuration
  - `config.py`: Configuration management

### Adding New Models

```python
from src.models.base import BaseModel, ModelConfig, ModelResponse

class MyCustomModel(BaseModel):
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        # Implement your model's generation logic
        pass
    
    def generate_sync(self, prompt: str, **kwargs) -> ModelResponse:
        # Implement synchronous version
        pass

# Register the model
from src.models.loader import ModelLoader
ModelLoader.register_model("my-model", MyCustomModel)
```

### Adding New Benchmarks

```python
from src.benchmark.base import BaseBenchmark, Task, TaskResult, AgentType
from src.benchmark.registry import benchmark

@benchmark(
    name="my_benchmark",
    agent_type=AgentType.SIMPLE_REFLEX,
    description="My custom benchmark"
)
class MyBenchmark(BaseBenchmark):
    def get_tasks(self) -> List[Task]:
        # Return list of tasks
        pass
    
    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        # Evaluate a single task
        pass
    
    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        # Calculate task score
        pass
```

## Evaluation Results

Results include:
- **Overall Score**: Weighted average across all tasks
- **Success Rate**: Percentage of successfully completed tasks  
- **Detailed Metrics**: Task-specific measurements
- **Response Analysis**: Token usage, latency, etc.

Example output:
```
Evaluation Results
Benchmark: traffic_light_simple
Model: gemini-1.5-flash
Overall Score: 0.95
Success Rate: 100.00%
Tasks Completed: 11
Average Execution Time: 1.23s
```

## 🔧 Configuration

Configuration can be managed through:
- Environment variables
- YAML/JSON config files
- Python configuration objects

Example config file:
```yaml
default_model: "gemini-1.5-pro"
api_keys:
  google: "your-gemini-api-key"
default_benchmark_config:
  collect_detailed_metrics: true
  save_responses: true
  random_seed: 42
log_level: "INFO"
```

## Examples

Run the comprehensive example:
```bash
python example_usage.py
```

This will:
1. Load a Gemini model
2. Show available benchmarks
3. Run the traffic light benchmark
4. Display detailed results

## 📝 Development

### Project Structure
```
AgenticEvals/
├── src/
│   ├── models/          # Model implementations
│   ├── benchmark/       # Benchmark framework  
│   ├── benchmarks/      # Specific benchmarks
│   └── utils/           # Utilities
├── requirements.txt     # Dependencies
├── setup.py            # Package setup
├── example_usage.py    # Full example
├── quick_start.py      # Quick test
└── README.md          # This file
```

### Running Tests
```bash
pytest src/tests/
```

### Code Formatting
```bash
black src/
isort src/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your benchmark or model implementation
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 References

Based on the implementation plan for NeurIPS 2025 Datasets & Benchmarks Track submission. See `plan.md` for detailed project specification.