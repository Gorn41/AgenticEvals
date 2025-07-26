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

### 3. Run Evaluations

```bash
python3 run.py --model gemma-3-27b-it
```

## Advanced Usage

### Plotting and Saving Results

To generate plots and save detailed results to CSV files, use the `--plot` flag. This will create a `results/<model_name>/` directory containing all output files.

```bash
python3 run.py --model gemma-3-27b-it --plot
```

This will produce the following files inside the `results/gemma-3-27b-it/` directory:

- `benchmark_performance_gemma-3-27b-it.png`: A plot showing performance metrics for each benchmark.
- `agent_type_performance_gemma-3-27b-it.png`: A plot showing aggregated performance metrics for each agent type.
- `benchmark_results_gemma-3-27b-it.csv`: A detailed breakdown of metrics for each task.
- `agent_type_results_gemma-3-27b-it.csv`: Aggregated metrics for each agent type.

### Running Specific Benchmarks

You can run one or more specific benchmarks by providing their names as arguments. The order of arguments is flexible.

```bash
python3 run.py --model gemma-3-27b-it inventory_management simple_reflex_email --plot
```

### Verbose Output

For more detailed diagnostic information from each task, use the `--verbose` flag:

```bash
python3 run.py --model gemma-3-27b-it --verbose
```

## Project Structure

```
AgenticEvals/
├── src/
│   ├── benchmark/        # Core benchmark framework (base classes, loader, registry)
│   ├── benchmarks/       # Implementations of specific benchmarks
│   ├── models/           # Support for different language models
│   └── utils/            # Utility functions (logging, config)
├── tests/                # Test suite for the project
├── run.py          # Main script for running evaluations
└── requirements.txt      # Project dependencies
```

## Supported Models

- **Google Gemini**: Support for various Gemini models.
- **Gemma**: Open-source models from Google.

The framework is designed to be easily extendable with new model providers.

## Benchmark Types

### Simple Reflex Agent
- **simple_reflex_example**: Basic stimulus-response scenarios.
- **simple_reflex_email**: Email processing tasks.

### Model-Based Reflex Agent
- **model_based_maze**: Navigation in a partially observable environment.
- **inventory_management**: Inventory tracking and restocking.

### Goal-Based Agent
- **hotel_booking**: Multi-step planning and booking.
- **pathfinding**: Find the shortest path in a directed, weighted graph.

### Utility-Based Agent
- **task_scheduling**: Complex task scheduling with constraints.

### Learning Agent
- **ball_drop**: Physics-based prediction task.

## Development

### Adding New Models

1.  Create a new model class that inherits from `BaseModel`.
2.  Implement the required methods for model interaction.
3.  Update the model loader in `src/models/loader.py` to include the new model.

### Adding New Benchmarks

1.  Create a new benchmark file in the `src/benchmarks/` directory.
2.  Define a new benchmark class that inherits from `BaseBenchmark`.
3.  Use the `@benchmark` decorator to register the new benchmark.
4.  Implement the `get_tasks`, `evaluate_task`, and other methods from `BaseBenchmark`.

Example:
```python
from src.benchmark.base import BaseBenchmark, Task, TaskResult
from src.benchmark.registry import benchmark
from src.models.base import BaseModel

@benchmark(name="my_new_benchmark", agent_type=AgentType.SIMPLE_REFLEX)
class MyNewBenchmark(BaseBenchmark):
    def get_tasks(self) -> List[Task]:
        # Return a list of Task objects
        pass

    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        # Evaluate a single task and return a TaskResult
        pass

    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        # Calculate the score for a single task
        pass
```

### Creating Custom Evaluation Scripts

You can create new evaluation scripts to run different combinations of benchmarks or models. The `run.py` script serves as a good starting point.

Here is a basic example of how to structure a new evaluation script:

```python
import asyncio
from src.models.loader import load_model_from_name
from src.benchmark.loader import load_benchmark

async def main():
    # Load your desired model
    model = load_model_from_name("gemma-3-4b-it")

    # Load the benchmark you want to run
    benchmark = load_benchmark("simple_reflex_email")

    # Run the benchmark and get the results
    results = await benchmark.run_benchmark(model)

    # Print the results
    print(f"Overall Score: {results.overall_score}")
    print(f"Success Rate: {results.get_success_rate()}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new feature branch.
3.  Add your changes and include tests.
4.  Ensure all tests pass by running `pytest`.
5.  Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
