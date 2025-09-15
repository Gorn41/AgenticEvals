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

Note that it is highly recommended to use Linux, Unix or Windows Subsystem for Linux (WSL) to run this benchmark. You may run into some installation issues when using windows. If you do, make sure you have a C++ compiler installed. If issues persist, try installing requirements.txt without vLLM by commenting out vLLM in requirements.txt before running pip install -r requirements.txt, though this will prevent you from being able to run LLMs locally for this benchmark, meaning you will only have to rely on LLM API calls. If using MacOS, please remove vLLM from the requirements.txt file before running pip install -r requirements.txt; again note that this will prevent you from being able to run LLMs locally for this benchmark. Also note that the instructions below assumes you are using bash; the instructions you actually have to run may differ depending on the shell you are using. However, the general steps should be the same, just with differences in language.

```bash
git clone https://github.com/Gorn41/AgenticEvals.git
cd AgenticEvals
python3 -m venv .venv
source .venv/bin/activate
# IMPORTANT: remove vLLM from requirements.txt before running the line below if using MacOS
# or if issues with installation persist.
pip install -r requirements.txt
```

### 2. Environment Setup (API keys and local Selenium MCP)

```bash
python3 setup_env.py
# Follow the prompts to set up your Gemini API key. The script will also set SELENIUM_MCP_URL
# (defaults to ws://127.0.0.1:7007 if you press Enter).
```

You must run a Selenium MCP server at `SELENIUM_MCP_URL` before running `local_web_navigation`.
The benchmark expects MCP tools: `browser.navigate`, `browser.click`, `browser.type`, `browser.submit`,
`browser.clearCookies`, `browser.getDomSummary`.

You can also set API keys and environment variables manually:
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

### 3. Run Evaluations

Note that it is recommend that you close any Chrome/ChromeDriver processes before running the benchmark.

```bash
python3 run.py --model gemma-3-27b-it
```

## Advanced Usage

### Plotting and Saving Results

To generate plots and save detailed results to CSV files, use the `--plot` flag. This will create a `results/<model_name>/` directory containing all output files. The plots include error bars that represent the standard deviation of metrics (score, time, and tokens) across the tasks within each benchmark, providing insight into the model's performance consistency.

```bash
python3 run.py --model gemma-3-27b-it --plot
```

This will produce the following files inside the `results/gemma-3-27b-it/` directory:

- `benchmark_performance_gemma-3-27b-it.png`: A plot showing performance metrics for each benchmark, with error bars for standard deviation.
- `agent_type_performance_gemma-3-27b-it.png`: A plot showing aggregated performance metrics for each agent type, with error bars.
- `benchmark_results_gemma-3-27b-it.csv`: A detailed breakdown of metrics for each task.
- `agent_type_results_gemma-3-27b-it.csv`: Aggregated metrics for each agent type, including standard deviation.

### Running Specific Tasks

You can run one or more specific tasks by providing their names as arguments. Use names of tasks under each agent type in Benchmark Types section below. The order of arguments is flexible.

```bash
python3 run.py --model gemma-3-27b-it inventory_management email_autoresponder fraud_detection --plot
```

### Verbose Output

For more detailed diagnostic information from each task, use the `--verbose` flag:

```bash
python3 run.py --model gemma-3-27b-it --verbose
```

### Local Inference via vLLM (Open Source Models)

You can run supported open-source models locally using vLLM instead of remote APIs. Install vLLM (added to requirements.txt) and run:

```bash
# Example using Gemma-3 weights from Hugging Face
python3 run.py --local --model google/gemma-3-4b-it --wait-seconds 0
```

You can also provide a YAML config to control vLLM engine parameters and sampling settings. A default `vllm_config.yaml` is included at the repo root; edit it as needed:

```yaml
# vllm_config.yaml
temperature: 0.3
max_tokens: 32768
additional_params:
  hf_model: google/gemma-3-4b-it  # optional if provided via --model
  dtype: bfloat16
  tensor_parallel_size: 1
  trust_remote_code: true
  gpu_memory_utilization: 0.9
```

Then run:

```bash
python3 run.py --local --model google/gemma-3-4b-it --vllm-config vllm_config.yaml
```

Notes:
- Proprietary endpoints (e.g., `gemini*`) cannot be loaded via vLLM and will raise an error.
- vLLM loads models from Hugging Face or local paths specified via `additional_params.hf_model`.
- Some HF models require authentication; set `HUGGING_FACE_HUB_TOKEN` in `.env` if needed. Use `python3 setup_env.py` to add it interactively.

### Validation and Cross-Validation

Run with built-in cross-validation:

```bash
python3 run.py --model gemma-3-4b-it --validate --wait-seconds 0
```

Provide a validation config listing external validation benchmarks to execute (names must be registered via the validation registry):

```yaml
# validation_benchmarks.yaml
validation_benchmarks:
  - name: ev_charging_policy
```

Run with config:

```bash
python3 run.py --model gemma-3-4b-it --validate --validation-config validation_benchmarks.yaml
```

This runs only the listed validation benchmarks (e.g., `ev_charging_policy`) when used with `--validation-only`. Predictions are computed from current agent-type aggregates.

### Validation-Only Mode (run only validation benchmarks)

Use `--validation-only` to execute only validation benchmarks listed in `--validation-config` and print prediction proxies.

```bash
python3 run.py --model gemma-3-4b-it \
  --validate \
  --validation-config validation_benchmarks.yaml \
  --validation-only
```


### Running a Subset with Validation

You can run only certain benchmarks alongside validation by passing their names before the flags:

```bash
python3 run.py --model gemma-3-4b-it fraud_detection inventory_management \
  --validate --validation-config validation_benchmarks.yaml
```

### Configure Wait Times Between Calls/Tasks

Multi-step benchmarks wait between API calls to respect rate limits, and the runner waits between tasks. You can configure this with `--wait-seconds` (default 15.0). Set to `0` to disable waits.

```bash
python3 run.py --model gemma-3-27b-it --wait-seconds 0
```

This value is propagated to all benchmarks and used for any internal delays.

### CLI Flags Quick Reference

- `--model <name>`: model identifier (e.g., `gemma-3-4b-it` or HF id with `--local`).
- `--local`: run locally via vLLM (open-source models only).
- `--vllm-config <path>`: YAML with vLLM engine/sampling params.
- `--wait-seconds <float>`: delay between scenarios/turns (default 15.0; set 0 to disable).
- `--plot`: generate plots and CSVs under `results/<model>/`.
- `--verbose`: print additional per-task diagnostics.
- `--validate`: enable validation/cross-validation reporting.
- `--validation-config <path>`: YAML listing validation benchmarks and their mapped agent types.
- `--validation-only`: skip running benchmarks and compute validation proxies from existing results.

## Project Structure

```
AgenticEvals/
├── ENVIRONMENT_SETUP.md
├── LICENSE
├── README.md
├── requirements.txt
├── pytest.ini
├── setup.py
├── setup_env.py                # Main env setup script
├── run.py                      # Main entrypoint to run evaluations
├── run_tests.py                # Helper to run unit tests
├── quick_start.py              # Minimal example runner
├── example_usage.py            # Example: loading a model and running a benchmark
├── plot_agent_type_rankings.py # Plotting utility script
├── plot_agent_type_results.py  # Plotting utility script
├── plot_benchmark_results.py   # Plotting utility script
├── MMAU_setup.py               # Setup for MMAU validation
├── run_MMAU_validation.sh      # Convenience script to run MMAU validation 
├── vllm_config.yaml            # vLLM configuration file
├── validation_benchmarks.yaml  # validation benchmark configuration file
├── results/                    # Generated results, plots, CSVs
│   └── <model_name>/           # E.g., gemma-3-27b-it, contains CSVs and plots
├── src/
│   ├── __init__.py
│   ├── benchmark/              # Core benchmark framework
│   │   ├── __init__.py
│   │   ├── base.py             # BaseBenchmark, Task, TaskResult, AgentType
│   │   ├── loader.py           # load_benchmark utilities
│   │   └── registry.py         # @benchmark decorator and registry
│   ├── benchmarks/             # Individual benchmark implementations
│   │   ├── __init__.py
│   │   ├── ball_drop.py
│   │   ├── ecosystem.py
│   │   ├── event_conflict_detection.py
│   │   ├── hotel_booking.py
│   │   ├── inventory_management.py
│   │   ├── local_web_app.py
│   │   ├── local_web_navigation.py
│   │   ├── manufacturing_line_optimization.py
│   │   ├── textual_maze_navigation.py
│   │   ├── shortest_path_planning.py
│   │   ├── portfolio_optimization.py
│   │   ├── selenium_mcp_server.py
│   │   ├── email_autoresponder.py
│   │   ├── traffic_light.py
│   │   ├── fraud_detection.py
│   │   ├── simulated_market.py
│   │   └── task_scheduling.py
│   ├── models/                 # Model interfaces and providers
│   │   ├── __init__.py
│   │   ├── base.py             # BaseModel framework
│   │   ├── gemini.py           # Gemini/Gemma model bindings
│   │   └── loader.py           # model loading utilities
│   └── utils/                  # Shared utilities
│       ├── __init__.py
│       ├── config.py           # Config handling (env, defaults)
│       └── logging.py          # Structured logging utilities
├── tests/                      # Unit test suite
│   ├── __init__.py
│   ├── README.md
│   ├── conftest.py
│   ├── test_benchmark/
│   │   ├── __init__.py
│   │   ├── test_base.py
│   │   └── test_loader.py
│   ├── test_benchmarks/
│   │   ├── __init__.py
│   │   └── test_simple_reflex_example.py
│   ├── test_models/
│   │   ├── __init__.py
│   │   ├── test_base.py
│   │   └── test_gemini.py
│   └── test_utils/
│       ├── __init__.py
│       ├── test_config.py
│       └── test_logging.py
└── TESTING.md                # Testing guide and conventions
```

## Supported Models

- **Google Gemini**: Support for various Gemini models.
- **Gemma**: Open-source models from Google.

The framework is designed to be easily extendable with new model providers.

## Benchmark Types

### Simple Reflex Agent
The key characteristics of a simple Reflex Agents/Tasks include being able to respond to the current state (no internal model or memory of past states nor prediction of future states), not requiring any learning, and emphasizing reaction speed.

- **traffic_light**: Basic stimulus-response scenarios. This is a basic reaction task emphasizing response speed.
- **email_autoresponder**: Email processing tasks. This task emphasizes the ability to react to signals amongst distractors within natural language.
- **fraud_detection**: Binary fraud vs. legitimate classification from multi-line, log-like scenarios. This task emphasizes the ability to react to signals amongst distractors within structured data.

### Model-Based Reflex Agent
The key characteristics of model-based reflex agents/tasks include maintaining an adaptable internal model of the environment and the ability to use this internal model to make informed decisions. This requires the agent to have good context memory and recall.

- **textual_maze_navigation**: Navigation in a partially observable environment. This task emphasizes the ability to build an internal model of the environment’s state space from partial observations and context memory and recall.
- **inventory_management**: Inventory tracking and restocking. This task emphasizes the ability to build an internal model of the environment’s transition function from partial observations.
- **event_conflict_detection**: Multi-turn distributed-systems incident tagging. This task emphasizes the ability to iteratively improve hypotheses of the environment model through extended reasoning.

### Goal-Based Agent
The key characteristics of goal-based agents/tasks include decision-making based on what actions would help the agent better achieve a goal, the ability to effectively plan for future actions and/or adjust plans.

- **hotel_booking**: Multi-step planning and booking. This task emphasizes planning comprehensiveness and efficiency.
- **shortest_path_planning**: Find the shortest path in a directed, weighted graph. This task emphasises the ability to do multi-goal long-horizon planning.
- **local_web_navigation**: Structured meta-planning benchmark on a deterministic local site. This task emphasizes the ability to perform meta-planning because in order to do well, the agent must plan during the exploration phase to optimize planning in the exploitation phase.

### Utility-Based Agent
The key characteristics of utility-based agents/tasks include decision-making based on outcome optimization where outcome is determined by various factors (i.e. optimize utility where the utility is a function of factors such as costs, risk, benefits, etc.).

- **task_scheduling**: Complex task scheduling with constraints. This task emphasizes the ability to perform constrained utility maximization.
- **portfolio_optimization**: Task on allocating capital to maximize profit based on a news forecast. This task emphasizes the ability to perform utility maximization from information inferred from natural language.
- **manufacturing_line_optimization**: Manufacturing parameters optimization. This task emphasizes the ability to perform multi-objective Pareto optimization.

### Learning Agent
The key characteristics of learning agents/tasks include continuous learning from data from the environment to improve decision-making, being able to learn from past data to generalize and make decisions on unseen data, and being flexible and able to adapt to and learn from data from a wide variety of tasks or environment dynamics.

- **ball_drop**: Physics-based prediction task. TThis task emphasizes the ability of an agent to learn and generalize to make decisions on unseen data from fixed but unknown environmental dynamics as well as learning involving in-context memory.
- **simulated_market**: A trading agent that learns to adapt its strategy in a simulated market using Retrieval-Augmented Generation (RAG). This task emphasizes agent flexibility and its ability to learn from a wide range of environmental dynamics as well as learning involving retrieval augmented generation (RAG).
- **ecosystem**: Knowledge-graph-based ecosystem dynamics learning. This task emphasizes agent ability to learn using a knowledge graph.

## Validation Tasks (not included in main agent-type aggregates)

- **ev_charging_policy**: EV Charging Policy Optimization (Multi-Turn). A 10-turn benchmark where the model builds an internal model of consistent EV arrival patterns and selects discrete policy parameters to maximize a weighted utility. Associated agent types for validation mapping: `model_based_reflex`, `utility_based`.

- **MMAU Benchmark Tasks**: Tasks from the MMAU benchmark (https://arxiv.org/abs/2407.18961). Covers a wide range of agent types. See below for special instructions on running MMAU Benchmark Tasks for validation. It is highly recommended that you do the following in a new workspace, i.e. in a separate directory to the root of the AgenticEvals root. This is because the process requires creating a new virtual environment and .env files.

```bash
# make sure you are working from a new empty directory.
mkdir MMAU_validation
cd MMAU_validation
# clone the modified MMAU benchmartk
git clone https://github.com/Gorn41/axlearn.git
# deactivate any virtual environment that might currently be active and create 
# a new one specifically for MMAU benchmark tasks
deactivate
python3 -m venv MMAUvenv
source MMAUvenv/bin/activate
# install the dependencies for running the MMAU benchmark
cd axlearn
pip install ".[mmau]"
cd ..
# Copy MMAU setup and runner scripts from the AgenticEvals root into the 
# new MMAU_validation directory
cp PATH_TO_AgenticEvals/MMAU_setup.py .
cp PATH_TO_AgenticEvals/run_MMAU_validation.sh .
# Run the setup script associated with MMAU (note that you will need 
# Google Application Credentials, Vertex AI details and an OpenAI key for this)
python3 MMAU_setup.py
# To easily run all MMAU validation tasks on a specific model do the following:
chmod +x run_MMAU_validation.sh
./run_MMAU_validation.sh --model <name>
# E.g. ./run_MMAU_validation.sh --model gemini-2.5-flash
# Make sure you increase API usage limit otherwise you may be rate limited
# Results can be found in the outputs/ folder
```

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
    benchmark = load_benchmark("email_autoresponder")

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
