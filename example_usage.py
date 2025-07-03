#!/usr/bin/env python3
"""
Example usage of LLM-AgentTypeEval with Gemini model.

This script demonstrates how to:
1. Load and configure a Gemini model
2. Load a benchmark
3. Run evaluation
4. Display results
"""

import asyncio
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from models.loader import load_gemini
from benchmark.loader import load_benchmark, get_available_benchmarks
from utils.logging import setup_logging, get_logger
from utils.config import get_config_manager

# Set up logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)


async def main():
    """Main example function."""
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        print("You can get a Gemini API key from: https://makersuite.google.com/app/apikey")
        return
    
    print("LLM-AgentTypeEval Example with Gemini")
    print("=" * 50)
    
    try:
        # 1. Load Gemini model
        print("\nLoading Gemini model...")
        model = load_gemini(
            model_name="gemini-1.5-flash",  # Use flash for faster responses
            temperature=0.1,  # Low temperature for consistent responses
            api_key=api_key
        )
        print(f"Model loaded: {model.model_name}")
        
        # 2. Show available benchmarks
        print("\nAvailable benchmarks:")
        available = get_available_benchmarks()
        for agent_type, benchmarks in available.items():
            print(f"  {agent_type.value}: {benchmarks}")
        
        # 3. Load a specific benchmark
        print("\nLoading traffic light benchmark...")
        benchmark = load_benchmark("traffic_light_simple")
        print(f"Benchmark loaded: {benchmark.benchmark_name}")
        
        # 4. Show benchmark info
        info = benchmark.get_benchmark_info()
        print(f"Agent type: {info['agent_type']}")
        print(f"Description: {info['description']}")
        
        # 5. Get tasks preview
        tasks = benchmark.get_tasks()
        print(f"Total tasks: {len(tasks)}")
        print(f"Sample task: {tasks[0].name}")
        
        # 6. Run the benchmark
        print("\nRunning benchmark evaluation...")
        print("This may take a few moments...")
        
        result = await benchmark.run_benchmark(model)
        
        # 7. Display results
        print("\nEvaluation Results")
        print("=" * 30)
        print(f"Benchmark: {result.benchmark_name}")
        print(f"Model: {result.model_name}")
        print(f"Agent Type: {result.agent_type.value}")
        print(f"Timestamp: {result.timestamp}")
        print()
        print(f"Overall Score: {result.overall_score:.2f}")
        print(f"Success Rate: {result.get_success_rate():.2%}")
        print(f"Tasks Completed: {len(result.task_results)}")
        print(f"Tasks Successful: {result.summary_metrics['num_tasks_successful']}")
        print(f"Tasks Failed: {result.summary_metrics['num_tasks_failed']}")
        print(f"Average Execution Time: {result.summary_metrics['average_execution_time']:.2f}s")
        
        # 8. Show detailed task results
        print("\nDetailed Task Results")
        print("-" * 40)
        for task_result in result.task_results[:5]:  # Show first 5 tasks
            status = "PASS" if task_result.success else "FAIL"
            print(f"[{status}] {task_result.task_name}: {task_result.score:.2f}")
            if task_result.model_response:
                response_preview = task_result.model_response.text[:50]
                if len(task_result.model_response.text) > 50:
                    response_preview += "..."
                print(f"Response: '{response_preview}'")
            print(f"Execution time: {task_result.execution_time:.2f}s")
            print()
        
        if len(result.task_results) > 5:
            print(f"... and {len(result.task_results) - 5} more tasks")
        
        # 9. Show metrics summary
        print("\nMetrics Summary")
        print("-" * 20)
        for metric_name in ["word_count", "follows_instructions", "exact_match"]:
            summary = result.get_metric_summary(metric_name)
            if summary:
                print(f"{metric_name}: mean={summary.get('mean', 0):.2f}, "
                      f"min={summary.get('min', 0):.2f}, "
                      f"max={summary.get('max', 0):.2f}")
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 