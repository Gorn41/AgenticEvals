#!/usr/bin/env python3
"""
Example usage of AgenticEvals with Gemini model.

This script demonstrates the full functionality of the benchmark system.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from models.loader import load_gemini, get_available_models
    from benchmark.loader import load_benchmark, get_available_benchmarks
    from utils.config import get_config_manager
    from utils.logging import get_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)

logger = get_logger(__name__)


async def main():
    """Main demonstration function."""
    print("=" * 60)
    print("AgenticEvals Example with Gemini")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: No API key found!")
        print("\nPlease set your Gemini API key:")
        print("1. Run 'python3 setup_env.py' for guided setup")
        print("2. Or set manually: export GOOGLE_API_KEY='your-key-here'")
        return
    
    print(f"SUCCESS: API key found: {api_key[:8]}...")
    print()
    
    try:
        # Load configuration
        config_manager = get_config_manager()
        print("SUCCESS: Configuration loaded successfully")
        
        # Show available models
        print("\nAvailable Models:")
        available_models = get_available_models()
        for model_name in available_models:
            print(f"  - {model_name}")
        
        # Show available benchmarks
        print("\nAvailable Benchmarks:")
        available_benchmarks = get_available_benchmarks()
        for agent_type, benchmark_list in available_benchmarks.items():
            print(f"  {agent_type.value}:")
            for benchmark_name in benchmark_list:
                print(f"    - {benchmark_name}")
        
        # Load model
        print("\nLoading Gemini model...")
        model = load_gemini(
            model_name="gemini-2.5-flash",
            api_key=api_key,
            temperature=0.1  # Low temperature for more consistent results
        )
        print(f"SUCCESS: Model loaded: {model.model_name}")
        
        # Test model with simple prompt
        print("\nTesting model with simple prompt...")
        test_response = await model.generate("What is 2+2? Answer with just the number.")
        print(f"   Model response: '{test_response.text}'")
        if test_response.prompt_tokens:
            print(f"   Tokens: {test_response.prompt_tokens} prompt + {test_response.completion_tokens} completion = {test_response.total_tokens} total")
        
        # Load and run benchmark  
        print("\nLoading traffic light benchmark...")
        benchmark = load_benchmark("traffic_light_simple")
        print(f"SUCCESS: Benchmark loaded: {benchmark.benchmark_name}")
        
        # Show benchmark info
        info = benchmark.get_benchmark_info()
        print(f"   Agent type: {info['agent_type']}")
        print(f"   Description: {info['description']}")
        
        # Show available tasks
        tasks = benchmark.get_tasks()
        print(f"   Available tasks: {len(tasks)}")
        for i, task in enumerate(tasks[:3], 1):  # Show first 3
            print(f"     {i}. {task.name}: {task.description}")
        if len(tasks) > 3:
            print(f"     ... and {len(tasks) - 3} more")
        
        # Run benchmark
        print(f"\nRunning benchmark with {len(tasks)} tasks...")
        print("   This may take a moment...")
        
        result = await benchmark.run_benchmark(model)
        
        # Display results
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        
        print(f"Benchmark: {result.benchmark_name}")
        print(f"Model: {result.model_name}")
        print(f"Agent Type: {result.agent_type.value}")
        print(f"Overall Score: {result.overall_score:.3f}")
        print(f"Success Rate: {result.get_success_rate():.1%}")
        
        # Summary statistics
        stats = result.get_summary_statistics()
        print(f"Tasks Completed: {stats['total_tasks']}")
        print(f"Successful Tasks: {stats['successful_tasks']}")
        print(f"Failed Tasks: {stats['total_tasks'] - stats['successful_tasks']}")
        print(f"Average Execution Time: {stats.get('average_execution_time', 0):.2f}s")
        
        # Show detailed task results
        print("\nDetailed Task Results:")
        print("-" * 60)
        
        for i, task_result in enumerate(result.task_results, 1):
            status = "PASS" if task_result.success else "FAIL"
            print(f"{status} Task {i}: {task_result.task_name}")
            print(f"   Score: {task_result.score:.3f}")
            if task_result.model_response:
                response_text = task_result.model_response.text.strip()
                # Truncate long responses
                if len(response_text) > 50:
                    response_text = response_text[:47] + "..."
                print(f"   Response: '{response_text}'")
            if task_result.execution_time:
                print(f"   Time: {task_result.execution_time:.2f}s")
            if task_result.error_message:
                print(f"   Error: {task_result.error_message}")
            print()
        
        # Summary metrics
        if hasattr(result, 'summary_metrics') and result.summary_metrics:
            print("Summary Metrics:")
            for metric, value in result.summary_metrics.items():
                if isinstance(value, float):
                    print(f"   {metric}: {value:.3f}")
                else:
                    print(f"   {metric}: {value}")
        
        print("\n" + "=" * 60)
        print("SUCCESS: Example completed successfully!")
        print("\nNext steps:")
        print("- Try running other benchmarks")
        print("- Experiment with different model parameters")
        print("- Check out the documentation for creating custom benchmarks")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: Error during execution: {e}")
        logger.exception("Error in example usage")
        print("\nTroubleshooting tips:")
        print("1. Check your API key is valid")
        print("2. Ensure you have internet connectivity")
        print("3. Try running 'python3 setup_env.py' to reconfigure")


if __name__ == "__main__":
    asyncio.run(main()) 