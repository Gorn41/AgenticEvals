#!/usr/bin/env python3
"""
Interactive Traffic Light Benchmark Runner

This script provides an interactive way to run the traffic light benchmark.
It first tests a single task, then asks if you want to run the full benchmark.
"""

import asyncio
import os
import sys
from pathlib import Path

def load_env_file():
    """Load environment variables from .env file."""
    print("Loading environment from .env file...")
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# Setup and validation
print("AgenticEvals - Traffic Light Benchmark")
print("=" * 50)

load_env_file()

# Check API key
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: No API key found!")
    print("Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
    print("You can add it to your .env file or export it in your shell")
    sys.exit(1)

print(f"API key found: {api_key[:10]}...")

# Setup Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    from src.benchmark.loader import load_benchmark
    from src.models.loader import load_gemini
    print("Imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

print("\nLoading Gemini model...")
try:
    model = load_gemini("gemini-2.5-flash", api_key=api_key, temperature=0.1)
except Exception as e:
    print(f"Model loading error: {e}")
    sys.exit(1)

print(f"Model loaded: {model.model_name}")

print("\nLoading traffic light benchmark...")
benchmark = load_benchmark("traffic_light_simple")
print(f"Benchmark loaded: {benchmark.benchmark_name}")
print(f"   Agent type: {benchmark.agent_type.value}")
print(f"   Description: {benchmark.get_benchmark_info().get('description', 'No description available')}")

# Get tasks and show info
tasks = benchmark.get_tasks()
print(f"   Total tasks: {len(tasks)}")

# Test single task first
async def test_single_task():
    print(f"\nTesting single task...")
    task = tasks[0]  # Test first task
    print(f"   Task: {task.name}")
    
    # Show prompt preview
    prompt_preview = task.prompt[:200] + "..." if len(task.prompt) > 200 else task.prompt
    print(f"   Prompt preview: {prompt_preview}")
    
    # Run single task
    result = await benchmark.evaluate_task(task, model)
    
    # Show result
    status = "PASS" if result.success else "FAIL"
    print(f"   Result: {status}")
    print(f"   Score: {result.score:.3f}")
    print(f"   Expected: '{task.expected_output}'")
    
    return result

# Ask about full benchmark
async def run_full_benchmark():
    try:
        print(f"\nRun full benchmark with all {len(tasks)} tasks? (y/n): ", end="")
        response = input().lower().strip()
        
        if response in ['y', 'yes']:
            print("\nRunning full benchmark...")
            result = await benchmark.run_benchmark(model)
            show_full_results(result)
        else:
            print("Single task test completed!")
    except KeyboardInterrupt:
        print("\nInterrupted by user")

def show_full_results(result):
    """Display full benchmark results."""
    try:
        print("BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Benchmark: {result.benchmark_name}")
        print(f"Model: {result.model_name}")
        print(f"Agent Type: {result.agent_type.value}")
        print(f"Execution Time: {result.execution_metadata.get('total_execution_time', 0):.2f}s")
        print(f"Overall Score: {result.overall_score:.3f}")
        print(f"Success Rate: {result.get_success_rate():.1%}")
        
        # Task summary
        passed = sum(1 for task_result in result.task_results if task_result.success)
        failed = len(result.task_results) - passed
        print(f"\nTask Summary:")
        print(f"   Passed: {passed}")
        print(f"   Failed: {failed}")
        print(f"   Total: {len(result.task_results)}")
        
        # Individual results
        print(f"\nDetailed Results:")
        for i, task_result in enumerate(result.task_results, 1):
            status = "PASS" if task_result.success else "FAIL"
            response = "None"
            if task_result.model_response:
                response = task_result.model_response.text.strip()
            
            print(f"   Task {i}: {status} - '{response}' (score: {task_result.score:.3f})")
            if task_result.error_message:
                print(f"            Error: {task_result.error_message}")
        
        print("Full benchmark completed!")
        
    except Exception as e:
        print(f"Error displaying results: {e}")

# Main execution
async def main():
    try:
        await test_single_task()
        await run_full_benchmark()
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 