#!/usr/bin/env python3
"""
Quick start script for LLM-AgentTypeEval with minimal setup.
Run this after installing dependencies to test the system.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def quick_demo():
    """Quick demonstration of the system."""
    print("LLM-AgentTypeEval Quick Start")
    print("=" * 40)
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Missing Gemini API Key!")
        print("Please set your Gemini API key:")
        print("export GOOGLE_API_KEY='your-gemini-api-key-here'")
        print("\nGet your key from: https://makersuite.google.com/app/apikey")
        return
    
    try:
        # Import after path setup
        from models.gemini import GeminiModel
        from models.base import ModelConfig
        from benchmarks.simple_reflex_example import TrafficLightBenchmark
        from benchmark.base import BenchmarkConfig, AgentType
        
        print("Imports successful")
        
        # Create model
        print("\nInitializing Gemini model...")
        model_config = ModelConfig(
            model_name="gemini-1.5-flash",
            api_key=api_key,
            temperature=0.1
        )
        model = GeminiModel(model_config)
        print(f"Model initialized: {model.model_name}")
        
        # Create benchmark
        print("\nSetting up traffic light benchmark...")
        benchmark_config = BenchmarkConfig(
            benchmark_name="traffic_light_simple",
            agent_type=AgentType.SIMPLE_REFLEX,
            num_tasks=3  # Run just 3 tasks for quick demo
        )
        benchmark = TrafficLightBenchmark(benchmark_config)
        
        tasks = benchmark.get_tasks()
        print(f"Benchmark ready with {len(tasks)} total tasks (running {benchmark_config.num_tasks})")
        
        # Test a single task first
        print("\nTesting single task...")
        test_task = tasks[0]
        print(f"Task: {test_task.name}")
        print(f"Prompt preview: {test_task.prompt[:100]}...")
        
        # Generate response
        result = await benchmark.evaluate_task(test_task, model)
        
        print(f"\nTask Result:")
        print(f"Success: {'PASS' if result.success else 'FAIL'}")
        print(f"Score: {result.score:.2f}")
        print(f"Response: '{result.model_response.text.strip()}'")
        print(f"Expected: '{test_task.expected_output}'")
        print(f"Time: {result.execution_time:.2f}s")
        
        # Run full benchmark
        print(f"\nRunning full benchmark ({benchmark_config.num_tasks} tasks)...")
        full_result = await benchmark.run_benchmark(model)
        
        print(f"\nBenchmark Complete!")
        print(f"Overall Score: {full_result.overall_score:.2f}")
        print(f"Success Rate: {full_result.get_success_rate():.1%}")
        print(f"Avg Time: {full_result.summary_metrics['average_execution_time']:.2f}s")
        
        print("\nQuick start successful! System is working.")
        print("Next steps:")
        print("- Run 'python example_usage.py' for full demo")
        print("- Check 'src/benchmarks/' for more benchmarks")
        print("- Modify benchmarks or create new ones")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Try running: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error: {e}")
        print("Check your API key and internet connection")

if __name__ == "__main__":
    asyncio.run(quick_demo()) 