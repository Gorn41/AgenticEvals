#!/usr/bin/env python3
"""
Quick start script for AgenticEvals with minimal setup.

This script provides a fast way to test if everything is working.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

print("AgenticEvals Quick Start")
print("=" * 30)

# Check API key
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("ERROR: No Gemini API key found!")
    print("\nPlease set your API key:")
    print("export GOOGLE_API_KEY='your-gemini-api-key-here'")
    print("\nOr run the setup script:")
    print("python3 setup_env.py")
    sys.exit(1)

print(f"SUCCESS: API key found: {api_key[:8]}...")

# Test imports
try:
    from src.models.loader import load_model_from_name
    from src.benchmark.loader import load_benchmark
    print("SUCCESS: Imports successful")
except ImportError as e:
    print(f"ERROR: Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

async def main():
    """Quick test function."""
    try:
        # Load model
        print("Loading model...")
        model = load_model_from_name("gemini-2.5-flash", api_key=api_key, temperature=0.1)
        print(f"SUCCESS: Model loaded: {model.model_name}")
        
        # Test model
        print("Testing model...")
        response = await model.generate("Say 'Hello from AgenticEvals!'")
        print(f"   Model response: {response.text}")
        
        # Load benchmark
        print("Loading benchmark...")
        benchmark = load_benchmark("traffic_light")
        print(f"SUCCESS: Benchmark loaded: {benchmark.benchmark_name}")
        
        # Test one task
        tasks = benchmark.get_tasks()
        if tasks:
            print("Testing one traffic light task...")
            task = tasks[0]  # Get first task
            result = await benchmark.evaluate_task(task, model)
            print(f"   Task: {task.name}")
            print(f"   Success: {result.success}")
            print(f"   Score: {result.score:.3f}")
            if result.model_response:
                print(f"   Response: '{result.model_response.text}'")
        
        print("\nSUCCESS: Quick start successful!")
        print("\nNext steps:")
        print("- Run 'python3 example_usage.py' for a full demo")
        print("- Run 'python3 run_tests.py --unit' to run tests")
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nTroubleshooting:")
        print("- Check your API key is valid")
        print("- Try running 'python3 setup_env.py'")

if __name__ == "__main__":
    asyncio.run(main()) 