#!/usr/bin/env python3
"""
Simple Traffic Light Benchmark Test Script

This script runs the traffic light benchmark automatically without user interaction.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.benchmark.loader import load_benchmark
from src.models.loader import load_gemini

async def main():
    """Run the traffic light benchmark automatically."""
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: No API key found! Set GOOGLE_API_KEY environment variable.")
            return
        
        # Load model
        model = load_gemini("gemini-2.5-flash", api_key=api_key, temperature=0.1)
        
        # Load benchmark
        print("Running Traffic Light Benchmark")
        print("=" * 40)
        benchmark = load_benchmark("traffic_light_simple")
        
        print(f"Model: {model.model_name}")
        print(f"Tasks: {len(benchmark.get_tasks())}")
        
        # Run benchmark
        print("\nRunning benchmark...")
        result = await benchmark.run_benchmark(model)
        
        # Display results
        print(f"\nRESULTS:")
        print(f"Overall Score: {result.overall_score:.3f}")
        print(f"Success Rate: {result.get_success_rate() * 100:.1f}%")
        print(f"Tasks Passed: {sum(1 for r in result.task_results if r.success)}/{len(result.task_results)}")
        
        print(f"\nTask Details:")
        for i, task_result in enumerate(result.task_results, 1):
            status = "PASS" if task_result.success else "FAIL"
            response = task_result.model_response.text.strip() if task_result.model_response else "None"
            print(f"{status} Task {i}: '{response}' (score: {task_result.score:.3f})")
        
        print("\nBenchmark completed!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 