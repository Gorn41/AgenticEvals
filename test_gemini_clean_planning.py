#!/usr/bin/env python3
"""
Test Gemma or Gemini on the clean hotel booking benchmark (pure search space planning).
"""

import sys
import os
import asyncio
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    print("Falling back to system environment variables...")

from src.benchmarks.hotel_booking import HotelBookingBenchmark
from src.models.gemini import GeminiModel
from src.models.base import ModelConfig
from src.benchmark.base import BenchmarkConfig, AgentType

async def test_gemini_planning():
    """Test Gemini/Gemma on pure search space planning."""
    print("=== Testing Gemini/Gemma on Hotel Booking (Search Space Planning) ===")
    
    # Create benchmark
    config = BenchmarkConfig(
        benchmark_name="hotel_booking_goal_based",
        agent_type=AgentType.GOAL_BASED
    )
    benchmark = HotelBookingBenchmark(config)
    
    # Initialize Gemini/Gemma model
    gemini_config = ModelConfig(
        model_name="gemma-3-4b-it",  
        api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        temperature=0.1,  # Low temperature for consistent planning
        max_tokens=15000   # Enough for search steps but prevents verbose explanations
    )
    gemini = GeminiModel(gemini_config)
    
    print(f"Using model: {gemini_config.model_name}")
    print(f"Max tokens: {gemini_config.max_tokens}")
    
    # Run benchmark with delays to avoid rate limits
    print("\n Running planning benchmark on Gemini/Gemma with 15s delays...")
    
    # Get tasks
    tasks = benchmark.get_tasks()
    task_results = []
    
    import time
    
    print(f"Total tasks: {len(tasks)}")
    
    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] Running: {task.name}")
        
        try:
            # Run individual task
            task_result = await benchmark.evaluate_task(task, gemini)
            task_results.append(task_result)
            print(f"  Score: {task_result.score:.3f}")
            
            # Add delay between tasks (except for the last task)
            if i < len(tasks):
                print(f"  Waiting 15 seconds before next test...")
                await asyncio.sleep(15)
                
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            # Create a failed task result
            from src.benchmark.base import TaskResult
            error_result = TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                agent_type=benchmark.agent_type,
                success=False,
                score=0.0,
                metrics={"error": str(e)},
                execution_time=0.0
            )
            task_results.append(error_result)
            
            # Still add delay after errors to avoid overwhelming the API
            if i < len(tasks):
                print(f"  Waiting 15 seconds before next test...")
                await asyncio.sleep(15)
    
    # Calculate overall metrics manually
    overall_score = sum(result.score for result in task_results) / len(task_results) if task_results else 0.0
    
    # Create benchmark result manually
    from src.benchmark.base import BenchmarkResult
    result = BenchmarkResult(
        benchmark_name=benchmark.benchmark_name,
        agent_type=benchmark.agent_type,
        model_name=gemini.model_name,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        task_results=task_results,
        overall_score=overall_score,
        summary_metrics={
            "total_execution_time": 0.0,  # We're not tracking this precisely with delays
            "num_tasks": len(task_results),
        }
    )
    
    # Check for truncated/blocked responses
    blocked_count = 0
    truncated_count = 0
    error_count = 0
    for task_result in result.task_results:
        if task_result.model_response:
            if task_result.model_response.metadata.get("blocked", False):
                blocked_count += 1
            elif task_result.model_response.metadata.get("truncated", False):
                truncated_count += 1
            elif "RESPONSE_TRUNCATED" in task_result.model_response.text:
                truncated_count += 1
            elif "RESPONSE_BLOCKED" in task_result.model_response.text:
                blocked_count += 1
        # Only count actual errors (API failures, etc.), not low scores
        if task_result.metrics.get("error"):
            error_count += 1
    
    if blocked_count > 0:
        print(f"WARNING: {blocked_count} responses were blocked by safety filters")
        print("   Consider using a different model or adjusting safety settings")
    
    if truncated_count > 0:
        print(f"WARNING: {truncated_count} responses were truncated (hit max_tokens limit)")
        print("   Consider increasing max_tokens parameter")
    
    if error_count > 0:
        print(f"WARNING: {error_count} tasks failed with errors")
    
    # Print overall results - focus on scores
    print(f"\nOVERALL RESULTS:")
    print(f"Average Score: {result.overall_score:.3f}")
    print(f"Total Tasks: {len(result.task_results)}")
    print(f"Blocked Responses: {blocked_count}/{len(result.task_results)}")
    print(f"Truncated Responses: {truncated_count}/{len(result.task_results)}")
    print(f"Error Count: {error_count}/{len(result.task_results)}")
    
    # Print detailed task results
    print(f"\n DETAILED PLANNING RESULTS:")
    print("=" * 80)
    
    for i, task_result in enumerate(result.task_results, 1):
        print(f"\n{i}. {task_result.task_name}")
        print(f"   Score: {task_result.score:.3f}")
        
        # Show planning breakdown
        metrics = task_result.metrics
        if "coverage" in metrics:
            print(f"   Planning Quality: {metrics.get('planning_quality', 'unknown')}")
            print(f"   Coverage: {metrics['coverage']:.3f} (fraction of required combinations found)")
            print(f"   Efficiency: {metrics['efficiency']:.3f} (search count optimality)")
            print(f"   Expected searches: {metrics['expected_searches']}")
            print(f"   Actual searches: {metrics['actual_searches']}")
            
            # Show constraint inference success
            if metrics.get('constraint_inference_success', False):
                print(f"   Successfully inferred constraints from natural language")
            else:
                print(f"   Failed to infer constraints from natural language")
        
        # Show errors if any
        if task_result.metrics.get("error"):
            print(f"   ERROR: {task_result.metrics['error']}")
        
        # Show model's search plan
        elif task_result.model_response and task_result.model_response.text:
            print(f"   Model Response:")
            
            # Extract just the search plan part
            response_text = task_result.model_response.text
            if "SEARCH PLAN:" in response_text:
                plan_part = response_text.split("SEARCH PLAN:")[1].strip()
                # Show first few lines of the plan
                plan_lines = plan_part.split('\n')[:6]  # First 6 lines
                for line in plan_lines:
                    if line.strip():
                        print(f"     {line.strip()}")
                if len(plan_part.split('\n')) > 6:
                    print(f"     ... (truncated)")
            else:
                # Show first 150 characters if no clear plan
                excerpt = response_text[:150] + "..." if len(response_text) > 150 else response_text
                print(f"     {excerpt}")
        
        print("-" * 80)
    
    # Summary analysis
    print(f"\nPLANNING ANALYSIS:")
    
    # Calculate planning quality distribution
    quality_counts = {"perfect": 0, "good": 0, "fair": 0, "poor": 0}
    coverage_scores = []
    efficiency_scores = []
    actual_scores = []  # Track all scores including 0.0
    
    for task_result in result.task_results:
        actual_scores.append(task_result.score)
        metrics = task_result.metrics
        if "planning_quality" in metrics:
            quality = metrics["planning_quality"]
            quality_counts[quality] += 1
        
        if "coverage" in metrics:
            coverage_scores.append(metrics["coverage"])
        if "efficiency" in metrics:
            efficiency_scores.append(metrics["efficiency"])
    
    print(f"Planning Quality Distribution:")
    for quality, count in quality_counts.items():
        percentage = (count / len(result.task_results)) * 100
        print(f"  {quality.capitalize()}: {count}/{len(result.task_results)} ({percentage:.1f}%)")
    
    # Score distribution
    high_scores = sum(1 for s in actual_scores if s >= 0.8)
    medium_scores = sum(1 for s in actual_scores if 0.3 <= s < 0.8)
    low_scores = sum(1 for s in actual_scores if 0.0 < s < 0.3)
    zero_scores = sum(1 for s in actual_scores if s == 0.0)
    
    print(f"\nScore Distribution:")
    print(f"  High (0.8-1.0): {high_scores}/{len(actual_scores)} ({high_scores/len(actual_scores)*100:.1f}%)")
    print(f"  Medium (0.3-0.8): {medium_scores}/{len(actual_scores)} ({medium_scores/len(actual_scores)*100:.1f}%)")
    print(f"  Low (0.0-0.3): {low_scores}/{len(actual_scores)} ({low_scores/len(actual_scores)*100:.1f}%)")
    print(f"  Zero (0.0): {zero_scores}/{len(actual_scores)} ({zero_scores/len(actual_scores)*100:.1f}%)")
    
    if coverage_scores:
        avg_coverage = sum(coverage_scores) / len(coverage_scores)
        print(f"\nAverage Coverage: {avg_coverage:.3f}")
        print(f"  (How well Gemma/Gemini infers required search combinations)")
    
    if efficiency_scores:
        avg_efficiency = sum(efficiency_scores) / len(efficiency_scores)
        print(f"Average Efficiency: {avg_efficiency:.3f}")
        print(f"  (How close Gemma/Gemini gets to optimal search count)")
    
    # Constraint inference success rate
    inference_successes = sum(1 for r in result.task_results 
                             if r.metrics.get('constraint_inference_success', False))
    inference_rate = inference_successes / len(result.task_results)
    print(f"\nConstraint Inference Success Rate: {inference_rate:.3f}")
    print(f"  (How often Gemma/Gemini successfully infers constraints from stories)")
    
    return result

if __name__ == "__main__":
    try:
        # Check for API key from .env file
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("No Gemini API key found in environment!")
            print("\nPlease set your API key:")
            print("Option 1: Set environment variable:")
            print("  export GOOGLE_API_KEY=your_actual_api_key")
            print("  export GEMINI_API_KEY=your_actual_api_key")
            print("Option 2: Create a .env file with:")
            print("  GOOGLE_API_KEY=your_actual_api_key")
            print("  GEMINI_API_KEY=your_actual_api_key")
            print("\nTo get a Gemini API key:")
            print(" Go to https://ai.google.dev/")
            print(" Create a new API key")
            sys.exit(1)
            
        print(" Found Gemini API key, proceeding with test...")
        result = asyncio.run(test_gemini_planning())
        
        print(f"\n Test completed successfully!")
        print(f" Gemma/Gemini achieved {result.overall_score:.3f} average score on search space planning")
        
    except Exception as e:
        print(f"\n Test failed: {e}")
        import traceback
        traceback.print_exc() 