#!/usr/bin/env python3
"""
Comprehensive test of Gemma on all available benchmarks.
"""

import os
import sys
import asyncio
import time
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add src to path for imports
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.benchmark.loader import load_benchmark, get_available_benchmarks
from src.models.loader import load_model_from_name
from src.benchmark.base import TaskResult, BaseBenchmark

async def test_benchmark(model, benchmark_name: str) -> Dict[str, Any]:
    """Test a model on a given benchmark."""
    print(f"\n Running Benchmark: {benchmark_name}")
    print("=" * 60)
    
    # Load benchmark
    try:
        benchmark = load_benchmark(benchmark_name)
    except ValueError as e:
        print(f"   [ERROR] Could not load benchmark: {e}")
        return {
            'benchmark_name': benchmark_name,
            'error': str(e)
        }
        
    tasks = benchmark.get_tasks()
    
    print(f"Agent Type: {benchmark.agent_type.value}")
    print(f"Total Tasks: {len(tasks)}")
    
    results = []
    total_start_time = time.time()
    
    for i, task in enumerate(tasks):
        print(f"\n Task {i+1}/{len(tasks)}: {task.name}")
        
        try:
            # Run the task
            result = await benchmark.evaluate_task(task, model)
            results.append(result)
            
            # Show results
            status = "[PASS]" if result.success else "[FAIL]"
            print(f"   Result: {status} (Score: {result.score:.3f})")
            
            if result.model_response and result.model_response.text:
                response_text = result.model_response.text.strip()
                print(f"   Model Response: {response_text[:100]}..." if len(response_text) > 100 else f"   Model Response: {response_text}")
            
            if result.metrics:
                print(f"   Output Tokens: {result.metrics.get('output_tokens', 'N/A')}")
            
            if benchmark_name == "inventory_management" and "turn_scores" in result.metrics:
                turn_scores_str = ", ".join([f"{s:.2f}" for s in result.metrics['turn_scores']])
                print(f"   Turn Scores: [{turn_scores_str}]")

            if result.execution_time is not None:
                print(f"   Execution Time: {result.execution_time:.2f}s")
            
        except Exception as e:
            print(f"   [ERROR] {e}")
            # Create a failed result for consistency
            result = TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                agent_type=benchmark.agent_type,
                success=False,
                score=0.0,
                metrics={},
                execution_time=0.0,
                error_message=str(e)
            )
            results.append(result)
        
        # Delay between tasks
        if i < len(tasks) - 1:
            print(f"   ...waiting 15s before next task...")
            await asyncio.sleep(15)
    
    total_time = time.time() - total_start_time
    
    # Calculate summary statistics
    successful_tasks = [r for r in results if r.success]
    avg_score = sum(r.score for r in results) / len(results) if results else 0.0
    avg_time = sum(r.execution_time for r in results if r.execution_time is not None) / len(results) if results else 0.0
    avg_output_tokens = sum(r.metrics.get('output_tokens') or 0 for r in results if r.metrics) / len(results) if results else 0.0
    
    summary = {
        'benchmark_name': benchmark_name,
        'total_tasks': len(tasks),
        'successful_tasks': len(successful_tasks),
        'success_rate': len(successful_tasks) / len(tasks) if tasks else 0.0,
        'average_score': avg_score,
        'average_time': avg_time,
        'average_output_tokens': avg_output_tokens,
        'total_time': total_time,
        'results': results
    }
    
    print(f"\n {benchmark_name.upper()} BENCHMARK SUMMARY:")
    print(f"   Success Rate: {summary['success_rate']:.1%} ({summary['successful_tasks']}/{summary['total_tasks']})")
    print(f"   Average Score: {summary['average_score']:.3f}")
    print(f"   Average Time per Task: {summary['average_time']:.2f}s")
    print(f"   Average Output Tokens per Task: {summary['average_output_tokens']:.1f}")
    print(f"   Total Time (with delays): {summary['total_time']:.2f}s")
    
    return summary

def plot_results(all_summaries: List[Dict[str, Any]], agent_type_results: Dict[str, Dict[str, Any]]):
    """
    Plot the results of the evaluation and save them to CSV files.
    """
    # Save per-task results
    with open('benchmark_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Benchmark', 'Task Name', 'Success', 'Score', 'Execution Time', 'Output Tokens'])
        for summary in all_summaries:
            for result in summary['results']:
                writer.writerow([
                    summary['benchmark_name'],
                    result.task_name,
                    result.success,
                    result.score,
                    result.execution_time,
                    result.metrics.get('output_tokens', 0)
                ])
    print("Benchmark results saved to benchmark_results.csv")

    # Save aggregated agent type results
    with open('agent_type_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Agent Type', 'Weighted Average Score', 'Average Execution Time', 'Average Output Tokens'])
        for agent_type, data in agent_type_results.items():
            avg_score = np.average(data['scores']) if data['scores'] else 0.0
            avg_time = np.average(data['execution_times']) if data['execution_times'] else 0.0
            avg_tokens = np.average(data['output_tokens']) if data['output_tokens'] else 0.0
            writer.writerow([agent_type, avg_score, avg_time, avg_tokens])
    print("Agent type results saved to agent_type_results.csv")

    print("\nPlotting results...")
    
    # Plot for individual benchmark performance
    fig1, axs1 = plt.subplots(2, 2, figsize=(15, 12))
    fig1.suptitle('Benchmark Performance Analysis', fontsize=16)
    
    benchmarks = [s['benchmark_name'] for s in all_summaries]
    
    # Success Rate by Benchmark
    success_rates = [s['success_rate'] for s in all_summaries]
    axs1[0, 0].bar(benchmarks, success_rates, color='skyblue')
    axs1[0, 0].set_title('Success Rate by Benchmark')
    axs1[0, 0].set_ylabel('Success Rate')
    axs1[0, 0].tick_params(axis='x', rotation=45)
    
    # Average Score by Benchmark
    avg_scores = [s['average_score'] for s in all_summaries]
    axs1[0, 1].bar(benchmarks, avg_scores, color='lightgreen')
    axs1[0, 1].set_title('Average Score by Benchmark')
    axs1[0, 1].set_ylabel('Average Score')
    axs1[0, 1].tick_params(axis='x', rotation=45)
    
    # Average Time per Task
    avg_times = [s['average_time'] for s in all_summaries]
    axs1[1, 0].bar(benchmarks, avg_times, color='salmon')
    axs1[1, 0].set_title('Average Time per Task (s)')
    axs1[1, 0].set_ylabel('Seconds')
    axs1[1, 0].tick_params(axis='x', rotation=45)
    
    # Average Output Tokens
    avg_tokens = [s['average_output_tokens'] for s in all_summaries]
    axs1[1, 1].bar(benchmarks, avg_tokens, color='gold')
    axs1[1, 1].set_title('Average Output Tokens per Task')
    axs1[1, 1].set_ylabel('Tokens')
    axs1[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('benchmark_performance.png')
    print("Benchmark performance plot saved to benchmark_performance.png")
    
    # Plot for aggregated agent type performance
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle('Aggregated Performance by Agent Type', fontsize=16)
    
    agent_types = list(agent_type_results.keys())
    
    # Weighted Average Score by Agent Type
    avg_scores_by_type = [np.average(d['scores']) for d in agent_type_results.values()]
    axs2[0].bar(agent_types, avg_scores_by_type, color='cornflowerblue')
    axs2[0].set_title('Weighted Average Score')
    axs2[0].set_ylabel('Score')
    
    # Average Execution Time by Agent Type
    avg_times_by_type = [np.average(d['execution_times']) for d in agent_type_results.values()]
    axs2[1].bar(agent_types, avg_times_by_type, color='mediumseagreen')
    axs2[1].set_title('Average Execution Time (s)')
    axs2[1].set_ylabel('Seconds')
    
    # Average Output Tokens by Agent Type
    avg_tokens_by_type = [np.average(d['output_tokens']) for d in agent_type_results.values()]
    axs2[2].bar(agent_types, avg_tokens_by_type, color='lightcoral')
    axs2[2].set_title('Average Output Tokens')
    axs2[2].set_ylabel('Tokens')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('agent_type_performance.png')
    print("Agent type performance plot saved to agent_type_performance.png")
    plt.show()

async def main(benchmarks_to_run: Optional[List[str]] = None, plot: bool = False):
    """Run comprehensive evaluation of Gemma on all benchmarks."""
    
    # Load environment variables
    load_dotenv()
    
    print("Comprehensive Gemma Evaluation")
    print("=" * 70)
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] No API key found!")
        print("Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        return
    
    print(f"API key found: {api_key[:10]}...")
    
    try:
        # Load Gemma model
        print(f"\n Loading gemma-3-27b-it...")
        model = load_model_from_name("gemma-3-27b-it", api_key=api_key, temperature=0.3, max_tokens=30000)
        print(f"Model loaded: {model.model_name}")
        
        # Get all available benchmarks
        if benchmarks_to_run:
            all_benchmark_names = benchmarks_to_run
        else:
            available_benchmarks = get_available_benchmarks()
            all_benchmark_names = [name for sublist in available_benchmarks.values() for name in sublist]
        
        all_summaries = []
        
        for benchmark_name in all_benchmark_names:
            summary = await test_benchmark(model, benchmark_name)
            if 'error' not in summary:
                all_summaries.append(summary)

        # Overall summary
        print(f"\n\nOVERALL EVALUATION SUMMARY")
        print("=" * 70)
        
        for summary in all_summaries:
            print(f"\n BENCHMARK: {summary['benchmark_name']}")
            print(f"   - Success Rate: {summary['success_rate']:.1%}")
            print(f"   - Average Score: {summary['average_score']:.3f}")
            print(f"   - Avg Time: {summary['average_time']:.2f}s")
            print(f"   - Avg Output Tokens: {summary['average_output_tokens']:.1f}")

        total_tasks = sum(s['total_tasks'] for s in all_summaries)
        total_successful = sum(s['successful_tasks'] for s in all_summaries)
        overall_success_rate = total_successful / total_tasks if total_tasks > 0 else 0.0
        
        # Weighted average score
        if total_tasks > 0:
            weighted_avg_score = sum(s['average_score'] * (s['total_tasks'] / total_tasks) for s in all_summaries)
        else:
            weighted_avg_score = 0.0
        
        total_time = sum(s['total_time'] for s in all_summaries)
        
        print(f"\n COMBINED RESULTS:")
        print(f"   Total Benchmarks Tested: {len(all_summaries)}")
        print(f"   Total Tasks: {total_tasks}")
        print(f"   Overall Success Rate: {overall_success_rate:.1%} ({total_successful}/{total_tasks})")
        print(f"   Weighted Average Score: {weighted_avg_score:.3f}")
        print(f"   Total Evaluation Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        
        # Aggregate results by agent type
        agent_type_results = {}
        for summary in all_summaries:
            agent_type = summary['results'][0].agent_type.value
            if agent_type not in agent_type_results:
                agent_type_results[agent_type] = {
                    'scores': [],
                    'execution_times': [],
                    'output_tokens': [],
                    'num_tasks': 0
                }
            
            agent_type_results[agent_type]['scores'].extend([r.score for r in summary['results']])
            agent_type_results[agent_type]['execution_times'].extend([r.execution_time for r in summary['results'] if r.execution_time is not None])
            agent_type_results[agent_type]['output_tokens'].extend([r.metrics.get('output_tokens', 0) for r in summary['results']])
            agent_type_results[agent_type]['num_tasks'] += summary['total_tasks']
            
        print("\n AGGREGATE METRICS BY AGENT TYPE:")
        print("=" * 70)
        for agent_type, data in agent_type_results.items():
            avg_score = np.average(data['scores'], weights=([1]*len(data['scores']))) if data['scores'] else 0.0
            avg_time = np.average(data['execution_times']) if data['execution_times'] else 0.0
            avg_tokens = np.average(data['output_tokens']) if data['output_tokens'] else 0.0
            print(f"\n Agent Type: {agent_type}")
            print(f"   - Weighted Average Score: {avg_score:.3f}")
            print(f"   - Average Execution Time: {avg_time:.2f}s")
            print(f"   - Average Output Tokens: {avg_tokens:.1f}")

        print(f"\nEvaluation complete! Model tested on {total_tasks} tasks across {len(all_summaries)} benchmarks.")
        
        if plot:
            plot_results(all_summaries, agent_type_results)
        
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gemma evaluation on specified benchmarks.")
    parser.add_argument(
        "benchmarks",
        nargs="*",
        help="Optional: One or more benchmark names to run. If not provided, all benchmarks will be run."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, plot the results."
    )
    args = parser.parse_args()

    asyncio.run(main(benchmarks_to_run=args.benchmarks or None, plot=args.plot)) 