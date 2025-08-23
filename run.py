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
from src.benchmarks.local_web_app import stop_server as stop_flask_server
from src.benchmarks.selenium_mcp_server import stop_mcp_server
from src.models.loader import load_model_from_name
from src.benchmark.base import TaskResult, BaseBenchmark

async def test_benchmark(model, benchmark_name: str, verbose: bool = False) -> Dict[str, Any]:
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
            
            if result.metrics:
                print(f"   Output Tokens: {result.metrics.get('output_tokens', 'N/A')}")
            
            if verbose:

                if benchmark_name == "textual_maze_navigation" and "model_move_outputs" in result.metrics:
                    print("   Model Move Outputs:")
                    for i, output in enumerate(result.metrics["model_move_outputs"]):
                        print(f"     Turn {i+1}: {output.strip()}")

                if benchmark_name == "inventory_management" and "turn_scores" in result.metrics:
                    turn_scores_str = ", ".join([f"{s:.2f}" for s in result.metrics['turn_scores']])
                    print(f"   Turn Scores: [{turn_scores_str}]")

                if benchmark_name == "shortest_path_planning" and "optimal_path" in result.metrics:
                    print(f"   Optimal Path: {result.metrics.get('optimal_path')} (Weight: {result.metrics.get('optimal_weight')})")
                    print(f"   Model Path:   {result.metrics.get('model_path', 'N/A')} (Weight: {result.metrics.get('model_path_weight', 'N/A')})")
                    print(f"   Model Raw:    '{result.metrics.get('model_path_raw', 'N/A')}'")
                
                # Additional verbose diagnostics for specific benchmarks
                if benchmark_name == "fraud_detection":
                    if result.metrics:
                        print(f"   Follows Instructions: {result.metrics.get('follows_instructions', 'N/A')}")
                        print(f"   Exact Match: {result.metrics.get('exact_match', 'N/A')}")
                        print(f"   Cleaned Response: {result.metrics.get('cleaned_response', 'N/A')}")

                if result.model_response and result.model_response.text:
                    response_text = result.model_response.text.strip()
                    print(f"   Model Response: {response_text}")

                if benchmark_name == "event_conflict_detection" and result.metrics:
                    m = result.metrics
                    # Final summary metrics
                    print(f"   Ground Truth Tags: {m.get('ground_truth_tags', 'N/A')}")
                    print(f"   Final Predicted Tags: {m.get('final_predicted_tags', 'N/A')}")
                    print(f"   Final Precision/Recall/F1: {m.get('final_precision','N/A'):.3f}/{m.get('final_recall','N/A'):.3f}/{m.get('final_f1','N/A'):.3f}")
                    if 'final_parse_failed' in m:
                        print(f"   Final Parse Failed: {m.get('final_parse_failed')}")
                    # Dynamics across turns
                    if 'per_turn_precision' in m and 'per_turn_recall' in m and 'per_turn_f1' in m:
                        turn_p = ", ".join(f"{x:.2f}" for x in m['per_turn_precision'])
                        turn_r = ", ".join(f"{x:.2f}" for x in m['per_turn_recall'])
                        turn_f1 = ", ".join(f"{x:.2f}" for x in m['per_turn_f1'])
                        print(f"   Per-Turn Precision: [{turn_p}]")
                        print(f"   Per-Turn Recall:    [{turn_r}]")
                        print(f"   Per-Turn F1:        [{turn_f1}]")
                    if 'predicted_tags_by_turn' in m:
                        print(f"   Predicted Tags by Turn: {m['predicted_tags_by_turn']}")
                    if 'flip_flops' in m:
                        print(f"   Flip-Flops: {m['flip_flops']}")
                    if 'monotonic_recall' in m:
                        print(f"   Monotonic Recall: {m['monotonic_recall']:.3f}")
                    if 'spurious_persistence' in m:
                        print(f"   Spurious Persistence: {m['spurious_persistence']:.3f}")

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
    scores = [r.score for r in results]
    execution_times = [r.execution_time for r in results if r.execution_time is not None]
    output_tokens = [r.metrics.get('output_tokens', 0) for r in results if r.metrics]

    summary = {
        'benchmark_name': benchmark_name,
        'agent_type': benchmark.agent_type.value,
        'total_tasks': len(tasks),
        'successful_tasks': len(successful_tasks),
        'success_rate': len(successful_tasks) / len(tasks) if tasks else 0.0,
        'average_score': np.mean(scores) if scores else 0.0,
        'std_dev_score': np.std(scores) if scores else 0.0,
        'average_time': np.mean(execution_times) if execution_times else 0.0,
        'std_dev_time': np.std(execution_times) if execution_times else 0.0,
        'average_output_tokens': np.mean(output_tokens) if output_tokens else 0.0,
        'std_dev_output_tokens': np.std(output_tokens) if output_tokens else 0.0,
        'total_time': total_time,
        'results': results
    }
    
    print(f"\n {benchmark_name.upper()} BENCHMARK SUMMARY:")
    print(f"   Success Rate: {summary['success_rate']:.1%} ({summary['successful_tasks']}/{summary['total_tasks']})")
    print(f"   Average Score: {summary['average_score']:.3f} (±{summary['std_dev_score']:.3f})")
    print(f"   Average Time per Task: {summary['average_time']:.2f}s (±{summary['std_dev_time']:.2f}s)")
    print(f"   Average Output Tokens per Task: {summary['average_output_tokens']:.1f} (±{summary['std_dev_output_tokens']:.1f})")
    print(f"   Total Time (with delays): {summary['total_time']:.2f}s")
    
    return summary

def plot_results(all_summaries: List[Dict[str, Any]], agent_type_results: Dict[str, Dict[str, Any]], model_name: str):
    """
    Plot the results of the evaluation and save them to CSV files.
    """
    results_dir = Path(f"results/{model_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    file_suffix = f"_{model_name}"

    # Save per-task results
    with open(results_dir / f'benchmark_results{file_suffix}.csv', 'w', newline='') as f:
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
    print(f"Benchmark results saved to {results_dir / f'benchmark_results{file_suffix}.csv'}")

    # Save aggregated agent type results
    with open(results_dir / f'agent_type_results{file_suffix}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Agent Type', 'Weighted Average Score', 'Std Dev Score', 'Average Execution Time', 'Std Dev Time', 'Average Output Tokens', 'Std Dev Tokens'])
        for agent_type, data in agent_type_results.items():
            writer.writerow([
                agent_type, 
                data['mean_score'], data['std_score'],
                data['mean_time'], data['std_time'],
                data['mean_tokens'], data['std_tokens']
            ])
    print(f"Agent type results saved to {results_dir / f'agent_type_results{file_suffix}.csv'}")

    print("\nPlotting results...")
    
    # Plot for individual benchmark performance
    fig1, axs1 = plt.subplots(1, 3, figsize=(18, 6))
    fig1.suptitle(f'Benchmark Performance Analysis for {model_name}', fontsize=16)
    
    benchmarks = [s['benchmark_name'] for s in all_summaries]
    
    # Average Score by Benchmark
    avg_scores = [s['average_score'] for s in all_summaries]
    std_scores = [s['std_dev_score'] for s in all_summaries]
    axs1[0].bar(benchmarks, avg_scores, yerr=std_scores, color='lightgreen', capsize=5)
    axs1[0].set_title('Average Score by Benchmark')
    axs1[0].set_ylabel('Average Score')
    axs1[0].tick_params(axis='x', rotation=45)
    
    # Average Time per Task
    avg_times = [s['average_time'] for s in all_summaries]
    std_times = [s['std_dev_time'] for s in all_summaries]
    axs1[1].bar(benchmarks, avg_times, yerr=std_times, color='salmon', capsize=5)
    axs1[1].set_title('Average Time per Task (s)')
    axs1[1].set_ylabel('Seconds')
    axs1[1].tick_params(axis='x', rotation=45)
    
    # Average Output Tokens
    avg_tokens = [s['average_output_tokens'] for s in all_summaries]
    std_tokens = [s['std_dev_output_tokens'] for s in all_summaries]
    axs1[2].bar(benchmarks, avg_tokens, yerr=std_tokens, color='gold', capsize=5)
    axs1[2].set_title('Average Output Tokens per Task')
    axs1[2].set_ylabel('Tokens (log scale)')
    axs1[2].set_yscale('log')
    axs1[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(results_dir / f'benchmark_performance{file_suffix}.png')
    print(f"Benchmark performance plot saved to {results_dir / f'benchmark_performance{file_suffix}.png'}")
    
    # Plot for aggregated agent type performance
    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 6))
    fig2.suptitle(f'Aggregated Performance by Agent Type for {model_name}', fontsize=16)
    
    agent_types = list(agent_type_results.keys())
    
    # Weighted Average Score by Agent Type
    avg_scores_by_type = [d['mean_score'] for d in agent_type_results.values()]
    std_scores_by_type = [d['std_score'] for d in agent_type_results.values()]
    axs2[0].bar(agent_types, avg_scores_by_type, yerr=std_scores_by_type, color='cornflowerblue', capsize=5)
    axs2[0].set_title('Weighted Average Score')
    axs2[0].set_ylabel('Score')
    axs2[0].tick_params(axis='x', rotation=45)
    
    # Average Execution Time by Agent Type
    avg_times_by_type = [d['mean_time'] for d in agent_type_results.values()]
    std_times_by_type = [d['std_time'] for d in agent_type_results.values()]
    axs2[1].bar(agent_types, avg_times_by_type, yerr=std_times_by_type, color='mediumseagreen', capsize=5)
    axs2[1].set_title('Average Execution Time (s)')
    axs2[1].set_ylabel('Seconds')
    axs2[1].tick_params(axis='x', rotation=45)
    
    # Average Output Tokens by Agent Type
    avg_tokens_by_type = [d['mean_tokens'] for d in agent_type_results.values()]
    std_tokens_by_type = [d['std_tokens'] for d in agent_type_results.values()]
    axs2[2].bar(agent_types, avg_tokens_by_type, yerr=std_tokens_by_type, color='lightcoral', capsize=5)
    axs2[2].set_title('Average Output Tokens')
    axs2[2].set_ylabel('Tokens (log scale)')
    if any(v > 0 for v in avg_tokens_by_type):
      axs2[2].set_yscale('log')
    axs2[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(results_dir / f'agent_type_performance{file_suffix}.png')
    print(f"Agent type performance plot saved to {results_dir / f'agent_type_performance{file_suffix}.png'}")
    plt.show()

async def main(model_name: str, benchmarks_to_run: Optional[List[str]] = None, plot: bool = False, verbose: bool = False):
    """Run comprehensive evaluation of a model on all benchmarks."""
    
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
        # Load the specified model
        print(f"\n Loading {model_name}...")
        model = load_model_from_name(model_name, api_key=api_key, temperature=0.3, max_tokens=30000)
        print(f"Model loaded: {model.model_name}")
        
        # Get all available benchmarks
        if benchmarks_to_run:
            all_benchmark_names = benchmarks_to_run
        else:
            available_benchmarks = get_available_benchmarks()
            all_benchmark_names = [name for sublist in available_benchmarks.values() for name in sublist]
        
        all_summaries = []
        
        for benchmark_name in all_benchmark_names:
            summary = await test_benchmark(model, benchmark_name, verbose=verbose)
            if 'error' not in summary:
                all_summaries.append(summary)

        # Overall summary
        print(f"\n\nOVERALL EVALUATION SUMMARY")
        print("=" * 70)
        
        for summary in all_summaries:
            print(f"\n BENCHMARK: {summary['benchmark_name']}")
            print(f"   - Success Rate: {summary['success_rate']:.1%}")
            print(f"   - Average Score: {summary['average_score']:.3f} (±{summary['std_dev_score']:.3f})")
            print(f"   - Avg Time: {summary['average_time']:.2f}s (±{summary['std_dev_time']:.2f}s)")
            print(f"   - Avg Output Tokens: {summary['average_output_tokens']:.1f} (±{summary['std_dev_output_tokens']:.1f})")

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
        # Auto-stop dev servers if local_web_navigation was run
        try:
            if any(s['benchmark_name'] == 'local_web_navigation' for s in all_summaries):
                stop_flask_server('127.0.0.1', 5005)
                stop_mcp_server()
        except Exception:
            pass
        
        # Aggregate results by agent type
        agent_type_results = {}
        for summary in all_summaries:
            agent_type = summary['agent_type']
            if agent_type not in agent_type_results:
                agent_type_results[agent_type] = {'scores': [], 'execution_times': [], 'output_tokens': [], 'num_tasks': 0}
            
            agent_type_results[agent_type]['scores'].extend([r.score for r in summary['results']])
            agent_type_results[agent_type]['execution_times'].extend([r.execution_time for r in summary['results'] if r.execution_time is not None])
            agent_type_results[agent_type]['output_tokens'].extend([r.metrics.get('output_tokens', 0) for r in summary['results']])
            agent_type_results[agent_type]['num_tasks'] += summary['total_tasks']

        agent_type_aggregated = {}
        for agent_type, data in agent_type_results.items():
            agent_type_aggregated[agent_type] = {
                'mean_score': np.mean(data['scores']) if data['scores'] else 0.0,
                'std_score': np.std(data['scores']) if data['scores'] else 0.0,
                'mean_time': np.mean(data['execution_times']) if data['execution_times'] else 0.0,
                'std_time': np.std(data['execution_times']) if data['execution_times'] else 0.0,
                'mean_tokens': np.mean(data['output_tokens']) if data['output_tokens'] else 0.0,
                'std_tokens': np.std(data['output_tokens']) if data['output_tokens'] else 0.0,
            }
        
        print("\n AGGREGATE METRICS BY AGENT TYPE:")
        print("=" * 70)
        for agent_type, data in agent_type_aggregated.items():
            print(f"\n Agent Type: {agent_type}")
            print(f"   - Avg Score: {data['mean_score']:.3f} (±{data['std_score']:.3f})")
            print(f"   - Avg Execution Time: {data['mean_time']:.2f}s (±{data['std_time']:.2f}s)")
            print(f"   - Avg Output Tokens: {data['mean_tokens']:.1f} (±{data['std_tokens']:.1f})")

        print(f"\nEvaluation complete! Model tested on {total_tasks} tasks across {len(all_summaries)} benchmarks.")
        
        if plot:
            if not agent_type_aggregated:
                print("Skipping plots due to no aggregated agent type data.")
            else:
                plot_results(all_summaries, agent_type_aggregated, model_name)
        
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, print detailed diagnostic information for each task."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-3-27b-it",
        help="The model to test. Defaults to gemma-3-27b-it."
    )
    args = parser.parse_args()

    asyncio.run(main(
        model_name=args.model,
        benchmarks_to_run=args.benchmarks or None,
        plot=args.plot,
        verbose=args.verbose
    ))
