#!/usr/bin/env python3
"""
Comprehensive test of Gemini Pro 2.5 on all email and maze tasks.
Tests both Simple Reflex (email) and Model-Based Reflex (maze) agent types.
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.benchmark.loader import load_benchmark
from src.models.loader import load_gemini
from src.benchmark.base import TaskResult

async def test_email_tasks(model, benchmark_name: str = "email_autoresponder_simple") -> Dict[str, Any]:
    """Test model on all email tasks."""
    print(f"\n[EMAIL] Testing Email Auto-Responder Tasks")
    print("=" * 60)
    
    # Load email benchmark
    benchmark = load_benchmark(benchmark_name)
    tasks = benchmark.get_tasks()
    
    print(f"Benchmark: {benchmark.benchmark_name}")
    print(f"Agent Type: {benchmark.agent_type.value}")
    print(f"Total Tasks: {len(tasks)}")
    
    results = []
    total_start_time = time.time()
    
    for i, task in enumerate(tasks):
        print(f"\n[TASK] Task {i+1}/{len(tasks)}: {task.name}")
        print(f"   Scenario: {task.metadata.get('scenario', 'unknown')}")
        print(f"   Expected: {task.expected_output}")
        print(f"   Difficulty: {task.metadata.get('difficulty', 'unknown')}")
        
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
                print(f"   Word Count: {result.metrics.get('word_count', 'N/A')}")
                print(f"   Follows Instructions: {result.metrics.get('follows_instructions', 'N/A')}")
                print(f"   Tokens Used: {result.metrics.get('tokens_used', 'N/A')}")
            
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
        
        # Small delay between tasks
        await asyncio.sleep(0.5)
    
    total_time = time.time() - total_start_time
    
    # Calculate summary statistics
    successful_tasks = [r for r in results if r.success]
    failed_tasks = [r for r in results if not r.success]
    
    avg_score = sum(r.score for r in results) / len(results) if results else 0.0
    avg_time = sum(r.execution_time for r in results) / len(results) if results else 0.0
    
    summary = {
        'benchmark_name': benchmark_name,
        'total_tasks': len(tasks),
        'successful_tasks': len(successful_tasks),
        'failed_tasks': len(failed_tasks),
        'success_rate': len(successful_tasks) / len(tasks) if tasks else 0.0,
        'average_score': avg_score,
        'average_time': avg_time,
        'total_time': total_time,
        'results': results
    }
    
    print(f"\n[SUMMARY] EMAIL BENCHMARK SUMMARY:")
    print(f"   Success Rate: {summary['success_rate']:.1%} ({summary['successful_tasks']}/{summary['total_tasks']})")
    print(f"   Average Score: {summary['average_score']:.3f}")
    print(f"   Average Time per Task: {summary['average_time']:.2f}s")
    print(f"   Total Time: {summary['total_time']:.2f}s")
    
    return summary

async def test_maze_tasks(model, benchmark_name: str = "textual_maze_navigation") -> Dict[str, Any]:
    """Test model on all maze tasks."""
    print(f"\n[MAZE] Testing Maze Navigation Tasks")
    print("=" * 60)
    
    # Load maze benchmark
    benchmark = load_benchmark(benchmark_name)
    tasks = benchmark.get_tasks()
    
    print(f"Benchmark: {benchmark.benchmark_name}")
    print(f"Agent Type: {benchmark.agent_type.value}")
    print(f"Total Tasks: {len(tasks)}")
    
    results = []
    total_start_time = time.time()
    
    for i, task in enumerate(tasks):
        print(f"\n[TASK] Task {i+1}/{len(tasks)}: {task.name}")
        print(f"   Difficulty: {task.metadata.get('difficulty', 'unknown')}")
        print(f"   Size: {task.metadata.get('maze_size', 'unknown')}")
        print(f"   Optimal Path: {task.metadata.get('optimal_path_length', 'unknown')} moves")
        print(f"   Max Allowed: {task.metadata.get('max_allowed_moves', 'unknown')} moves")
        
        # Show maze layout
        if 'maze_layout' in task.metadata:
            maze = task.metadata['maze_layout']
            print(f"   Layout:")
            for row in maze.grid:
                print(f"     {' '.join(row)}")
        
        try:
            # Run the task
            print(f"   [RUNNING] Running navigation... (this may take a while)")
            result = await benchmark.evaluate_task(task, model)
            results.append(result)
            
            # Show results
            status = "[SUCCESS]" if result.success else "[FAILED]"
            print(f"   Result: {status} (Score: {result.score:.3f})")
            
            if result.success and result.metadata:
                moves_made = result.metadata.get('moves_made', 'Unknown')
                optimal_moves = task.metadata.get('optimal_path_length', 'Unknown')
                print(f"   Moves Made: {moves_made}")
                print(f"   Efficiency: {optimal_moves}/{moves_made} = {optimal_moves/max(moves_made,1):.1%}")
                
                if 'path_taken' in result.metadata:
                    path = result.metadata['path_taken']
                    if len(path) <= 10:
                        print(f"   Path: {' → '.join(map(str, path))}")
                    else:
                        print(f"   Path: {' → '.join(map(str, path[:3]))} ... {' → '.join(map(str, path[-3:]))}")
            
            if result.metrics:
                print(f"   Total Tokens: {result.metrics.get('total_tokens_used', 'N/A')}")
                print(f"   Avg Tokens/Move: {result.metrics.get('avg_tokens_per_move', 'N/A'):.1f}")
                print(f"   Unique Cells: {result.metrics.get('unique_cells_visited', 'N/A')}")
                print(f"   Conversation Turns: {result.metrics.get('total_conversation_turns', 'N/A')}")
            
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
        
        # Small delay between tasks
        await asyncio.sleep(1.0)
    
    total_time = time.time() - total_start_time
    
    # Calculate summary statistics
    successful_tasks = [r for r in results if r.success]
    failed_tasks = [r for r in results if not r.success]
    
    avg_score = sum(r.score for r in results) / len(results) if results else 0.0
    avg_time = sum(r.execution_time for r in results) / len(results) if results else 0.0
    
    # Maze-specific statistics
    successful_results = [r for r in results if r.success and r.metadata]
    if successful_results:
        total_moves = sum(r.metadata.get('moves_made', 0) for r in successful_results)
        total_optimal = sum(tasks[i].metadata.get('optimal_path_length', 0) for i, r in enumerate(results) if r.success)
        overall_efficiency = total_optimal / max(total_moves, 1) if total_moves > 0 else 0.0
    else:
        overall_efficiency = 0.0
    
    summary = {
        'benchmark_name': benchmark_name,
        'total_tasks': len(tasks),
        'successful_tasks': len(successful_tasks),
        'failed_tasks': len(failed_tasks),
        'success_rate': len(successful_tasks) / len(tasks) if tasks else 0.0,
        'average_score': avg_score,
        'average_time': avg_time,
        'total_time': total_time,
        'overall_efficiency': overall_efficiency,
        'results': results
    }
    
    print(f"\n[SUMMARY] MAZE BENCHMARK SUMMARY:")
    print(f"   Success Rate: {summary['success_rate']:.1%} ({summary['successful_tasks']}/{summary['total_tasks']})")
    print(f"   Average Score: {summary['average_score']:.3f}")
    print(f"   Overall Efficiency: {summary['overall_efficiency']:.1%}")
    print(f"   Average Time per Task: {summary['average_time']:.2f}s")
    print(f"   Total Time: {summary['total_time']:.2f}s")
    
    return summary

async def main():
    """Run comprehensive evaluation of Gemini Pro 2.5 on all tasks."""
    
    # Load environment variables
    load_dotenv()
    
    print("Comprehensive Gemini Pro 2.5 Evaluation")
    print("=" * 70)
    print("Testing on ALL email and maze tasks")
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] No API key found!")
        print("Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        return
    
    print(f"[OK] API key found: {api_key[:10]}...")
    
    try:
        # Load Gemini Pro 2.5 model
        print(f"\n[LOADING] Loading Gemini Pro 2.5...")
        model = load_gemini("gemini-2.5-pro", api_key=api_key, temperature=0.3)
        print(f"[OK] Model loaded: {model.model_name}")
        
        # Test email tasks
        email_summary = await test_email_tasks(model)
        
        # Test maze tasks
        maze_summary = await test_maze_tasks(model)
        
        # Overall summary
        print(f"\n[RESULTS] OVERALL EVALUATION SUMMARY")
        print("=" * 70)
        
        print(f"\n[EMAIL] EMAIL BENCHMARK:")
        print(f"   Tasks: {email_summary['total_tasks']}")
        print(f"   Success Rate: {email_summary['success_rate']:.1%}")
        print(f"   Average Score: {email_summary['average_score']:.3f}")
        print(f"   Total Time: {email_summary['total_time']:.1f}s")
        
        print(f"\n[MAZE] MAZE BENCHMARK:")
        print(f"   Tasks: {maze_summary['total_tasks']}")
        print(f"   Success Rate: {maze_summary['success_rate']:.1%}")
        print(f"   Average Score: {maze_summary['average_score']:.3f}")
        print(f"   Path Efficiency: {maze_summary['overall_efficiency']:.1%}")
        print(f"   Total Time: {maze_summary['total_time']:.1f}s")
        
        # Combined statistics
        total_tasks = email_summary['total_tasks'] + maze_summary['total_tasks']
        total_successful = email_summary['successful_tasks'] + maze_summary['successful_tasks']
        overall_success_rate = total_successful / total_tasks if total_tasks > 0 else 0.0
        
        # Weighted average score
        email_weight = email_summary['total_tasks'] / total_tasks
        maze_weight = maze_summary['total_tasks'] / total_tasks
        weighted_avg_score = (email_summary['average_score'] * email_weight + 
                            maze_summary['average_score'] * maze_weight)
        
        total_time = email_summary['total_time'] + maze_summary['total_time']
        
        print(f"\nCOMBINED RESULTS:")
        print(f"   Total Tasks: {total_tasks}")
        print(f"   Overall Success Rate: {overall_success_rate:.1%} ({total_successful}/{total_tasks})")
        print(f"   Weighted Average Score: {weighted_avg_score:.3f}")
        print(f"   Total Evaluation Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        
        print(f"\nEvaluation complete! Model tested on {total_tasks} tasks across 2 agent types.")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the comprehensive evaluation
    asyncio.run(main()) 