"""
Test script for the Goal-Based Agent benchmark using Gemini Pro.

This script demonstrates the hotel booking planning benchmark that tests
a model's ability to create comprehensive plans to achieve specific goals.
"""

import asyncio
import os
import time
from typing import Dict, Any

from src.models.loader import load_gemini
from src.benchmark.loader import load_benchmark


async def test_goal_based_hotel_booking(model, benchmark_name: str = "hotel_booking_goal_based") -> Dict[str, Any]:
    """Test model on hotel booking planning tasks."""
    print(f"\n Testing Goal-Based Hotel Booking Planning")
    print("=" * 60)
    
    # Load benchmark
    benchmark = load_benchmark(benchmark_name)
    tasks = benchmark.get_tasks()
    
    print(f"   Total Tasks: {len(tasks)}")
    print(f"   Agent Type: Goal-Based")
    print(f"   Focus: Planning and goal-directed reasoning")
    
    results = []
    total_start_time = time.time()
    
    for i, task in enumerate(tasks):
        print(f"\n Task {i+1}/{len(tasks)}: {task.name}")
        print(f"   Scenario: {task.metadata.get('scenario', 'unknown')}")
        print(f"   Difficulty: {task.metadata.get('difficulty', 'unknown')}")
        print(f"   Type: {task.metadata.get('scenario_type', 'unknown')}")
        
        try:
            # Run the task
            print(f"   Running planning task... (this may take a while)")
            result = await benchmark.evaluate_task(task, model)
            results.append(result)
            
            # Display results
            if result.success:
                print(f"   [SUCCESS] Score: {result.score:.3f}")
            else:
                print(f"   [FAILED] Score: {result.score:.3f}")
            
            # Show key metrics
            if result.metrics:
                print(f"   Planning Accuracy: {result.metrics.get('planning_accuracy', 0):.3f}")
                print(f"   Execution Success: {result.metrics.get('execution_success', False)}")
                print(f"   Template Followed: {result.metrics.get('template_followed', False)}")
                
                # Show plan quality indicators
                location_correct = result.metrics.get('location_correct', False)
                guests_correct = result.metrics.get('guests_correct', False) 
                budget_correct = result.metrics.get('budget_correct', False)
                print(f"   Requirements: Location={location_correct}, Guests={guests_correct}, Budget={budget_correct}")
            
            print(f"   Execution Time: {result.execution_time:.1f}s")
            
        except Exception as e:
            print(f"   [ERROR] Failed to evaluate task: {e}")
            # Create error result
            from src.benchmark.base import TaskResult
            error_result = TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                agent_type=benchmark.agent_type,
                success=False,
                score=0.0,
                metrics={},
                execution_time=0.0,
                error_message=str(e)
            )
            results.append(error_result)
    
    total_time = time.time() - total_start_time
    
    # Calculate summary statistics
    successful_tasks = sum(1 for result in results if result.success)
    total_tasks = len(results)
    success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
    average_score = sum(result.score for result in results) / total_tasks if total_tasks > 0 else 0
    
    # Calculate planning-specific metrics
    planning_accuracies = [r.metrics.get('planning_accuracy', 0) for r in results if r.metrics]
    avg_planning_accuracy = sum(planning_accuracies) / len(planning_accuracies) if planning_accuracies else 0
    
    execution_successes = [r.metrics.get('execution_success', False) for r in results if r.metrics]
    execution_success_rate = sum(execution_successes) / len(execution_successes) if execution_successes else 0
    
    # Template compliance
    template_compliance = [r.metrics.get('template_followed', False) for r in results if r.metrics]
    template_compliance_rate = sum(template_compliance) / len(template_compliance) if template_compliance else 0
    
    summary = {
        'benchmark_name': benchmark_name,
        'total_tasks': total_tasks,
        'successful_tasks': successful_tasks,
        'success_rate': success_rate,
        'average_score': average_score,
        'average_planning_accuracy': avg_planning_accuracy,
        'execution_success_rate': execution_success_rate,
        'template_compliance_rate': template_compliance_rate,
        'total_time': total_time,
        'results': results
    }
    
    print(f"\n GOAL-BASED BENCHMARK SUMMARY:")
    print(f"   Success Rate: {summary['success_rate']:.1%} ({summary['successful_tasks']}/{summary['total_tasks']})")
    print(f"   Average Score: {summary['average_score']:.3f}")
    print(f"   Planning Accuracy: {summary['average_planning_accuracy']:.3f}")
    print(f"   Execution Success Rate: {summary['execution_success_rate']:.1%}")
    print(f"   Template Compliance: {summary['template_compliance_rate']:.1%}")
    print(f"   Total Time: {summary['total_time']:.1f}s ({summary['total_time']/60:.1f} minutes)")
    
    return summary


async def main():
    """Main function to run the goal-based benchmark evaluation."""
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: No API key found!")
        print("Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        print("You can run: python3 setup_env.py")
        return
    
    print(f"API key found: {api_key[:10]}...")
    
    try:
        # Load Gemini Pro 2.5 model
        print(f"\n Loading Gemini Pro 2.5...")
        model = load_gemini("gemini-2.5-pro", api_key=api_key, temperature=0.3)
        print(f"Model loaded: {model.model_name}")
        
        # Test goal-based hotel booking tasks
        print(f"\n" + "="*80)
        print(f"GOAL-BASED AGENT EVALUATION: HOTEL BOOKING PLANNING")
        print(f"="*80)
        
        goal_summary = await test_goal_based_hotel_booking(model)
        
        # Overall summary
        print(f"\nOVERALL EVALUATION SUMMARY")
        print("=" * 70)
        
        print(f"\n GOAL-BASED PLANNING BENCHMARK:")
        print(f"   Tasks: {goal_summary['total_tasks']}")
        print(f"   Success Rate: {goal_summary['success_rate']:.1%}")
        print(f"   Average Score: {goal_summary['average_score']:.3f}")
        print(f"   Planning Accuracy: {goal_summary['average_planning_accuracy']:.3f}")
        print(f"   Execution Success: {goal_summary['execution_success_rate']:.1%}")
        print(f"   Template Compliance: {goal_summary['template_compliance_rate']:.1%}")
        print(f"   Total Time: {goal_summary['total_time']:.1f}s")
        
        # Performance insights
        print(f"\n PERFORMANCE INSIGHTS:")
        
        if goal_summary['success_rate'] > 0.8:
            print("Strong performance on goal-based planning tasks")
        elif goal_summary['success_rate'] > 0.6:
            print("Moderate performance on goal-based planning tasks")
        else:
            print("Weak performance on goal-based planning tasks")
            
        if goal_summary['average_planning_accuracy'] > 0.8:
            print("Excellent requirement extraction and planning accuracy")
        elif goal_summary['average_planning_accuracy'] > 0.6:
            print("Good requirement extraction with some gaps")
        else:
            print("Poor requirement extraction - may struggle with goal understanding")
            
        if goal_summary['execution_success_rate'] > 0.8:
            print("Plans are highly executable and practical")
        elif goal_summary['execution_success_rate'] > 0.6:
            print("Plans are moderately executable with some issues")
        else:
            print("Plans often fail when executed - may lack practical constraints")
            
        if goal_summary['template_compliance_rate'] > 0.8:
            print("Excellent structured response adherence")
        else:
            print("Poor template following - may need better instruction compliance")
        
        # Recommendations
        print(f"\n RECOMMENDATIONS:")
        
        if goal_summary['average_planning_accuracy'] < 0.7:
            print("Planning: Consider improving requirement extraction prompting")
            print("   - Model may need clearer instructions for parsing user requests")
            print("   - Review failed cases for pattern analysis")
        
        if goal_summary['execution_success_rate'] < 0.7:
            print("Execution: Plans need to be more practical and constraint-aware")
            print("   - Model may not understand business constraints well")
            print("   - Could benefit from better constraint reasoning")
        
        if goal_summary['template_compliance_rate'] < 0.8:
            print("Structure: Improve template following and structured output")
            print("   - Model may benefit from examples of proper format")
            print("   - Consider more explicit formatting instructions")
        
        if goal_summary['success_rate'] < 0.8:
            print("Overall: Goal-based reasoning needs improvement")
            print("   - Consider breaking down complex planning into smaller steps")
            print("   - May benefit from chain-of-thought prompting")
        
        print(f"\nEvaluation complete! Model tested on {goal_summary['total_tasks']} goal-based planning tasks.")
        print(f"Focus: Strategic planning, requirement extraction, and constraint handling.")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 