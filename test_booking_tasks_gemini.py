"""
Comprehensive test of Gemini on all goal-based hotel booking benchmark tasks.
"""

import asyncio
import os
import time
from src.models.loader import load_gemini
from src.benchmark.loader import load_benchmark


async def test_all_tasks():
    """Test Gemini on all 5 benchmark tasks."""
    print("TESTING GEMINI ON ALL HOTEL BOOKING TASKS")
    print("="*80)
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: No API key found!")
        return
    
    # Load model and benchmark
    model = load_gemini("gemini-2.5-pro", api_key=api_key, temperature=0.3)
    benchmark = load_benchmark("hotel_booking_goal_based")
    
    print("SUCCESS: Model and benchmark loaded")
    
    # Get all tasks
    tasks = benchmark.get_tasks()
    print(f"SUCCESS: Found {len(tasks)} tasks to test")
    print()
    
    results = []
    total_time = 0
    
    for i, task in enumerate(tasks, 1):
        print(f"TASK {i}/5: {task.name}")
        print(f"   Difficulty: {task.metadata['difficulty']}")
        print(f"   Expected searches: {task.metadata['expected_searches']}")
        print(f"   Goal: {task.metadata['goal']}")
        
        # Show task requirements
        requirements = task.metadata['requirements']
        print(f"   Guests: {requirements['guests']}")
        print(f"   Budget: ${requirements['max_budget_total']}")
        print(f"   Special needs: {requirements.get('special_needs', [])}")
        
        # Run evaluation
        print("   Running evaluation...")
        start_time = time.time()
        
        try:
            result = await benchmark.evaluate_task(task, model)
            end_time = time.time()
            task_time = end_time - start_time
            total_time += task_time
            
            print(f"   SUCCESS: Completed in {task_time:.1f}s")
            print(f"   Success: {'PASS' if result.success else 'FAIL'} {result.success}")
            print(f"   Score: {result.score:.3f}")
            
            # Show key metrics
            if result.metrics:
                print(f"   Search steps: {result.metrics.get('search_steps_planned', 'N/A')}")
                print(f"   Goal achieved: {result.metrics.get('goal_achieved', 'N/A')}")
                print(f"   Coverage: {result.metrics.get('search_completeness', 0):.1%}")
                print(f"   Rooms found: {result.metrics.get('total_rooms_found', 'N/A')}")
            
            # Show model response preview
            if result.model_response and result.model_response.text:
                response = result.model_response.text
                if "SEARCH PLAN:" in response:
                    plan_start = response.find("SEARCH PLAN:")
                    plan_part = response[plan_start:plan_start+200] + "..."
                    print(f"   Model output: {plan_part}")
            
            results.append(result)
            
        except Exception as e:
            print(f"   ERROR: FAILED: {e}")
            end_time = time.time()
            total_time += end_time - start_time
            results.append(None)
        
        print()
    
    # Calculate overall statistics
    print("="*80)
    print("OVERALL RESULTS")
    print("="*80)
    
    successful_results = [r for r in results if r and r.success]
    all_results = [r for r in results if r is not None]
    
    success_rate = len(successful_results) / len(tasks) * 100 if tasks else 0
    avg_score = sum(r.score for r in all_results) / len(all_results) if all_results else 0
    
    print(f"SUCCESS RATE: {success_rate:.1f}% ({len(successful_results)}/{len(tasks)})")
    print(f"AVERAGE SCORE: {avg_score:.3f}")
    print(f"TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print()
    
    # Show per-difficulty breakdown
    print("BREAKDOWN BY DIFFICULTY:")
    difficulty_stats = {}
    
    for i, result in enumerate(results):
        if result:
            difficulty = tasks[i].metadata['difficulty']
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {'total': 0, 'successful': 0, 'scores': []}
            
            difficulty_stats[difficulty]['total'] += 1
            if result.success:
                difficulty_stats[difficulty]['successful'] += 1
            difficulty_stats[difficulty]['scores'].append(result.score)
    
    for difficulty, stats in difficulty_stats.items():
        success_rate = stats['successful'] / stats['total'] * 100
        avg_score = sum(stats['scores']) / len(stats['scores'])
        print(f"   {difficulty.capitalize():8}: {success_rate:5.1f}% success, {avg_score:.3f} avg score ({stats['successful']}/{stats['total']})")
    
    print()
    
    # Show detailed task results
    print("DETAILED TASK RESULTS:")
    for i, (task, result) in enumerate(zip(tasks, results), 1):
        if result:
            status = "PASS" if result.success else "FAIL"
            print(f"   Task {i}: {task.name}")
            print(f"            {status} | Score: {result.score:.3f} | Steps: {result.metrics.get('search_steps_planned', 'N/A') if result.metrics else 'N/A'}")
        else:
            print(f"   Task {i}: {task.name}")
            print(f"            ERROR | Score: 0.000 | Steps: N/A")
    
    print()
    
    # Performance insights
    print("PERFORMANCE INSIGHTS:")
    
    if success_rate >= 80:
        print("   Excellent performance! Model handles systematic search planning very well.")
    elif success_rate >= 60:
        print("   Good performance! Model generally understands search planning.")
    elif success_rate >= 40:
        print("   Moderate performance! Model sometimes struggles with complex search planning.")
    else:
        print("   Poor performance! Model has difficulty with systematic search planning.")
    
    if avg_score >= 0.8:
        print("   High quality plans - good goal achievement and coverage.")
    elif avg_score >= 0.6:
        print("   Decent quality plans - room for improvement in coverage or execution.")
    else:
        print("   Plans need work - low goal achievement or poor coverage.")
    
    # Specific analysis
    basic_tasks = [r for i, r in enumerate(results) if r and tasks[i].metadata['difficulty'] == 'basic']
    complex_tasks = [r for i, r in enumerate(results) if r and tasks[i].metadata['difficulty'] in ['complex', 'extreme']]
    
    if basic_tasks and all(r.success for r in basic_tasks):
        print("   Handles simple scenarios well.")
    
    if complex_tasks and any(r.success for r in complex_tasks):
        print("   Can handle some complex multi-variable scenarios.")
    
    print("\nBENCHMARK EVALUATION COMPLETE!")
    return results


if __name__ == "__main__":
    asyncio.run(test_all_tasks()) 