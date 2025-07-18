"""
Self-contained test script for validating the scenarios in the
TaskSchedulingBenchmark.
"""

import asyncio
from dataclasses import dataclass, field, asdict
from itertools import combinations, permutations
from typing import List, Dict, Any, Optional, Tuple

# --- Data Structures (Copied from task_scheduling.py) ---

@dataclass
class Job:
    """Represents a job with its properties."""
    id: str
    duration: int
    reward: int
    deadline: Optional[int] = None
    dependencies: List[str] = field(default_factory=list)

@dataclass
class TimeSlot:
    """Represents a time slot with its properties."""
    id: str
    start: int
    end: int
    cost: int = 0

@dataclass
class ScheduledJob:
    """Represents a job assigned to a specific time slot."""
    job_id: str
    slot_id: str

    def __str__(self):
        return f"({self.job_id}, {self.slot_id})"

@dataclass
class Schedule:
    """Represents a complete schedule of jobs."""
    assignments: List[ScheduledJob]

    def __str__(self):
        return ", ".join(map(str, self.assignments))

# --- Scenarios (Copied from task_scheduling.py) ---

def get_scenarios() -> List[Dict[str, Any]]:
    """Defines the scenarios for the benchmark."""
    # (Scenario list copied from the benchmark file)
    return [
            {
                "name": "Simple Rewards",
                "description": "Assign jobs to slots to maximize rewards. No complex constraints.",
                "difficulty": "simple",
                "jobs": [
                    Job(id="J1", duration=2, reward=10),
                    Job(id="J2", duration=1, reward=5),
                    Job(id="J3", duration=3, reward=12),
                ],
                "slots": [
                    TimeSlot(id="S1", start=0, end=5),
                    TimeSlot(id="S2", start=0, end=5),
                    TimeSlot(id="S3", start=0, end=5),
                ]
            },
            {
                "name": "With Deadlines",
                "description": "Jobs have deadlines that must be met.",
                "difficulty": "medium",
                "jobs": [
                    Job(id="J1", duration=2, reward=20, deadline=4),
                    Job(id="J2", duration=3, reward=30, deadline=5),
                    Job(id="J3", duration=1, reward=10, deadline=2),
                ],
                "slots": [
                    TimeSlot(id="S1", start=0, end=5),
                    TimeSlot(id="S2", start=0, end=5),
                    TimeSlot(id="S3", start=0, end=5),
                ]
            },
            {
                "name": "With Costs",
                "description": "Time slots have costs that reduce utility.",
                "difficulty": "medium",
                "jobs": [
                    Job(id="J1", duration=2, reward=25),
                    Job(id="J2", duration=2, reward=25),
                ],
                "slots": [
                    TimeSlot(id="S1", start=0, end=4, cost=10),
                    TimeSlot(id="S2", start=0, end=4, cost=5),
                ]
            },
            {
                "name": "With Dependencies",
                "description": "Some jobs must be completed before others.",
                "difficulty": "hard",
                "jobs": [
                    Job(id="J1", duration=2, reward=20),
                    Job(id="J2", duration=2, reward=30, dependencies=["J1"]),
                    Job(id="J3", duration=1, reward=10),
                ],
                "slots": [
                    TimeSlot(id="S1", start=0, end=5),
                    TimeSlot(id="S2", start=0, end=5),
                    TimeSlot(id="S3", start=0, end=5),
                ]
            },
            {
                "name": "Complex Scenario",
                "description": "A mix of deadlines, costs, and dependencies.",
                "difficulty": "hard",
                "jobs": [
                    Job(id="J1", duration=1, reward=15, deadline=3),
                    Job(id="J2", duration=2, reward=25, dependencies=["J1"]),
                    Job(id="J3", duration=1, reward=10, deadline=4, dependencies=["J1"]),
                ],
                "slots": [
                    TimeSlot(id="S1", start=0, end=5, cost=5),
                    TimeSlot(id="S2", start=0, end=5, cost=2),
                    TimeSlot(id="S3", start=0, end=5, cost=0),
                ]
            },
            {
                "name": "More Jobs, Limited Slots",
                "description": "Select the most profitable subset of jobs when there are not enough time slots for all.",
                "difficulty": "medium",
                "jobs": [
                    Job(id="J1", duration=2, reward=20),
                    Job(id="J2", duration=2, reward=25),
                    Job(id="J3", duration=1, reward=10),
                    Job(id="J4", duration=3, reward=30),
                ],
                "slots": [
                    TimeSlot(id="S1", start=0, end=5),
                    TimeSlot(id="S2", start=0, end=5),
                    TimeSlot(id="S3", start=0, end=5),
                ]
            },
            {
                "name": "High Cost vs. Tight Deadline",
                "description": "A high-reward job has a tight deadline that can only be met by using an expensive time slot.",
                "difficulty": "medium",
                "jobs": [
                    Job(id="J1", duration=2, reward=50, deadline=2),
                    Job(id="J2", duration=3, reward=40),
                    Job(id="J3", duration=2, reward=10),
                ],
                "slots": [
                    TimeSlot(id="S1", start=0, end=2, cost=30),
                    TimeSlot(id="S2", start=0, end=5, cost=5),
                    TimeSlot(id="S3", start=2, end=5, cost=5),
                ]
            },
            {
                "name": "Long Dependency Chain",
                "description": "Tests planning for a sequence of dependent tasks (J1->J2->J3).",
                "difficulty": "hard",
                "jobs": [
                    Job(id="J1", duration=1, reward=10),
                    Job(id="J2", duration=2, reward=20, dependencies=["J1"]),
                    Job(id="J3", duration=1, reward=30, dependencies=["J2"]),
                ],
                "slots": [
                    TimeSlot(id="S1", start=0, end=5),
                    TimeSlot(id="S2", start=0, end=5),
                    TimeSlot(id="S3", start=0, end=5),
                ]
            },
            {
                "name": "Unprofitable Job",
                "description": "One job's reward is lower than the cost of any available slot, so it should be left unscheduled.",
                "difficulty": "medium",
                "jobs": [
                    Job(id="J1", duration=2, reward=30),
                    Job(id="J2", duration=2, reward=30),
                    Job(id="J3", duration=2, reward=10),
                ],
                "slots": [
                    TimeSlot(id="S1", start=0, end=4, cost=15),
                    TimeSlot(id="S2", start=0, end=4, cost=15),
                    TimeSlot(id="S3", start=0, end=4, cost=15),
                ]
            },
            {
                "name": "Parallel Opportunities",
                "description": "Tests scheduling independent jobs in parallel.",
                "difficulty": "medium",
                "jobs": [
                    Job(id="J1", duration=2, reward=20),
                    Job(id="J2", duration=2, reward=20),
                    Job(id="J3", duration=2, reward=25),
                    Job(id="J4", duration=2, reward=25),
                ],
                "slots": [
                    TimeSlot(id="S1", start=0, end=2),
                    TimeSlot(id="S2", start=0, end=2),
                    TimeSlot(id="S3", start=2, end=4),
                    TimeSlot(id="S4", start=2, end=4),
                ]
            },
            {
                "name": "Gaps Between Slots",
                "description": "A job's duration must fit entirely within a single slot.",
                "difficulty": "medium",
                "jobs": [
                    Job(id="J1", duration=3, reward=40),
                    Job(id="J2", duration=1, reward=10),
                    Job(id="J3", duration=2, reward=20),
                ],
                "slots": [
                    TimeSlot(id="S1", start=0, end=2),
                    TimeSlot(id="S2", start=3, end=5),
                ]
            },
            {
                "name": "Impossible Duration",
                "description": "A job is longer than any available time slot and must be excluded.",
                "difficulty": "simple",
                "jobs": [
                    Job(id="J1", duration=5, reward=100),
                    Job(id="J2", duration=2, reward=20),
                    Job(id="J3", duration=2, reward=20),
                ],
                "slots": [
                    TimeSlot(id="S1", start=0, end=4),
                    TimeSlot(id="S2", start=0, end=4),
                    TimeSlot(id="S3", start=0, end=3),
                ]
            },
            {
                "name": "All Unprofitable",
                "description": "All jobs result in negative utility and should not be scheduled. Optimal utility is 0.",
                "difficulty": "hard",
                "jobs": [
                    Job(id="J1", duration=2, reward=10),
                    Job(id="J2", duration=3, reward=20),
                ],
                "slots": [
                    TimeSlot(id="S1", start=0, end=5, cost=25),
                    TimeSlot(id="S2", start=0, end=5, cost=30),
                ]
            },
            {
                "name": "Dependency and Deadline Interaction",
                "description": "A dependency forces an early schedule for one job to allow another to meet its deadline.",
                "difficulty": "hard",
                "jobs": [
                    Job(id="J1", duration=2, reward=20),
                    Job(id="J2", duration=2, reward=30, dependencies=["J1"], deadline=4),
                    Job(id="J3", duration=2, reward=50),
                ],
                "slots": [
                    TimeSlot(id="S1", start=0, end=2),
                    TimeSlot(id="S2", start=2, end=4),
                    TimeSlot(id="S3", start=0, end=5),
                ]
            },
            {
                "name": "Very Complex Scenario",
                "description": "A challenging mix of dependencies, costs, and tight deadlines with 5 jobs.",
                "difficulty": "very_hard",
                "jobs": [
                    Job(id="J1", duration=2, reward=30, deadline=4),
                    Job(id="J2", duration=1, reward=15),
                    Job(id="J3", duration=2, reward=25, dependencies=["J1"], deadline=5),
                    Job(id="J4", duration=1, reward=10, dependencies=["J2"]),
                    Job(id="J5", duration=2, reward=40),
                ],
                "slots": [
                    TimeSlot(id="S1", start=0, end=3, cost=5),
                    TimeSlot(id="S2", start=1, end=4, cost=10),
                    TimeSlot(id="S3", start=2, end=5, cost=0),
                    TimeSlot(id="S4", start=0, end=5, cost=15),
                    TimeSlot(id="S5", start=0, end=5, cost=20),
                ]
            }
        ]

# --- Solver and Utility Functions (Copied from task_scheduling.py) ---

def calculate_utility(schedule: Schedule, jobs: List[Job], slots: List[TimeSlot]) -> Tuple[int, bool]:
    """Calculate the total utility of a given schedule and check its validity."""
    job_map = {job.id: job for job in jobs}
    slot_map = {slot.id: slot for slot in slots}
    
    # Basic validation: valid IDs, no duplicate jobs or slots
    scheduled_jobs = set()
    scheduled_slots = set()
    for assignment in schedule.assignments:
        if assignment.job_id not in job_map or assignment.slot_id not in slot_map:
            return 0, False  # Invalid job or slot ID
        if assignment.job_id in scheduled_jobs or assignment.slot_id in scheduled_slots:
            return 0, False  # Duplicate job or slot assignment
        scheduled_jobs.add(assignment.job_id)
        scheduled_slots.add(assignment.slot_id)

    # Build maps for scheduled items
    assignment_map = {a.job_id: a for a in schedule.assignments}
    job_end_times = {}

    # Check timing constraints (duration, deadline)
    for job_id, assignment in assignment_map.items():
        job = job_map[job_id]
        slot = slot_map[assignment.slot_id]
        
        start_time = slot.start
        end_time = start_time + job.duration

        if end_time > slot.end:
            return 0, False  # Job doesn't fit in slot
        
        if job.deadline is not None and end_time > job.deadline:
            return 0, False  # Missed deadline

        job_end_times[job_id] = end_time

    # Check dependency constraints
    for job_id, assignment in assignment_map.items():
        job = job_map[job_id]
        slot = slot_map[assignment.slot_id]
        start_time = slot.start

        for dep_id in job.dependencies:
            if dep_id not in job_end_times or start_time < job_end_times[dep_id]:
                return 0, False # Dependency violation

    # If all checks pass, calculate utility
    total_reward = sum(job_map[a.job_id].reward for a in schedule.assignments)
    total_cost = sum(slot_map[a.slot_id].cost for a in schedule.assignments)
    
    return total_reward - total_cost, True

def solve_schedule(jobs: List[Job], slots: List[TimeSlot]) -> Tuple[Schedule, int, bool]:
    """
    Finds the optimal schedule using brute-force permutation of all job subsets.
    Returns the best schedule, max utility, and a boolean indicating if any valid schedule was found.
    """
    job_map = {job.id: job for job in jobs}
    
    best_schedule = Schedule(assignments=[])
    max_utility = 0
    any_valid_schedule_found = False

    job_ids = list(job_map.keys())
    slot_ids = [slot.id for slot in slots]

    # Check the case of scheduling nothing
    if 0 > max_utility:
        max_utility = 0
        best_schedule = Schedule(assignments=[])

    # Iterate over all possible numbers of jobs to schedule
    for k in range(1, len(job_ids) + 1):
        # Iterate over all combinations of k jobs
        for job_subset in combinations(job_ids, k):
            # Iterate over all permutations of these k jobs
            for job_perm in permutations(job_subset):
                # Iterate over all permutations of k slots
                if k > len(slot_ids):
                    continue
                for slot_perm in permutations(slot_ids, k):
                    assignments = [ScheduledJob(job_id=job_perm[i], slot_id=slot_perm[i]) for i in range(k)]
                    schedule = Schedule(assignments=assignments)
                    
                    utility, is_valid = calculate_utility(schedule, jobs, slots)
                    
                    if is_valid:
                        any_valid_schedule_found = True
                        if utility > max_utility:
                            max_utility = utility
                            best_schedule = schedule

    return best_schedule, max_utility, any_valid_schedule_found


# --- Main Execution Block ---

async def main():
    """
    This script is for testing and validating the scenarios in the
    TaskSchedulingBenchmark.
    """
    print("Running scenario validation for TaskSchedulingBenchmark...")
    
    scenarios = get_scenarios()
    
    for scenario in scenarios:
        print(f"\n--- SCENARIO: {scenario['name']} ---")
        jobs = [Job(**asdict(j)) for j in scenario['jobs']]
        slots = [TimeSlot(**asdict(s)) for s in scenario['slots']]
        
        optimal_schedule, optimal_utility, is_solvable = solve_schedule(jobs, slots)
        
        print(f"Optimal Utility: {optimal_utility}")
        if optimal_schedule.assignments:
            print(f"Optimal Schedule: {optimal_schedule}")
        else:
            print("Optimal Schedule: (No jobs scheduled)")

        if not is_solvable:
            print("INFO: No valid schedule is possible for this scenario.")

    print("\nValidation complete.")

if __name__ == "__main__":
    asyncio.run(main()) 