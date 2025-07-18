"""
Utility-Based Task Scheduling Benchmark for AgenticEvals.

This benchmark tests a model's ability to solve a simple optimization problem:
assigning jobs to time slots to maximize a cumulative utility score, which is
a function of job rewards and slot costs.

The model must:
1.  Parse a description of jobs and time slots with their associated rewards,
    durations, deadlines, costs, and dependencies.
2.  Generate a valid schedule that assigns jobs to time slots.
3.  Maximize the total utility of the schedule, where utility = sum of rewards - sum of costs.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from itertools import permutations, combinations
import re

from ..benchmark.base import BaseBenchmark, Task, TaskResult, BenchmarkConfig, AgentType
from ..benchmark.registry import benchmark
from ..models.base import BaseModel, ModelResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


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


@benchmark(
    name="task_scheduling_utility_based",
    agent_type=AgentType.UTILITY_BASED,
    description="Assigns jobs to time slots to maximize utility."
)
class TaskSchedulingBenchmark(BaseBenchmark):
    """Benchmark for utility-based task scheduling."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)

    def get_tasks(self) -> List[Task]:
        """Get all tasks for the task scheduling benchmark."""
        tasks = []
        scenarios = self._get_scenarios()
        for i, scenario in enumerate(scenarios):
            optimal_schedule, optimal_utility = self._solve_schedule(scenario['jobs'], scenario['slots'])
            task = Task(
                task_id=f"task_scheduling_{i+1}",
                name=f"Task Scheduling: {scenario['name']}",
                description=scenario['description'],
                prompt=self._create_prompt(scenario),
                expected_output=str(optimal_schedule),
                metadata={
                    "scenario": scenario,
                    "optimal_utility": optimal_utility,
                    "optimal_schedule": str(optimal_schedule),
                    "difficulty": scenario['difficulty'],
                }
            )
            tasks.append(task)
        return tasks

    def _get_scenarios(self) -> List[Dict[str, Any]]:
        """Defines the scenarios for the benchmark."""
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

    def _create_prompt(self, scenario: Dict[str, Any]) -> str:
        """Create a prompt for the given scenario."""
        prompt = f"You are a task scheduler. Your goal is to assign jobs to time slots to maximize the total utility.\n"
        prompt += f"Utility is calculated as the sum of rewards from completed jobs minus the sum of costs from used time slots.\n\n"
        prompt += "JOBS:\n"
        for job in scenario['jobs']:
            prompt += f"- Job {job.id}: duration={job.duration}, reward={job.reward}"
            if job.deadline is not None:
                prompt += f", deadline={job.deadline}"
            if job.dependencies:
                prompt += f", dependencies={job.dependencies}"
            prompt += "\n"

        prompt += "\nTIME SLOTS:\n"
        for slot in scenario['slots']:
            prompt += f"- Slot {slot.id}: start={slot.start}, end={slot.end}, cost={slot.cost}\n"

        prompt += "\nCONSTRAINTS:\n"
        prompt += "- A job's duration must fit within its assigned time slot (end_time <= slot_end).\n"
        prompt += "- A job scheduled in a time slot is assumed to start at the beginning of that slot (slot.start).\n"
        prompt += "- If a job has a deadline, its completion time must be at or before the deadline.\n"
        prompt += "- If a job has dependencies, it cannot start until all its dependencies are completed. A job can start at the same time a dependency finishes (e.g., if dependency J2 of job J1 finishes at time 3, job J1 can start at time 3).\n"
        prompt += "- Each time slot can only be assigned one job.\n"
        prompt += "- A job can only be assigned to one time slot.\n"
        prompt += "- It is not required to use all jobs or all time slots; you can leave some unscheduled if it maximizes utility.\n\n"
        prompt += "Provide your final schedule as a list of (job_id, slot_id) pairs, like this: (J1, S2), (J2, S1), (J3, S3)\n"
        prompt += "Do not include any other text or explanation in your response.\n"
        return prompt

    def _parse_schedule(self, response_text: str) -> Schedule:
        """Parse the model's response to extract the schedule."""
        assignments = []
        # This verbose regex is designed to be readable and flexible.
        # It handles parentheses or square brackets, optional quotes,
        # and whitespace variations.
        pattern = r"""
            [\(\[]              # Match an opening parenthesis or bracket
            \s*                 # Optional whitespace
            '?\"?(\w+)'?\"?   # Capture the job ID (e.g., 'J1' or "J1" or J1)
            \s*,\s*             # Match the comma separator with optional whitespace
            '?\"?(\w+)'?\"?   # Capture the slot ID
            \s*                 # Optional whitespace
            [\)\]]              # Match a closing parenthesis or bracket
        """
        matches = re.findall(pattern, response_text, re.VERBOSE)
        for job_id, slot_id in matches:
            assignments.append(ScheduledJob(job_id=job_id, slot_id=slot_id))
        return Schedule(assignments=assignments)

    def _calculate_utility(self, schedule: Schedule, jobs: List[Job], slots: List[TimeSlot]) -> Tuple[int, bool]:
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
                if dep_id not in job_end_times:
                    return 0, False # Dependency not scheduled
                if start_time < job_end_times[dep_id]:
                    return 0, False # Dependency violation

        # If all checks pass, calculate utility
        total_reward = sum(job_map[a.job_id].reward for a in schedule.assignments)
        total_cost = sum(slot_map[a.slot_id].cost for a in schedule.assignments)
        
        return total_reward - total_cost, True

    def _solve_schedule(self, jobs: List[Job], slots: List[TimeSlot]) -> Tuple[Schedule, int]:
        """Finds the optimal schedule using brute-force permutation of all job subsets."""
        job_map = {job.id: job for job in jobs}
        
        best_schedule = Schedule(assignments=[])
        # Start with 0 utility for the case where scheduling any job is a net loss
        max_utility = 0

        job_ids = list(job_map.keys())
        slot_ids = [slot.id for slot in slots]

        # Iterate over all possible numbers of jobs to schedule
        for k in range(len(job_ids), 0, -1):
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
                        
                        utility, is_valid = self._calculate_utility(schedule, jobs, slots)
                        
                        if is_valid and utility > max_utility:
                            max_utility = utility
                            best_schedule = schedule

        return best_schedule, max_utility

    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        """Evaluate a single task scheduling task."""
        start_time = time.time()
        model_response = await model.generate(task.prompt)
        execution_time = time.time() - start_time

        schedule = self._parse_schedule(model_response.text)
        
        scenario = task.metadata['scenario']
        jobs = [Job(**asdict(j)) for j in scenario['jobs']]
        slots = [TimeSlot(**asdict(s)) for s in scenario['slots']]

        achieved_utility, is_valid = self._calculate_utility(schedule, jobs, slots)
        optimal_utility = task.metadata['optimal_utility']
        
        score = 0
        if optimal_utility > 0 and is_valid:
            score = achieved_utility / optimal_utility
        elif optimal_utility == 0 and achieved_utility == 0 and is_valid:
            score = 1.0

        return TaskResult(
            task_id=task.task_id,
            task_name=task.name,
            agent_type=self.agent_type,
            success=is_valid and score > 0,
            score=max(0, score),
            metrics={
                "achieved_utility": achieved_utility,
                "optimal_utility": optimal_utility,
                "is_valid_schedule": is_valid,
                "optimality_gap": optimal_utility - achieved_utility,
                "output_tokens": model_response.completion_tokens,
            },
            model_response=model_response,
            execution_time=execution_time,
        )

    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        """Calculate score for a task response."""
        # This is handled in evaluate_task for consistency.
        return 0.0 