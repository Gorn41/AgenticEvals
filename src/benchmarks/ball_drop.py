"""
Learning Agent Benchmark: Ball Drop on a Moving Target.

This benchmark evaluates a model's ability to learn and adapt over a series of
episodes. The agent must learn the physics of a simple simulation to determine
the optimal time to drop a ball to hit a moving target.

The agent's learning is assessed based on its ability to improve its performance
(reduce miss distance) over time and to generalize to unseen scenarios.
"""

import time
import random
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import asyncio
import re # Added for regex parsing

from ..benchmark.base import BaseBenchmark, Task, TaskResult, BenchmarkConfig, AgentType
from ..benchmark.registry import benchmark
from ..models.base import BaseModel, ModelResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class EpisodeState:
    """Represents the state of a single ball drop episode."""
    episode_id: Union[int, str]
    initial_offset: float
    target_velocity: float
    ball_position: Tuple[float, float] = (0.0, 10.0) # (x, y)
    target_position: float = 0.0 # x position
    time_step: int = 0
    drop_time: Optional[float] = None
    outcome: Optional[float] = None # Miss distance

@dataclass
class AgentAction:
    """Represents the action taken by the agent."""
    predicted_drop_time: Optional[float]
    reasoning: str = ""

@dataclass
class TrialResult:
    """Stores the result of a single trial for an agent."""
    episode_id: Union[int, str]
    initial_offset: float
    target_velocity: float
    drop_time: Optional[float]
    displacement: float

@benchmark(
    name="ball_drop",
    agent_type=AgentType.LEARNING,
    description="Tests an agent's ability to learn through trial and error in a physics-based task."
)
class BallDropBenchmark(BaseBenchmark):
    """Benchmark for learning to drop a ball on a moving target."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.simulation_height = 10.0
        self.fall_time = 2.0 # Fixed fall time of 2 seconds
        self.environment_width = 40.0 # Environment from -20 to 20
        self.target_start_pos = -20.0
        self.memory: List[TrialResult] = []

    def get_tasks(self) -> List[Task]:
        """Generate a series of learning tasks, each representing a stage of learning."""
        tasks = []
        
        # Generate a consistent set of training and evaluation scenarios
        all_training_scenarios = self._generate_scenarios(50, "train")
        evaluation_scenarios = self._generate_scenarios(10, "eval")
        
        eval_interval = 5
        num_training_scenarios = len(all_training_scenarios)

        for i in range(eval_interval, num_training_scenarios + 1, eval_interval):
            # The new scenarios for this stage
            start_index = i - eval_interval
            new_training_scenarios = all_training_scenarios[start_index:i]
            
            task = Task(
                task_id=f"learning_stage_{i}_episodes",
                name=f"Ball Drop after {i} Training Episodes",
                description=f"Evaluate agent performance after training on {i} total episodes.",
                prompt="", # The prompt is generated within the evaluation logic
                metadata={
                    "new_training_scenarios": new_training_scenarios,
                    "evaluation_scenarios": evaluation_scenarios,
                    "total_episodes_so_far": i
                }
            )
            tasks.append(task)
            
        return tasks

    def _generate_scenarios(self, num_scenarios: int, scenario_type: str) -> List[Dict[str, float]]:
        """Generate a set of initial conditions for the ball drop task."""
        scenarios = []
        # Use a fixed seed for reproducibility based on scenario type
        seed = 123 if scenario_type == "train" else 456
        rng = random.Random(seed)
        
        for i in range(num_scenarios):
            # Target moves right, so velocity is positive
            velocity = round(rng.uniform(1.5, 2.0), 1)
            
            # To ensure the task is always solvable, the ball's offset must be
            # to the right of the target's starting position and within reach.
            min_offset = self.target_start_pos + 15 
            max_offset = self.environment_width / 2 - 15
            
            offset = round(rng.uniform(min_offset, max_offset), 0)
            scenarios.append({"initial_offset": offset, "target_velocity": velocity})
        return scenarios

    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        """Evaluate the agent's learning over a single stage of episodes."""
        start_time = time.time()
        
        training_scenarios = task.metadata["new_training_scenarios"]
        evaluation_scenarios = task.metadata["evaluation_scenarios"]
        
        total_output_tokens = 0
        total_api_calls = 0

        # --- Training Phase ---
        for i, scenario in enumerate(training_scenarios):
            # The episode_id should be based on the total number of episodes so far
            episode_id = task.metadata["total_episodes_so_far"] - len(training_scenarios) + i
            episode_result, metrics = await self._run_episode(
                episode_id, scenario, model, self.memory
            )
            self._update_memory(self.memory, episode_result)
            total_output_tokens += metrics["output_tokens"]
            total_api_calls += metrics["api_calls"]

        # --- Evaluation Phase ---
        eval_results = []
        for i, eval_scenario in enumerate(evaluation_scenarios):
            eval_episode_result, eval_metrics = await self._run_episode(
                f"eval_after_{task.metadata['total_episodes_so_far']}_{i}",
                eval_scenario,
                model,
                self.memory,
                is_eval=True
            )
            eval_results.append(eval_episode_result)
            total_output_tokens += eval_metrics["output_tokens"]
            total_api_calls += eval_metrics["api_calls"]

        # --- Final Calculation for this task ---
        total_duration = time.time() - start_time
        actual_execution_time = total_duration - (total_api_calls * 15)

        # Calculate score for each eval episode and average them
        episode_scores = []
        for result in eval_results:
            worst_possible_miss = max(
                abs(result.initial_offset - self.target_start_pos),
                abs(self.environment_width / 2 - result.initial_offset)
            )
            score = max(0, 1 - (abs(result.displacement) / worst_possible_miss)) if worst_possible_miss > 0 else 1.0
            episode_scores.append(score)
        
        final_score = sum(episode_scores) / len(episode_scores) if episode_scores else 0.0
        
        avg_eval_displacement = sum(abs(r.displacement) for r in eval_results) / len(eval_results) if eval_results else float('inf')
        
        summary_metrics = {
            "output_tokens": total_output_tokens,
            "total_api_calls": total_api_calls,
            "actual_execution_time": actual_execution_time,
            "avg_eval_displacement": avg_eval_displacement,
            "num_training_episodes": len(self.memory),
        }
        
        return TaskResult(
            task_id=task.task_id,
            task_name=task.name,
            agent_type=self.agent_type,
            success=final_score > 0.7,
            score=final_score,
            metrics=summary_metrics,
            execution_time=actual_execution_time
        )

    async def _run_episode(self, episode_id: Union[int, str], scenario: Dict[str, float], model: BaseModel, memory: List[TrialResult], is_eval: bool = False) -> Tuple[TrialResult, Dict[str, int]]:
        """Runs a single episode of the ball drop task."""
        state = EpisodeState(
            episode_id=episode_id,
            initial_offset=scenario["initial_offset"],
            target_velocity=scenario["target_velocity"],
            ball_position=(scenario["initial_offset"], self.simulation_height),
            target_position=self.target_start_pos,
        )

        prompt = self._create_prompt(state, memory)
        response = None
        action = None
        
        try:
            response = await model.generate(prompt)
            await asyncio.sleep(15)
            action = self._parse_action(response.text)
        except Exception as e:
            logger.error(f"API call failed during episode {episode_id}: {e}")
            # Action remains None if API call fails

        if action and action.predicted_drop_time is not None:
            # Use the predicted drop time directly
            state.drop_time = action.predicted_drop_time
            
            travel_time = state.drop_time + self.fall_time
            target_final_pos = state.target_position + (state.target_velocity * travel_time)
            target_final_pos = max(self.target_start_pos, min(target_final_pos, self.environment_width / 2))
            
            displacement = state.ball_position[0] - target_final_pos
        else:
            # Penalize with worst possible miss if action is None or drop time is None
            left_miss = state.ball_position[0] - self.target_start_pos
            right_miss = state.ball_position[0] - (self.environment_width / 2)
            displacement = left_miss if abs(left_miss) > abs(right_miss) else right_miss
            state.drop_time = None # Ensure drop time is None on failure

        state.outcome = abs(displacement)

        return TrialResult(
            episode_id=episode_id,
            initial_offset=state.initial_offset,
            target_velocity=state.target_velocity,
            drop_time=state.drop_time,
            displacement=displacement
        ), {
            "output_tokens": response.completion_tokens if response and response.completion_tokens else 0,
            "api_calls": 1
        }

    def _parse_action(self, response_text: str) -> AgentAction:
        """
        Parses the model's response to get the agent's action.
        It first looks for the specific format [prediction: <time>], then
        falls back to the last float in the string if the format is not found.
        """
        # 1. Primary parsing: Look for the specific format [prediction: <float>]
        # The units (s, sec, secs, seconds) are optional and ignored.
        primary_match = re.search(r'\[prediction:\s*([-+]?\d*\.?\d+)\s*(?:s|sec|secs|seconds)?\]', response_text, re.IGNORECASE)
        if primary_match:
            try:
                drop_time = float(primary_match.group(1))
                return AgentAction(predicted_drop_time=drop_time, reasoning=response_text)
            except (ValueError, IndexError):
                pass

        # 2. Fallback parsing: Find the last floating-point number in the response.
        fallback_matches = re.findall(r'[-+]?\d*\.?\d+', response_text)
        if fallback_matches:
            try:
                last_number_str = fallback_matches[-1]
                drop_time = float(last_number_str)
                return AgentAction(predicted_drop_time=drop_time, reasoning=response_text)
            except (ValueError, IndexError):
                pass

        # 3. If no number is found at all, return None for drop time.
        logger.error(f"parsing failed for response: {response_text}")
        return AgentAction(predicted_drop_time=None, reasoning=response_text)

    def _update_memory(self, memory: List[TrialResult], new_result: TrialResult):
        """Updates the agent's memory with the latest trial result."""
        memory.append(new_result)

    def _create_prompt(self, state: EpisodeState, memory: List[TrialResult]) -> str:
        """Creates the prompt for the agent based on the current state and memory."""
        prompt = f"""You are an agent learning to drop a ball on a moving target.
The environment is a 2D plane. The ball drops from a fixed height.

Environment Details:
- The horizontal axis ranges from {self.target_start_pos}m to {self.environment_width / 2}m.
- The ball is at a fixed height of {self.simulation_height}m.
- The target starts at the far left (x = {self.target_start_pos}m) and moves to the right.
- The target will stop if it hits the right boundary.
- Do not assume standard Earth physics (e.g., gravity). The physics of this world must be learned from experience.

Current State:
- Ball's horizontal starting position (offset): {state.initial_offset:.2f}m (0.0m is the center, positive offset is right of center and negative offset is left of center).
- Target's starting x-position: {state.target_position:.2f}m
- Target's velocity: {state.target_velocity:.2f}m/s (positive, moves to the right).
- Current time: 0.0s

Your task is to predict the optimal time in seconds (s) to drop the ball. I.e. the time at which the ball will land closest to the target.

Feedback from Previous Trials:
Displacement is the horizontal distance (in meters) between the ball and the target when the ball lands.
A positive displacement means the ball landed to the right of the target.
A negative displacement means the ball landed to the left of the target.
"""
        if not memory:
            prompt += "- No previous trials.\n"
        else:
            # Provide the full history of trials
            for trial in memory:
                prompt += f"- Trial {trial.episode_id}: offset={trial.initial_offset:.2f}m, vel={trial.target_velocity:.2f}m/s, drop_time={trial.drop_time:.2f}s, displacement={trial.displacement:.2f}m\n"

        prompt += "\nBased on the initial state and previous trials, what is the optimal time in seconds to drop the ball? Format your final answer as `[prediction: <time>]`, where `<time>` is a floating-point number with up to two decimal places (e.g., `[prediction: 1.23]`)."
        return prompt

    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        """
        Calculate score for a task response. For this benchmark, scoring is
        handled within evaluate_task, as it depends on performance over
        multiple episodes.
        """
        return 0.0 