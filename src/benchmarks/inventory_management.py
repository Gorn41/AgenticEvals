


"""
Inventory Management Model-Based Reflex Agent benchmark for AgenticEvals.

This benchmark tests a model's ability to manage inventory levels for multiple items,
each with its own depletion pattern and minimum threshold. The agent must deduce the
patterns and make optimal restocking decisions to minimize stock shortages over a
fixed number of turns.
"""

import time
import asyncio
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
import random
import json
import re # Added for regex fallback in _parse_restock_decision
import numpy as np

from ..benchmark.base import BaseBenchmark, Task, TaskResult, BenchmarkConfig, AgentType
from ..benchmark.registry import benchmark
from ..models.base import BaseModel, ModelResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ItemState:
    """Represents the state of a single inventory item."""
    name: str
    initial_stock: int
    current_stock: int
    threshold: int
    depletion_pattern: List[int]


@dataclass
class InventoryState:
    """Represents the overall state of the inventory."""
    items: Dict[str, ItemState]
    turn: int
    total_restock_capacity: int
    total_tokens_used: int
    restock_history: List[Dict[str, int]] = field(default_factory=list)

    @classmethod
    def from_scenario(cls, scenario_data: Dict[str, Any]) -> 'InventoryState':
        """Creates an InventoryState object from a scenario definition."""
        items = {
            name: ItemState(
                name=name,
                initial_stock=details["initial"],
                current_stock=details["initial"],
                threshold=details["threshold"],
                depletion_pattern=details["pattern"],
            )
            for name, details in scenario_data["items"].items()
        }
        return cls(
            items=items,
            turn=0,
            total_restock_capacity=scenario_data["restock_capacity"],
            total_tokens_used=0,
            restock_history=[]
        )


@benchmark(
    name="inventory_management",
    agent_type=AgentType.MODEL_BASED_REFLEX,
    description="Inventory management benchmark testing pattern detection and resource allocation."
)
class InventoryManagementBenchmark(BaseBenchmark):
    """
    Model-based reflex agent benchmark for inventory management.
    """

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)

    def _create_scenarios(self) -> List[Dict[str, Any]]:
        """Creates a list of scenarios for the inventory management tasks."""
        # Restock capacity is now per-turn
        scenarios = [
            {
                "name": "Scenario 1: Simple constant depletion",
                "items": {
                    "A": {"initial": 20, "threshold": 10, "pattern": [2] * 10},
                    "B": {"initial": 15, "threshold": 5, "pattern": [1] * 10},
                },
                "restock_capacity": 1,
            },
            {
                "name": "Scenario 2: Alternating and increasing depletion",
                "items": {
                    "A": {"initial": 30, "threshold": 10, "pattern": [2, 4] * 5},
                    "B": {"initial": 25, "threshold": 10, "pattern": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]},
                    "C": {"initial": 20, "threshold": 15, "pattern": [1] * 10},
                },
                "restock_capacity": 3,
            },
            {
                "name": "Scenario 3: High initial stock, aggressive depletion",
                "items": {
                    "A": {"initial": 50, "threshold": 20, "pattern": [5, 6, 7, 5, 6, 7, 5, 6, 7, 5]},
                    "B": {"initial": 60, "threshold": 30, "pattern": [4, 8] * 5},
                },
                "restock_capacity": 4,
            },
            {
                "name": "Scenario 4: Low capacity, tight thresholds",
                "items": {
                    "A": {"initial": 15, "threshold": 10, "pattern": [1, 2] * 5},
                    "B": {"initial": 12, "threshold": 8, "pattern": [1] * 10},
                    "C": {"initial": 10, "threshold": 5, "pattern": [1, 0] * 5},
                },
                "restock_capacity": 2,
            },
            {
                "name": "Scenario 5: Increasing Complexity and More Items",
                "items": {
                    "A": {"initial": 30, "threshold": 10, "pattern": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                    "B": {"initial": 40, "threshold": 15, "pattern": [1, 1, 2, 3, 5, 8, 1, 1, 2, 3]},
                    "C": {"initial": 25, "threshold": 10, "pattern": [2, 2, 4, 4, 6, 6, 4, 4, 2, 2]},
                    "D": {"initial": 20, "threshold": 18, "pattern": [1] * 10},
                },
                "restock_capacity": 6,
            },
            {
                "name": "Scenario 6: The Decoy and The Sleeper",
                "items": {
                    "A": {"initial": 20, "threshold": 5, "pattern": [5, 4, 3, 2, 1, 1, 1, 1, 1, 1]},
                    "B": {"initial": 30, "threshold": 10, "pattern": [1, 1, 1, 2, 2, 3, 4, 5, 6, 7]},
                    "C": {"initial": 25, "threshold": 10, "pattern": [1, 5] * 5},
                },
                "restock_capacity": 3,
            },
            {
                "name": "Scenario 7: Extreme Volatility, Zero Margin for Error",
                "items": {
                    "A": {"initial": 100, "threshold": 20, "pattern": [5, 10, 15, 2, 5, 10, 15, 2, 5, 10]},
                    "B": {"initial": 80, "threshold": 10, "pattern": [10, 1, 10, 1, 10, 1, 10, 1, 10, 1]},
                    "C": {"initial": 60, "threshold": 50, "pattern": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                    "D": {"initial": 40, "threshold": 30, "pattern": [2] * 10},
                },
                "restock_capacity": 15,
            },
            {
                "name": "Scenario 8: Immediate Danger (Starts Below Threshold)",
                "items": {
                    "A": {"initial": 5, "threshold": 15, "pattern": [2] * 10},
                    "B": {"initial": 20, "threshold": 10, "pattern": [1] * 10},
                    "C": {"initial": 25, "threshold": 10, "pattern": [2, 1] * 5},
                },
                "restock_capacity": 4,
            },
            {
                "name": "Scenario 9: Many Simple Items",
                "items": {
                    "A": {"initial": 15, "threshold": 10, "pattern": [1] * 10},
                    "B": {"initial": 12, "threshold": 8, "pattern": [1] * 10},
                    "C": {"initial": 10, "threshold": 5, "pattern": [1] * 10},
                    "D": {"initial": 20, "threshold": 15, "pattern": [1] * 10},
                    "E": {"initial": 18, "threshold": 12, "pattern": [1] * 10},
                },
                "restock_capacity": 4,
            },
            {
                "name": "Scenario 10",
                "items": {
                    "A": {"initial": 40, "threshold": 10, "pattern": [1, 1, 1, 1, 1, 5, 6, 7, 8, 9]},
                    "B": {"initial": 30, "threshold": 20, "pattern": [2] * 10},
                },
                "restock_capacity": 3,
            },
            {
                "name": "Scenario 11: Advanced Patterns (Fibonacci-like)",
                "items": {
                    "A": {"initial": 60, "threshold": 20, "pattern": [1, 2, 3, 5, 8, 5, 3, 2, 1, 1]},
                    "B": {"initial": 80, "threshold": 20, "pattern": [1, 2, 4, 8, 4, 2, 1, 1, 1, 1]},
                    "C": {"initial": 30, "threshold": 25, "pattern": [1] * 10},
                },
                "restock_capacity": 7,
            },
            {
                "name": "Scenario 12: Six Items, Starts Below",
                "items": {
                    "A": {"initial": 10, "threshold": 15, "pattern": [1] * 10},
                    "B": {"initial": 20, "threshold": 10, "pattern": [2, 3] * 5},
                    "C": {"initial": 30, "threshold": 20, "pattern": [1, 1, 1, 1, 5, 5, 5, 5, 1, 1]},
                    "D": {"initial": 40, "threshold": 30, "pattern": [1] * 10},
                    "E": {"initial": 50, "threshold": 40, "pattern": [1] * 10},
                    "F": {"initial": 60, "threshold": 50, "pattern": [1] * 10},
                },
                "restock_capacity": 10,
            },
            {
                "name": "Scenario 13: Illusion of Safety",
                "items": {
                    "A": {"initial": 100, "threshold": 20, "pattern": [8, 9, 10, 8, 9, 10, 8, 9, 10, 8]},
                    "B": {"initial": 120, "threshold": 30, "pattern": [10, 11, 12, 10, 11, 12, 10, 11, 12, 10]},
                    "C": {"initial": 80, "threshold": 15, "pattern": [7] * 10},
                },
                "restock_capacity": 9,
            },
            {
                "name": "Scenario 14: Just-In-Time Management",
                "items": {
                    "A": {"initial": 10, "threshold": 8, "pattern": [1, 2] * 5},
                    "B": {"initial": 8, "threshold": 6, "pattern": [1] * 10},
                    "C": {"initial": 6, "threshold": 4, "pattern": [1, 0] * 5},
                },
                "restock_capacity": 3,
            },
            {
                "name": "Scenario 15: The Final Exam",
                "items": {
                    "A": {"initial": 20, "threshold": 25, "pattern": [1, 2, 3, 2, 1, 1, 2, 3, 2, 1]},
                    "B": {"initial": 50, "threshold": 10, "pattern": [2, 4, 6, 8, 6, 4, 2, 1, 1, 1]},
                    "C": {"initial": 30, "threshold": 20, "pattern": [1] * 10},
                    "D": {"initial": 40, "threshold": 10, "pattern": [3] * 10},
                    "E": {"initial": 60, "threshold": 20, "pattern": [1, 1, 2, 2, 3, 3, 8, 9, 10, 11]},
                },
                "restock_capacity": 8,
            },
        ]
        return scenarios

    def get_tasks(self) -> List[Task]:
        """Get all inventory management tasks."""
        scenarios = self._create_scenarios()
        return [
            Task(
                task_id=f"inventory_{i+1}",
                name=f"Scenario {i+1}: {s['name']}",
                description=f"Manage inventory for {len(s['items'])} items over 10 turns.",
                prompt=self._create_initial_prompt(s),
                expected_output="Maintain inventory above thresholds.",
                evaluation_criteria={"method": "per_turn_normalized"},
                metadata=s
            ) for i, s in enumerate(scenarios)
        ]

    def _get_rules_text(self, state: InventoryState) -> str:
        """Generates the rules text block for prompts."""
        return f"""
Rules:
- The restock capacity is {state.total_restock_capacity} units and is constant for every turn. This restock capacity is shared among all items. So the sum of the restocked amounts across all items this turn cannot exceed this turn's capacity.
- Each turn, you first decide your restock plan, and then the items will deplete based on consistent, hidden patterns.
- Your goal is to restock items so that AFTER this turn's depletion, the stock for each item is as close as possible to its threshold, ideally at or above it.
- The total amount you restock across all items cannot exceed this turn's capacity. For example, if capacity is 10, a restock of `{{"A": 5, "B": 5}}` is VALID (5+5<=10), and a restock of `{{"A": 1, "B": 8}}` is also VALID (1+8<=10) but a restock of`{{"A": 5, "B": 6}}` is INVALID (5+6>10).
- Your response MUST be in JSON format, like: {{"A": 5, "B": 0}}
- Your response must be a single JSON object for the current turn only.
"""

    def _create_initial_prompt(self, scenario_data: Dict[str, Any]) -> str:
        """Creates the initial prompt for the inventory management task."""
        state = InventoryState.from_scenario(scenario_data)
        item_details = "\n".join(
            [f"- {item.name}: initial_stock={item.current_stock}, threshold={item.threshold}" for item in state.items.values()]
        )
        rules_text = self._get_rules_text(state)

        return f"""
Welcome to the Inventory Management Challenge!

This is a 10-turn simulation where you must manage inventory levels for several items.

Your task is to deduce hidden depletion patterns and make optimal restocking decisions each turn to keep stock levels above their minimum thresholds.

Initial State:
{item_details}
{rules_text}
IMPORTANT: If your restock plan is invalid (e.g., exceeds capacity), NO restocking will occur that turn, you will receive the worst possible score, and be penalized. Avoid this at all costs.

Turn 1/10:
Analyze the initial state and provide your restock plan for the first turn.
Your response must be a single JSON object.
"""

    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        """Evaluates a single inventory management task."""
        start_time = time.time()
        wait_seconds = float(self.config.additional_params.get("wait_seconds", 15.0)) if getattr(self, "config", None) else 15.0
        state = InventoryState.from_scenario(task.metadata)
        
        initial_prompt = self._create_initial_prompt(task.metadata)
        
        turn_scores = []
        total_tokens_used = 0
        delays_applied = 0
        accumulated_call_time = 0.0
        was_invalid_last_turn = False
        depletion_history: Dict[str, List[int]] = {name: [] for name in state.items.keys()}

        for turn in range(1, 11):
            state.turn = turn
            
            # Add delay for turns after the first one
            if turn > 1 and wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
                delays_applied += 1

            # Prepare the prompt for the current turn
            if turn == 1:
                prompt = initial_prompt
            else:
                item_details = "\n".join(
                    [f"- {item.name}: current_stock={item.current_stock}, threshold={item.threshold}" for item in state.items.values()]
                )
                depletion_history_str = "Depletion History (amount depleted each turn):\n" + "\n".join(
                    [f"- {name}: {', '.join(map(str, depletions))}" for name, depletions in depletion_history.items() if depletions]
                )
                rules_text = self._get_rules_text(state)

                prompt = f"""
Turn {turn}/10 of the Inventory Management Challenge.

{depletion_history_str}

Current State (before your restock this turn):
{item_details}
{rules_text}
"""
                if was_invalid_last_turn:
                    prompt += "\nIMPORTANT: Your last decision was invalid because it exceeded the restock capacity. NO restocking was performed. Make a valid decision this turn."
                
                prompt += "\nYour task is to decide which items to restock for this turn.\nYour response must be a single JSON object."


            # Generate model response
            call_started_at = time.time()
            response = await model.generate(prompt)
            accumulated_call_time += (time.time() - call_started_at)
            if response.completion_tokens:
                total_tokens_used += response.completion_tokens

            # Parse decision
            restock_decision = self._parse_restock_decision(response.text)

            # Calculate score for the turn
            turn_score = self._calculate_turn_score(state, restock_decision)
            turn_scores.append(turn_score)

            # Check if the decision was invalid for the next turn's prompt
            was_invalid_last_turn = sum(restock_decision.values()) > state.total_restock_capacity

            # Apply valid restock decisions
            if not was_invalid_last_turn:
                for item_name, restock_amount in restock_decision.items():
                    if item_name in state.items:
                        state.items[item_name].current_stock += restock_amount
            
            # Apply depletion
            stock_before_depletion = {name: item.current_stock for name, item in state.items.items()}
            for item in state.items.values():
                depletion_amount = item.depletion_pattern[turn - 1]
                item.current_stock = max(0, item.current_stock - depletion_amount)

            # Record actual depletion
            for name, item in state.items.items():
                actual_depletion = stock_before_depletion[name] - item.current_stock
                depletion_history[name].append(actual_depletion)

        # Final calculations
        # Sum of API call durations across turns; excludes intentional sleeps
        net_execution_time = accumulated_call_time
        final_score = self._calculate_final_score(turn_scores)
        success = final_score > 0.7  # Define success as an average score > 0.7

        return TaskResult(
            task_id=task.task_id,
            task_name=task.name,
            agent_type=self.agent_type,
            success=success,
            score=final_score,
            metrics={
                "output_tokens": total_tokens_used,
                "turn_scores": turn_scores
            },
            model_response=ModelResponse(
                text=f"Final score: {final_score:.3f}",
                total_tokens=total_tokens_used
            ),
            execution_time=net_execution_time,
            metadata={
                **task.metadata,
                "final_inventory_state": {name: item.current_stock for name, item in state.items.items()},
            }
        )

    def _parse_restock_decision(self, response_text: str) -> Dict[str, int]:
        """
        Parses the restock decision from the model's response.
        Expects a single, clean JSON object.
        """
        try:
            # Attempt to load the entire string as a JSON object
            decision = json.loads(response_text)
            if isinstance(decision, dict):
                # Validate that keys are strings and values are integers
                if all(isinstance(k, str) and isinstance(v, int) for k, v in decision.items()):
                    return {k: v for k, v in decision.items() if v > 0} # Filter out zero values
        except json.JSONDecodeError:
            # If direct parsing fails, try to find JSON within the text as a fallback
            pass

        # Fallback to regex if direct parsing fails or format is wrong
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                decision = json.loads(json_str)
                if isinstance(decision, dict):
                     # Validate that keys are strings and values are integers
                    if all(isinstance(k, str) and isinstance(v, int) for k, v in decision.items()):
                        return {k: v for k, v in decision.items() if v > 0} # Filter out zero values
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Could not parse JSON from response: {response_text}")
            return {}
        
        logger.warning(f"Could not parse a valid single JSON object from response: {response_text}")
        return {}

    def _calculate_achieved_sum_after_depletion(self, state: InventoryState, decision: Dict[str, int]) -> int:
        """
        Helper to calculate the sum of min(stock, threshold) after a restock decision and depletion.
        """
        # Create a temporary copy of the state to simulate the outcome
        temp_items = {
            name: ItemState(
                name=item.name,
                initial_stock=item.initial_stock,
                current_stock=item.current_stock,
                threshold=item.threshold,
                depletion_pattern=item.depletion_pattern
            ) for name, item in state.items.items()
        }

        # 1. Apply the restock decision
        for item_name, restock_amount in decision.items():
            if item_name in temp_items:
                temp_items[item_name].current_stock += restock_amount
        
        # 2. Apply depletion for the current turn
        for item in temp_items.values():
            depletion = item.depletion_pattern[state.turn - 1]
            item.current_stock = max(0, item.current_stock - depletion)
            
        # 3. Calculate the sum of min(stock, threshold)
        achieved_sum = sum(min(item.current_stock, item.threshold) for item in temp_items.values())
        return achieved_sum

    def _calculate_turn_score(self, state: InventoryState, model_decision: Dict[str, int]) -> float:
        """
        Calculates a normalized score for the model's decision in a single turn.
        The score is 1 for the best possible decision and 0 for the worst, based on post-depletion outcomes.
        """
        capacity = state.total_restock_capacity

        # 1. Validate the model's decision against capacity
        total_restocked = sum(model_decision.values())
        if total_restocked > capacity:
            return 0.0  # Immediate penalty for exceeding capacity

        # 2. Calculate the score for the model's actual decision
        model_achieved_sum = self._calculate_achieved_sum_after_depletion(state, model_decision)

        # 3. Determine the BEST possible decision
        # First, determine the post-depletion needs for every item
        items_in_need = []
        for item in state.items.values():
            stock_after_depletion_no_restock = item.current_stock - item.depletion_pattern[state.turn - 1]
            needed = item.threshold - stock_after_depletion_no_restock
            if needed > 0:
                items_in_need.append({'name': item.name, 'needed': needed})
        
        # Sort by the most needed items first
        sorted_items_in_need = sorted(items_in_need, key=lambda x: x['needed'], reverse=True)

        # Greedily allocate capacity to the most needy items
        best_decision = {item.name: 0 for item in state.items.values()}
        remaining_capacity_best = capacity
        for item_info in sorted_items_in_need:
            item_name = item_info['name']
            needed = item_info['needed']
            
            to_add = min(remaining_capacity_best, needed)
            best_decision[item_name] = to_add
            remaining_capacity_best -= to_add
            
            if remaining_capacity_best <= 0:
                break
                
        best_achieved_sum = self._calculate_achieved_sum_after_depletion(state, best_decision)

        # 4. Determine the WORST possible decision (i.e., doing nothing)
        worst_decision = {}
        worst_achieved_sum = self._calculate_achieved_sum_after_depletion(state, worst_decision)

        # 5. Normalize the model's score
        denominator = best_achieved_sum - worst_achieved_sum
        if denominator == 0:
            # If best and worst outcomes are the same, any valid decision is perfect.
            return 1.0
        
        score = (model_achieved_sum - worst_achieved_sum) / denominator
        return max(0.0, min(1.0, score))

    def _calculate_final_score(self, turn_scores: List[float]) -> float:
        """Calculates the final score as the average of turn scores."""
        if not turn_scores:
            return 0.0
        return sum(turn_scores) / len(turn_scores)
        
    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        """Calculate the final score for the task."""
        # The primary scoring is done per turn in evaluate_task.
        # This method returns the final averaged score passed in the model_response.
        if "final_score" in model_response.metadata:
            return model_response.metadata["final_score"]
        return 0.0


