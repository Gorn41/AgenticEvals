"""
Traffic Light Simple Reflex Agent benchmark for AgenticEvals.

This benchmark tests a model's ability to follow simple stimulus-response patterns,
specifically for traffic light scenarios where the model should respond with
appropriate actions based on the current signal.
"""

import time
import re
from typing import List, Dict, Any
from ..benchmark.base import BaseBenchmark, Task, TaskResult, BenchmarkConfig, AgentType
from ..benchmark.registry import benchmark
from ..models.base import BaseModel, ModelResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


@benchmark(
    name="traffic_light",
    agent_type=AgentType.SIMPLE_REFLEX,
    description="Traffic light response benchmark testing immediate rule-based responses"
)
class TrafficLightBenchmark(BaseBenchmark):
    """
    Simple reflex agent benchmark using traffic light scenarios.
    
    Tests the model's ability to provide immediate, rule-based responses
    to traffic light signals without requiring memory or planning.
    """
    
    def get_tasks(self) -> List[Task]:
        """Get all traffic light tasks."""
        tasks = []
        
        # Basic traffic light scenarios
        scenarios = [
            ("red", "stop", "red light means stop"),
            ("green", "go", "green light means go"),
            ("yellow", "caution", "yellow light means slow down/caution"),
            ("RED", "stop", "red light (uppercase) means stop"),
            ("GREEN", "go", "green light (uppercase) means go"),
            ("YELLOW", "caution", "yellow light (uppercase) means caution"),
        ]
        
        for i, (signal, expected, description) in enumerate(scenarios):
            task = Task(
                task_id=f"traffic_{i+1}",
                name=f"Traffic Light: {signal}",
                description=description,
                prompt=self._create_prompt(signal),
                expected_output=expected,
                evaluation_criteria={
                    "exact_match": True,
                    "case_insensitive": True,
                    "keywords": [expected]
                },
                metadata={
                    "signal": signal,
                    "expected_action": expected,
                    "difficulty": "basic"
                }
            )
            tasks.append(task)
        
        # Additional edge cases
        edge_cases = [
            ("red light", "stop", "red light with extra word"),
            ("green signal", "go", "green signal alternative phrasing"),
            ("amber", "caution", "amber instead of yellow"),
            ("flashing red", "stop", "flashing red light"),
            ("flashing yellow", "caution", "flashing yellow light"),
        ]
        
        for i, (signal, expected, description) in enumerate(edge_cases):
            task = Task(
                task_id=f"traffic_edge_{i+1}",
                name=f"Traffic Edge Case: {signal}",
                description=description,
                prompt=self._create_prompt(signal),
                expected_output=expected,
                evaluation_criteria={
                    "exact_match": False,
                    "case_insensitive": True,
                    "keywords": [expected]
                },
                metadata={
                    "signal": signal,
                    "expected_action": expected,
                    "difficulty": "edge_case"
                }
            )
            tasks.append(task)
        
        return tasks
    
    def _create_prompt(self, signal: str) -> str:
        """Create a prompt for the given traffic signal."""
        return f"""Traffic Signal Response Task:

You are at an intersection and see a {signal} traffic light.

Instructions:
- Respond with exactly one word: "stop", "go", or "caution"
- Do not provide explanations or additional text
- Base your response on standard traffic rules

Traffic light: {signal}

Your response:"""
    
    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        """Evaluate a single traffic light task."""
        start_time = time.time()
        
        try:
            # Generate model response
            model_response = await model.generate(task.prompt)
            execution_time = time.time() - start_time
            
            # Calculate score
            score = self.calculate_score(task, model_response)
            success = score > 0.5
            
            # Calculate detailed metrics
            metrics = self._calculate_detailed_metrics(task, model_response)
            
            return TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                agent_type=self.agent_type,
                success=success,
                score=score,
                metrics=metrics,
                model_response=model_response,
                execution_time=execution_time,
                metadata=task.metadata
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error evaluating task {task.task_id}: {e}")
            
            return TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                agent_type=self.agent_type,
                success=False,
                score=0.0,
                metrics={},
                execution_time=execution_time,
                error_message=str(e),
                metadata=task.metadata
            )
    
    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        """Calculate score for a traffic light task."""
        expected = task.expected_output.lower() if task.expected_output else ""
        response_text = model_response.text.strip().lower() if model_response.text else ""
        
        # Remove common punctuation
        response_text = re.sub(r'[.,!?;:]', '', response_text)
        
        # Exact match gets full score
        if response_text == expected:
            return 1.0
        
        # Partial match based on keywords
        if expected in response_text:
            # Check if the response is just the expected word plus minor additions
            words = response_text.split()
            if expected in words and len(words) <= 3:
                return 0.8
            elif expected in response_text:
                return 0.6
        
        # Check for synonyms
        synonyms = {
            "stop": ["halt", "wait", "brake"],
            "go": ["proceed", "drive", "move", "continue"],
            "caution": ["slow", "careful", "warning", "amber"]
        }
        
        if expected in synonyms:
            for synonym in synonyms[expected]:
                if synonym in response_text:
                    return 0.7
        
        return 0.0
    
    def _calculate_detailed_metrics(self, task: Task, model_response: ModelResponse) -> Dict[str, Any]:
        """Calculate detailed metrics for analysis."""
        response_text = model_response.text.strip() if model_response.text else ""
        expected = task.expected_output.lower() if task.expected_output else ""
        
        # Response analysis
        word_count = len(response_text.split())
        char_count = len(response_text)
        contains_expected = expected in response_text.lower() if expected and response_text else False
        
        # Check if response follows instructions (one word only)
        follows_instructions = word_count == 1
        
        # Extract actual response word
        first_word = response_text.split()[0].lower() if response_text.split() else ""
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "follows_instructions": follows_instructions,
            "contains_expected": contains_expected,
            "first_word": first_word,
            "exact_match": response_text.lower().strip() == expected,
            "response_latency": model_response.latency,
            "output_tokens": model_response.completion_tokens if model_response else 0,
        } 