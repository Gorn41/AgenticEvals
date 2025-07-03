"""
Base benchmark interface for LLM-AgentTypeEval.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time
import uuid

from pydantic import BaseModel as PydanticBaseModel
from ..models.base import BaseModel, ModelResponse


class AgentType(Enum):
    """Types of AI agents in the benchmark."""
    SIMPLE_REFLEX = "simple_reflex"
    MODEL_BASED_REFLEX = "model_based_reflex"
    GOAL_BASED = "goal_based"
    UTILITY_BASED = "utility_based"
    LEARNING = "learning"


class TaskResult(PydanticBaseModel):
    """Result from a single task execution."""
    task_id: str
    task_name: str
    agent_type: AgentType
    success: bool
    score: float
    metrics: Dict[str, Any]
    model_response: Optional[ModelResponse] = None
    execution_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


class BenchmarkResult(PydanticBaseModel):
    """Result from a complete benchmark execution."""
    benchmark_id: str
    benchmark_name: str
    agent_type: AgentType
    model_name: str
    timestamp: str
    task_results: List[TaskResult]
    overall_score: float
    summary_metrics: Dict[str, Any]
    execution_metadata: Dict[str, Any] = {}
    
    def get_success_rate(self) -> float:
        """Calculate overall success rate."""
        if not self.task_results:
            return 0.0
        return sum(1 for result in self.task_results if result.success) / len(self.task_results)
    
    def get_average_score(self) -> float:
        """Calculate average score across all tasks."""
        if not self.task_results:
            return 0.0
        return sum(result.score for result in self.task_results) / len(self.task_results)
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, float]:
        """Get summary statistics for a specific metric."""
        values = []
        for result in self.task_results:
            if metric_name in result.metrics:
                values.append(result.metrics[metric_name])
        
        if not values:
            return {}
        
        return {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }


class BenchmarkConfig(PydanticBaseModel):
    """Configuration for benchmark execution."""
    benchmark_name: str
    agent_type: AgentType
    num_tasks: Optional[int] = None
    random_seed: Optional[int] = None
    timeout_seconds: Optional[float] = None
    max_retries: int = 3
    collect_detailed_metrics: bool = True
    save_responses: bool = True
    additional_params: Dict[str, Any] = {}


@dataclass
class Task:
    """Represents a single task in a benchmark."""
    task_id: str
    name: str
    description: str
    prompt: str
    expected_output: Optional[str] = None
    evaluation_criteria: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.evaluation_criteria is None:
            self.evaluation_criteria = {}
        if self.metadata is None:
            self.metadata = {}


class BaseBenchmark(ABC):
    """Abstract base class for all agent type benchmarks."""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize the benchmark with given configuration."""
        self.config = config
        self.benchmark_name = config.benchmark_name
        self.agent_type = config.agent_type
        
    @abstractmethod
    def get_tasks(self) -> List[Task]:
        """Get all tasks for this benchmark."""
        pass
    
    @abstractmethod
    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        """Evaluate a single task with the given model."""
        pass
    
    @abstractmethod
    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        """Calculate score for a task based on model response."""
        pass
    
    async def run_benchmark(self, model: BaseModel) -> BenchmarkResult:
        """Run the complete benchmark on the given model."""
        benchmark_id = str(uuid.uuid4())
        start_time = time.time()
        
        tasks = self.get_tasks()
        if self.config.num_tasks:
            tasks = tasks[:self.config.num_tasks]
        
        task_results = []
        
        for task in tasks:
            try:
                result = await self.evaluate_task(task, model)
                task_results.append(result)
            except Exception as e:
                # Create a failed task result
                error_result = TaskResult(
                    task_id=task.task_id,
                    task_name=task.name,
                    agent_type=self.agent_type,
                    success=False,
                    score=0.0,
                    metrics={},
                    execution_time=0.0,
                    error_message=str(e)
                )
                task_results.append(error_result)
        
        # Calculate overall metrics
        overall_score = self._calculate_overall_score(task_results)
        summary_metrics = self._calculate_summary_metrics(task_results)
        
        execution_time = time.time() - start_time
        
        return BenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_name=self.benchmark_name,
            agent_type=self.agent_type,
            model_name=model.model_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            task_results=task_results,
            overall_score=overall_score,
            summary_metrics=summary_metrics,
            execution_metadata={
                "total_execution_time": execution_time,
                "num_tasks": len(task_results),
                "config": self.config.model_dump()
            }
        )
    
    def _calculate_overall_score(self, task_results: List[TaskResult]) -> float:
        """Calculate overall benchmark score."""
        if not task_results:
            return 0.0
        return sum(result.score for result in task_results) / len(task_results)
    
    def _calculate_summary_metrics(self, task_results: List[TaskResult]) -> Dict[str, Any]:
        """Calculate summary metrics across all tasks."""
        if not task_results:
            return {}
        
        return {
            "success_rate": sum(1 for r in task_results if r.success) / len(task_results),
            "average_score": sum(r.score for r in task_results) / len(task_results),
            "average_execution_time": sum(r.execution_time for r in task_results) / len(task_results),
            "num_tasks_completed": len(task_results),
            "num_tasks_successful": sum(1 for r in task_results if r.success),
            "num_tasks_failed": sum(1 for r in task_results if not r.success),
        }
    
    def get_benchmark_info(self) -> Dict[str, Any]:
        """Get information about this benchmark."""
        return {
            "benchmark_name": self.benchmark_name,
            "agent_type": self.agent_type.value,
            "description": self.__doc__ or "",
            "config": self.config.model_dump()
        } 