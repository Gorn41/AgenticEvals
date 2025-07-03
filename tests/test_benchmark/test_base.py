"""
Tests for benchmark base classes and interfaces.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

from benchmark.base import (
    AgentType, TaskResult, BenchmarkResult, BenchmarkConfig, 
    Task, BaseBenchmark
)
from models.base import ModelResponse, BaseModel


class TestAgentType:
    """Test the AgentType enum."""
    
    def test_agent_type_values(self):
        """Test that all expected agent types exist."""
        assert AgentType.SIMPLE_REFLEX.value == "simple_reflex"
        assert AgentType.MODEL_BASED_REFLEX.value == "model_based_reflex"
        assert AgentType.GOAL_BASED.value == "goal_based"
        assert AgentType.UTILITY_BASED.value == "utility_based"
        assert AgentType.LEARNING.value == "learning"


class TestTaskResult:
    """Test the TaskResult class."""
    
    def test_task_result_creation(self):
        """Test creating a TaskResult."""
        model_response = ModelResponse(text="stop", tokens_used=5, latency=0.8)
        
        result = TaskResult(
            task_id="test_task",
            task_name="Test Task",
            agent_type=AgentType.SIMPLE_REFLEX,
            success=True,
            score=0.95,
            metrics={"accuracy": 1.0, "response_time": 0.8},
            model_response=model_response,
            execution_time=0.8,
            metadata={"difficulty": "easy"}
        )
        
        assert result.task_id == "test_task"
        assert result.task_name == "Test Task"
        assert result.agent_type == AgentType.SIMPLE_REFLEX
        assert result.success is True
        assert result.score == 0.95
        assert result.metrics["accuracy"] == 1.0
        assert result.model_response == model_response
        assert result.execution_time == 0.8
        assert result.metadata["difficulty"] == "easy"
    
    def test_task_result_minimal(self):
        """Test TaskResult with minimal required fields."""
        result = TaskResult(
            task_id="minimal",
            task_name="Minimal Task",
            agent_type=AgentType.GOAL_BASED,
            success=False,
            score=0.0,
            metrics={},
            execution_time=1.0
        )
        
        assert result.task_id == "minimal"
        assert result.success is False
        assert result.model_response is None
        assert result.error_message is None
        assert result.metadata == {}


class TestBenchmarkResult:
    """Test the BenchmarkResult class."""
    
    def test_benchmark_result_creation(self):
        """Test creating a BenchmarkResult."""
        task_results = [
            TaskResult(
                task_id="task1",
                task_name="Task 1", 
                agent_type=AgentType.SIMPLE_REFLEX,
                success=True,
                score=1.0,
                metrics={},
                execution_time=0.5
            ),
            TaskResult(
                task_id="task2",
                task_name="Task 2",
                agent_type=AgentType.SIMPLE_REFLEX, 
                success=False,
                score=0.0,
                metrics={},
                execution_time=0.8
            )
        ]
        
        result = BenchmarkResult(
            benchmark_id="bench_123",
            benchmark_name="Test Benchmark",
            agent_type=AgentType.SIMPLE_REFLEX,
            model_name="test-model",
            timestamp="2024-01-01 12:00:00",
            task_results=task_results,
            overall_score=0.5,
            summary_metrics={"avg_score": 0.5}
        )
        
        assert result.benchmark_id == "bench_123"
        assert result.benchmark_name == "Test Benchmark"
        assert result.agent_type == AgentType.SIMPLE_REFLEX
        assert result.model_name == "test-model"
        assert len(result.task_results) == 2
        assert result.overall_score == 0.5
    
    def test_get_success_rate(self):
        """Test calculating success rate."""
        task_results = [
            TaskResult("t1", "Task 1", AgentType.SIMPLE_REFLEX, True, 1.0, {}, 0.5),
            TaskResult("t2", "Task 2", AgentType.SIMPLE_REFLEX, True, 0.8, {}, 0.6),
            TaskResult("t3", "Task 3", AgentType.SIMPLE_REFLEX, False, 0.0, {}, 0.7),
            TaskResult("t4", "Task 4", AgentType.SIMPLE_REFLEX, True, 0.9, {}, 0.4)
        ]
        
        result = BenchmarkResult(
            benchmark_id="test",
            benchmark_name="Test",
            agent_type=AgentType.SIMPLE_REFLEX,
            model_name="test",
            timestamp="2024-01-01",
            task_results=task_results,
            overall_score=0.675,
            summary_metrics={}
        )
        
        assert result.get_success_rate() == 0.75  # 3 out of 4 successful
    
    def test_get_success_rate_empty(self):
        """Test success rate with no tasks."""
        result = BenchmarkResult(
            benchmark_id="test",
            benchmark_name="Test",
            agent_type=AgentType.SIMPLE_REFLEX,
            model_name="test",
            timestamp="2024-01-01",
            task_results=[],
            overall_score=0.0,
            summary_metrics={}
        )
        
        assert result.get_success_rate() == 0.0
    
    def test_get_average_score(self):
        """Test calculating average score."""
        task_results = [
            TaskResult("t1", "Task 1", AgentType.SIMPLE_REFLEX, True, 1.0, {}, 0.5),
            TaskResult("t2", "Task 2", AgentType.SIMPLE_REFLEX, True, 0.8, {}, 0.6),
            TaskResult("t3", "Task 3", AgentType.SIMPLE_REFLEX, False, 0.2, {}, 0.7)
        ]
        
        result = BenchmarkResult(
            benchmark_id="test",
            benchmark_name="Test", 
            agent_type=AgentType.SIMPLE_REFLEX,
            model_name="test",
            timestamp="2024-01-01",
            task_results=task_results,
            overall_score=0.0,  # Will be calculated
            summary_metrics={}
        )
        
        assert result.get_average_score() == pytest.approx(0.667, rel=1e-2)
    
    def test_get_metric_summary(self):
        """Test getting metric summary."""
        task_results = [
            TaskResult("t1", "T1", AgentType.SIMPLE_REFLEX, True, 1.0, {"latency": 0.5}, 0.5),
            TaskResult("t2", "T2", AgentType.SIMPLE_REFLEX, True, 0.8, {"latency": 0.8}, 0.8),
            TaskResult("t3", "T3", AgentType.SIMPLE_REFLEX, True, 0.9, {"latency": 0.3}, 0.3),
            TaskResult("t4", "T4", AgentType.SIMPLE_REFLEX, True, 0.7, {"other": 1.0}, 0.4)
        ]
        
        result = BenchmarkResult(
            benchmark_id="test",
            benchmark_name="Test",
            agent_type=AgentType.SIMPLE_REFLEX,
            model_name="test", 
            timestamp="2024-01-01",
            task_results=task_results,
            overall_score=0.85,
            summary_metrics={}
        )
        
        latency_summary = result.get_metric_summary("latency")
        assert latency_summary["count"] == 3  # Only 3 tasks have latency
        assert latency_summary["mean"] == pytest.approx(0.533, rel=1e-2)
        assert latency_summary["min"] == 0.3
        assert latency_summary["max"] == 0.8
        
        # Non-existent metric
        empty_summary = result.get_metric_summary("nonexistent")
        assert empty_summary == {}


class TestBenchmarkConfig:
    """Test the BenchmarkConfig class."""
    
    def test_benchmark_config_creation(self):
        """Test creating a BenchmarkConfig."""
        config = BenchmarkConfig(
            benchmark_name="test_benchmark",
            agent_type=AgentType.GOAL_BASED,
            num_tasks=10,
            random_seed=42,
            timeout_seconds=30.0,
            max_retries=3,
            collect_detailed_metrics=True,
            save_responses=False,
            additional_params={"custom": "value"}
        )
        
        assert config.benchmark_name == "test_benchmark"
        assert config.agent_type == AgentType.GOAL_BASED
        assert config.num_tasks == 10
        assert config.random_seed == 42
        assert config.timeout_seconds == 30.0
        assert config.max_retries == 3
        assert config.collect_detailed_metrics is True
        assert config.save_responses is False
        assert config.additional_params["custom"] == "value"
    
    def test_benchmark_config_defaults(self):
        """Test BenchmarkConfig with default values."""
        config = BenchmarkConfig(
            benchmark_name="minimal",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        
        assert config.benchmark_name == "minimal"
        assert config.agent_type == AgentType.SIMPLE_REFLEX
        assert config.num_tasks is None
        assert config.random_seed is None
        assert config.timeout_seconds is None
        assert config.max_retries == 3
        assert config.collect_detailed_metrics is True
        assert config.save_responses is True
        assert config.additional_params == {}


class TestTask:
    """Test the Task dataclass."""
    
    def test_task_creation(self):
        """Test creating a Task."""
        task = Task(
            task_id="task_001",
            name="Traffic Light Test",
            description="Test response to red light",
            prompt="You see a red traffic light. What do you do?",
            expected_output="stop",
            evaluation_criteria={"exact_match": True, "case_sensitive": False},
            metadata={"signal_color": "red", "difficulty": "easy"}
        )
        
        assert task.task_id == "task_001"
        assert task.name == "Traffic Light Test"
        assert task.description == "Test response to red light"
        assert task.prompt == "You see a red traffic light. What do you do?"
        assert task.expected_output == "stop"
        assert task.evaluation_criteria["exact_match"] is True
        assert task.metadata["signal_color"] == "red"
    
    def test_task_minimal(self):
        """Test Task with minimal required fields."""
        task = Task(
            task_id="minimal",
            name="Minimal Task",
            description="Minimal test",
            prompt="Test prompt"
        )
        
        assert task.task_id == "minimal"
        assert task.name == "Minimal Task"
        assert task.expected_output is None
        # Post-init should set defaults
        assert task.evaluation_criteria == {}
        assert task.metadata == {}


class TestBaseBenchmark:
    """Test the BaseBenchmark abstract class."""
    
    def test_base_benchmark_is_abstract(self):
        """Test that BaseBenchmark cannot be instantiated directly."""
        config = BenchmarkConfig(
            benchmark_name="test",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        
        with pytest.raises(TypeError):
            BaseBenchmark(config)
    
    def test_concrete_implementation(self):
        """Test a concrete implementation of BaseBenchmark."""
        
        class ConcreteBenchmark(BaseBenchmark):
            def get_tasks(self):
                return [
                    Task("t1", "Task 1", "Test task", "Test prompt", "expected")
                ]
            
            async def evaluate_task(self, task, model):
                response = await model.generate(task.prompt)
                score = 1.0 if response.text == task.expected_output else 0.0
                return TaskResult(
                    task_id=task.task_id,
                    task_name=task.name,
                    agent_type=self.agent_type,
                    success=score > 0.5,
                    score=score,
                    metrics={},
                    model_response=response,
                    execution_time=0.5
                )
            
            def calculate_score(self, task, model_response):
                return 1.0 if model_response.text == task.expected_output else 0.0
        
        config = BenchmarkConfig(
            benchmark_name="concrete",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        
        benchmark = ConcreteBenchmark(config)
        assert benchmark.benchmark_name == "concrete"
        assert benchmark.agent_type == AgentType.SIMPLE_REFLEX
        assert benchmark.config == config
    
    async def test_run_benchmark(self):
        """Test running a complete benchmark."""
        
        class TestBenchmark(BaseBenchmark):
            def get_tasks(self):
                return [
                    Task("t1", "Task 1", "Test", "prompt1", "stop"),
                    Task("t2", "Task 2", "Test", "prompt2", "go")
                ]
            
            async def evaluate_task(self, task, model):
                response = await model.generate(task.prompt)
                score = 1.0 if response.text == task.expected_output else 0.0
                return TaskResult(
                    task_id=task.task_id,
                    task_name=task.name,
                    agent_type=self.agent_type,
                    success=score > 0.5,
                    score=score,
                    metrics={"test_metric": 1.0},
                    model_response=response,
                    execution_time=0.5
                )
            
            def calculate_score(self, task, model_response):
                return 1.0 if model_response.text == task.expected_output else 0.0
        
        config = BenchmarkConfig(
            benchmark_name="test_benchmark",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        benchmark = TestBenchmark(config)
        
        # Mock model
        mock_model = AsyncMock()
        mock_model.model_name = "test-model"
        mock_model.generate.side_effect = [
            ModelResponse(text="stop", tokens_used=5, latency=0.5),
            ModelResponse(text="go", tokens_used=3, latency=0.3)
        ]
        
        result = await benchmark.run_benchmark(mock_model)
        
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "test_benchmark"
        assert result.model_name == "test-model"
        assert result.agent_type == AgentType.SIMPLE_REFLEX
        assert len(result.task_results) == 2
        assert result.overall_score == 1.0  # Both tasks successful
        assert result.summary_metrics["num_tasks_completed"] == 2
        assert result.summary_metrics["num_tasks_successful"] == 2
    
    async def test_run_benchmark_with_task_limit(self):
        """Test running benchmark with task limit."""
        
        class TestBenchmark(BaseBenchmark):
            def get_tasks(self):
                return [
                    Task("t1", "Task 1", "Test", "prompt1", "stop"),
                    Task("t2", "Task 2", "Test", "prompt2", "go"),
                    Task("t3", "Task 3", "Test", "prompt3", "caution")
                ]
            
            async def evaluate_task(self, task, model):
                response = await model.generate(task.prompt)
                return TaskResult(
                    task_id=task.task_id,
                    task_name=task.name,
                    agent_type=self.agent_type,
                    success=True,
                    score=1.0,
                    metrics={},
                    model_response=response,
                    execution_time=0.5
                )
            
            def calculate_score(self, task, model_response):
                return 1.0
        
        config = BenchmarkConfig(
            benchmark_name="limited",
            agent_type=AgentType.SIMPLE_REFLEX,
            num_tasks=2  # Limit to 2 tasks
        )
        benchmark = TestBenchmark(config)
        
        mock_model = AsyncMock()
        mock_model.model_name = "test-model"
        mock_model.generate.return_value = ModelResponse(text="test", tokens_used=5, latency=0.5)
        
        result = await benchmark.run_benchmark(mock_model)
        
        # Should only run 2 tasks despite 3 being available
        assert len(result.task_results) == 2
    
    async def test_run_benchmark_with_error(self):
        """Test benchmark handling task evaluation errors."""
        
        class ErrorBenchmark(BaseBenchmark):
            def get_tasks(self):
                return [
                    Task("t1", "Good Task", "Test", "prompt1", "stop"),
                    Task("t2", "Bad Task", "Test", "prompt2", "go")
                ]
            
            async def evaluate_task(self, task, model):
                if task.task_id == "t2":
                    raise Exception("Evaluation failed")
                
                response = await model.generate(task.prompt)
                return TaskResult(
                    task_id=task.task_id,
                    task_name=task.name,
                    agent_type=self.agent_type,
                    success=True,
                    score=1.0,
                    metrics={},
                    model_response=response,
                    execution_time=0.5
                )
            
            def calculate_score(self, task, model_response):
                return 1.0
        
        config = BenchmarkConfig(
            benchmark_name="error_test",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        benchmark = ErrorBenchmark(config)
        
        mock_model = AsyncMock()
        mock_model.model_name = "test-model"
        mock_model.generate.return_value = ModelResponse(text="stop", tokens_used=5, latency=0.5)
        
        result = await benchmark.run_benchmark(mock_model)
        
        assert len(result.task_results) == 2
        # First task should succeed
        assert result.task_results[0].success is True
        # Second task should fail
        assert result.task_results[1].success is False
        assert result.task_results[1].error_message == "Evaluation failed"
        assert result.summary_metrics["num_tasks_successful"] == 1
        assert result.summary_metrics["num_tasks_failed"] == 1
    
    def test_get_benchmark_info(self):
        """Test getting benchmark information."""
        
        class InfoBenchmark(BaseBenchmark):
            """Test benchmark for info testing."""
            
            def get_tasks(self):
                return []
            
            async def evaluate_task(self, task, model):
                pass
            
            def calculate_score(self, task, model_response):
                return 0.0
        
        config = BenchmarkConfig(
            benchmark_name="info_test",
            agent_type=AgentType.UTILITY_BASED,
            max_retries=5
        )
        benchmark = InfoBenchmark(config)
        
        info = benchmark.get_benchmark_info()
        
        assert info["benchmark_name"] == "info_test"
        assert info["agent_type"] == "utility_based"
        assert "Test benchmark for info testing" in info["description"]
        assert "config" in info
        assert info["config"]["max_retries"] == 5 