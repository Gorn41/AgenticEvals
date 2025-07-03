"""
Tests for benchmark base classes.
"""

import pytest
import os
from typing import List

from benchmark.base import (
    BaseBenchmark, BenchmarkConfig, BenchmarkResult, TaskResult, Task, AgentType
)
from models.base import ModelResponse, BaseModel
from models.loader import load_gemini


class TestAgentType:
    """Test the AgentType enum."""
    
    def test_agent_types_exist(self):
        """Test that all expected agent types exist."""
        expected_types = [
            "SIMPLE_REFLEX",
            "MODEL_BASED_REFLEX", 
            "GOAL_BASED",
            "UTILITY_BASED",
            "LEARNING"
        ]
        
        for agent_type_name in expected_types:
            assert hasattr(AgentType, agent_type_name)
    
    def test_agent_type_values(self):
        """Test agent type string values."""
        assert AgentType.SIMPLE_REFLEX.value == "simple_reflex"
        assert AgentType.MODEL_BASED_REFLEX.value == "model_based_reflex"
        assert AgentType.GOAL_BASED.value == "goal_based"
        assert AgentType.UTILITY_BASED.value == "utility_based"
        assert AgentType.LEARNING.value == "learning"


class TestTaskResult:
    """Test the TaskResult class."""
    
    def test_task_result_creation(self):
        """Test creating a task result."""
        model_response = ModelResponse(
            text="Test response",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            finish_reason="completed"
        )
        
        result = TaskResult(
            task_id="task_1",
            task_name="Test Task",
            agent_type=AgentType.SIMPLE_REFLEX,
            success=True,
            score=0.85,
            metrics={"accuracy": 0.9},
            model_response=model_response,
            execution_time=1.2,
            metadata={"notes": "good performance"}
        )
        
        assert result.task_id == "task_1"
        assert result.task_name == "Test Task"
        assert result.agent_type == AgentType.SIMPLE_REFLEX
        assert result.success is True
        assert result.score == 0.85
        assert result.metrics == {"accuracy": 0.9}
        assert result.model_response == model_response
        assert result.execution_time == 1.2
        assert result.metadata == {"notes": "good performance"}
    
    def test_task_result_minimal(self):
        """Test TaskResult with minimal required fields."""
        result = TaskResult(
            task_id="minimal",
            success=False,
            score=0.2
        )
        
        assert result.task_id == "minimal"
        assert result.success is False
        assert result.score == 0.2
        assert result.task_name is None
        assert result.agent_type is None
        assert result.metrics == {}
        assert result.model_response is None
        assert result.execution_time is None
        assert result.metadata == {}


class TestBenchmarkResult:
    """Test the BenchmarkResult class."""
    
    def test_benchmark_result_creation(self):
        """Test creating a benchmark result."""
        task_results = [
            TaskResult(
                task_id="task_1",
                success=True,
                score=0.9,
                execution_time=1.0
            ),
            TaskResult(
                task_id="task_2", 
                success=False,
                score=0.3,
                execution_time=1.5
            )
        ]
        
        result = BenchmarkResult(
            benchmark_name="test_benchmark",
            model_name="test_model",
            agent_type=AgentType.SIMPLE_REFLEX,
            task_results=task_results,
            overall_score=0.6,
            metadata={"test_run": True}
        )
        
        assert result.benchmark_name == "test_benchmark"
        assert result.model_name == "test_model"
        assert result.agent_type == AgentType.SIMPLE_REFLEX
        assert len(result.task_results) == 2
        assert result.overall_score == 0.6
        assert result.metadata == {"test_run": True}
    
    def test_get_success_rate(self):
        """Test calculating success rate."""
        task_results = [
            TaskResult(task_id="1", success=True, score=1.0),
            TaskResult(task_id="2", success=True, score=0.8),
            TaskResult(task_id="3", success=False, score=0.2)
        ]
        
        result = BenchmarkResult(
            benchmark_name="test",
            model_name="test",
            agent_type=AgentType.SIMPLE_REFLEX,
            task_results=task_results,
            overall_score=0.667
        )
        
        success_rate = result.get_success_rate()
        assert success_rate == pytest.approx(0.667, rel=1e-2)
    
    def test_get_summary_statistics(self):
        """Test getting summary statistics."""
        task_results = [
            TaskResult(task_id="1", success=True, score=0.9, execution_time=1.0),
            TaskResult(task_id="2", success=True, score=0.8, execution_time=1.2),
            TaskResult(task_id="3", success=False, score=0.1, execution_time=0.8)
        ]
        
        result = BenchmarkResult(
            benchmark_name="test",
            model_name="test", 
            agent_type=AgentType.SIMPLE_REFLEX,
            task_results=task_results,
            overall_score=0.6
        )
        
        stats = result.get_summary_statistics()
        
        assert "total_tasks" in stats
        assert "successful_tasks" in stats
        assert "average_score" in stats
        assert "average_execution_time" in stats
        
        assert stats["total_tasks"] == 3
        assert stats["successful_tasks"] == 2
        assert stats["average_execution_time"] == pytest.approx(1.0, rel=1e-2)


class TestBenchmarkConfig:
    """Test the BenchmarkConfig class."""
    
    def test_config_creation_minimal(self):
        """Test creating benchmark config with minimal parameters."""
        config = BenchmarkConfig(
            benchmark_name="test_benchmark",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        
        assert config.benchmark_name == "test_benchmark"
        assert config.agent_type == AgentType.SIMPLE_REFLEX
        assert config.max_retries == 3  # Default value
        assert config.collect_detailed_metrics is True  # Default value
    
    def test_config_creation_full(self):
        """Test creating benchmark config with all parameters."""
        config = BenchmarkConfig(
            benchmark_name="test_benchmark",
            agent_type=AgentType.MODEL_BASED_REFLEX,
            num_tasks=10,
            random_seed=42,
            timeout_seconds=30.0,
            max_retries=5,
            collect_detailed_metrics=False,
            save_responses=True,
            additional_params={"custom": "value"}
        )
        
        assert config.benchmark_name == "test_benchmark"
        assert config.agent_type == AgentType.MODEL_BASED_REFLEX
        assert config.num_tasks == 10
        assert config.random_seed == 42
        assert config.timeout_seconds == 30.0
        assert config.max_retries == 5
        assert config.collect_detailed_metrics is False
        assert config.save_responses is True
        assert config.additional_params == {"custom": "value"}


class TestTask:
    """Test the Task class."""
    
    def test_task_creation(self):
        """Test creating a task."""
        task = Task(
            task_id="task_1",
            name="Test Task",
            description="A test task",
            prompt="Test prompt",
            expected_output="Expected result",
            metadata={"category": "test"}
        )
        
        assert task.task_id == "task_1"
        assert task.name == "Test Task"
        assert task.description == "A test task"
        assert task.prompt == "Test prompt"
        assert task.expected_output == "Expected result"
        assert task.metadata == {"category": "test"}
    
    def test_task_minimal_creation(self):
        """Test creating a task with minimal parameters."""
        task = Task(
            task_id="minimal_task",
            prompt="Simple prompt",
            expected_output="Simple output"
        )
        
        assert task.task_id == "minimal_task"
        assert task.prompt == "Simple prompt"
        assert task.expected_output == "Simple output"
        assert task.name is None
        assert task.description is None
        assert task.metadata == {}


class TestBaseBenchmark:
    """Test the BaseBenchmark abstract class."""
    
    def test_abstract_methods_exist(self):
        """Test that BaseBenchmark has the required abstract methods."""
        abstract_methods = BaseBenchmark.__abstractmethods__
        
        expected_methods = {'get_tasks', 'evaluate_task', 'calculate_score'}
        assert expected_methods.issubset(abstract_methods)
    
    def test_cannot_instantiate_base_benchmark(self):
        """Test that BaseBenchmark cannot be instantiated directly."""
        config = BenchmarkConfig(
            benchmark_name="test",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        
        with pytest.raises(TypeError):
            BaseBenchmark(config)
    
    def test_benchmark_info_structure(self):
        """Test the structure of benchmark info method."""
        
        class ConcreteBenchmark(BaseBenchmark):
            """Test benchmark for validation."""
            
            def get_tasks(self):
                return []
            
            async def evaluate_task(self, task, model):
                return TaskResult(task_id="test", success=True, score=1.0)
            
            def calculate_score(self, task, model_response):
                return 1.0
        
        config = BenchmarkConfig(
            benchmark_name="test_info",
            agent_type=AgentType.UTILITY_BASED,
            max_retries=5
        )
        benchmark = ConcreteBenchmark(config)
        
        info = benchmark.get_benchmark_info()
        
        assert isinstance(info, dict)
        assert info["benchmark_name"] == "test_info"
        assert info["agent_type"] == "utility_based"
        assert "Test benchmark for validation" in info["description"]
        assert "config" in info
        assert info["config"]["max_retries"] == 5


class TestBenchmarkIntegration:
    """Integration tests that require API keys."""
    
    @pytest.mark.integration
    def test_model_loading_for_benchmarks(self):
        """Test that models can be loaded for benchmark use."""
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("No API key available for integration test")
        
        model = load_gemini("gemini-2.5-flash", api_key=api_key, temperature=0.1)
        
        assert model is not None
        assert hasattr(model, 'generate')
        assert hasattr(model, 'generate_sync')
        assert model.model_name == "gemini-2.5-flash"
    
    @pytest.mark.integration
    async def test_benchmark_interface_compatibility(self):
        """Test that real models work with benchmark interface."""
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("No API key available for integration test")
        
        # Test model response format compatibility
        model = load_gemini("gemini-2.5-flash", api_key=api_key, temperature=0.1)
        response = await model.generate("What is 2+2?")
        
        # Verify response has expected structure for benchmarks
        assert hasattr(response, 'text')
        assert isinstance(response.text, str)
        assert len(response.text) > 0
        
        # Test creating TaskResult with real response
        task_result = TaskResult(
            task_id="integration_test",
            task_name="Math Test",
            agent_type=AgentType.SIMPLE_REFLEX,
            success=True,
            score=1.0,
            model_response=response,
            execution_time=0.5
        )
        
        assert task_result.model_response == response
        assert task_result.task_id == "integration_test" 