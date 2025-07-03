"""
Tests for the traffic light simple reflex benchmark.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from benchmarks.simple_reflex_example import TrafficLightBenchmark
from benchmark.base import BenchmarkConfig, AgentType, Task
from models.base import ModelResponse


class TestTrafficLightBenchmark:
    """Test the TrafficLightBenchmark class."""
    
    def test_init(self):
        """Test TrafficLightBenchmark initialization."""
        config = BenchmarkConfig(
            benchmark_name="traffic_light_simple",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        
        benchmark = TrafficLightBenchmark(config)
        assert benchmark.benchmark_name == "traffic_light_simple"
        assert benchmark.agent_type == AgentType.SIMPLE_REFLEX
    
    def test_get_tasks(self):
        """Test getting all tasks."""
        config = BenchmarkConfig(
            benchmark_name="traffic_light_simple",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        benchmark = TrafficLightBenchmark(config)
        
        tasks = benchmark.get_tasks()
        
        assert len(tasks) > 0
        assert all(isinstance(task, Task) for task in tasks)
        
        # Check basic traffic light scenarios are included
        task_signals = [task.metadata.get("signal", "").lower() for task in tasks]
        assert "red" in task_signals
        assert "green" in task_signals
        assert "yellow" in task_signals
    
    def test_create_prompt(self):
        """Test prompt creation."""
        config = BenchmarkConfig(
            benchmark_name="traffic_light_simple",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        benchmark = TrafficLightBenchmark(config)
        
        prompt = benchmark._create_prompt("red")
        
        assert "red" in prompt.lower()
        assert "traffic light" in prompt.lower()
        assert "stop" in prompt.lower() or "go" in prompt.lower() or "caution" in prompt.lower()
    
    async def test_evaluate_task_success(self):
        """Test successful task evaluation."""
        config = BenchmarkConfig(
            benchmark_name="traffic_light_simple",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        benchmark = TrafficLightBenchmark(config)
        
        # Create a test task
        task = Task(
            task_id="test_red",
            name="Red Light Test",
            description="Test red light response",
            prompt="You see a red traffic light. What do you do?",
            expected_output="stop",
            metadata={"signal": "red", "difficulty": "basic"}
        )
        
        # Mock model with correct response
        mock_model = AsyncMock()
        mock_model.generate.return_value = ModelResponse(
            text="stop",
            tokens_used=5,
            latency=0.5,
            metadata={}
        )
        
        result = await benchmark.evaluate_task(task, mock_model)
        
        assert result.success is True
        assert result.score > 0.5
        assert result.task_id == "test_red"
        assert result.task_name == "Red Light Test"
        assert result.agent_type == AgentType.SIMPLE_REFLEX
        assert result.model_response.text == "stop"
    
    async def test_evaluate_task_failure(self):
        """Test task evaluation with incorrect response."""
        config = BenchmarkConfig(
            benchmark_name="traffic_light_simple",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        benchmark = TrafficLightBenchmark(config)
        
        task = Task(
            task_id="test_red",
            name="Red Light Test", 
            description="Test red light response",
            prompt="You see a red traffic light. What do you do?",
            expected_output="stop",
            metadata={"signal": "red"}
        )
        
        # Mock model with incorrect response
        mock_model = AsyncMock()
        mock_model.generate.return_value = ModelResponse(
            text="go",  # Wrong answer
            tokens_used=3,
            latency=0.3
        )
        
        result = await benchmark.evaluate_task(task, mock_model)
        
        assert result.success is False
        assert result.score <= 0.5
        assert result.model_response.text == "go"
    
    async def test_evaluate_task_error_handling(self):
        """Test task evaluation with model error."""
        config = BenchmarkConfig(
            benchmark_name="traffic_light_simple",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        benchmark = TrafficLightBenchmark(config)
        
        task = Task(
            task_id="test_error",
            name="Error Test",
            description="Test error handling",
            prompt="Test prompt",
            expected_output="stop"
        )
        
        # Mock model that raises exception
        mock_model = AsyncMock()
        mock_model.generate.side_effect = Exception("Model error")
        
        result = await benchmark.evaluate_task(task, mock_model)
        
        assert result.success is False
        assert result.score == 0.0
        assert result.error_message == "Model error"
        assert result.model_response is None
    
    def test_calculate_score_exact_match(self):
        """Test score calculation with exact match."""
        config = BenchmarkConfig(
            benchmark_name="traffic_light_simple",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        benchmark = TrafficLightBenchmark(config)
        
        task = Task("t1", "Test", "desc", "prompt", expected_output="stop")
        response = ModelResponse(text="stop")
        
        score = benchmark.calculate_score(task, response)
        assert score == 1.0
    
    def test_calculate_score_case_insensitive(self):
        """Test score calculation is case insensitive."""
        config = BenchmarkConfig(
            benchmark_name="traffic_light_simple", 
            agent_type=AgentType.SIMPLE_REFLEX
        )
        benchmark = TrafficLightBenchmark(config)
        
        task = Task("t1", "Test", "desc", "prompt", expected_output="stop")
        response = ModelResponse(text="STOP")
        
        score = benchmark.calculate_score(task, response)
        assert score == 1.0
    
    def test_calculate_score_with_extra_words(self):
        """Test score calculation with extra words."""
        config = BenchmarkConfig(
            benchmark_name="traffic_light_simple",
            agent_type=AgentType.SIMPLE_REFLEX  
        )
        benchmark = TrafficLightBenchmark(config)
        
        task = Task("t1", "Test", "desc", "prompt", expected_output="stop")
        response = ModelResponse(text="I should stop")
        
        score = benchmark.calculate_score(task, response)
        assert 0.5 < score < 1.0  # Partial credit
    
    def test_calculate_score_synonym(self):
        """Test score calculation with synonyms."""
        config = BenchmarkConfig(
            benchmark_name="traffic_light_simple",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        benchmark = TrafficLightBenchmark(config)
        
        task = Task("t1", "Test", "desc", "prompt", expected_output="stop")
        response = ModelResponse(text="halt")  # Synonym for stop
        
        score = benchmark.calculate_score(task, response)
        assert score == 0.7  # Synonym score
    
    def test_calculate_score_no_match(self):
        """Test score calculation with no match."""
        config = BenchmarkConfig(
            benchmark_name="traffic_light_simple",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        benchmark = TrafficLightBenchmark(config)
        
        task = Task("t1", "Test", "desc", "prompt", expected_output="stop")
        response = ModelResponse(text="dance")  # Completely wrong
        
        score = benchmark.calculate_score(task, response)
        assert score == 0.0
    
    def test_calculate_detailed_metrics(self):
        """Test detailed metrics calculation."""
        config = BenchmarkConfig(
            benchmark_name="traffic_light_simple",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        benchmark = TrafficLightBenchmark(config)
        
        task = Task("t1", "Test", "desc", "prompt", expected_output="stop")
        response = ModelResponse(
            text="stop",
            tokens_used=5,
            latency=0.8
        )
        
        metrics = benchmark._calculate_detailed_metrics(task, response)
        
        assert metrics["word_count"] == 1
        assert metrics["follows_instructions"] is True
        assert metrics["contains_expected"] is True
        assert metrics["exact_match"] is True
        assert metrics["first_word"] == "stop"
        assert metrics["response_latency"] == 0.8
        assert metrics["tokens_used"] == 5
    
    def test_calculate_detailed_metrics_verbose_response(self):
        """Test detailed metrics with verbose response."""
        config = BenchmarkConfig(
            benchmark_name="traffic_light_simple",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        benchmark = TrafficLightBenchmark(config)
        
        task = Task("t1", "Test", "desc", "prompt", expected_output="stop")
        response = ModelResponse(text="I need to stop at the red light")
        
        metrics = benchmark._calculate_detailed_metrics(task, response)
        
        assert metrics["word_count"] == 9
        assert metrics["follows_instructions"] is False  # More than 1 word
        assert metrics["contains_expected"] is True
        assert metrics["exact_match"] is False
        assert metrics["first_word"] == "i" 