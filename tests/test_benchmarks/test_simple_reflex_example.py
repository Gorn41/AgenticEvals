"""
Tests for the traffic light simple reflex benchmark.
"""

import pytest
import os

from benchmarks.simple_reflex_example import TrafficLightBenchmark
from benchmark.base import BenchmarkConfig, AgentType, Task
from models.base import ModelResponse
from models.loader import load_gemini


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
    
    def test_calculate_score_no_match(self):
        """Test score calculation with no match."""
        config = BenchmarkConfig(
            benchmark_name="traffic_light_simple",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        benchmark = TrafficLightBenchmark(config)
        
        task = Task("t1", "Test", "desc", "prompt", expected_output="stop")
        response = ModelResponse(text="run a marathon")
        
        score = benchmark.calculate_score(task, response)
        assert score == 0.0
    
    def test_benchmark_info(self):
        """Test getting benchmark information."""
        config = BenchmarkConfig(
            benchmark_name="traffic_light_simple",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        benchmark = TrafficLightBenchmark(config)
        
        info = benchmark.get_benchmark_info()
        
        assert isinstance(info, dict)
        assert "name" in info
        assert "description" in info
        assert "agent_type" in info


class TestTrafficLightBenchmarkIntegration:
    """Integration tests requiring API keys."""
    
    @pytest.mark.integration
    async def test_full_benchmark_run(self):
        """Test running the full benchmark with a real model."""
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("No API key available for integration test")
        
        # Create benchmark
        config = BenchmarkConfig(
            benchmark_name="traffic_light_simple",
            agent_type=AgentType.SIMPLE_REFLEX,
            num_tasks=3  # Limit for faster testing
        )
        benchmark = TrafficLightBenchmark(config)
        
        # Load model
        model = load_gemini("gemini-2.5-flash", api_key=api_key, temperature=0.1)
        
        # Run benchmark
        result = await benchmark.run_benchmark(model)
        
        # Validate results
        assert result is not None
        assert result.benchmark_name == "traffic_light_simple"
        assert result.model_name == "gemini-2.5-flash"
        assert len(result.task_results) <= 3
        assert 0.0 <= result.overall_score <= 1.0
    
    @pytest.mark.integration
    async def test_single_task_evaluation(self):
        """Test evaluating a single task with real model."""
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            pytest.skip("No API key available for integration test")
        
        config = BenchmarkConfig(
            benchmark_name="traffic_light_simple",
            agent_type=AgentType.SIMPLE_REFLEX
        )
        benchmark = TrafficLightBenchmark(config)
        
        # Create a red light task
        task = Task(
            task_id="test_red",
            name="Red Light Test",
            description="Test red light response",
            prompt="You see a red traffic light. What should you do? Answer in one word.",
            expected_output="stop",
            metadata={"signal": "red", "difficulty": "basic"}
        )
        
        # Load model
        model = load_gemini("gemini-2.5-flash", api_key=api_key, temperature=0.1)
        
        # Evaluate task
        result = await benchmark.evaluate_task(task, model)
        
        # Validate result
        assert result is not None
        assert result.task_id == "test_red"
        assert result.model_response is not None
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0 