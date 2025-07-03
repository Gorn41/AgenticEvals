"""
Benchmark loader for AgenticEvals.
"""

import os
import importlib
from typing import Dict, Type, List, Optional
from pathlib import Path

from .base import BaseBenchmark, BenchmarkConfig, AgentType
from .registry import BenchmarkRegistry
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BenchmarkLoader:
    """Factory class for loading different benchmark types."""
    
    def __init__(self):
        """Initialize the benchmark loader."""
        self.registry = BenchmarkRegistry()
        self._auto_discover_benchmarks()
    
    def _auto_discover_benchmarks(self):
        """Automatically discover and register benchmarks in the benchmarks package."""
        benchmarks_dir = Path(__file__).parent.parent / "benchmarks"
        
        if not benchmarks_dir.exists():
            logger.warning(f"Benchmarks directory not found: {benchmarks_dir}")
            return
        
        # Import all benchmark modules
        for benchmark_file in benchmarks_dir.glob("*.py"):
            if benchmark_file.name.startswith("_"):
                continue
            
            module_name = f"src.benchmarks.{benchmark_file.stem}"
            try:
                importlib.import_module(module_name)
                logger.info(f"Discovered benchmark module: {module_name}")
            except ImportError as e:
                logger.warning(f"Failed to import benchmark module {module_name}: {e}")
    
    def get_available_benchmarks(self) -> Dict[AgentType, List[str]]:
        """Get available benchmarks grouped by agent type."""
        return self.registry.get_benchmarks_by_type()
    
    def get_benchmark_info(self, benchmark_name: str) -> Dict:
        """Get information about a specific benchmark."""
        benchmark_class = self.registry.get_benchmark_class(benchmark_name)
        if not benchmark_class:
            raise ValueError(f"Benchmark not found: {benchmark_name}")
        
        # Check if the benchmark class has a static info method
        if hasattr(benchmark_class, 'get_static_info'):
            return benchmark_class.get_static_info()
        
        # Otherwise get info from class attributes or docstring
        agent_type = self.registry.get_benchmark_agent_type(benchmark_name)
        
        info = {
            "name": benchmark_name,
            "agent_type": agent_type.value if agent_type else None,
            "class_name": benchmark_class.__name__,
            "description": benchmark_class.__doc__ or "No description available"
        }
        
        # Add any additional class-level information
        if hasattr(benchmark_class, 'BENCHMARK_DESCRIPTION'):
            info["description"] = benchmark_class.BENCHMARK_DESCRIPTION
        
        if hasattr(benchmark_class, 'BENCHMARK_VERSION'):
            info["version"] = benchmark_class.BENCHMARK_VERSION
            
        return info
    
    def load_benchmark(self, benchmark_name: str, **config_kwargs) -> BaseBenchmark:
        """Load a benchmark by name with given configuration."""
        benchmark_class = self.registry.get_benchmark_class(benchmark_name)
        if not benchmark_class:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. Available benchmarks: {self.registry.list_benchmarks()}")
        
        # Build benchmark configuration
        config = self._build_benchmark_config(benchmark_name, **config_kwargs)
        
        # Instantiate benchmark
        benchmark = benchmark_class(config)
        
        logger.info(f"Loaded benchmark: {benchmark_name}")
        return benchmark
    
    def _build_benchmark_config(self, benchmark_name: str, **kwargs) -> BenchmarkConfig:
        """Build benchmark configuration with defaults."""
        agent_type = self.registry.get_benchmark_agent_type(benchmark_name)
        
        config_dict = {
            "benchmark_name": benchmark_name,
            "agent_type": agent_type,
            "num_tasks": kwargs.get("num_tasks"),
            "random_seed": kwargs.get("random_seed"),
            "timeout_seconds": kwargs.get("timeout_seconds"),
            "max_retries": kwargs.get("max_retries", 3),
            "collect_detailed_metrics": kwargs.get("collect_detailed_metrics", True),
            "save_responses": kwargs.get("save_responses", True),
            "additional_params": kwargs.get("additional_params", {}),
        }
        
        return BenchmarkConfig(**config_dict)
    
    def load_benchmarks_by_type(self, agent_type: AgentType, **config_kwargs) -> List[BaseBenchmark]:
        """Load all benchmarks for a specific agent type."""
        benchmark_names = self.registry.get_benchmarks_for_type(agent_type)
        benchmarks = []
        
        for name in benchmark_names:
            try:
                benchmark = self.load_benchmark(name, **config_kwargs)
                benchmarks.append(benchmark)
            except Exception as e:
                logger.error(f"Failed to load benchmark {name}: {e}")
        
        return benchmarks
    
    def load_all_benchmarks(self, **config_kwargs) -> Dict[AgentType, List[BaseBenchmark]]:
        """Load all available benchmarks grouped by agent type."""
        all_benchmarks = {}
        
        for agent_type in AgentType:
            benchmarks = self.load_benchmarks_by_type(agent_type, **config_kwargs)
            if benchmarks:
                all_benchmarks[agent_type] = benchmarks
        
        return all_benchmarks
    
    def load_from_config_file(self, config_path: str) -> BaseBenchmark:
        """Load benchmark from a configuration file."""
        import yaml
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        benchmark_name = config_data.get("benchmark_name")
        if not benchmark_name:
            raise ValueError("benchmark_name is required in config file")
        
        # Remove benchmark_name from kwargs since it's passed separately
        config_kwargs = {k: v for k, v in config_data.items() if k != "benchmark_name"}
        
        return self.load_benchmark(benchmark_name, **config_kwargs)


# Global loader instance
_benchmark_loader = None


def get_benchmark_loader() -> BenchmarkLoader:
    """Get the global benchmark loader instance."""
    global _benchmark_loader
    if _benchmark_loader is None:
        _benchmark_loader = BenchmarkLoader()
    return _benchmark_loader


# Convenience functions
def load_benchmark(benchmark_name: str, **kwargs) -> BaseBenchmark:
    """Convenience function to load a benchmark by name."""
    return get_benchmark_loader().load_benchmark(benchmark_name, **kwargs)


def get_available_benchmarks() -> Dict[AgentType, List[str]]:
    """Convenience function to get available benchmarks."""
    return get_benchmark_loader().get_available_benchmarks()


def load_benchmarks_by_type(agent_type: AgentType, **kwargs) -> List[BaseBenchmark]:
    """Convenience function to load benchmarks by agent type."""
    return get_benchmark_loader().load_benchmarks_by_type(agent_type, **kwargs) 