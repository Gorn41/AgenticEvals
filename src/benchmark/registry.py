"""
Benchmark registry for AgenticEvals.
"""

from typing import Dict, Type, List, Optional, Set
from collections import defaultdict

from .base import BaseBenchmark, AgentType
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BenchmarkRegistry:
    """Registry for managing available benchmarks."""
    
    def __init__(self):
        """Initialize the benchmark registry."""
        self._benchmarks: Dict[str, Type[BaseBenchmark]] = {}
        self._agent_type_map: Dict[str, AgentType] = {}
        self._description_map: Dict[str, str] = {}
    
    def register(self, 
                 name: str, 
                 benchmark_class: Type[BaseBenchmark], 
                 agent_type: AgentType,
                 description: str = ""):
        """Register a benchmark class."""
        if name in self._benchmarks:
            logger.warning(f"Overriding existing benchmark: {name}")
        
        self._benchmarks[name] = benchmark_class
        self._agent_type_map[name] = agent_type
        self._description_map[name] = description
        
        logger.info(f"Registered benchmark: {name} (type: {agent_type.value})")
    
    def unregister(self, name: str):
        """Unregister a benchmark."""
        if name in self._benchmarks:
            del self._benchmarks[name]
            del self._agent_type_map[name]
            del self._description_map[name]
            logger.info(f"Unregistered benchmark: {name}")
        else:
            logger.warning(f"Benchmark not found for unregistration: {name}")
    
    def get_benchmark_class(self, name: str) -> Optional[Type[BaseBenchmark]]:
        """Get benchmark class by name."""
        return self._benchmarks.get(name)
    
    def get_benchmark_agent_type(self, name: str) -> Optional[AgentType]:
        """Get agent type for a benchmark."""
        return self._agent_type_map.get(name)
    
    def get_benchmark_description(self, name: str) -> str:
        """Get description for a benchmark."""
        return self._description_map.get(name, "")
    
    def list_benchmarks(self) -> List[str]:
        """List all registered benchmark names."""
        return list(self._benchmarks.keys())
    
    def get_benchmarks_for_type(self, agent_type: AgentType) -> List[str]:
        """Get all benchmark names for a specific agent type."""
        return [name for name, atype in self._agent_type_map.items() if atype == agent_type]
    
    def get_benchmarks_by_type(self) -> Dict[AgentType, List[str]]:
        """Get all benchmarks grouped by agent type."""
        result = defaultdict(list)
        for name, agent_type in self._agent_type_map.items():
            result[agent_type].append(name)
        return dict(result)
    
    def get_all_agent_types(self) -> Set[AgentType]:
        """Get all agent types that have registered benchmarks."""
        return set(self._agent_type_map.values())
    
    def is_registered(self, name: str) -> bool:
        """Check if a benchmark is registered."""
        return name in self._benchmarks
    
    def get_registry_info(self) -> Dict:
        """Get complete registry information."""
        return {
            "total_benchmarks": len(self._benchmarks),
            "benchmarks_by_type": self.get_benchmarks_by_type(),
            "benchmark_info": {
                name: {
                    "agent_type": self._agent_type_map[name].value,
                    "description": self._description_map[name],
                    "class": self._benchmarks[name].__name__
                }
                for name in self._benchmarks.keys()
            }
        }


# Global registry instance
_global_registry = BenchmarkRegistry()


def get_registry() -> BenchmarkRegistry:
    """Get the global benchmark registry."""
    return _global_registry


def register_benchmark(name: str, 
                      benchmark_class: Type[BaseBenchmark], 
                      agent_type: AgentType,
                      description: str = ""):
    """Register a benchmark in the global registry."""
    _global_registry.register(name, benchmark_class, agent_type, description)


def unregister_benchmark(name: str):
    """Unregister a benchmark from the global registry."""
    _global_registry.unregister(name)


# Decorator for easy benchmark registration
def benchmark(name: str, agent_type: AgentType, description: str = ""):
    """Decorator to register a benchmark class."""
    def decorator(cls: Type[BaseBenchmark]) -> Type[BaseBenchmark]:
        register_benchmark(name, cls, agent_type, description)
        return cls
    return decorator 


# -------------------- Validation Benchmark Registry --------------------

class ValidationBenchmarkRegistry:
    """Registry for validation-only benchmarks (can map to multiple agent types)."""

    def __init__(self):
        self._benchmarks: Dict[str, Type[BaseBenchmark]] = {}
        self._agent_types_map: Dict[str, List[AgentType]] = {}
        self._description_map: Dict[str, str] = {}

    def register(self, name: str, benchmark_class: Type[BaseBenchmark], agent_types: List[AgentType], description: str = ""):
        if name in self._benchmarks:
            logger.warning(f"Overriding existing validation benchmark: {name}")
        self._benchmarks[name] = benchmark_class
        self._agent_types_map[name] = list(agent_types or [])
        self._description_map[name] = description
        logger.info(f"Registered validation benchmark: {name} (types: {[t.value for t in agent_types]})")

    def unregister(self, name: str):
        if name in self._benchmarks:
            del self._benchmarks[name]
            del self._agent_types_map[name]
            del self._description_map[name]
            logger.info(f"Unregistered validation benchmark: {name}")
        else:
            logger.warning(f"Validation benchmark not found for unregistration: {name}")

    def get_benchmark_class(self, name: str) -> Optional[Type[BaseBenchmark]]:
        return self._benchmarks.get(name)

    def get_agent_types(self, name: str) -> List[AgentType]:
        return list(self._agent_types_map.get(name, []))

    def get_description(self, name: str) -> str:
        return self._description_map.get(name, "")

    def list_benchmarks(self) -> List[str]:
        return list(self._benchmarks.keys())

    def get_registry_info(self) -> Dict:
        return {
            "total_validation_benchmarks": len(self._benchmarks),
            "validation_benchmarks": {
                name: {
                    "agent_types": [t.value for t in self._agent_types_map.get(name, [])],
                    "description": self._description_map.get(name, ""),
                    "class": self._benchmarks[name].__name__,
                }
                for name in self._benchmarks.keys()
            }
        }


_validation_registry = ValidationBenchmarkRegistry()


def get_validation_registry() -> ValidationBenchmarkRegistry:
    return _validation_registry


def register_validation_benchmark(name: str, benchmark_class: Type[BaseBenchmark], agent_types: List[AgentType], description: str = ""):
    _validation_registry.register(name, benchmark_class, agent_types, description)


def unregister_validation_benchmark(name: str):
    _validation_registry.unregister(name)


def validation_benchmark(name: str, agent_types: List[AgentType], description: str = ""):
    """Decorator to register a validation benchmark class with multiple agent types."""
    def decorator(cls: Type[BaseBenchmark]) -> Type[BaseBenchmark]:
        register_validation_benchmark(name, cls, agent_types, description)
        return cls
    return decorator