"""
Benchmark implementations for LLM-AgentTypeEval.
"""

# Import example benchmarks to auto-register them
try:
    from . import simple_reflex_example
except ImportError:
    pass

__all__ = [] 