"""
Benchmark implementations for AgenticEvals.
"""

# Import all benchmark modules to ensure they register themselves
from . import simple_reflex_example
from . import simple_reflex_email
from . import model_based_maze

__all__ = [
    "simple_reflex_example",
    "simple_reflex_email",
    "model_based_maze"
] 