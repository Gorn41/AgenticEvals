"""
AgenticEvals: A benchmark for evaluating LLMs across classic AI agent types.
"""

__version__ = "0.1.0"
__author__ = "Nattaput (Gorn) Namchittai"
__email__ = "gorn41@outlook.com"

from .models import load_gemini
from .benchmark import load_benchmark, get_available_benchmarks

__all__ = [
    "load_gemini",
    "load_benchmark", 
    "get_available_benchmarks"
] 