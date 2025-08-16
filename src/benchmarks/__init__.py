"""
Benchmark implementations for AgenticEvals.
"""

# Import all benchmark modules to ensure they register themselves
from . import simple_reflex_example
from . import simple_reflex_email
from . import model_based_maze
from . import hotel_booking
from . import portfolio_optimization
from . import pathfinding
from . import inventory_management
from . import ball_drop
from . import task_scheduling
from . import simulated_market_learning
from . import simple_reflex_fraud_detection
from . import event_conflict_detection
from . import local_web_navigation
from . import manufacturing_optimization

__all__ = [
    "simple_reflex_example",
    "simple_reflex_email",
    "model_based_maze",
    "hotel_booking",
    "portfolio_optimization",
    "pathfinding",
    "inventory_management",
    "ball_drop",
    "task_scheduling",
    "simulated_market_learning",
    "simple_reflex_fraud_detection",
    "event_conflict_detection",
    "local_web_navigation",
    "manufacturing_optimization",
] 