"""
Benchmark implementations for AgenticEvals.
"""

# Import all benchmark modules to ensure they register themselves
from . import traffic_light
from . import email_autoresponder
from . import textual_maze_navigation
from . import hotel_booking
from . import portfolio_optimization
from . import shortest_path_planning
from . import inventory_management
from . import ball_drop
from . import task_scheduling
from . import simulated_market
from . import fraud_detection
from . import event_conflict_detection
from . import local_web_navigation
from . import manufacturing_line_optimization
from . import ecosystem
from . import ev_charging_policy

__all__ = [
    "traffic_light",
    "email_autoresponder",
    "textual_maze_navigation",
    "hotel_booking",
    "portfolio_optimization",
    "shortest_path_planning",
    "inventory_management",
    "ball_drop",
    "task_scheduling",
    "simulated_market",
    "fraud_detection",
    "event_conflict_detection",
    "local_web_navigation",
    "manufacturing_line_optimization",
    "ecosystem",
    "ev_charging_policy",
]