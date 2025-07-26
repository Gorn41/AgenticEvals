"""
Goal-Based Pathfinding Benchmark for AgenticEvals.

This benchmark tests a model's ability to find the shortest path in a directed,
weighted graph, given a full adjacency matrix. It evaluates planning and
reasoning over structured data.
"""

import heapq
import itertools
import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..benchmark.base import (AgentType, BaseBenchmark, BenchmarkConfig, Task,
                              TaskResult)
from ..benchmark.registry import benchmark
from ..models.base import BaseModel, ModelResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Graph:
    """Represents a directed, weighted graph."""
    nodes: List[str]
    adjacency_matrix: List[List[Optional[int]]]

    def get_weight(self, from_node: str, to_node: str) -> Optional[int]:
        """Gets the weight of the edge between two nodes."""
        try:
            from_idx = self.nodes.index(from_node)
            to_idx = self.nodes.index(to_node)
            return self.adjacency_matrix[from_idx][to_idx]
        except (ValueError, IndexError):
            return None


def dijkstra(graph: Graph, start_node: str) -> Dict[str, Tuple[Optional[int], Optional[List[str]]]]:
    """
    Dijkstra's algorithm to find the shortest paths from a single source node.
    Returns a dictionary mapping each node to a tuple of (total_weight, path).
    """
    if start_node not in graph.nodes:
        return {}

    distances = {node: (float('inf'), []) for node in graph.nodes}
    distances[start_node] = (0, [start_node])
    
    priority_queue = [(0, start_node, [start_node])]  # (weight, node, path)

    while priority_queue:
        current_weight, current_node, current_path = heapq.heappop(priority_queue)

        if current_weight > distances[current_node][0]:
            continue

        from_idx = graph.nodes.index(current_node)
        for to_idx, weight in enumerate(graph.adjacency_matrix[from_idx]):
            if weight is not None:
                to_node = graph.nodes[to_idx]
                distance = current_weight + weight
                if distance < distances[to_node][0]:
                    new_path = current_path + [to_node]
                    distances[to_node] = (distance, new_path)
                    heapq.heappush(priority_queue, (distance, to_node, new_path))
    
    # Final cleanup for nodes that are unreachable
    for node, (dist, path) in distances.items():
        if dist == float('inf'):
            distances[node] = (None, None)
            
    return distances


def find_optimal_multi_goal_path(graph: Graph, start_node: str, goal_nodes: List[str]) -> Tuple[Optional[int], Optional[List[str]]]:
    """
    Finds the shortest path from a start node that visits all goal nodes.

    This is done by calculating all-pairs shortest paths first, and then checking
    all permutations of the goal order to find the minimum total travel distance.
    """
    if not goal_nodes:
        return 0, [start_node]
    
    # 1. Calculate all-pairs shortest paths using Dijkstra from each relevant node
    all_pairs_paths = {}
    nodes_to_query = [start_node] + goal_nodes
    for node in nodes_to_query:
        all_pairs_paths[node] = dijkstra(graph, node)

    best_total_weight = float('inf')
    best_full_path = None

    # 2. Iterate through all permutations of goal nodes
    for permutation in itertools.permutations(goal_nodes):
        current_total_weight = 0
        current_full_path = []
        is_possible = True
        
        # Path from start to the first goal in the permutation
        first_goal = permutation[0]
        start_to_first_goal_w, start_to_first_goal_p = all_pairs_paths[start_node][first_goal]
        
        if start_to_first_goal_w is None:
            is_possible = False
        else:
            current_total_weight += start_to_first_goal_w
            current_full_path.extend(start_to_first_goal_p[:-1])

        # Path between consecutive goals in the permutation
        if is_possible:
            for i in range(len(permutation) - 1):
                from_goal = permutation[i]
                to_goal = permutation[i+1]
                
                inter_goal_w, inter_goal_p = all_pairs_paths[from_goal][to_goal]

                if inter_goal_w is None:
                    is_possible = False
                    break
                else:
                    current_total_weight += inter_goal_w
                    current_full_path.extend(inter_goal_p[:-1])
            
        if is_possible:
            # Add the final goal to the path
            current_full_path.append(permutation[-1])
            
            if current_total_weight < best_total_weight:
                best_total_weight = current_total_weight
                best_full_path = current_full_path

    if best_total_weight == float('inf'):
        return None, None
        
    return best_total_weight, best_full_path


@benchmark(
    name="shortest_path_planning",
    agent_type=AgentType.GOAL_BASED,
    description="Find the shortest path in a directed, weighted graph."
)
class PathfindingBenchmark(BaseBenchmark):
    """
    Benchmark for shortest pathfinding in a directed, weighted graph.
    """
    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)

    def get_tasks(self) -> List[Task]:
        """Defines the scenarios for the pathfinding benchmark."""
        scenarios = [
            # 1. Increased complexity from the original "Simple 4-Node Path"
            {
                "name": "Dense 4-Node with a Feeder Loop",
                "difficulty": "easy",
                "graph": {
                    "nodes": ["A", "B", "C", "D"],
                    "adjacency_matrix": [
                        [None, 2, 9, None],
                        [None, None, 1, 6],
                        [None, 4, None, 1],
                        [1, None, None, None] # Loop back to A
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["D"]
            },
            # 2. Increased complexity from "5-Node with a Trap"
            {
                "name": "5-Node with Multiple Traps",
                "difficulty": "medium",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E"],
                    "adjacency_matrix": [
                        [None, 1, 15, 4, None],
                        [None, None, None, 1, 10],
                        [None, 2, None, 5, None],
                        [None, None, 1, None, 3],
                        [None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["E"]
            },
            # 3. Increased complexity from "Medium 6-Node Graph"
            {
                "name": "Dense 6-Node with Crossroads",
                "difficulty": "medium",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F"],
                    "adjacency_matrix": [
                        [None, 2, 4, None, 9, None],
                        [None, None, 1, 7, None, None],
                        [None, 1, None, 3, None, None],
                        [None, None, None, None, 1, 5],
                        [None, None, 2, None, None, 1],
                        [None, None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["F"]
            },
            # 4. Increased complexity from "Graph with Dead End"
            {
                "name": "Graph with Multiple Dead Ends",
                "difficulty": "medium",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E"],
                    "adjacency_matrix": [
                        [None, 1, None, None, 12],
                        [None, None, 2, 9, None],
                        [None, None, None, None, 3],
                        [None, None, 1, None, None],
                        [None, None, None, 2, None] # Dead end
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["E"]
            },
            # 5. more tempting fake paths
            {
                "name": "Goal with Sinks",
                "difficulty": "hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E"],
                    "adjacency_matrix": [
                        [None, 1, 1, None, None],
                        [None, None, None, 1, None],
                        [None, None, None, None, 1],
                        [None, None, None, None, None], # Sink
                        [None, None, None, None, None]  # Sink
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["D"]
            },
            # 6. Denser version of "Simple Multi-Goal"
            {
                "name": "Dense Multi-Goal",
                "difficulty": "hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E"],
                    "adjacency_matrix": [
                        [None, 1, 10, None, 9],
                        [None, None, 2, None, 3],
                        [None, None, None, 3, None],
                        [None, None, None, None, 1],
                        [2, 1, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["C", "E"]
            },
            # 7. Denser version of "Complex Multi-Goal Order Matters"
            {
                "name": "Very Dense Multi-Goal",
                "difficulty": "very_hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F"],
                    "adjacency_matrix": [
                        [None, 2, 10, None, None, 20],
                        [None, None, None, 2, 15, None],
                        [None, None, None, None, 2, 5],
                        [2, None, 1, None, None, 8],
                        [None, 1, None, 3, None, 1],
                        [None, None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["F", "C"]
            },
            # 8. Denser 8-node graph
            {
                "name": "Very Dense 8-Node Graph",
                "difficulty": "hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G", "H"],
                    "adjacency_matrix": [
                        [None, 1, 2, 8, None, None, None, None],
                        [None, None, 3, 4, 5, None, None, None],
                        [None, None, None, None, None, 6, 7, None],
                        [None, None, None, None, 3, None, 12, None],
                        [None, 2, None, None, None, None, 1, None],
                        [None, None, None, 7, None, None, None, 8],
                        [None, None, None, None, None, 2, None, 2],
                        [None, None, None, None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["H"]
            },
            # 9. Denser "Dense 6-Node Graph"
            {
                "name": "Extremely Dense 6-Node Graph",
                "difficulty": "hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F"],
                    "adjacency_matrix": [
                        [None, 2, 9, 1, 10, 12],
                        [3, None, 3, None, 1, 5],
                        [None, 2, None, 2, None, 8],
                        [None, 6, 1, None, 4, 2],
                        [None, 3, None, None, None, 1],
                        [None, None, None, 2, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["F"]
            },
            # 10. Denser "Graph with Cycle Trap"
            {
                "name": "Graph with Multiple Cycle Traps",
                "difficulty": "very_hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F"],
                    "adjacency_matrix": [
                        [None, 3, None, None, 10, None],
                        [None, None, 1, 8, None, None],
                        [None, None, None, 1, None, 20],
                        [None, 1, None, None, 1, 7],
                        [None, None, 1, None, None, 1],
                        [None, None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["F"]
            },
            # 11. Denser "Three-Goal Challenge"
            {
                "name": "Dense Three-Goal Challenge",
                "difficulty": "very_hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G"],
                    "adjacency_matrix": [
                        [None, 1, None, 2, 15, None, None],
                        [None, None, 5, None, 7, None, None],
                        [None, None, None, None, 1, 2, 10],
                        [None, 3, 8, None, None, None, None],
                        [None, None, None, 2, None, None, 1],
                        [None, None, 3, None, None, None, None],
                        [None, None, None, 1, None, 1, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["B", "F", "G"]
            },
            # 12. Denser "Expansive 10-Node Graph"
            {
                "name": "Dense 10-Node Graph",
                "difficulty": "extremely_hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                    "adjacency_matrix": [
                        [None, 1, None, 12, None, 7, None, None, 20, None],
                        [None, None, 2, 3, None, None, None, 18, None, None],
                        [None, None, None, None, None, None, None, None, 4, None],
                        [None, None, None, None, 1, None, 15, None, None, None],
                        [None, None, 3, None, None, 1, None, None, None, None],
                        [None, None, None, 2, None, None, 1, None, 10, None],
                        [None, None, None, None, 2, None, None, 1, None, None],
                        [None, None, None, None, None, None, None, None, 1, 1],
                        [None, None, None, None, None, None, 3, None, None, 2],
                        [None, None, None, None, None, None, None, 1, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["J"]
            },
            # 13. Denser "Deceptive Path Weights"
            {
                "name": "Very Deceptive Path Weights",
                "difficulty": "extremely_hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G", "H"],
                    "adjacency_matrix": [
                        [None, 15, 15, 2, None, None, None, 20],
                        [None, None, None, None, None, None, None, 1],
                        [None, None, None, None, None, None, None, 2],
                        [None, 2, None, None, 2, None, 18, None],
                        [None, None, 2, None, None, 2, None, None],
                        [None, None, None, None, None, None, 2, None],
                        [None, None, None, 3, None, None, None, 2],
                        [None, None, None, None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["H"]
            },
            # 14. Denser "The Trifecta"
            {
                "name": "The Trifecta",
                "difficulty": "extremely_hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G", "H"],
                    "adjacency_matrix": [
                        [None, 1, 30, 30, None, None, 15, None],
                        [None, None, 2, None, 25, None, None, None],
                        [None, None, None, 3, 10, None, None, None],
                        [None, None, None, None, None, 22, None, 4],
                        [None, None, 4, None, None, 1, None, 1],
                        [1, None, None, None, None, None, 1, None],
                        [None, 12, None, None, 1, None, None, None],
                        [None, None, None, 5, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["D", "F", "H"]
            },
            # 15. Denser "The Maze"
            {
                "name": "The 4x3 Maze",
                "difficulty": "extremely_hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"],
                    "adjacency_matrix": [
                        [None, 1, None, None, 1, None, None, None, None, None, None, None], # A
                        [None, None, 1, None, None, 1, None, None, None, None, None, None], # B
                        [None, None, None, 1, None, None, 1, None, None, None, None, None], # C
                        [None, None, None, None, None, None, None, 1, None, None, None, None], # D
                        [None, 1, None, None, None, 1, None, None, 1, None, None, None], # E
                        [None, None, 1, None, None, None, 1, None, None, 1, None, None], # F
                        [None, None, None, 1, None, None, None, 1, None, None, 1, None], # G
                        [None, None, None, None, None, None, None, None, None, None, None, 1], # H
                        [None, None, None, None, None, None, None, None, None, 1, None, None], # I
                        [None, None, None, None, None, None, None, None, None, None, 1, None], # J
                        [None, None, None, None, None, None, None, 1, None, None, None, 1], # K
                        [None, None, None, None, None, None, None, None, None, None, None, None]  # L
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["L", "D"]
            },
            # 16. The Gauntlet: 15 nodes, 4 goals, very dense
            {
                "name": "The Gauntlet",
                "difficulty": "masterclass",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"],
                    "adjacency_matrix": [
                        [None, 1, 1, 20, 20, None, None, None, None, None, None, None, None, None, None],
                        [None, None, None, 2, 1, None, None, None, None, None, None, None, None, None, None],
                        [None, 2, None, None, None, 1, 1, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, 1, None, None, 50, None, None], # D (Goal) -> J
                        [None, None, None, None, None, None, 2, 2, None, None, None, None, None, None, None],
                        [None, None, None, None, 3, None, None, None, 1, None, None, None, None, None, None],
                        [None, None, None, 50, None, None, None, None, None, None, 1, None, None, None, None], # G (Goal) -> D, K
                        [None, None, None, None, None, 2, 1, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, 1, None, None, None, 2, 2, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, 1, 1, None], # J (Goal) -> M, N
                        [None, None, None, None, None, None, None, None, None, 2, None, None, None, None, 1],
                        [None, None, None, None, None, None, None, None, None, None, 1, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], # M (Goal)
                        [None, None, None, None, None, None, None, None, None, None, None, 1, 1, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, 1, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["D", "G", "J", "M"]
            },
            # 17. The Labyrinth: 16 nodes, 5 goals
            {
                "name": "The Labyrinth",
                "difficulty": "masterclass",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"],
                    "adjacency_matrix": [
                        [None, 2, 2, None, None, None, None, None, None, None, None, 20, None, None, None, None],
                        [None, None, None, 3, 3, None, None, None, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, 4, 4, None, None, None, None, None, None],
                        [None, None, None, None, None, 1, None, None, None, None, None, None, None, None, None, None],
                        [None, None, None, 1, None, None, 1, None, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, 2, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, 2, None, None, None, None, None, None, 2, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, 1, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, 1, None, None, None, None, None, None],
                        [None, None, None, None, None, None, 1, None, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, 1, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, 1, None, None],
                        [None, None, None, None, 2, None, None, None, None, None, None, None, None, None, 1, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1],
                        [None, None, None, None, None, None, None, None, None, 1, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["C", "E", "I", "M", "P"]
            },
            # 18. The Web: 18 nodes, 4 goals, extremely connected
            {
                "name": "The Web",
                "difficulty": "masterclass",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R"],
                    "adjacency_matrix": [
                        [None, 1, 1, 1, 1, None, None, None, None, None, None, None, None, None, None, None, None, None],
                        [None, None, 2, None, None, None, 20, 20, None, None, None, None, None, None, None, None, None, None],
                        [None, None, None, 2, None, None, None, None, 20, 20, None, None, None, None, None, None, None, None],
                        [None, None, None, None, 2, None, None, None, None, None, 20, 20, None, None, None, None, None, None],
                        [None, None, None, None, None, 2, None, None, None, None, None, None, 20, 20, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1, 1, None, None], # F (Goal)
                        [10, None, None, None, None, None, None, 3, None, None, None, None, None, None, None, None, None, None],
                        [None, 10, None, None, None, None, None, None, 3, None, None, None, None, None, None, None, None, None],
                        [None, None, 10, None, None, None, None, None, None, 3, None, None, None, None, None, None, None, None],
                        [None, None, None, 10, None, None, None, None, None, None, 3, None, None, None, None, None, None, None],
                        [None, None, None, None, 10, None, None, None, None, None, None, 3, None, None, None, None, None, None],
                        [None, None, None, None, None, 10, None, None, None, None, None, None, 3, None, None, None, None, None], # K (Goal)
                        [None, None, None, None, None, None, None, None, None, None, None, 10, None, 3, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, 10, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 4, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 4], # P (Goal)
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]  # R (Goal)
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["F", "K", "P", "R"]
            },
            # 19. The Citadel: 20 nodes, 5 goals, layered graph
            {
                "name": "The Citadel",
                "difficulty": "masterclass",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"],
                    "adjacency_matrix": [
                        [None, 1, None, None, 1, None, None, None, None, None, 10, None, None, None, None, None, None, None, None, None], # A (Outer Ring)
                        [None, None, 1, None, None, None, None, None, None, None, None, 10, None, None, None, None, None, None, None, None], # B
                        [None, None, None, 1, None, None, None, None, None, None, None, None, 10, None, None, None, None, None, None, None], # C
                        [None, None, None, None, 1, None, None, None, None, None, None, None, None, 10, None, None, None, None, None, None], # D
                        [1, None, None, None, None, None, None, None, None, None, None, None, None, None, 10, None, None, None, None, None], # E (Outer Ring Goal)
                        [None, None, None, None, None, None, 1, None, None, 1, None, None, None, None, None, 15, None, None, None, None], # F (Middle Ring)
                        [None, None, None, None, None, None, None, 1, None, None, None, None, None, None, None, None, 15, None, None, None], # G
                        [None, None, None, None, None, None, None, None, 1, None, None, None, None, None, None, None, None, 15, None, None], # H
                        [None, None, None, None, None, 1, None, None, None, None, None, None, None, None, None, None, None, None, 15, None], # I
                        [None, None, None, None, None, None, None, None, 1, None, None, None, None, None, None, None, None, None, None, 15], # J (Middle Ring Goal)
                        [10, None, None, None, None, None, None, None, None, None, None, 2, None, None, 2, None, None, None, None, None], # K (Inner Ring)
                        [None, 10, None, None, None, None, None, None, None, None, None, None, 2, None, None, None, None, None, None, None], # L
                        [None, None, 10, None, None, None, None, None, None, None, None, None, None, 2, None, None, None, None, None, None], # M
                        [None, None, None, 10, None, None, None, None, None, None, None, None, None, None, 2, None, None, None, None, None], # N
                        [None, None, None, None, 10, None, None, None, None, None, 2, None, None, None, None, None, None, None, None, None], # O (Inner Ring Goal)
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 3, None, None, 3], # P (Core)
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 3, None, None], # Q
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 3, None], # R
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 3, None, None, None, None], # S
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], # T (Core Goal)
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["E", "J", "O", "T", "S"]
            },
            # 20. The Penultimate Test: 20 nodes, 6 goals, with a few low-cost "secret" paths
            {
                "name": "The Penultimate Test",
                "difficulty": "masterclass",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"],
                    "adjacency_matrix": [
                        [None, 50, 50, 50, 50, 50, 50, 50, 1, None, None, None, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, 1, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1],
                        [None, 100, 100, 100, 100, 100, 100, 100, None, 1, 1, 1, 1, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, 1, 1, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1, 1, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1, 1, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 1, 1, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["N", "O", "P", "Q", "R", "S", "T"]
            },
            # 21. Fully Connected Chaos: 15 nodes, 5 goals, fully connected with high variance weights
            {
                "name": "Fully Connected Chaos",
                "difficulty": "masterclass",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"],
                    "adjacency_matrix": [
                        [None, 11, 237, 233, 107, 106, 175, 177, 13, 248, 110, 241, 151, 101, 134],
                        [14, None, 153, 218, 169, 135, 12, 112, 132, 158, 77, 18, 150, 249, 140],
                        [113, 161, None, 192, 10, 222, 115, 235, 133, 50, 16, 118, 100, 201, 62],
                        [223, 227, 21, None, 70, 78, 234, 170, 40, 148, 28, 1, 22, 149, 83],
                        [15, 194, 183, 114, None, 244, 23, 120, 211, 19, 128, 141, 236, 109, 215],
                        [213, 182, 168, 56, 163, None, 41, 186, 152, 232, 165, 30, 93, 245, 179],
                        [54, 2, 91, 124, 242, 195, None, 187, 85, 212, 53, 121, 181, 17, 185],
                        [208, 92, 246, 190, 8, 230, 20, None, 203, 160, 214, 184, 73, 224, 206],
                        [24, 5, 205, 108, 166, 125, 143, 60, None, 157, 172, 80, 207, 7, 219],
                        [138, 202, 126, 61, 164, 4, 173, 94, 9, None, 142, 155, 39, 217, 111],
                        [247, 98, 193, 162, 176, 103, 127, 26, 210, 86, None, 225, 45, 87, 228],
                        [229, 238, 59, 159, 197, 136, 90, 79, 178, 58, 198, None, 231, 154, 49],
                        [44, 104, 250, 137, 240, 199, 122, 33, 146, 220, 68, 46, None, 191, 6],
                        [174, 117, 43, 200, 31, 25, 188, 52, 226, 97, 116, 209, 189, None, 167],
                        [129, 3, 221, 64, 145, 131, 89, 27, 204, 171, 96, 243, 75, 55, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["E", "H", "K", "N", "O"]
            }
        ]

        tasks = []
        for i, scenario in enumerate(scenarios):
            graph = Graph(nodes=scenario["graph"]["nodes"], adjacency_matrix=scenario["graph"]["adjacency_matrix"])
            start_node = scenario["start_node"]
            goal_nodes = scenario["goal_nodes"]

            optimal_weight, optimal_path = find_optimal_multi_goal_path(graph, start_node, goal_nodes)

            task = Task(
                task_id=f"pathfinding_{i+1}",
                name=scenario["name"],
                description=f"Find the shortest path from {start_node} to {goal_nodes}",
                prompt=self._create_prompt(graph, start_node, goal_nodes),
                evaluation_criteria={
                    "optimal_weight": optimal_weight,
                    "optimal_path": optimal_path
                },
                metadata={
                    "difficulty": scenario["difficulty"],
                    "graph": scenario["graph"],
                    "start_node": start_node,
                    "goal_nodes": goal_nodes
                }
            )
            tasks.append(task)
        return tasks

    def _create_prompt(self, graph: Graph, start_node: str, goal_nodes: List[str]) -> str:
        """Creates a formatted prompt for the model."""
        
        matrix_pretty = json.dumps(graph.adjacency_matrix, indent=4)
        
        # Adjust prompt for single vs. multi-goal
        path_clarification = ""
        if len(goal_nodes) == 1:
            goal_description = f"the goal node '{goal_nodes[0]}'"
            example_format = "A->B->D->E"
        else:
            goal_description = f"all of the following goal nodes: {goal_nodes}"
            path_clarification = "\n\nFor multiple goals, you must provide a single continuous path starting at the start node that visits all goal nodes in the most efficient order. The goals are not necessarily in the most efficient order, so you must figure out the most efficient order such that the path is as short as possible. The path ends when the last goal is visited."
            example_format = "A->C->E->B->D"

        return f"""
TASK: Find the shortest path in a directed, weighted graph.

DESCRIPTION:
You are given a graph represented by an adjacency matrix. Your task is to find the sequence of nodes that forms the shortest valid continuous path from the start node to {goal_description}. The 'shortest' path is defined as the path with the minimum possible sum of weights, not necessarily the path with the fewest nodes.{path_clarification}

GRAPH DEFINITION:
- Nodes: {graph.nodes}
- Adjacency Matrix: The value at matrix[row][col] is the weight (distance) of the directed edge from `nodes[row]` to `nodes[col]`. `null` means no direct edge exists.
{matrix_pretty}

YOUR TASK:
- Start Node: {start_node}
- Goal Node(s): {goal_nodes}

OUTPUT FORMAT:
First, provide a step-by-step reasoning of how you found the shortest path. Explain your choices, especially when dealing with multiple goals or deceptive paths.

After your reasoning, provide the final answer on a new line in the following format:
[answer: <path>]

The <path> should be a sequence of nodes separated by '->'. The sequence must include the start node at the beginning and a goal node at the end. For example: [answer: A->C->E->B->D]

If you determine that no valid path exists to reach the goal(s), output [answer: None] instead.

IMPORTANT:
- Your objective is to find the path with the minimum possible total weight (total distance).
- Each step in your path MUST correspond to a valid, directed edge in the graph. You cannot travel between nodes where an edge does not exist.
- Your final answer must be enclosed in the [answer: ...] tag.

"""

    def _parse_path_response(self, response_text: str) -> List[str]:
        """
        Parses the model's text response to extract the path sequence, handling "None" responses.
        """
        if not response_text:
            return []

        # 1. Look for [answer: <path>] tag first.
        answer_match = re.search(r'\[answer:\s*([^\]]+)\]', response_text, re.IGNORECASE)
        if answer_match:
            content = answer_match.group(1).strip()
            if content.lower() == 'none':
                return ['None']
            # If tag exists, we trust its content and don't fall back.
            return self._parse_string_for_path(content)

        # 2. Fallback logic if tag is not found.
        # Find the end position of the last match for any path-like pattern
        last_path_pos = -1
        last_path_str = ""

        path_patterns = [
            r'[A-Z](?:\s*->\s*[A-Z])+',  # Arrow-based paths
            r'[A-Z](?:\s*[A-Z])+'       # Space/contiguous capital letter paths
        ]
        
        for pattern in path_patterns:
            matches = list(re.finditer(pattern, response_text))
            if matches:
                last_match = matches[-1]
                if last_match.end() > last_path_pos:
                    last_path_pos = last_match.end()
                    last_path_str = last_match.group(0)

        # Find the start position of the last match for "none"
        last_none_pos = -1
        none_matches = list(re.finditer(r'\bnone\b', response_text, re.IGNORECASE))
        if none_matches:
            last_none_pos = none_matches[-1].start()

        # Compare positions
        if last_none_pos > last_path_pos:
            return ['None']
        
        if last_path_str:
            return re.findall(r'[A-Z]', last_path_str)

        return []

    def _parse_string_for_path(self, text: str) -> List[str]:
        """Helper function to parse a string and extract the most likely path."""
        # Pre-process to handle newlines.
        processed_text = text.replace('\n', ' ')

        # Fallback 1: Find all path-like strings (e.g., A -> B -> C) and take the last one.
        # This regex handles optional spaces around the arrow.
        path_matches = re.findall(r'[A-Z](?:\s*->\s*[A-Z])+', processed_text)
        if path_matches:
            last_path = path_matches[-1]
            return re.findall(r'[A-Z]', last_path)

        # Fallback 2: Find all sequences of capital letters (with or without spaces) and take the last one.
        # This handles formats like "A B C" and "ABC".
        capital_matches = re.findall(r'[A-Z](?:\s*[A-Z])+', processed_text)
        if capital_matches:
            last_match = capital_matches[-1]
            # Extract letters from the match string 'A B C' or 'ABC'
            return re.findall(r'[A-Z]', last_match)

        return []


    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        """Evaluates the model's performance on a pathfinding task."""
        start_time = time.time()
        
        try:
            model_response = await model.generate(task.prompt)
            execution_time = time.time() - start_time
            
            # 1. Parse the model's response
            path_nodes = self._parse_path_response(model_response.text)
            
            # 2. Get ground truth from task metadata
            graph = Graph(**task.metadata['graph'])
            start_node = task.metadata['start_node']
            goal_nodes = set(task.metadata['goal_nodes'])
            optimal_weight = task.evaluation_criteria['optimal_weight']

            # 3. Handle "None" responses
            if path_nodes == ['None']:
                if optimal_weight is None:
                    # Correctly said None when no path exists
                    return self._create_task_result(task, model_response, execution_time, 1.0, "success", "Correctly identified unreachable goal.")
                else:
                    # Incorrectly said None when a path does exist
                    return self._create_task_result(task, model_response, execution_time, 0.0, "invalid_path", "Incorrectly stated no path exists.", {"model_path_raw": model_response.text})
            
            # 4. Handle unparseable or empty responses
            if not path_nodes:
                return self._create_task_result(task, model_response, execution_time, 0.0, "invalid_path", "Response was empty or unparseable.", {"model_path_raw": model_response.text})
            
            # If optimal_weight is None but model provides a path, it's a failure.
            if optimal_weight is None and path_nodes:
                return self._create_task_result(task, model_response, execution_time, 0.0, "invalid_path", "Model hallucinated a path to an unreachable goal.", {"model_path_raw": model_response.text, "model_path": "->".join(path_nodes)})

            # 5. Initial validation of the path
            if path_nodes[0] != start_node:
                return self._create_task_result(task, model_response, execution_time, 0.0, "invalid_path", "Path did not start at the correct node.", {"model_path_raw": model_response.text, "model_path": "->".join(path_nodes)})
            
            if path_nodes[-1] not in goal_nodes:
                return self._create_task_result(task, model_response, execution_time, 0.0, "invalid_path", "Path did not end at a valid goal node.", {"model_path_raw": model_response.text, "model_path": "->".join(path_nodes)})

            # Check if all required goals were visited
            if not goal_nodes.issubset(set(path_nodes)):
                return self._create_task_result(task, model_response, execution_time, 0.0, "invalid_path", "Path did not visit all required goal nodes.", {"model_path_raw": model_response.text, "model_path": "->".join(path_nodes)})

            # 6. Calculate the weight of the model's path and check for invalid edges
            model_path_weight = 0
            for i in range(len(path_nodes) - 1):
                from_node, to_node = path_nodes[i], path_nodes[i+1]
                weight = graph.get_weight(from_node, to_node)
                if weight is None:
                    return self._create_task_result(task, model_response, execution_time, 0.0, "invalid_path", f"Path contains an invalid edge from {from_node} to {to_node}.", {"model_path_raw": model_response.text, "model_path": "->".join(path_nodes)})
                model_path_weight += weight
            
            # 7. Calculate score based on weight comparison
            score = 0.0
            if optimal_weight is not None and model_path_weight is not None:
                if model_path_weight == 0:
                    # If model path weight is 0, score is 1.0 only if optimal is also 0.
                    score = 1.0 if optimal_weight == 0 else 0.0
                else:
                    # Score is the ratio of optimal weight to model's path weight.
                    score = optimal_weight / model_path_weight

            # Final score should be clamped between 0 and 1.
            score = max(0.0, min(1.0, score))

            # 8. Determine success based on whether the model reached the goal properly (or correctly said no path exists)
            success = score > 0.0
            status = "success" if success else "failure"
            message = (
                "Path evaluated successfully and met threshold."
                if success
                else "Path was valid, but did not meet performance threshold (score <= 0.7)."
            )

            # 9. Create final result object
            return self._create_task_result(
                task,
                model_response,
                execution_time,
                score,
                status,
                message,
                extra_metrics={
                    "model_path_weight": model_path_weight,
                    "model_path": path_nodes
                }
            )

        except Exception as e:
            logger.error(f"Error evaluating task {task.task_id}: {str(e)}")
            return self._create_task_result(task, None, time.time() - start_time, 0.0, "error", str(e))

    def _create_task_result(self, task: Task, model_response: Optional[ModelResponse], execution_time: float, score: float, status: str, message: str, extra_metrics: Dict = {}) -> TaskResult:
        """Helper to create a TaskResult object."""
        
        metrics = {
            "status": status,
            "message": message,
            "difficulty": task.metadata.get("difficulty"),
            "optimal_weight": task.evaluation_criteria.get("optimal_weight"),
            "optimal_path": "->".join(task.evaluation_criteria.get("optimal_path", []) or []),
            "output_tokens": model_response.completion_tokens if model_response else 0,
            "model_path_raw": model_response.text if model_response else ""
        }
        metrics.update(extra_metrics)

        return TaskResult(
            task_id=task.task_id,
            task_name=task.name,
            agent_type=self.agent_type,
            success=(status == "success"),
            score=score,
            metrics=metrics,
            model_response=model_response,
            execution_time=execution_time,
        )

    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        """This method is handled by evaluate_task for this benchmark."""
        return 0.0 