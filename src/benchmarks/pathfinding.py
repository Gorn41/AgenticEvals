"""
Goal-Based Pathfinding Benchmark for AgenticEvals.

This benchmark tests a model's ability to find the shortest path in a directed,
weighted graph, given a full adjacency matrix. It evaluates planning and
reasoning over structured data.
"""

import heapq
import itertools
import json
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
            # Simple, small graph
            {
                "name": "Simple 4-Node Path",
                "difficulty": "easy",
                "graph": {
                    "nodes": ["A", "B", "C", "D"],
                    "matrix": [
                        [None, 1, 3, None],
                        [None, None, 1, 4],
                        [None, None, None, 1],
                        [None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["D"]
            },
            # A slightly more complex graph where the direct path isn't the shortest
            {
                "name": "5-Node with a Trap",
                "difficulty": "easy",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E"],
                    "matrix": [
                        [None, 1, 8, None, None],
                        [None, None, None, 1, None],
                        [None, 2, None, 5, None],
                        [None, None, 1, None, 3],
                        [None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["E"]
            },
            # Medium complexity with more nodes and edges
            {
                "name": "Medium 6-Node Graph",
                "difficulty": "medium",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F"],
                    "matrix": [
                        [None, 2, 4, None, None, None],
                        [None, None, 1, 7, None, None],
                        [None, None, None, 3, None, None],
                        [None, None, None, None, 1, 5],
                        [None, None, None, None, None, 1],
                        [None, None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["F"]
            },
            # A graph with a dead end
            {
                "name": "Graph with Dead End",
                "difficulty": "medium",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E"],
                    "matrix": [
                        [None, 1, None, None, None],
                        [None, None, 2, 9, None],
                        [None, None, None, None, 3], # Path to E (goal)
                        [None, None, 1, None, None], # Dead end path from D
                        [None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["E"]
            },
            # Unreachable goal
            {
                "name": "Unreachable Goal",
                "difficulty": "hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D"],
                    "matrix": [
                        [None, 1, None, None],
                        [None, None, 1, None],
                        [None, None, None, None],
                        [None, None, 1, None] # D is a sink with no path to it
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["D"]
            },
            # Multi-goal scenario
            {
                "name": "Simple Multi-Goal",
                "difficulty": "hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E"],
                    "matrix": [
                        [None, 1, None, None, 9],
                        [None, None, 2, None, None],
                        [None, None, None, 3, None],
                        [None, None, None, None, 1],
                        [None, 1, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["C", "E"]
            },
            # More complex multi-goal where order matters
            {
                "name": "Complex Multi-Goal Order Matters",
                "difficulty": "very_hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F"],
                    "matrix": [
                        [None, 2, 10, None, None, None], # A->B (2), A->C (10)
                        [None, None, None, 2, None, None], # B->D (2)
                        [None, None, None, None, 2, None], # C->E (2)
                        [None, None, 1, None, None, 8], # D->C (1), D->F (8)
                        [None, 1, None, None, None, 1], # E->B (1), E->F (1)
                        [None, None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["F", "C"]
            }
        ]

        harder_scenarios = [
            # Scenario 8: Larger single-goal graph
            {
                "name": "Large 8-Node Graph",
                "difficulty": "hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G", "H"],
                    "matrix": [
                        [None, 1, 2, None, None, None, None, None],
                        [None, None, None, 4, 5, None, None, None],
                        [None, None, None, None, None, 6, 7, None],
                        [None, None, None, None, 3, None, None, None],
                        [None, None, None, None, None, None, 1, None],
                        [None, None, None, None, None, None, None, 8],
                        [None, None, None, None, None, 2, None, 2],
                        [None, None, None, None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["H"]
            },
            # Scenario 9: Dense graph with many paths
            {
                "name": "Dense 6-Node Graph",
                "difficulty": "hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F"],
                    "matrix": [
                        [None, 2, 9, 1, None, None],
                        [None, None, 3, None, 1, 5],
                        [None, 2, None, None, None, 8],
                        [None, 6, 1, None, 4, 2],
                        [None, None, None, None, None, 1],
                        [None, None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["F"]
            },
            # Scenario 10: Graph with a low-cost cycle trap
            {
                "name": "Graph with Cycle Trap",
                "difficulty": "hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F"],
                    "matrix": [
                        [None, 3, None, None, 10, None],
                        [None, None, 1, None, None, None],
                        [None, None, None, 1, None, None],
                        [None, 1, None, None, None, 7], # Cycle back to B
                        [None, None, None, None, None, 1], # Direct but expensive path
                        [None, None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["F"]
            },
            # Scenario 11: Very hard multi-goal with 3 goals
            {
                "name": "Three-Goal Challenge",
                "difficulty": "very_hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G"],
                    "matrix": [
                        [None, 1, None, 2, None, None, None],
                        [None, None, 5, None, None, None, None],
                        [None, None, None, None, 1, 2, None],
                        [None, 3, None, None, None, None, None],
                        [None, None, None, None, None, None, 1],
                        [None, None, None, None, None, None, None],
                        [None, None, None, 1, None, 1, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["B", "F", "G"]
            },
            # Scenario 12: Very large 10-node graph
            {
                "name": "Expansive 10-Node Graph",
                "difficulty": "very_hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                    "matrix": [
                        [None, 1, None, None, None, 7, None, None, None, None],
                        [None, None, 2, 3, None, None, None, None, None, None],
                        [None, None, None, None, None, None, None, None, 4, None],
                        [None, None, None, None, 1, None, None, None, None, None],
                        [None, None, None, None, None, 1, None, None, None, None],
                        [None, None, None, 2, None, None, 1, None, None, None],
                        [None, None, None, None, None, None, None, 1, None, None],
                        [None, None, None, None, None, None, None, None, 1, 1],
                        [None, None, None, None, None, None, None, None, None, 2],
                        [None, None, None, None, None, None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["J"]
            },
            # Scenario 13: Deceptive path weights (long path is better)
            {
                "name": "Deceptive Path Weights",
                "difficulty": "very_hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G", "H"],
                    "matrix": [
                        [None, 12, 12, 2, None, None, None, None],
                        [None, None, None, None, None, None, None, 1], # A->B->H is 13
                        [None, None, None, None, None, None, None, 2], # A->C->H is 14
                        [None, None, None, None, 2, None, None, None],
                        [None, None, None, None, None, 2, None, None],
                        [None, None, None, None, None, None, 2, None],
                        [None, None, None, None, None, None, None, 2], # A->D->E->F->G->H is 10
                        [None, None, None, None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["H"]
            },
            # Scenario 14: Extremely hard multi-goal
            {
                "name": "The Trifecta",
                "difficulty": "extremely_hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G", "H"],
                    "matrix": [
                        [None, 1, None, None, None, None, 15, None],
                        [None, None, 2, None, None, None, None, None],
                        [None, None, None, 3, 10, None, None, None],
                        [None, None, None, None, None, None, None, 4],
                        [None, None, None, None, None, 1, None, 1],
                        [1, None, None, None, None, None, 1, None],
                        [None, None, None, None, 1, None, None, None],
                        [None, None, None, None, None, None, None, None]
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["D", "F", "H"]
            },
            # Scenario 15: The Maze
            {
                "name": "The 3x3 Maze",
                "difficulty": "extremely_hard",
                "graph": {
                    "nodes": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
                    "matrix": [
                        [None, 1, None, 1, None, None, None, None, None], # A
                        [None, None, 1, None, 1, None, None, None, None], # B
                        [None, None, None, None, None, 1, None, None, None], # C
                        [None, None, None, None, 1, None, 1, None, None], # D
                        [None, None, None, None, None, None, None, 1, None], # E
                        [None, None, 10, None, None, None, None, None, 1], # F (C is a trap)
                        [None, None, None, None, None, None, None, 1, None], # G
                        [None, None, None, None, None, None, None, None, 1], # H
                        [None, None, None, None, None, None, None, None, None], # I
                    ]
                },
                "start_node": "A",
                "goal_nodes": ["I", "C"]
            }
        ]
        scenarios.extend(harder_scenarios)

        tasks = []
        for i, scenario in enumerate(scenarios):
            graph = Graph(nodes=scenario["graph"]["nodes"], adjacency_matrix=scenario["graph"]["matrix"])
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
        if len(goal_nodes) == 1:
            goal_description = f"the goal node '{goal_nodes[0]}'"
            path_description = f"a path from '{start_node}' to '{goal_nodes[0]}'."
            example_format = "A->B->D->E"
        else:
            goal_description = f"all of the following goal nodes: {goal_nodes}"
            path_description = f"a single path that starts at '{start_node}' and visits all of the goal nodes in the most efficient order."
            example_format = "A->C->E->B->D"

        return f"""
TASK: Find the shortest path in a directed, weighted graph.

DESCRIPTION:
You are given a graph represented by an adjacency matrix. Your task is to find the sequence of nodes that forms the shortest path from the start node to {goal_description}. The path must be valid, meaning each step must correspond to a directed edge in the graph.

GRAPH DEFINITION:
- Nodes: {graph.nodes}
- Adjacency Matrix: The value at matrix[row][col] is the weight of the directed edge from `nodes[row]` to `nodes[col]`. `null` means no direct edge exists.
{matrix_pretty}

YOUR TASK:
- Start Node: {start_node}
- Goal Node(s): {goal_nodes}

OUTPUT FORMAT:
- Provide ONLY the sequence of nodes, separated by '->'.
- The sequence must include the start node at the beginning and a goal node at the end.
- Example: {example_format}

IMPORTANT:
- You must find the path with the minimum possible total weight.
- Your response should contain ONLY the path string. Do not include any other text, explanations, or headers.

OUTPUT THE PATH:
"""

    def _parse_path_response(self, response_text: str) -> List[str]:
        """Parses the model's text response to extract the path sequence."""
        if not response_text:
            return []
        
        # Normalize to a common delimiter '->'
        response_text = response_text.replace(" -> ", "->").replace(" , ", "->").replace(",", "->").replace(" ", "->")
        
        # Find the most likely path string, even if there's extra text
        path_match = re.search(r'([A-Z](?:->[A-Z])+)', response_text)
        
        if path_match:
            path_str = path_match.group(1)
            return [node.strip() for node in path_str.split('->')]
        
        # Fallback for simple sequences of letters
        # Handles "ABDE" or "A B D E"
        letters_only = "".join(re.findall(r'[A-Z]', response_text))
        if len(letters_only) > 1:
            return list(letters_only)
            
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
            
            # 3. Initial validation of the path
            if not path_nodes:
                return self._create_task_result(task, model_response, execution_time, 0.0, "invalid_path", "Response was empty or unparseable.")
            
            if path_nodes[0] != start_node:
                return self._create_task_result(task, model_response, execution_time, 0.0, "invalid_path", "Path did not start at the correct node.")
            
            if path_nodes[-1] not in goal_nodes:
                return self._create_task_result(task, model_response, execution_time, 0.0, "invalid_path", "Path did not end at a valid goal node.")

            # Check if all required goals were visited
            if not goal_nodes.issubset(set(path_nodes)):
                return self._create_task_result(task, model_response, execution_time, 0.0, "invalid_path", "Path did not visit all required goal nodes.")

            # 4. Calculate the weight of the model's path and check for invalid edges
            model_path_weight = 0
            for i in range(len(path_nodes) - 1):
                from_node, to_node = path_nodes[i], path_nodes[i+1]
                weight = graph.get_weight(from_node, to_node)
                if weight is None:
                    return self._create_task_result(task, model_response, execution_time, 0.0, "invalid_path", f"Path contains an invalid edge from {from_node} to {to_node}.")
                model_path_weight += weight
            
            # 5. Calculate the score
            final_score = 0.0
            if optimal_weight is not None and optimal_weight > 0:
                # If path is optimal, score is 1.0
                if model_path_weight == optimal_weight:
                    final_score = 1.0
                # If path is more than 3x optimal, score is 0.0
                elif model_path_weight >= optimal_weight * 3:
                    final_score = 0.0
                # Otherwise, linear scaling
                else:
                    # The score decreases as the model's path weight increases.
                    # The range of "badness" is from optimal_weight (good) to 3*optimal_weight (bad)
                    weight_range = (optimal_weight * 3) - optimal_weight
                    # How far into the "bad" range is the model's path?
                    model_deviation = model_path_weight - optimal_weight
                    # Normalize this deviation to a 0-1 scale (0 is best, 1 is worst)
                    penalty_ratio = model_deviation / weight_range
                    # The final score is 1 minus this penalty
                    final_score = 1.0 - penalty_ratio
            elif optimal_weight == 0 and model_path_weight == 0:
                final_score = 1.0 # Correctly found zero-cost path
            
            return self._create_task_result(task, model_response, execution_time, final_score, "success", "Path evaluated successfully.",
                extra_metrics={
                    "model_path_weight": model_path_weight,
                    "model_path": "->".join(path_nodes)
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