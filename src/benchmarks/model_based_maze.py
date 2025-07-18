"""
Textual Maze Navigation Model-Based Reflex Agent benchmark for AgenticEvals.

This benchmark tests a model's ability to navigate through a maze using partial observations
while maintaining internal state to avoid loops and reach the goal efficiently.
"""

import time
import json
import asyncio
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
from ..benchmark.base import BaseBenchmark, Task, TaskResult, BenchmarkConfig, AgentType
from ..benchmark.registry import benchmark
from ..models.base import BaseModel, ModelResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MazeState:
    """Represents the current state of maze navigation."""
    position: Tuple[int, int]
    visited_cells: Set[Tuple[int, int]]
    goal_position: Tuple[int, int]
    move_count: int
    total_tokens_used: int
    
    
@dataclass 
class MazeLayout:
    """Represents a maze layout."""
    grid: List[List[str]]  # '#' = wall, '.' = open, 'S' = start, 'G' = goal
    start_pos: Tuple[int, int]
    goal_pos: Tuple[int, int]
    optimal_path_length: int


@benchmark(
    name="textual_maze_navigation",
    agent_type=AgentType.MODEL_BASED_REFLEX,
    description="Textual maze navigation benchmark testing internal state management and memory"
)
class TextualMazeBenchmark(BaseBenchmark):
    """
    Model-based reflex agent benchmark using textual maze navigation.
    
    Tests the model's ability to maintain internal state, remember visited locations,
    and navigate efficiently through a maze using only partial observations.
    """
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
    
    def _create_maze_layouts(self) -> List[MazeLayout]:
        """Create different maze layouts for testing."""
        layouts = []
        
        # 1. Simple 4x4 maze (original)
        simple_4x4 = [
            ['S', '.', '#', '.'],
            ['.', '.', '#', '.'],
            ['#', '.', '.', '.'],
            ['#', '#', '.', 'G']
        ]
        layouts.append(MazeLayout(simple_4x4, (0, 0), (3, 3), 5))
        
        # 2. Medium 5x5 maze (original)
        medium_5x5 = [
            ['S', '.', '#', '.', '.'],
            ['.', '.', '#', '.', '#'],
            ['.', '#', '.', '.', '.'],
            ['.', '#', '#', '#', '.'],
            ['.', '.', '.', '.', 'G']
        ]
        layouts.append(MazeLayout(medium_5x5, (0, 0), (4, 4), 8))
        
        # 3. Complex 6x6 maze (original)
        complex_6x6 = [
            ['S', '.', '#', '.', '.', '.'],
            ['.', '.', '#', '.', '#', '.'],
            ['#', '.', '.', '.', '#', '.'],
            ['#', '#', '#', '.', '#', '.'],
            ['.', '.', '.', '.', '.', '.'],
            ['#', '#', '#', '.', '.', 'G']
        ]
        layouts.append(MazeLayout(complex_6x6, (0, 0), (5, 5), 10))
        
        # 4. Challenging 7x7 maze with multiple dead ends
        challenging_7x7 = [
            ['S', '.', '#', '.', '.', '.', '.'],
            ['.', '.', '#', '.', '#', '#', '.'],
            ['#', '.', '.', '.', '#', '.', '.'],
            ['#', '#', '#', '.', '#', '.', '#'],
            ['.', '.', '.', '.', '.', '.', '#'],
            ['.', '#', '#', '#', '#', '.', '.'],
            ['.', '.', '.', '.', '.', '.', 'G']
        ]
        layouts.append(MazeLayout(challenging_7x7, (0, 0), (6, 6), 12))
        
        # 5. Large 8x8 maze with spiral pattern
        large_8x8 = [
            ['S', '.', '.', '.', '.', '.', '.', '.'],
            ['#', '#', '#', '#', '#', '#', '#', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '#', '#', '#', '#', '#', '#', '#'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['#', '#', '#', '#', '#', '#', '#', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '#', '#', '#', '#', '#', '.', 'G']
        ]
        layouts.append(MazeLayout(large_8x8, (0, 0), (7, 7), 15))
        
        # 6. Tricky 6x6 maze with narrow passages
        tricky_6x6 = [
            ['S', '#', '.', '.', '.', '.'],
            ['.', '#', '.', '#', '#', '.'],
            ['.', '.', '.', '#', '.', '.'],
            ['#', '#', '.', '#', '.', '#'],
            ['.', '.', '.', '.', '.', '#'],
            ['.', '#', '#', '#', '.', 'G']
        ]
        layouts.append(MazeLayout(tricky_6x6, (0, 0), (5, 5), 9))
        
        # 7. Advanced 9x9 maze for maximum challenge
        advanced_9x9 = [
            ['S', '.', '#', '.', '.', '.', '#', '.', '.'],
            ['.', '.', '#', '.', '#', '.', '#', '.', '.'],
            ['#', '.', '.', '.', '#', '.', '.', '.', '#'],
            ['#', '#', '#', '.', '#', '#', '#', '.', '#'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '#', '#', '#', '.', '#', '#', '#', '.'],
            ['.', '.', '.', '#', '.', '#', '.', '.', '.'],
            ['#', '#', '.', '.', '.', '.', '.', '#', '.'],
            ['.', '.', '.', '#', '#', '#', '.', '.', 'G']
        ]
        layouts.append(MazeLayout(advanced_9x9, (0, 0), (8, 8), 16))
        
        # 8. Compact but complex 5x5 maze
        compact_complex_5x5 = [
            ['S', '#', '.', '#', '.'],
            ['.', '#', '.', '#', '.'],
            ['.', '.', '.', '.', '.'],
            ['#', '.', '#', '.', '#'],
            ['.', '.', '#', '.', 'G']
        ]
        layouts.append(MazeLayout(compact_complex_5x5, (0, 0), (4, 4), 6))
        
        return layouts
    
    def get_tasks(self) -> List[Task]:
        """Get all maze navigation tasks."""
        tasks = []
        maze_layouts = self._create_maze_layouts()
        
        difficulty_levels = [
            ("simple", "Basic navigation with clear paths"),
            ("medium", "Moderate complexity with some dead ends"),
            ("complex", "Advanced navigation requiring careful planning"),
            ("challenging", "Large maze with multiple dead ends"),
            ("large", "Very large maze with spiral patterns"),
            ("tricky", "Compact but complex with narrow passages"),
            ("advanced", "Maximum challenge with complex routing"),
            ("compact_complex", "Small but intricate maze design")
        ]
        
        for i, (maze, (difficulty, description)) in enumerate(zip(maze_layouts, difficulty_levels)):
            task = Task(
                task_id=f"maze_{i+1}",
                name=f"Maze Navigation: {difficulty.title()} {len(maze.grid)}x{len(maze.grid[0])}",
                description=description,
                prompt=self._create_initial_prompt(maze),
                expected_output="SUCCESS",
                evaluation_criteria={
                    "goal_reached": True,
                    "efficiency_threshold": 3.0  # Fail if more than 3x optimal moves
                },
                metadata={
                    "maze_layout": maze,
                    "difficulty": difficulty,
                    "maze_size": f"{len(maze.grid)}x{len(maze.grid[0])}",
                    "optimal_path_length": maze.optimal_path_length,
                    "max_allowed_moves": maze.optimal_path_length * 3,  # 3x optimal before failure
                    "challenge_level": i + 1
                }
            )
            tasks.append(task)
        
        return tasks
    
    def _create_initial_prompt(self, maze: MazeLayout) -> str:
        """Create the initial prompt for maze navigation."""
        return f"""Maze Navigation Task:

You are navigating through a maze to reach the goal. You can only see your immediate surroundings.

Rules:
- You start at position S and need to reach position G
- You can move: UP, DOWN, LEFT, RIGHT
- '#' represents walls (cannot move there)
- '.' represents open spaces (can move there)
- You cannot move outside the maze boundaries
- Avoid revisiting cells you've already been to (maintain internal memory)

Current view:
{self._get_local_view(maze, maze.start_pos)}

Your position: {maze.start_pos}
Goal: Find and reach position G

Respond with exactly one move: UP, DOWN, LEFT, or RIGHT
Your move:"""
    
    def _get_local_view(self, maze: MazeLayout, position: Tuple[int, int], view_radius: int = 1) -> str:
        """Get the local view around the current position."""
        row, col = position
        grid = maze.grid
        view_lines = []
        
        for r in range(row - view_radius, row + view_radius + 1):
            line = ""
            for c in range(col - view_radius, col + view_radius + 1):
                if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]):
                    line += "#"  # Out of bounds = wall
                elif (r, c) == position:
                    line += "X"  # Current position
                else:
                    line += grid[r][c]
                line += " "
            view_lines.append(line.strip())
        
        return "\n".join(view_lines)
    
    def _create_continuation_prompt(self, maze: MazeLayout, state: MazeState, last_move: str) -> str:
        """Create a prompt for the next move in the maze."""
        # Create a list of visited cells for the model to remember
        visited_list = sorted(list(state.visited_cells))
        visited_str = ", ".join([f"({r},{c})" for r, c in visited_list])
        
        # Show recent moves for context (last 3 moves)
        recent_context = ""
        if state.move_count > 1:
            recent_context = f"\nRecent moves: Move {max(1, state.move_count-2)} to {state.move_count}"
        
        return f"""Maze Navigation (Move {state.move_count + 1}):

CONTEXT:
- Goal: Reach position {state.goal_position} marked as 'G'
- You are at position {state.position}
- Previous move: {last_move}
- Total moves made: {state.move_count}

MEMORY - Cells you have visited:
{visited_str}

IMPORTANT: Avoid revisiting cells you've already been to! Try to explore new areas.

Current view (3x3 around your position):
{self._get_local_view(maze, state.position)}

Legend: S=Start, G=Goal, #=Wall, .=Open path, X=Your current position

STRATEGY: 
1. Look for the goal (G) in your current view
2. If you see G, move toward it
3. If no G visible, explore new areas (avoid visited cells)
4. Prefer directions that lead to unexplored territory

Respond with exactly one move: UP, DOWN, LEFT, or RIGHT
Your move:"""
    
    def _parse_move(self, response_text: str) -> str:
        """Parse the move from model response."""
        response = response_text.strip().upper()
        valid_moves = ["UP", "DOWN", "LEFT", "RIGHT"]
        
        for move in valid_moves:
            if move in response:
                return move
        
        # If no valid move found, return first word as fallback
        return response.split()[0] if response.split() else "INVALID"
    
    def _apply_move(self, position: Tuple[int, int], move: str) -> Tuple[int, int]:
        """Apply a move to get new position."""
        row, col = position
        
        if move == "UP":
            return (row - 1, col)
        elif move == "DOWN":
            return (row + 1, col)
        elif move == "LEFT":
            return (row, col - 1)
        elif move == "RIGHT":
            return (row, col + 1)
        else:
            return position  # Invalid move
    
    def _is_valid_position(self, maze: MazeLayout, position: Tuple[int, int]) -> bool:
        """Check if a position is valid (within bounds and not a wall)."""
        row, col = position
        grid = maze.grid
        
        if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]):
            return False
        
        return grid[row][col] != '#'
    
    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        """Evaluate a single maze navigation task."""
        start_time = time.time()
        maze = task.metadata["maze_layout"]
        
        # Initialize maze state
        state = MazeState(
            position=maze.start_pos,
            visited_cells={maze.start_pos},
            goal_position=maze.goal_pos,
            move_count=0,
            total_tokens_used=0
        )
        
        max_moves = task.metadata["max_allowed_moves"]
        success = False
        path_taken = [maze.start_pos]
        conversation_history = []
        delays_applied = 0
        
        try:
            # Initial move
            initial_response = await model.generate(task.prompt)
            conversation_history.append(("initial", task.prompt, initial_response.text))
            
            if initial_response.completion_tokens:
                state.total_tokens_used += initial_response.completion_tokens
            
            # Parse and apply first move
            move = self._parse_move(initial_response.text)
            new_position = self._apply_move(state.position, move)
            
            if self._is_valid_position(maze, new_position):
                state.position = new_position
                state.visited_cells.add(new_position)
                path_taken.append(new_position)
                state.move_count += 1
                
                # Check if goal reached
                if new_position == maze.goal_pos:
                    success = True
            
            # Continue navigation until goal reached or max moves
            while not success and state.move_count < max_moves:
                # Add 15s delay to respect rate limits
                await asyncio.sleep(15)
                delays_applied += 1
                
                # Create continuation prompt
                continuation_prompt = self._create_continuation_prompt(maze, state, move)
                response = await model.generate(continuation_prompt)
                conversation_history.append(("continuation", continuation_prompt, response.text))
                
                if response.completion_tokens:
                    state.total_tokens_used += response.completion_tokens
                
                # Parse and apply move
                move = self._parse_move(response.text)
                new_position = self._apply_move(state.position, move)
                
                if self._is_valid_position(maze, new_position):
                    state.position = new_position
                    state.visited_cells.add(new_position)
                    path_taken.append(new_position)
                    state.move_count += 1
                    
                    # Check if goal reached
                    if new_position == maze.goal_pos:
                        success = True
                        break
                else:
                    # Invalid move - count it but don't change position
                    state.move_count += 1
                
            total_execution_time = time.time() - start_time
            net_execution_time = total_execution_time - (delays_applied * 15)
            
            # Calculate score
            score = self.calculate_score(task, success, state, path_taken)
            
            # Calculate detailed metrics
            metrics = self._calculate_detailed_metrics(task, state, path_taken, success, conversation_history)
            
            return TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                agent_type=self.agent_type,
                success=success,
                score=score,
                metrics=metrics,
                model_response=ModelResponse(
                    text=f"Goal reached: {success}, Moves: {state.move_count}, Path: {path_taken}",
                    total_tokens=state.total_tokens_used
                ),
                execution_time=net_execution_time,
                metadata={
                    **task.metadata,
                    "path_taken": path_taken,
                    "moves_made": state.move_count,
                    "goal_reached": success,
                    "conversation_history": conversation_history
                }
            )
            
        except Exception as e:
            total_execution_time = time.time() - start_time
            net_execution_time = total_execution_time - (delays_applied * 15)
            logger.error(f"Error evaluating maze task {task.task_id}: {e}")
            
            return TaskResult(
                task_id=task.task_id,
                task_name=task.name,
                agent_type=self.agent_type,
                success=False,
                score=0.0,
                metrics={},
                execution_time=net_execution_time,
                error_message=str(e),
                metadata=task.metadata
            )
    
    def calculate_score(self, task: Task, success: bool, state: MazeState, path_taken: List[Tuple[int, int]]) -> float:
        """Calculate score for maze navigation task with continuous efficiency scaling."""
        if not success:
            return 0.0
        
        optimal_length = task.metadata["optimal_path_length"]
        actual_length = state.move_count
        efficiency_threshold = task.evaluation_criteria.get("efficiency_threshold", 3.0)
        
        # If took more than threshold times optimal moves, task fails
        if actual_length > optimal_length * efficiency_threshold:
            return 0.0
        
        # Continuous efficiency scaling: score = 1 - (excess_moves / optimal_moves)
        # Perfect path (actual = optimal) gets score 1.0
        # Path with 2x optimal moves gets score 0.5
        # Path with 3x optimal moves gets score 0.0
        excess_moves = actual_length - optimal_length
        efficiency_penalty = excess_moves / optimal_length
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, 1.0 - efficiency_penalty))
        
        return score
    
    def _calculate_detailed_metrics(self, task: Task, state: MazeState, path_taken: List[Tuple[int, int]], 
                                  success: bool, conversation_history: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        """Calculate detailed metrics for analysis."""
        optimal_length = task.metadata["optimal_path_length"]
        
        # Path efficiency metrics
        unique_cells = len(set(path_taken))
        total_moves = len(path_taken) - 1  # Subtract starting position
        revisit_count = total_moves - unique_cells + 1 if total_moves > 0 else 0
        
        # Memory usage metrics
        avg_tokens_per_move = state.total_tokens_used / max(state.move_count, 1)
        
        # Conversation analysis
        total_prompts = len(conversation_history)
        
        return {
            "goal_reached": success,
            "total_moves": state.move_count,
            "optimal_moves": optimal_length,
            "efficiency_ratio": optimal_length / max(state.move_count, 1),
            "unique_cells_visited": unique_cells,
            "cells_revisited": revisit_count,
            "path_efficiency": unique_cells / max(total_moves, 1),
            "output_tokens": state.total_tokens_used,
            "avg_tokens_per_move": avg_tokens_per_move,
            "total_conversation_turns": total_prompts,
            "memory_footprint_per_move": avg_tokens_per_move,
            "success_rate": 1.0 if success else 0.0,
            "path_taken": path_taken,
            "maze_size": task.metadata["maze_size"],
            "difficulty": task.metadata["difficulty"]
        } 