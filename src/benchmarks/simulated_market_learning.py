"""
Learning Agent Benchmark: Simulated Market Trading with RAG.

This benchmark evaluates a model's ability to learn and adapt its trading strategy
in a simulated market environment. The agent must learn the underlying patterns of
the market through a series of episodes and improve its performance over time.

The agent's learning is facilitated by a Retrieval-Augmented Generation (RAG)
mechanism, where it can query a memory of past trading episodes to inform its
current decisions.
"""

import time
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import asyncio
import re
import ast
import warnings

# Suppress a specific FutureWarning from torch triggered by sentence-transformers
warnings.filterwarnings("ignore", category=FutureWarning, message=".*`encoder_attention_mask` is deprecated.*")

from sentence_transformers import SentenceTransformer, util

from ..benchmark.base import BaseBenchmark, Task, TaskResult, BenchmarkConfig, AgentType
from ..benchmark.registry import benchmark
from ..models.base import BaseModel, ModelResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)

# --- Data Structures ---

@dataclass
class MarketState:
    """Represents the state of the market at a given time step."""
    step: int
    price: float

@dataclass
class Trade:
    """Represents a single trade action taken by the agent."""
    action: str  # 'BUY', 'SELL', 'HOLD'

@dataclass
class EpisodeResult:
    """Stores the result of a single trading episode for the agent's memory."""
    episode_id: int
    market_condition: str
    pnl: float

@dataclass
class AgentAction:
    """Represents the parsed action from the model's response."""
    action: str  # 'BUY', 'SELL', or 'HOLD'
    reasoning: str



@benchmark(
    name="simulated_market_learning",
    agent_type=AgentType.LEARNING,
    description="Tests an agent's ability to learn a trading strategy in a simulated market with RAG."
)
class SimulatedMarketLearningBenchmark(BaseBenchmark):
    """
    Benchmark for learning a trading strategy in a simulated market environment.
    """

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.memory_corpus: List[EpisodeResult] = []
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_tasks(self) -> List[Task]:
        """Generate a series of tasks, each representing a different market regime."""
        tasks = []
        scenarios = [
            {"name": "Gentle Bull Market", "pattern": "gentle_bull"},
            {"name": "Strong Bull Market", "pattern": "strong_bull"},
            {"name": "Gentle Bear Market", "pattern": "gentle_bear"},
            {"name": "Strong Bear Market", "pattern": "strong_bear"},
            {"name": "Volatile Market", "pattern": "volatile"},
            {"name": "Mean-Reverting Market", "pattern": "mean_reverting"},
            {"name": "Sudden Spike", "pattern": "spike"},
            {"name": "Sudden Crash", "pattern": "crash"},
            {"name": "Standard Bull", "pattern": "bull"},
            {"name": "Standard Bear", "pattern": "bear"},
        ]

        for i, scenario in enumerate(scenarios):
            task = Task(
                task_id=f"market_learning_{i+1}",
                name=f"Market Learning: {scenario['name']}",
                description=f"Learn to trade in a simulated {scenario['name'].lower()}.",
                prompt="",  # The prompt is generated dynamically within the evaluation logic
                metadata={
                    "market_pattern": scenario['pattern'],
                    "num_training_episodes": 20,
                    "num_test_episodes": 5,
                    "episode_length": 50, # 50 time steps per episode
                }
            )
            tasks.append(task)
            
        return tasks

    def _generate_market_data(self, pattern: str, length: int, seed: int) -> List[float]:
        """Generates a sequence of market prices based on a deterministic pattern."""
        np.random.seed(seed)
        prices = [100.0]
        
        if pattern == "bull":
            for _ in range(1, length):
                change = np.random.normal(0.2, 0.5) # Increased drift
                prices.append(max(1.0, prices[-1] + change))
        elif pattern == "bear":
            for _ in range(1, length):
                change = np.random.normal(-0.2, 0.5) # Increased drift
                prices.append(max(1.0, prices[-1] + change))
        elif pattern == "volatile":
            for _ in range(1, length):
                change = np.random.normal(0, 1.5)
                prices.append(max(1.0, prices[-1] + change))
        elif pattern == "crash":
            crash_point = int(length * np.random.uniform(0.6, 0.9))
            for i in range(1, length):
                if i < crash_point:
                    change = np.random.normal(0.01, 0.2)
                    prices.append(max(1.0, prices[-1] + change))
                else:
                    change = np.random.normal(-2.0, 1.0)
                    prices.append(max(1.0, prices[-1] + change))
        elif pattern == "spike":
            spike_point = int(length * np.random.uniform(0.6, 0.9))
            for i in range(1, length):
                if i < spike_point:
                    change = np.random.normal(-0.01, 0.2)
                    prices.append(max(1.0, prices[-1] + change))
                else:
                    change = np.random.normal(2.0, 1.0)
                    prices.append(max(1.0, prices[-1] + change))
        elif pattern == "mean_reverting":
            mean = 100.0
            for _ in range(1, length):
                reversion = (mean - prices[-1]) * 0.4 # Increased reversion strength
                change = np.random.normal(reversion, 0.5)
                prices.append(max(1.0, prices[-1] + change))
        elif pattern == "gentle_bull":
            for _ in range(1, length):
                change = np.random.normal(0.1, 0.2)
                prices.append(max(1.0, prices[-1] + change))
        elif pattern == "gentle_bear":
            for _ in range(1, length):
                change = np.random.normal(-0.1, 0.2)
                prices.append(max(1.0, prices[-1] + change))
        elif pattern == "strong_bull":
            for _ in range(1, length):
                change = np.random.normal(0.4, 0.6)
                prices.append(max(1.0, prices[-1] + change))
        elif pattern == "strong_bear":
            for _ in range(1, length):
                change = np.random.normal(-0.4, 0.6)
                prices.append(max(1.0, prices[-1] + change))
        else: # Default to a simple random walk
            for _ in range(1, length):
                change = np.random.normal(0, 1.0)
                prices.append(max(1.0, prices[-1] + change))

        return prices
    
    def _calculate_optimal_pnl(self, prices: List[float], capital: float = 10000.0) -> float:
        """
        Calculates the maximum possible profit from a single buy and sell transaction.
        """
        min_price = float('inf')
        max_profit = 0.0
        for price in prices:
            if price < min_price:
                min_price = price
            elif price - min_price > max_profit:
                max_profit = price - min_price
        
        # Scale profit by the number of shares that could have been bought with initial capital
        return max_profit * (capital / min_price) if min_price > 0 and min_price != float('inf') else 0.0

    def _calculate_baseline_pnl(self, prices: List[float], capital: float = 10000.0) -> float:
        """Calculates the PnL of a simple buy-and-hold strategy."""
        if not prices or prices[0] <= 0:
            return 0.0
        
        shares_bought = capital / prices[0]
        final_value = shares_bought * prices[-1]
        return final_value - capital

    async def _run_episode(self, episode_id: int, market_data: List[float], model: BaseModel, is_training: bool, capital: float = 10000.0) -> Tuple[EpisodeResult, Dict]:
        """Runs a single episode of the trading task step-by-step."""
        cash = capital
        shares = 0
        total_output_tokens = 0
        api_calls = 0
        
        for step in range(len(market_data) - 1):
            historical_prices = market_data[:step+1]
            current_price = market_data[step]
            
            prompt = self._create_prompt(historical_prices, cash, shares)
            
            if self.memory_corpus:
                prompt += self._retrieve_memories(historical_prices)

            response = await model.generate(prompt)
            api_calls += 1
            await asyncio.sleep(15) # Prevent rate limiting
            
            total_output_tokens += response.completion_tokens or 0
            
            action = self._parse_action(response.text)
            
            # Agent uses full capital and can only hold one position at a time
            if action.action == 'BUY' and shares == 0 and cash >= current_price:
                shares_to_buy = cash / current_price
                shares = shares_to_buy
                cash = 0
            elif action.action == 'SELL' and shares > 0:
                cash += shares * current_price
                shares = 0
                
        final_value = cash + shares * market_data[-1]
        pnl = final_value - capital
        
        # For memory, we need a summary of the market
        market_condition_summary = self._summarize_market(market_data)

        episode_result = EpisodeResult(
            episode_id=episode_id,
            market_condition=market_condition_summary,
            pnl=pnl
        )
        
        return episode_result, {"output_tokens": total_output_tokens, "api_calls": api_calls}

    def _summarize_market(self, prices: List[float]) -> str:
        """Creates a textual summary of the market's behavior."""
        if len(prices) < 2:
            return "Market history is too short to determine a trend."

        price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
        volatility = np.std(np.diff(prices)) / np.mean(prices) if np.mean(prices) > 0 else 0
        
        summary = ""
        if price_change > 0.05: summary += "Uptrend. "
        elif price_change < -0.05: summary += "Downtrend. "
        else: summary += "Sideways trend. "
        
        if volatility > 0.02: summary += "High volatility."
        elif volatility < 0.005: summary += "Low volatility."
        else: summary += "Moderate volatility."
        
        return summary

    def _create_prompt(self, historical_prices: List[float], cash: float, shares: float) -> str:
        """Creates the prompt for the agent for a single time step."""
        prices_str = ", ".join([f"{p:.2f}" for p in historical_prices])
        return f"""
You are a trading agent. You must decide whether to BUY, SELL, or HOLD at the current time step.
You will be given the history of prices up to the current moment, and your current portfolio status.

**Current State:**
- **Portfolio:** {shares:.2f} shares, ${cash:.2f} cash
- **Price History:** [{prices_str}]
- **Current Price:** {historical_prices[-1]:.2f}

**Your Task:**
Decide on your action for the current time step. Your response must be one of 'BUY', 'SELL', or 'HOLD'.
Provide a brief reasoning for your choice, then state your final action in the format `[Action: <action>]`.

**Example Response:**
The price has been trending up, so I will buy now.
[Action: BUY]
"""

    def _retrieve_memories(self, historical_prices: List[float], top_k: int = 3) -> str:
        """Retrieves the most relevant past episodes using RAG."""
        if not self.memory_corpus:
            return ""

        memory_docs = [f"Market: {m.market_condition} PnL: {m.pnl:.2f}" for m in self.memory_corpus]
        memory_embeddings = self.embedding_model.encode(memory_docs, convert_to_tensor=True)
        
        # Create an embedding for the current situation's summary
        query_summary = self._summarize_market(historical_prices)
        query_embedding = self.embedding_model.encode(query_summary, convert_to_tensor=True)
        
        cosine_scores = util.pytorch_cos_sim(query_embedding, memory_embeddings)[0]
        top_results = np.argpartition(-cosine_scores.cpu(), range(min(top_k, len(self.memory_corpus))))[:min(top_k, len(self.memory_corpus))]
        
        retrieved_memories = "\n\n**Retrieved Memories from Past Episodes:**\n"
        for idx in top_results:
            retrieved_memories += f"- {memory_docs[idx]}\n"
            
        return retrieved_memories

    def _parse_action(self, response_text: str) -> AgentAction:
        """Parses the model's response to extract the structured action."""
        # Primary parsing: Look for the specific format [Action: <action>]
        primary_match = re.search(r'\[Action:\s*(BUY|SELL|HOLD)\s*\]', response_text, re.IGNORECASE)
        if primary_match:
            action = primary_match.group(1).upper()
            return AgentAction(action=action, reasoning=response_text)

        # Fallback parsing: Find the last mention of "Action: <action>"
        fallback_matches = re.findall(r'Action:\s*(BUY|SELL|HOLD)', response_text, re.IGNORECASE)
        if fallback_matches:
            action = fallback_matches[-1].upper()
            return AgentAction(action=action, reasoning=response_text)

        # Default to HOLD if no action is found
        return AgentAction(action='HOLD', reasoning=response_text)


    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        """Evaluates the agent's learning over a series of training and testing episodes."""
        start_time = time.time()
        
        # Reset memory for each scenario to ensure clean learning evaluation
        self.memory_corpus = []
        
        num_training = task.metadata["num_training_episodes"]
        num_test = task.metadata["num_test_episodes"]
        episode_length = task.metadata["episode_length"]
        market_pattern = task.metadata["market_pattern"]
        
        total_output_tokens = 0
        total_api_calls = 0

        # --- Training Phase ---
        for i in range(num_training):
            market_data = self._generate_market_data(market_pattern, episode_length, seed=i)
            episode_result, metrics = await self._run_episode(i, market_data, model, is_training=True)
            self.memory_corpus.append(episode_result)
            total_output_tokens += metrics.get("output_tokens", 0)
            total_api_calls += metrics.get("api_calls", 0)

        # --- Testing Phase ---
        test_results = []
        for i in range(num_test):
            market_data = self._generate_market_data(market_pattern, episode_length, seed=num_training + i)
            episode_result, metrics = await self._run_episode(num_training + i, market_data, model, is_training=False)
            test_results.append(episode_result)
            total_output_tokens += metrics.get("output_tokens", 0)
            total_api_calls += metrics.get("api_calls", 0)
            
        # --- Final Score Calculation ---
        agent_pnl = sum(r.pnl for r in test_results)
        
        # Calculate optimal and baseline PnL on the same data used for testing
        optimal_pnl = sum(self._calculate_optimal_pnl(data) for data in [self._generate_market_data(market_pattern, episode_length, seed=num_training + i) for i in range(num_test)])
        baseline_pnl = sum(self._calculate_baseline_pnl(data) for data in [self._generate_market_data(market_pattern, episode_length, seed=num_training + i) for i in range(num_test)])


        # Normalize score
        if optimal_pnl - baseline_pnl == 0:
            score = 1.0 if agent_pnl >= baseline_pnl else 0.0
        else:
            score = (agent_pnl - baseline_pnl) / (optimal_pnl - baseline_pnl)
        
        final_score = max(0.0, min(1.0, score)) # Clamp score between 0 and 1

        actual_execution_time = (time.time() - start_time) - (total_api_calls * 15)

        return TaskResult(
            task_id=task.task_id,
            task_name=task.name,
            agent_type=self.agent_type,
            success=final_score > 0.7,
            score=final_score,
            metrics={
                "agent_pnl": agent_pnl,
                "optimal_pnl": optimal_pnl,
                "baseline_pnl": baseline_pnl,
                "output_tokens": total_output_tokens,
                "num_training_episodes": num_training,
                "num_test_episodes": num_test,
            },
            execution_time=actual_execution_time
        )

    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        """Scoring is handled within evaluate_task for this benchmark."""
        return 0.0

