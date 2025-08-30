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
    actions_summary: str = ""
    num_buys: int = 0
    num_sells: int = 0
    final_shares: float = 0.0

@dataclass
class AgentAction:
    """Represents the parsed action from the model's response."""
    action: str  # 'BUY', 'SELL', or 'HOLD'
    reasoning: str



@benchmark(
    name="simulated_market",
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
        accumulated_call_time = 0.0
        actions_taken: List[str] = []
        event_actions: List[str] = []  # Only BUY/SELL with time and price
        
        total_steps = len(market_data)
        wait_seconds = float(self.config.additional_params.get("wait_seconds", 15.0)) if getattr(self, "config", None) else 15.0
        for step in range(total_steps - 1):
            historical_prices = market_data[:step+1]
            current_price = market_data[step]
            
            prompt = self._create_prompt(historical_prices, cash, shares, current_step=step, total_steps=total_steps)
            
            if self.memory_corpus:
                prompt += self._retrieve_memories(historical_prices)
            call_started_at = time.time()
            response = await model.generate(prompt)
            accumulated_call_time += (time.time() - call_started_at)
            api_calls += 1
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds) # Prevent rate limiting
            
            total_output_tokens += response.completion_tokens or 0
            
            action = self._parse_action(response.text)
            actions_taken.append(action.action)
            
            # Agent uses full capital and can only hold one position at a time
            if action.action == 'BUY' and shares == 0 and cash >= current_price:
                shares_to_buy = cash / current_price
                shares = shares_to_buy
                cash = 0
                event_actions.append(f"BUY at step {step} price {current_price:.2f}")
            elif action.action == 'SELL' and shares > 0:
                cash += shares * current_price
                shares = 0
                event_actions.append(f"SELL at step {step} price {current_price:.2f}")
                
        final_value = cash + shares * market_data[-1]
        pnl = final_value - capital
        
        # For memory, we need a summary of the market
        market_condition_summary = self._summarize_market(market_data)
        actions_summary = "; ".join(event_actions) if event_actions else "None"
        num_buys = sum(1 for a in actions_taken if a == 'BUY')
        num_sells = sum(1 for a in actions_taken if a == 'SELL')

        episode_result = EpisodeResult(
            episode_id=episode_id,
            market_condition=market_condition_summary,
            pnl=pnl,
            actions_summary=actions_summary,
            num_buys=num_buys,
            num_sells=num_sells,
            final_shares=float(shares)
        )
        
        return episode_result, {"output_tokens": total_output_tokens, "api_calls": api_calls, "api_time": accumulated_call_time}

    def _summarize_market(self, prices: List[float]) -> str:
        """Creates a richer textual summary of the market's behavior."""
        if len(prices) < 2:
            return "Market history too short: insufficient data for summary."

        # Basic features
        start_price = prices[0]
        end_price = prices[-1]
        price_change = (end_price - start_price) / start_price if start_price > 0 else 0.0
        diffs = np.diff(prices)
        mean_price = float(np.mean(prices)) if len(prices) > 0 else 0.0
        volatility = float(np.std(diffs) / mean_price) if mean_price > 0 else 0.0

        # Magnitude and timing features
        max_price = max(prices)
        min_price = min(prices)
        max_idx = int(np.argmax(prices))
        min_idx = int(np.argmin(prices))
        range_pct = ((max_price - min_price) / start_price) if start_price > 0 else 0.0

        # Early/late spike/crash indicators
        horizon = len(prices)
        early_window = prices[: max(2, horizon // 3)]
        late_window = prices[max(0, 2 * horizon // 3 - 1) :]
        early_change = (early_window[-1] - early_window[0]) / early_window[0] if early_window and early_window[0] > 0 else 0.0
        late_change = (late_window[-1] - late_window[0]) / late_window[0] if late_window and late_window[0] > 0 else 0.0

        # Trend label
        if price_change > 0.05:
            trend = "Uptrend"
        elif price_change < -0.05:
            trend = "Downtrend"
        else:
            trend = "Sideways"

        # Volatility label
        if volatility > 0.02:
            vol_label = "High volatility"
        elif volatility < 0.005:
            vol_label = "Low volatility"
        else:
            vol_label = "Moderate volatility"

        timing_note = ""
        if max_idx >= int(0.6 * horizon):
            timing_note += " Late peak."
        elif max_idx <= int(0.4 * horizon):
            timing_note += " Early peak."
        if min_idx >= int(0.6 * horizon):
            timing_note += " Late trough."
        elif min_idx <= int(0.4 * horizon):
            timing_note += " Early trough."

        return (
            f"{trend}. {vol_label}. "
            f"Change: {price_change:.3f}. RangePct: {range_pct:.3f}. "
            f"EarlyChange: {early_change:.3f}. LateChange: {late_change:.3f}." + timing_note
        )

    def _create_prompt(self, historical_prices: List[float], cash: float, shares: float, current_step: int, total_steps: int) -> str:
        """Creates the prompt for the agent for a single time step."""
        prices_str = ", ".join([f"{p:.2f}" for p in historical_prices])
        steps_remaining = max(0, total_steps - current_step - 1)

        return f"""
You are a trading agent. You must decide whether to BUY, SELL, or HOLD at the current time step.
You will be given the history of prices up to the current moment, and your current portfolio status.

Important rules for this episode:
- The episode length is fixed: {total_steps} steps (indexed 0 to {total_steps-1}).
- You are currently at step {current_step}. Steps remaining after this decision: {steps_remaining}.
- At the final step, your portfolio is marked-to-market: final value = cash + shares * last_price.
- You will NOT be forced to sell at the end; unsold shares are valued at the last price.
- Your objective is to maximize final portfolio value.

**Current State:**
- **Portfolio:** {shares:.2f} shares, ${cash:.2f} cash
- **Price History:** [{prices_str}]
- **Current Price:** {historical_prices[-1]:.2f}

**Your Task:**
Decide on your action for the current time step. Your response must be one of 'BUY', 'SELL', or 'HOLD'.
Provide a brief reasoning for your choice, then state your final action in the format `[Action: <action>]`.

If retrieved memories are provided below, consider how similar past conditions affected outcomes and adapt your decision accordingly.

**Example Response:**
Given the uptrend and limited steps remaining, I prefer to hold.
[Action: HOLD]
"""

    def _retrieve_memories(self, historical_prices: List[float], top_k: int = 5) -> str:
        """Retrieves the most relevant past episodes using RAG."""
        if not self.memory_corpus:
            return ""

        # Enrich memory docs with compact numeric features to improve semantic separation
        def _features_from_summary(summary: str) -> str:
            # Extract approximate numeric cues present in summary text (best effort)
            return summary

        memory_docs = [
            (
                f"Market: {m.market_condition} | Actions: {m.actions_summary} | "
                f"Buys: {m.num_buys}, Sells: {m.num_sells}, FinalShares: {m.final_shares:.2f} | "
                f"Outcome PnL: {m.pnl:.2f}"
            )
            for m in self.memory_corpus
        ]
        memory_embeddings = self.embedding_model.encode(memory_docs, convert_to_tensor=True)
        
        # Create an embedding for the current situation's summary
        query_summary = self._summarize_market(historical_prices)
        query_embedding = self.embedding_model.encode(query_summary, convert_to_tensor=True)
        
        cosine_scores = util.pytorch_cos_sim(query_embedding, memory_embeddings)[0]
        cosine_scores_np = cosine_scores.detach().cpu().numpy()
        k = min(top_k, len(self.memory_corpus))
        top_results = np.argpartition(-cosine_scores_np, range(k))[:k]
        
        retrieved_memories = (
            "\n\n**Retrieved Memories from Past Episodes (most similar first):**\n"
            "Legend: 'BUY at step S price P' means a buy action executed at step S at price P; "
            "'SELL at step S price P' means a sell action at step S at price P.\n"
        )
        # Order top results by score descending
        sorted_top = sorted([(int(i), float(cosine_scores_np[int(i)])) for i in top_results], key=lambda x: -x[1])
        for idx, score in sorted_top:
            retrieved_memories += f"- {memory_docs[idx]} (sim={score:.3f})\n"
            
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
        accumulated_call_time = 0.0

        # --- Training Phase ---
        for i in range(num_training):
            market_data = self._generate_market_data(market_pattern, episode_length, seed=i)
            episode_result, metrics = await self._run_episode(i, market_data, model, is_training=True)
            self.memory_corpus.append(episode_result)
            total_output_tokens += metrics.get("output_tokens", 0)
            total_api_calls += metrics.get("api_calls", 0)
            accumulated_call_time += metrics.get("api_time", 0.0)

        # --- Testing Phase ---
        test_results = []
        for i in range(num_test):
            market_data = self._generate_market_data(market_pattern, episode_length, seed=num_training + i)
            episode_result, metrics = await self._run_episode(num_training + i, market_data, model, is_training=False)
            test_results.append(episode_result)
            total_output_tokens += metrics.get("output_tokens", 0)
            total_api_calls += metrics.get("api_calls", 0)
            accumulated_call_time += metrics.get("api_time", 0.0)
            
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

        # Sum of API call durations across all steps/episodes; excludes intentional sleeps
        actual_execution_time = accumulated_call_time

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

