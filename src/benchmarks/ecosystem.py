"""
Learning Agent Benchmark: Fictional Ecosystem Dynamics Learning with Knowledge Graphs

This benchmark evaluates a model's ability to learn ecosystem relationships through
building and maintaining a knowledge graph of species interactions in a completely 
fictional biological system. The agent must discover predator-prey relationships, 
growth rates, and carrying capacity effects by updating its internal knowledge graph
across multiple episodes.

The agent's learning is assessed based on population prediction accuracy. The knowledge
graph serves as the learning mechanism, not the evaluation target.
"""

import time
import random
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import asyncio
import re

import networkx as nx

from ..benchmark.base import BaseBenchmark, Task, TaskResult, BenchmarkConfig, AgentType
from ..benchmark.registry import benchmark
from ..models.base import BaseModel, ModelResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Species:
    """Represents a fictional species in the ecosystem."""
    name: str
    population: int


@dataclass
class EcosystemState:
    """Complete state of the ecosystem."""
    species: Dict[str, Species]
    carrying_capacity: int = 500
    time_step: int = 0


@dataclass
class EcosystemEpisode:
    """Represents a single learning episode."""
    episode_id: Union[int, str]
    initial_populations: Dict[str, int]
    intervention: Dict[str, Any]
    predicted_populations: Optional[Dict[str, int]] = None
    predicted_relationships: Optional[Dict[str, List[str]]] = None
    actual_populations: Optional[Dict[str, int]] = None


@dataclass
class EcosystemTrialResult:
    """Stores the result of a single trial for an agent."""
    episode_id: Union[int, str]
    intervention_type: str
    species_affected: str
    initial_populations: Dict[str, int]
    predicted_populations: Dict[str, int]
    predicted_relationships: Dict[str, List[str]]
    actual_populations: Dict[str, int]
    prediction_error: float


class EcosystemKnowledgeGraph:
    """
    Knowledge graph to track learned ecosystem relationships.
    This serves as the agent's persistent memory and learning mechanism.
    """

    def __init__(self, carrying_capacity: int = 500, confidence_decay_existing: float = 0.7, confidence_new_weight: float = 0.3):
        self.graph = nx.DiGraph()
        self.relationship_confidence: Dict[Tuple[str, str], float] = {}
        self.species_growth_estimates: Dict[str, float] = {}
        self.carrying_capacity: int = carrying_capacity
        self.conf_decay_existing: float = confidence_decay_existing
        self.conf_new_weight: float = confidence_new_weight

    def update_relationship(self, predator: str, prey: str, confidence: float):
        """Add or update a predator-prey relationship with confidence score."""
        if self.graph.has_edge(predator, prey):
            current_conf = self.relationship_confidence.get((predator, prey), 0.0)
            new_conf = (current_conf * self.conf_decay_existing) + (confidence * self.conf_new_weight)
        else:
            new_conf = confidence

        self.graph.add_edge(predator, prey, relationship="eats")
        self.relationship_confidence[(predator, prey)] = new_conf
        self.graph[predator][prey]['weight'] = new_conf

    def remove_relationship(self, predator: str, prey: str):
        """Remove a relationship from the graph."""
        if self.graph.has_edge(predator, prey):
            self.graph.remove_edge(predator, prey)
            if (predator, prey) in self.relationship_confidence:
                del self.relationship_confidence[(predator, prey)]

    def get_prey_list(self, predator: str) -> List[str]:
        """Get list of species that the predator eats."""
        if predator not in self.graph:
            return []
        return list(self.graph.successors(predator))

    def get_predators_of(self, prey: str) -> List[str]:
        """Get list of species that eat the given prey."""
        if prey not in self.graph:
            return []
        return list(self.graph.predecessors(prey))

    def get_relationship_confidence(self, predator: str, prey: str) -> float:
        """Get confidence score for a specific relationship."""
        return self.relationship_confidence.get((predator, prey), 0.0)

    def get_ecosystem_structure(self) -> Dict[str, List[str]]:
        """Get current understanding of who eats whom."""
        structure: Dict[str, List[str]] = {}
        all_species = set(self.graph.nodes())
        for species in all_species:
            structure[species] = self.get_prey_list(species)
        return structure

    def classify_species_type(self, species: str) -> str:
        """Classify species as producer, herbivore, or carnivore based on relationships."""
        prey_list = self.get_prey_list(species)
        if not prey_list:
            return "producer"
        for prey in prey_list:
            if self.get_prey_list(prey):
                return "carnivore"
        return "herbivore"

    def estimate_population_change(self, species: str, current_pop: int, ecosystem_state: Dict[str, int]) -> int:
        """Use knowledge graph to estimate population change for a species."""
        if current_pop <= 0:
            return 0

        growth_rate = self.species_growth_estimates.get(species, 0.15)
        new_pop = current_pop * (1 + growth_rate)

        for predator in self.get_predators_of(species):
            if predator in ecosystem_state and ecosystem_state[predator] > 0:
                confidence = self.get_relationship_confidence(predator, species)
                predation_effect = confidence * 0.02 * ecosystem_state[predator]
                new_pop *= (1 - min(0.8, predation_effect))

        total_pop = sum(ecosystem_state.values())
        threshold_capacity = int(0.9 * self.carrying_capacity)
        if total_pop > threshold_capacity:
            competition_factor = threshold_capacity / total_pop
            new_pop *= competition_factor

        return max(0, int(round(new_pop)))

    def calculate_accuracy_vs_ground_truth(self, ground_truth_matrix: Dict[str, List[str]]) -> float:
        """Calculate how accurate the knowledge graph is vs ground truth (for analysis only)."""
        if not ground_truth_matrix:
            return 1.0

        correct_relationships = 0
        total_possible_relationships = 0

        all_species = set(ground_truth_matrix.keys())
        for predator in all_species:
            for prey in all_species:
                if predator == prey:
                    continue
                total_possible_relationships += 1
                true_eats = prey in ground_truth_matrix.get(predator, [])
                predicted_eats = prey in self.get_prey_list(predator)
                if true_eats == predicted_eats:
                    correct_relationships += 1

        return correct_relationships / total_possible_relationships if total_possible_relationships > 0 else 1.0


class EcosystemSolver:
    """
    Perfect deterministic solver that acts as the 'physics engine' for ecosystem dynamics.
    This provides ground truth that the learning agent must discover through experience.
    """

    def __init__(self, carrying_capacity: int = 500, predation_efficiency: float = 0.015, episode_steps: int = 10):
        self.predator_prey_matrix: Dict[str, List[str]] = {
            "Species_A": ["Species_B"],
            "Species_B": [],
            "Species_C": ["Species_A"],
            "Species_D": ["Species_B", "Species_C"],
            "Species_E": ["Species_A", "Species_D"],
        }

        self.base_growth_rates: Dict[str, float] = {
            "Species_A": 0.15,
            "Species_B": 0.25,
            "Species_C": 0.10,
            "Species_D": 0.08,
            "Species_E": 0.05,
        }

        self.predation_efficiency: float = predation_efficiency
        self.total_carrying_capacity: int = carrying_capacity
        self.episode_steps: int = episode_steps

    def get_ground_truth_relationships(self) -> Dict[str, List[str]]:
        return self.predator_prey_matrix.copy()

    def solve_ecosystem_change(self, initial_populations: Dict[str, int], intervention: Dict[str, Any]) -> Dict[str, int]:
        current_pops = self._apply_intervention(initial_populations.copy(), intervention)
        for _ in range(self.episode_steps):
            current_pops = self._simulate_one_step(current_pops)
        return current_pops

    def _apply_intervention(self, populations: Dict[str, int], intervention: Dict[str, Any]) -> Dict[str, int]:
        if intervention["type"] == "add_species":
            species = intervention["species"]
            count = intervention["count"]
            populations[species] = populations.get(species, 0) + count
        elif intervention["type"] == "remove_species":
            species = intervention["species"]
            populations[species] = 0
        elif intervention["type"] == "double_population":
            species = intervention["species"]
            if species in populations:
                populations[species] *= 2
        elif intervention["type"] == "halve_population":
            species = intervention["species"]
            if species in populations:
                populations[species] = populations[species] // 2
        return populations

    def _simulate_one_step(self, current_populations: Dict[str, int]) -> Dict[str, int]:
        new_populations: Dict[str, int] = {}
        total_current_pop = sum(current_populations.values())

        for species, population in current_populations.items():
            if population <= 0:
                new_populations[species] = 0
                continue

            new_pop = float(population)
            growth_rate = self.base_growth_rates.get(species, 0.1)
            new_pop *= (1 + growth_rate)

            for predator, prey_list in self.predator_prey_matrix.items():
                if species in prey_list and current_populations.get(predator, 0) > 0:
                    predator_count = current_populations[predator]
                    predation_pressure = self.predation_efficiency * predator_count
                    predation_pressure = min(0.9, predation_pressure)
                    new_pop *= (1 - predation_pressure)

            if total_current_pop > self.total_carrying_capacity:
                competition_factor = self.total_carrying_capacity / total_current_pop
                new_pop *= competition_factor

            new_populations[species] = max(0, int(round(new_pop)))

        return new_populations


@benchmark(
    name="ecosystem",
    agent_type=AgentType.LEARNING,
    description="Tests an agent's ability to learn ecosystem dynamics through knowledge graph construction."
)
class EcosystemLearningBenchmark(BaseBenchmark):
    """Benchmark for learning ecosystem dynamics through knowledge graph-based learning."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        # Parameters (configurable via additional_params)
        params = self.config.additional_params or {}
        self.params = {
            "episode_steps": params.get("episode_steps", 10),
            "carrying_capacity": params.get("carrying_capacity", 500),
            "predation_efficiency": params.get("predation_efficiency", 0.015),
            "confidence_decay_existing": params.get("confidence_decay_existing", 0.7),
            "confidence_new_weight": params.get("confidence_new_weight", 0.3),
            "evidence_min_support": params.get("evidence_min_support", 3),
            "capacity_guard_threshold": params.get("capacity_guard_threshold", 0.9),
            "accuracy_gate_for_boost": params.get("accuracy_gate_for_boost", 0.8),
        }

        self.solver = EcosystemSolver(
            carrying_capacity=self.params["carrying_capacity"],
            predation_efficiency=self.params["predation_efficiency"],
            episode_steps=self.params["episode_steps"],
        )
        self.memory: List[EcosystemTrialResult] = []
        self.agent_knowledge_graph = EcosystemKnowledgeGraph(
            carrying_capacity=self.params["carrying_capacity"],
            confidence_decay_existing=self.params["confidence_decay_existing"],
            confidence_new_weight=self.params["confidence_new_weight"],
        )

        # Fair generalization: train and evaluate on the same species set
        self.training_species = ["Species_A", "Species_B", "Species_C", "Species_D", "Species_E"]
        self.evaluation_species = ["Species_A", "Species_B", "Species_C", "Species_D", "Species_E"]

        # Evidence tracking for relationship inference
        self._evidence_support: Dict[Tuple[str, str], int] = {}
        self._evidence_oppose: Dict[Tuple[str, str], int] = {}

    def get_tasks(self) -> List[Task]:
        tasks: List[Task] = []
        # Increase training scenarios to yield 10 tasks with interval 8: 80/8 = 10
        all_training_scenarios = self._generate_scenarios(80, "train")

        # Build a set of training scenario keys for deduplication
        def _scenario_key(s: Dict[str, Any]) -> Tuple:
            init_tuple = tuple(sorted(s["initial_populations"].items()))
            intervention = s["intervention"]
            itype = intervention.get("type")
            species = intervention.get("species")
            count = intervention.get("count", None)
            return (init_tuple, itype, species, count)

        training_keys = { _scenario_key(s) for s in all_training_scenarios }

        # Generate evaluation scenarios ensuring none are identical to any training scenario
        desired_eval = 8
        evaluation_scenarios: List[Dict[str, Any]] = []
        eval_keys = set()
        seed_base = 456
        attempt = 0
        while len(evaluation_scenarios) < desired_eval and attempt < 100:
            candidate_batch = self._generate_scenarios(desired_eval, "eval", seed_override=seed_base + attempt)
            for s in candidate_batch:
                key = _scenario_key(s)
                if key in training_keys or key in eval_keys:
                    continue
                evaluation_scenarios.append(s)
                eval_keys.add(key)
                if len(evaluation_scenarios) >= desired_eval:
                    break
            attempt += 1

        eval_interval = 8
        num_training_scenarios = len(all_training_scenarios)
        for i in range(eval_interval, num_training_scenarios + 1, eval_interval):
            start_index = i - eval_interval
            new_training_scenarios = all_training_scenarios[start_index:i]

            task = Task(
                task_id=f"ecosystem_stage_{i}_episodes",
                name=f"Ecosystem Learning after {i} Training Episodes",
                description=f"Evaluate ecosystem prediction using knowledge graph after {i} total episodes.",
                prompt="",
                metadata={
                    "new_training_scenarios": new_training_scenarios,
                    "evaluation_scenarios": evaluation_scenarios,
                    "total_episodes_so_far": i,
                },
            )
            tasks.append(task)
        return tasks

    def _generate_scenarios(self, num_scenarios: int, scenario_type: str, seed_override: Optional[int] = None) -> List[Dict[str, Any]]:
        scenarios: List[Dict[str, Any]] = []
        if seed_override is not None:
            seed = seed_override
        else:
            seed = 123 if scenario_type == "train" else 456
        rng = random.Random(seed)

        species_list = self.training_species if scenario_type == "train" else self.evaluation_species
        for _ in range(num_scenarios):
            initial_populations: Dict[str, int] = {}
            for species in species_list:
                initial_populations[species] = rng.randint(20, 80)

            intervention_type = rng.choice([
                "add_species",
                "remove_species",
                "double_population",
                "halve_population",
            ])
            target_species = rng.choice(species_list)

            if intervention_type == "add_species":
                intervention = {"type": "add_species", "species": target_species, "count": rng.randint(10, 30)}
            elif intervention_type == "remove_species":
                intervention = {"type": "remove_species", "species": target_species}
            elif intervention_type == "double_population":
                intervention = {"type": "double_population", "species": target_species}
            else:
                intervention = {"type": "halve_population", "species": target_species}

            scenarios.append({"initial_populations": initial_populations, "intervention": intervention})

        return scenarios

    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        start_time = time.time()

        training_scenarios = task.metadata["new_training_scenarios"]
        evaluation_scenarios = task.metadata["evaluation_scenarios"]

        total_output_tokens = 0
        total_api_calls = 0
        total_api_time = 0.0

        # Training phase
        for i, scenario in enumerate(training_scenarios):
            episode_id = task.metadata["total_episodes_so_far"] - len(training_scenarios) + i
            episode_result, metrics = await self._run_episode(episode_id, scenario, model)
            self._update_memory(self.memory, episode_result)
            self._update_agent_knowledge_graph(episode_result)
            total_output_tokens += metrics["output_tokens"]
            total_api_calls += metrics["api_calls"]
            total_api_time += metrics.get("api_time", 0.0)

        # Evaluation phase
        eval_results: List[EcosystemTrialResult] = []
        for i, eval_scenario in enumerate(evaluation_scenarios):
            eval_episode_result, eval_metrics = await self._run_episode(
                f"eval_after_{task.metadata['total_episodes_so_far']}_{i}", eval_scenario, model, is_eval=True
            )
            eval_results.append(eval_episode_result)
            total_output_tokens += eval_metrics["output_tokens"]
            total_api_calls += eval_metrics["api_calls"]
            total_api_time += eval_metrics.get("api_time", 0.0)

        # Measure execution as the sum of API call durations (excludes sleep delays)
        actual_execution_time = total_api_time

        episode_scores: List[float] = []
        for result in eval_results:
            prediction_score = self._calculate_prediction_score(result.predicted_populations, result.actual_populations)
            episode_scores.append(prediction_score)

        final_score = sum(episode_scores) / len(episode_scores) if episode_scores else 0.0
        avg_eval_error = (
            sum(r.prediction_error for r in eval_results) / len(eval_results) if eval_results else float("inf")
        )

        ground_truth_relationships = self.solver.get_ground_truth_relationships()
        overall_kg_accuracy = self.agent_knowledge_graph.calculate_accuracy_vs_ground_truth(ground_truth_relationships)

        summary_metrics = {
            "output_tokens": total_output_tokens,
            "total_api_calls": total_api_calls,
            "actual_execution_time": actual_execution_time,
            "avg_eval_prediction_error": avg_eval_error,
            "overall_knowledge_graph_accuracy": overall_kg_accuracy,
            "num_training_episodes": len(self.memory),
            "knowledge_graph_size": len(self.agent_knowledge_graph.graph.edges()),
        }

        return TaskResult(
            task_id=task.task_id,
            task_name=task.name,
            agent_type=self.agent_type,
            success=final_score > 0.6,
            score=final_score,
            metrics=summary_metrics,
            execution_time=actual_execution_time,
        )

    async def _run_episode(
        self, episode_id: Union[int, str], scenario: Dict[str, Any], model: BaseModel, is_eval: bool = False
    ) -> Tuple[EcosystemTrialResult, Dict[str, int]]:
        initial_populations = scenario["initial_populations"]
        intervention = scenario["intervention"]

        prompt = self._create_prompt(initial_populations, intervention)
        response: Optional[ModelResponse] = None
        predicted_populations: Dict[str, int] = {}
        predicted_relationships: Dict[str, List[str]] = {}

        try:
            api_start = time.time()
            response = await model.generate(prompt)
            api_time = time.time() - api_start
            wait_seconds = float(self.config.additional_params.get("wait_seconds", 15.0)) if getattr(self, "config", None) else 15.0
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            predicted_populations, predicted_relationships = self._parse_prediction(response.text)
        except Exception as e:
            logger.error(f"API call failed during episode {episode_id}: {e}")

        actual_populations = self.solver.solve_ecosystem_change(initial_populations, intervention)
        prediction_error = self._calculate_prediction_error(predicted_populations, actual_populations)

        return (
            EcosystemTrialResult(
                episode_id=episode_id,
                intervention_type=intervention["type"],
                species_affected=intervention.get("species", "none"),
                initial_populations=initial_populations,
                predicted_populations=predicted_populations,
                predicted_relationships=predicted_relationships,
                actual_populations=actual_populations,
                prediction_error=prediction_error,
            ),
            {
                "output_tokens": (response.completion_tokens if response and response.completion_tokens else 0),
                "api_calls": 1,
                "api_time": (api_time if 'api_time' in locals() else 0.0),
            },
        )

    def _create_prompt(self, initial_populations: Dict[str, int], intervention: Dict[str, Any]) -> str:
        prompt = f"""You are a research ecologist with a knowledge graph of ecosystem relationships.

CRITICAL INSTRUCTIONS:
- Use ONLY your current knowledge graph to make predictions
- This is a simulated ecosystem with artificial species (Species_A, Species_B, etc.)
- Do NOT assume any real-world biological knowledge applies
- You will be evaluated ONLY on population prediction accuracy

Your Current Knowledge Graph - Species Relationships You've Discovered:
{self._format_knowledge_graph_state()}

Ecosystem Rules (apply these using your knowledge graph):
- Total carrying capacity: {self.solver.total_carrying_capacity} individuals
- Species compete for limited resources
- Population changes follow consistent patterns over {self.solver.episode_steps} time steps
- Your knowledge graph represents predator-prey relationships you've learned

Current Ecosystem State (before intervention):"""

        for species, population in sorted(initial_populations.items()):
            prompt += f"\n- {species}: {population} individuals"

        prompt += f"\n\nIntervention Being Applied:"

        if intervention["type"] == "add_species":
            prompt += f"\n- Add {intervention['count']} {intervention['species']} individuals to the ecosystem"
        elif intervention["type"] == "remove_species":
            prompt += f"\n- Remove all {intervention['species']} individuals from the ecosystem"
        elif intervention["type"] == "double_population":
            prompt += f"\n- Double the population of {intervention['species']}"
        elif intervention["type"] == "halve_population":
            prompt += f"\n- Reduce the population of {intervention['species']} by half"

        prompt += """

Your Tasks:
1. PREDICT FINAL POPULATIONS: Use your knowledge graph to predict populations after {self.solver.episode_steps} time steps
2. UPDATE KNOWLEDGE GRAPH: Refine your understanding of relationships based on any new insights

EVALUATION: You will be scored ONLY on population prediction accuracy (0-1 scale).

Use your knowledge graph to apply these dynamics:
- Predators reduce prey populations
- All species have base growth rates  
- Carrying capacity limits total population
- Competition occurs when resources are scarce

FORMAT YOUR RESPONSE EXACTLY AS:
[populations: Species_A:123,Species_B:456,Species_C:789]
[relationships: Species_A:Species_B,Species_C:Species_A,Species_D:Species_B|Species_C]

Where:
- populations: Your predicted final population for each species (THIS IS WHAT'S SCORED)
- relationships: Your updated knowledge graph (predator:prey, use | for multiple prey)
- Include ALL species from initial state in populations, predict 0 if extinct"""

        return prompt

    def _format_knowledge_graph_state(self) -> str:
        structure = self.agent_knowledge_graph.get_ecosystem_structure()
        if not structure:
            return "- No relationships discovered yet - you must learn through experience"

        kg_text = ""
        for species in sorted(structure.keys()):
            prey_list = structure[species]
            species_type = self.agent_knowledge_graph.classify_species_type(species)
            if prey_list:
                confidence_info: List[str] = []
                for prey in prey_list:
                    conf = self.agent_knowledge_graph.get_relationship_confidence(species, prey)
                    confidence_info.append(f"{prey}(conf:{conf:.2f})")
                kg_text += f"\n- {species} ({species_type}) eats: {', '.join(confidence_info)}"
            else:
                kg_text += f"\n- {species} ({species_type}) eats: nothing (likely producer)"
        return kg_text if kg_text else "- No relationships discovered yet"

    def _parse_prediction(self, response_text: str) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
        populations: Dict[str, int] = {}
        relationships: Dict[str, List[str]] = {}

        pop_match = re.search(r"\[populations:\s*([^\]]+)\]", response_text, re.IGNORECASE)
        if pop_match:
            pop_str = pop_match.group(1)
            pairs = re.findall(r"(Species_[A-Z]):\s*(\d+)", pop_str, re.IGNORECASE)
            for species, population_str in pairs:
                try:
                    populations[species] = int(population_str)
                except ValueError:
                    pass

        rel_match = re.search(r"\[relationships:\s*([^\]]+)\]", response_text, re.IGNORECASE)
        if rel_match:
            rel_str = rel_match.group(1)
            pred_pairs = re.findall(r"(Species_[A-Z]):((?:Species_[A-Z](?:\|Species_[A-Z])*)?)", rel_str, re.IGNORECASE)
            for predator, prey_str in pred_pairs:
                if prey_str:
                    prey_list = [p.strip() for p in prey_str.split('|') if p.strip()]
                    relationships[predator] = prey_list
                else:
                    relationships[predator] = []

        return populations, relationships

    def _calculate_prediction_error(self, predicted: Dict[str, int], actual: Dict[str, int]) -> float:
        if not predicted or not actual:
            return float('inf')
        total_error = 0.0
        species_count = 0
        for species in actual.keys():
            actual_pop = actual[species]
            predicted_pop = predicted.get(species, 0)
            error = abs(predicted_pop - actual_pop)
            total_error += error
            species_count += 1
        return total_error / species_count if species_count > 0 else float('inf')

    def _calculate_prediction_score(self, predicted_populations: Dict[str, int], actual_populations: Dict[str, int]) -> float:
        if not predicted_populations or not actual_populations:
            return 0.0
        total_error = 0.0
        total_max_error = 0.0
        species_count = 0
        for species in actual_populations.keys():
            actual_pop = actual_populations[species]
            predicted_pop = predicted_populations.get(species, 0)
            actual_error = abs(predicted_pop - actual_pop)
            if actual_pop == 0:
                max_error = max(predicted_pop, 100) if predicted_pop > 0 else 100
            else:
                max_error = max(actual_pop, actual_pop * 3)
            total_error += actual_error
            total_max_error += max_error
            species_count += 1
        if total_max_error == 0 or species_count == 0:
            return 1.0 if total_error == 0 else 0.0
        normalized_error = total_error / total_max_error
        return max(0.0, 1.0 - normalized_error)

    def _update_memory(self, memory: List[EcosystemTrialResult], new_result: EcosystemTrialResult):
        memory.append(new_result)

    def _update_agent_knowledge_graph(self, trial_result: EcosystemTrialResult):
        initial = trial_result.initial_populations
        actual = trial_result.actual_populations
        predicted_rels = trial_result.predicted_relationships
        prediction_accuracy = 1.0 / (1.0 + trial_result.prediction_error / 20.0)

        affected = trial_result.species_affected
        total_pop = sum(actual.values())
        capacity_guard = self.params["capacity_guard_threshold"] * self.solver.total_carrying_capacity

        # 1) Accumulate evidence based on targeted interventions and directional changes
        for predator in initial.keys():
            for prey in initial.keys():
                if predator == prey:
                    continue
                key = (predator, prey)
                initial_pred = initial.get(predator, 0)
                initial_prey = initial.get(prey, 0)
                final_pred = actual.get(predator, 0)
                final_prey = actual.get(prey, 0)
                if initial_pred == 0 or initial_prey == 0:
                    continue
                change_pred = (final_pred - initial_pred) / initial_pred
                change_prey = (final_prey - initial_prey) / initial_prey

                targeted = (predator == affected) or (prey == affected)
                if not targeted:
                    continue

                if total_pop >= capacity_guard:
                    # Under capacity stress, do not infer predation
                    self._evidence_oppose[key] = self._evidence_oppose.get(key, 0) + 1
                    continue

                # Predation-like signals: predator up & prey down, or predator down & prey up
                if (change_pred > 0.05 and change_prey < -0.1) or (change_pred < -0.05 and change_prey > 0.1):
                    self._evidence_support[key] = self._evidence_support.get(key, 0) + 1
                else:
                    # Non-predation-like signal
                    self._evidence_oppose[key] = self._evidence_oppose.get(key, 0) + 1

        # 2) Apply evidence thresholds to update/remove relationships
        min_support = int(self.params["evidence_min_support"]) or 1
        for key, supp in list(self._evidence_support.items()):
            opp = self._evidence_oppose.get(key, 0)
            predator, prey = key
            if supp >= min_support and supp > opp:
                # Confidence shaped by margin between support and oppose
                conf = min(0.95, 0.2 + 0.15 * (supp - opp))
                self.agent_knowledge_graph.update_relationship(predator, prey, conf)
            elif opp >= min_support and opp > supp:
                self.agent_knowledge_graph.remove_relationship(predator, prey)

        # 3) Incorporate model-predicted relationships only if some evidence exists and accuracy is high
        if prediction_accuracy >= self.params["accuracy_gate_for_boost"]:
            for predator, prey_list in predicted_rels.items():
                for prey in prey_list:
                    if self._evidence_support.get((predator, prey), 0) >= 1:
                        boost = min(0.1, prediction_accuracy * 0.1)
                        self.agent_knowledge_graph.update_relationship(predator, prey, boost)

        # 4) Estimate producer growth rates (no predators, no prey)
        for species in initial.keys():
            prey_list = self.agent_knowledge_graph.get_prey_list(species)
            predator_list = self.agent_knowledge_graph.get_predators_of(species)
            if not prey_list and not predator_list:
                initial_pop = initial.get(species, 0)
                final_pop = actual.get(species, 0)
                if initial_pop > 0:
                    growth_factor = final_pop / initial_pop
                    estimated_growth = (growth_factor - 1.0) / max(1, self.solver.episode_steps)
                    current_estimate = self.agent_knowledge_graph.species_growth_estimates.get(species, 0.15)
                    new_estimate = (current_estimate * self.params["confidence_decay_existing"]) + (estimated_growth * self.params["confidence_new_weight"])
                    self.agent_knowledge_graph.species_growth_estimates[species] = max(0.0, min(0.5, new_estimate))

    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        return 0.0


