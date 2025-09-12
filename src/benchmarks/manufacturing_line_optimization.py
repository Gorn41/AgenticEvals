"""
Utility-Based Manufacturing Line Optimization Benchmark for AgenticEvals.

This benchmark evaluates a model's ability to optimize manufacturing line
parameters against multiple competing objectives using a weighted utility
function. The model must reason about trade-offs (throughput, quality,
energy efficiency, longevity) and output a final configuration in a strict
JSON format enclosed in a [FINAL ANSWER: ...] tag.

Scoring follows a re-scaled utility measure per scenario so that:
  - Score = 1.0  -> model matches the optimal feasible configuration
  - Score = 0.0  -> model chooses the worst valid configuration (or invalid)
  - 0 < Score < 1 -> relative performance between worst and best valid

The benchmark uses a discrete parameter grid (≈111k combinations per scenario)
and an internal solver to compute objective-wise min/max over the feasible set,
the optimal utility U_max, and the worst feasible utility U_min. During
evaluation, score = max(0, (Achieved - U_min)) / (U_max - U_min) for valid
solutions; invalid outputs score 0.0.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..benchmark.base import (
    AgentType,
    BaseBenchmark,
    BenchmarkConfig,
    Task,
    TaskResult,
)
from ..benchmark.registry import benchmark
from ..models.base import BaseModel, ModelResponse
from ..utils.logging import get_logger

logger = get_logger(__name__)


# Discrete parameter grid (≈ 111,000 points per scenario)
DISCRETE_PARAMETERS: Dict[str, List[float]] = {
    "line_speed": [float(v) for v in range(50, 201, 5)],  # 50, 55, ..., 200
    "qc_strictness": [i / 2.0 for i in range(2, 21)],     # 1.0, 1.5, ..., 10.0
    "power_setting": [float(v) for v in range(60, 101, 2)],  # 60, 62, ..., 100
    "maintenance_freq": [i / 2.0 for i in range(2, 11)],  # 1.0, 1.5, ..., 5.0
}


@dataclass(frozen=True)
class Parameters:
    """Manufacturing line parameter tuple."""

    speed: float         # units/hour
    qc: float            # QC strictness (1.0–10.0)
    power: float         # % (60–100)
    maintenance: float   # cycles/day (1.0–5.0)


@dataclass(frozen=True)
class ObjectiveRaw:
    """Raw objective values before normalization."""

    throughput: float  # higher is better
    quality: float     # rate in [0, 1], higher is better
    efficiency: float  # higher is better (inverse energy/unit)
    longevity: float   # higher is better (inverse wear)


@dataclass(frozen=True)
class ObjectiveMinMax:
    """Per-objective min/max across feasible parameter grid."""

    throughput_min: float
    throughput_max: float
    quality_min: float
    quality_max: float
    efficiency_min: float
    efficiency_max: float
    longevity_min: float
    longevity_max: float


def _float_in_allowed(value: float, allowed: List[float], tol: float = 1e-9) -> bool:
    for a in allowed:
        if abs(value - a) <= tol:
            return True
    return False


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _normalize(x: float, x_min: float, x_max: float) -> float:
    if x_max - x_min <= 1e-12:
        # Degenerate range; return midpoint to avoid division by zero.
        return 0.5
    return (x - x_min) / (x_max - x_min)


def _iter_parameter_grid() -> Iterable[Parameters]:
    for speed in DISCRETE_PARAMETERS["line_speed"]:
        for qc in DISCRETE_PARAMETERS["qc_strictness"]:
            for power in DISCRETE_PARAMETERS["power_setting"]:
                for maintenance in DISCRETE_PARAMETERS["maintenance_freq"]:
                    yield Parameters(
                        speed=speed, qc=qc, power=power, maintenance=maintenance
                    )


def _compute_objectives_raw(
    params: Parameters,
    alpha: float,
    beta: float,
    gamma: float,
    k_energy: float,
    h_wear: float,
) -> ObjectiveRaw:
    """Computes raw objective values from physics-inspired relationships."""
    speed = params.speed
    qc = params.qc
    power = params.power
    maintenance = params.maintenance

    # Throughput falls with stricter QC
    throughput = speed * (1.0 - alpha * (qc - 1.0) / 9.0)

    # Quality improves with stricter QC but falls with higher speed
    quality = _clamp(
        0.6 + beta * (qc - 1.0) / 9.0 - gamma * (speed - 50.0) / 150.0,
        0.0,
        1.0,
    )

    # Energy per unit rises with power and speed; efficiency is inverse
    energy_per_unit = (power / 80.0) ** 1.2 * (1.0 + k_energy * (speed - 100.0) / 100.0)
    # Guard against tiny or negative values (the expression should be positive)
    energy_per_unit = max(1e-9, energy_per_unit)
    efficiency = 1.0 / energy_per_unit

    # Wear increases with power and speed; maintenance mitigates wear; longevity is inverse
    wear = (speed / 100.0) ** 1.3 * (power / 80.0) ** 1.1 / (1.0 + h_wear * (maintenance - 1.0))
    wear = max(1e-9, wear)
    longevity = 1.0 / wear

    return ObjectiveRaw(
        throughput=throughput,
        quality=quality,
        efficiency=efficiency,
        longevity=longevity,
    )


def _weighted_utility(
    raw: ObjectiveRaw,
    minmax: ObjectiveMinMax,
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Returns weighted utility and component-wise normalized objectives."""
    t_n = _normalize(raw.throughput, minmax.throughput_min, minmax.throughput_max)
    q_n = _normalize(raw.quality, minmax.quality_min, minmax.quality_max)
    e_n = _normalize(raw.efficiency, minmax.efficiency_min, minmax.efficiency_max)
    l_n = _normalize(raw.longevity, minmax.longevity_min, minmax.longevity_max)

    utility = (
        weights["T"] * t_n
        + weights["Q"] * q_n
        + weights["E"] * e_n
        + weights["L"] * l_n
    )

    return utility, {"T": t_n, "Q": q_n, "E": e_n, "L": l_n}


@benchmark(
    name="manufacturing_line_optimization",
    agent_type=AgentType.UTILITY_BASED,
    description="Optimize manufacturing parameters to maximize a weighted utility across throughput, quality, energy efficiency, and equipment longevity.",
)
class ManufacturingOptimizationBenchmark(BaseBenchmark):
    """Benchmark for multi-objective manufacturing line optimization."""

    # Default physics coefficients; tuned to avoid objective collapse
    COEFFS = {
        "alpha": 0.30,  # Throughput sensitivity to QC
        "beta": 0.35,   # Quality sensitivity to QC
        "gamma": 0.25,  # Quality sensitivity to speed
        "k_energy": 0.20,  # Energy sensitivity to speed
        "h_wear": 0.50,    # Maintenance mitigation on wear
    }

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)

    def get_tasks(self) -> List[Task]:
        tasks: List[Task] = []

        # Scenario list: weights sum to 1.0; quality threshold enforced
        scenarios: List[Dict[str, Any]] = [
            {"name": "Scenario 1", "weights": {"T": 0.6, "Q": 0.2, "E": 0.1, "L": 0.1}, "quality_threshold": 0.65},
            {"name": "Scenario 2", "weights": {"T": 0.5, "Q": 0.3, "E": 0.1, "L": 0.1}, "quality_threshold": 0.65},
            {"name": "Scenario 3", "weights": {"T": 0.1, "Q": 0.7, "E": 0.1, "L": 0.1}, "quality_threshold": 0.75},
            {"name": "Scenario 4", "weights": {"T": 0.2, "Q": 0.6, "E": 0.1, "L": 0.1}, "quality_threshold": 0.75},
            {"name": "Scenario 5", "weights": {"T": 0.2, "Q": 0.2, "E": 0.5, "L": 0.1}, "quality_threshold": 0.80},
            {"name": "Scenario 6", "weights": {"T": 0.3, "Q": 0.2, "E": 0.4, "L": 0.1}, "quality_threshold": 0.80},
            {"name": "Scenario 7", "weights": {"T": 0.2, "Q": 0.2, "E": 0.1, "L": 0.5}, "quality_threshold": 0.65},
            {"name": "Scenario 8", "weights": {"T": 0.3, "Q": 0.2, "E": 0.1, "L": 0.4}, "quality_threshold": 0.65},
            {"name": "Scenario 9", "weights": {"T": 0.25, "Q": 0.25, "E": 0.25, "L": 0.25}, "quality_threshold": 0.70},
            {"name": "Scenario 10", "weights": {"T": 0.4, "Q": 0.4, "E": 0.1, "L": 0.1}, "quality_threshold": 0.70},
        ]

        for i, scenario in enumerate(scenarios):
            prep = self._prepare_scenario(scenario)

            # Build prompt and expected output string for readability
            prompt = self._create_prompt(
                weights=scenario["weights"],
                quality_threshold=scenario["quality_threshold"],
                minmax=prep["objective_minmax"],
                coeffs=self.COEFFS,
            )

            expected_json = json.dumps(
                {
                    "speed": prep["optimal_params"].speed,
                    "qc": prep["optimal_params"].qc,
                    "power": prep["optimal_params"].power,
                    "maintenance": prep["optimal_params"].maintenance,
                }
            )
            expected_output = f"[FINAL ANSWER: {expected_json}]"

            task = Task(
                task_id=f"manufacturing_opt_{i+1}",
                name=f"Manufacturing Optimization: {scenario['name']}",
                description=f"Optimize parameters given explicit objective weights. Quality threshold: {scenario['quality_threshold']:.2f}",
                prompt=prompt,
                expected_output=expected_output,
                metadata={
                    "weights": scenario["weights"],
                    "quality_threshold": scenario["quality_threshold"],
                    "objective_minmax": prep["objective_minmax"].__dict__,
                    "U_max": prep["U_max"],
                    "U_min": prep["U_min"],
                    "optimal_params": {
                        "speed": prep["optimal_params"].speed,
                        "qc": prep["optimal_params"].qc,
                        "power": prep["optimal_params"].power,
                        "maintenance": prep["optimal_params"].maintenance,
                    },
                    "optimal_components": prep["optimal_components"],
                    "coeffs": self.COEFFS,
                },
            )
            tasks.append(task)

        return tasks

    def _prepare_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Compute min/max per objective, and U_min/U_max/optimal parameters for the scenario."""
        weights = scenario["weights"]
        q_threshold = float(scenario.get("quality_threshold", 0.85))

        alpha = self.COEFFS["alpha"]
        beta = self.COEFFS["beta"]
        gamma = self.COEFFS["gamma"]
        k_energy = self.COEFFS["k_energy"]
        h_wear = self.COEFFS["h_wear"]

        # First pass: determine per-objective min/max over feasible points
        t_min = math.inf
        t_max = -math.inf
        q_min = math.inf
        q_max = -math.inf
        e_min = math.inf
        e_max = -math.inf
        l_min = math.inf
        l_max = -math.inf

        feasible_count = 0
        for p in _iter_parameter_grid():
            raw = _compute_objectives_raw(p, alpha, beta, gamma, k_energy, h_wear)
            if raw.quality + 1e-12 < q_threshold:
                continue
            feasible_count += 1
            if raw.throughput < t_min:
                t_min = raw.throughput
            if raw.throughput > t_max:
                t_max = raw.throughput
            if raw.quality < q_min:
                q_min = raw.quality
            if raw.quality > q_max:
                q_max = raw.quality
            if raw.efficiency < e_min:
                e_min = raw.efficiency
            if raw.efficiency > e_max:
                e_max = raw.efficiency
            if raw.longevity < l_min:
                l_min = raw.longevity
            if raw.longevity > l_max:
                l_max = raw.longevity

        if feasible_count == 0:
            raise ValueError(
                f"No feasible parameter configurations found for scenario '{scenario['name']}' with quality_threshold={q_threshold}."
            )

        # Ensure each objective has non-degenerate range across feasible grid
        eps = 1e-12
        if (t_max - t_min) <= eps or (q_max - q_min) <= eps or (e_max - e_min) <= eps or (l_max - l_min) <= eps:
            raise ValueError(
                "Degenerate objective range detected. Adjust coefficients/thresholds so each of T,Q,E,L varies over the feasible grid. "
                f"Ranges -> T: {t_min:.6f}..{t_max:.6f}, Q: {q_min:.6f}..{q_max:.6f}, E: {e_min:.6f}..{e_max:.6f}, L: {l_min:.6f}..{l_max:.6f}"
            )

        minmax = ObjectiveMinMax(
            throughput_min=t_min,
            throughput_max=t_max,
            quality_min=q_min,
            quality_max=q_max,
            efficiency_min=e_min,
            efficiency_max=e_max,
            longevity_min=l_min,
            longevity_max=l_max,
        )

        # Second pass: compute weighted normalized utility; collect U_min / U_max and the argmax
        U_max = -math.inf
        U_min = math.inf
        optimal_params: Optional[Parameters] = None
        optimal_components: Dict[str, float] = {}

        for p in _iter_parameter_grid():
            raw = _compute_objectives_raw(p, alpha, beta, gamma, k_energy, h_wear)
            if raw.quality + 1e-12 < q_threshold:
                continue
            u, comps = _weighted_utility(raw, minmax, weights)
            if u > U_max:
                U_max = u
                optimal_params = p
                optimal_components = comps
            if u < U_min:
                U_min = u

        assert optimal_params is not None and U_max != -math.inf

        return {
            "objective_minmax": minmax,
            "U_max": float(U_max),
            "U_min": float(U_min),
            "optimal_params": optimal_params,
            "optimal_components": optimal_components,
        }

    def _create_prompt(
        self,
        weights: Dict[str, float],
        quality_threshold: float,
        minmax: ObjectiveMinMax,
        coeffs: Dict[str, float],
    ) -> str:
        """Create the instruction prompt with explicit equations, constants, normalization, allowed values, and answer format."""
        def fmt_list(values: List[float]) -> str:
            # Show full list for determinism and clarity
            return ", ".join(f"{v:g}" for v in values)

        prompt = []
        prompt.append("Manufacturing Line Optimization Task\n")
        prompt.append("You are a manufacturing optimization agent. Think step by step to analyze the trade-offs between throughput (T), quality (Q), energy efficiency (E), and equipment longevity (L). Your goal is to MAXIMIZE the weighted utility defined below.\n")
        prompt.append("After your reasoning, provide the final configuration as JSON in the required answer box.\n\n")

        prompt.append("Objective Weights (sum to 1.0):\n")
        prompt.append(f"- Throughput (T): {weights['T']}\n")
        prompt.append(f"- Quality Rate (Q): {weights['Q']}\n")
        prompt.append(f"- Energy Efficiency (E): {weights['E']}\n")
        prompt.append(f"- Equipment Longevity (L): {weights['L']}\n\n")

        # Equations and constants
        prompt.append("Utility and Normalization:\n")
        prompt.append("- Utility = w_T * T_norm + w_Q * Q_norm + w_E * E_norm + w_L * L_norm\n")
        prompt.append("- Normalization for each objective X ∈ {T,Q,E,L}: X_norm = (X_raw − X_min) / (X_max − X_min).\n")
        prompt.append("  If X_max == X_min, X_norm is treated as 0.5.\n")
        prompt.append("- The following per-objective min/max are computed over the feasible grid for this scenario:\n")
        prompt.append(
            f"  T_min={minmax.throughput_min:.6f}, T_max={minmax.throughput_max:.6f}; "
            f"Q_min={minmax.quality_min:.6f}, Q_max={minmax.quality_max:.6f}; "
            f"E_min={minmax.efficiency_min:.6f}, E_max={minmax.efficiency_max:.6f}; "
            f"L_min={minmax.longevity_min:.6f}, L_max={minmax.longevity_max:.6f}\n\n"
        )

        prompt.append("Objective Equations and Constants (use these EXACT formulas):\n")
        prompt.append(
            f"- Throughput (raw): T_raw = speed * (1 − α * (qc − 1) / 9), with α={coeffs['alpha']:.3f}\n"
        )
        prompt.append(
            f"- Quality (raw, clamped to [0,1]): Q_raw = clamp(0.6 + β * (qc − 1) / 9 − γ * (speed − 50) / 150, 0, 1), with β={coeffs['beta']:.3f}, γ={coeffs['gamma']:.3f}\n"
        )
        prompt.append(
            f"- Energy per unit: C = (power/80)^1.2 * (1 + k * (speed − 100)/100), with k={coeffs['k_energy']:.3f}.\n"
        )
        prompt.append("  Energy efficiency (raw): E_raw = 1 / C\n")
        prompt.append(
            f"- Wear: W = (speed/100)^1.3 * (power/80)^1.1 / (1 + h * (maintenance − 1)), with h={coeffs['h_wear']:.3f}.\n"
        )
        prompt.append("  Equipment longevity (raw): L_raw = 1 / W\n")
        prompt.append(
            f"- Feasibility constraint: Q_raw must be ≥ quality_threshold = {quality_threshold:.2f}. Only feasible settings are allowed.\n\n"
        )

        prompt.append("Constraints:\n")
        prompt.append(f"- Minimum quality threshold: {quality_threshold:.2f}\n")
        prompt.append("- Parameters must be chosen only from the allowed discrete values below.\n\n")

        prompt.append("Allowed Values (use these with the SAME JSON keys):\n")
        prompt.append(f"- speed (units/hour): {{{fmt_list(DISCRETE_PARAMETERS['line_speed'])}}}\n")
        prompt.append(f"- qc: {{{fmt_list(DISCRETE_PARAMETERS['qc_strictness'])}}}\n")
        prompt.append(f"- power (%): {{{fmt_list(DISCRETE_PARAMETERS['power_setting'])}}}\n")
        prompt.append(f"- maintenance (cycles/day): {{{fmt_list(DISCRETE_PARAMETERS['maintenance_freq'])}}}\n\n")

        prompt.append("Output Requirements:\n")
        prompt.append("- First, write your reasoning and calculations, including how you compute raw objectives, apply normalization using the scenario-specific min/max above, and combine them with the weights to get Utility. Aim to MAXIMIZE Utility.\n")
        prompt.append("- Then, on a new line, output the final configuration in this exact format:\n")
        example = {
            "speed": 150,
            "qc": 8.0,
            "power": 90,
            "maintenance": 3.0,
        }
        prompt.append(f"  [FINAL ANSWER: {json.dumps(example)}]\n")
        prompt.append("- The JSON must include the keys speed, qc, power, maintenance with numeric values from the allowed sets.\n")
        prompt.append("- Use plain decimal numbers only (no scientific notation).\n")
        prompt.append("- Do not add any extra text after the answer box.\n")

        return "".join(prompt)

    def _parse_parameters(self, response_text: str) -> Tuple[Optional[Parameters], Dict[str, Any]]:
        """Parse the model's response for parameters using robust fallbacks.

        Returns (Parameters or None if invalid, diagnostics dict).
        """
        diagnostics: Dict[str, Any] = {}

        # 1) Primary: last [FINAL ANSWER: {...}] JSON block
        try:
            matches = list(
                re.finditer(r"\[\s*FINAL\s+ANSWER\s*:\s*(\{.*?\})\s*\]", response_text, re.IGNORECASE | re.DOTALL)
            )
            if matches:
                last = matches[-1]
                json_str = last.group(1)
                data = json.loads(json_str)
                params = self._params_from_mapping(data)
                if params is not None:
                    diagnostics["parse_strategy"] = "final_answer_json"
                    return params, diagnostics
        except Exception as e:
            diagnostics["final_answer_json_error"] = str(e)

        # 2) Secondary: scan from the end for any JSON object that includes required keys
        try:
            json_candidates = list(re.finditer(r"(\{[^{}]*\})", response_text, re.DOTALL))
            for m in reversed(json_candidates):
                cand = m.group(1)
                try:
                    data = json.loads(cand)
                except Exception:
                    continue
                params = self._params_from_mapping(data)
                if params is not None:
                    diagnostics["parse_strategy"] = "raw_json_anywhere"
                    return params, diagnostics
        except Exception as e:
            diagnostics["raw_json_error"] = str(e)

        # 3) Tertiary: key=value fallback from back to front (any order)
        try:
            def find_last_number_for(key: str) -> Optional[float]:
                # Find the last occurrence of key=number (allow spaces)
                pattern = rf"{key}\s*=\s*([0-9]+(?:\.[0-9]+)?)"
                ms = list(re.finditer(pattern, response_text, re.IGNORECASE))
                if not ms:
                    return None
                return float(ms[-1].group(1))

            speed_v = find_last_number_for("speed")
            qc_v = find_last_number_for("qc")
            power_v = find_last_number_for("power")
            maint_v = find_last_number_for("maintenance")

            if None not in (speed_v, qc_v, power_v, maint_v):
                data = {"speed": speed_v, "qc": qc_v, "power": power_v, "maintenance": maint_v}
                params = self._params_from_mapping(data)
                if params is not None:
                    diagnostics["parse_strategy"] = "kv_pairs_from_end"
                    return params, diagnostics
        except Exception as e:
            diagnostics["kv_pairs_error"] = str(e)

        diagnostics["parse_strategy"] = "failed"
        return None, diagnostics

    def _params_from_mapping(self, data: Dict[str, Any]) -> Optional[Parameters]:
        """Validate mapping -> Parameters against discrete sets."""
        try:
            speed = float(data["speed"])  # units/hour
            qc = float(data["qc"])  # 1.0–10.0 step 0.5
            power = float(data["power"])  # 60–100 step 2
            maintenance = float(data["maintenance"])  # 1.0–5.0 step 0.5
        except Exception:
            return None

        if not _float_in_allowed(speed, DISCRETE_PARAMETERS["line_speed"]):
            return None
        if not _float_in_allowed(qc, DISCRETE_PARAMETERS["qc_strictness"]):
            return None
        if not _float_in_allowed(power, DISCRETE_PARAMETERS["power_setting"]):
            return None
        if not _float_in_allowed(maintenance, DISCRETE_PARAMETERS["maintenance_freq"]):
            return None

        return Parameters(speed=speed, qc=qc, power=power, maintenance=maintenance)

    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        """Evaluate a single manufacturing optimization task."""
        start_time = time.time()
        model_response = await model.generate(task.prompt)
        execution_time = float(model_response.latency or 0.0)

        weights = task.metadata["weights"]
        q_threshold = float(task.metadata["quality_threshold"])
        minmax_dict = task.metadata["objective_minmax"]
        minmax = ObjectiveMinMax(
            throughput_min=minmax_dict["throughput_min"],
            throughput_max=minmax_dict["throughput_max"],
            quality_min=minmax_dict["quality_min"],
            quality_max=minmax_dict["quality_max"],
            efficiency_min=minmax_dict["efficiency_min"],
            efficiency_max=minmax_dict["efficiency_max"],
            longevity_min=minmax_dict["longevity_min"],
            longevity_max=minmax_dict["longevity_max"],
        )
        U_max = float(task.metadata["U_max"])
        U_min = float(task.metadata["U_min"])

        # Parse response parameters
        parsed_params, diagnostics = self._parse_parameters(model_response.text)

        alpha = self.COEFFS["alpha"]
        beta = self.COEFFS["beta"]
        gamma = self.COEFFS["gamma"]
        k_energy = self.COEFFS["k_energy"]
        h_wear = self.COEFFS["h_wear"]

        is_valid = False
        achieved_utility = 0.0
        model_components = {"T": 0.0, "Q": 0.0, "E": 0.0, "L": 0.0}
        feasibility_reason = ""

        if parsed_params is not None:
            raw = _compute_objectives_raw(parsed_params, alpha, beta, gamma, k_energy, h_wear)
            if raw.quality + 1e-12 >= q_threshold:
                is_valid = True
                achieved_utility, model_components = _weighted_utility(raw, minmax, weights)
            else:
                feasibility_reason = "quality_below_threshold"
        else:
            feasibility_reason = "parse_failed"

        # Score re-scaling; invalid -> 0.0
        score = 0.0
        if is_valid:
            denom = max(1e-12, U_max - U_min)
            score = max(0.0, (achieved_utility - U_min)) / denom

        # Success criterion: valid and near-optimal
        success = is_valid and score >= 0.95

        # Prepare metrics
        optimal_params = task.metadata["optimal_params"]
        optimal_components = task.metadata["optimal_components"]

        metrics: Dict[str, Any] = {
            "achieved_utility": achieved_utility if is_valid else 0.0,
            "optimal_utility": U_max,
            "worst_utility": U_min,
            "optimality_gap": (U_max - achieved_utility) if is_valid else U_max - U_min,
            "is_valid_parameters": is_valid,
            "feasibility_reason": feasibility_reason,
            "quality_threshold": q_threshold,
            "model_components": model_components,
            "optimal_components": optimal_components,
            "chosen_parameters": (
                {
                    "speed": parsed_params.speed,
                    "qc": parsed_params.qc,
                    "power": parsed_params.power,
                    "maintenance": parsed_params.maintenance,
                }
                if parsed_params is not None
                else {}
            ),
            "optimal_parameters": optimal_params,
            "output_tokens": model_response.completion_tokens,
            "parse_diagnostics": diagnostics,
        }

        return TaskResult(
            task_id=task.task_id,
            task_name=task.name,
            agent_type=self.agent_type,
            success=success,
            score=max(0.0, min(1.0, score)),
            metrics=metrics,
            model_response=model_response,
            execution_time=execution_time,
        )

    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        # Scoring computed during evaluate_task
        return 0.0


