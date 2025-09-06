"""
EV Charging Policy Optimization (Multi-Turn) for AgenticEvals.

This benchmark mirrors the manufacturing_line_optimization scoring design:
- Discrete policy grid
- Explicit objective equations and weights in the prompt
- Solver computes per-objective min/max over feasible set and U_min/U_max
- Final score = max(0, (U_achieved - U_min)) / max(1e-12, U_max - U_min)
- Invalid output or parameters outside discrete sets => score 0

Differences:
- Multi-turn (10 turns): each turn reveals additional realized arrivals
- Model is expected to build an internal model of the arrival pattern
- Only the final (turn 10) output is scored; earlier turns provide history
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..benchmark.base import (
    AgentType,
    BaseBenchmark,
    BenchmarkConfig,
    Task,
    TaskResult,
)
from ..models.base import BaseModel, ModelResponse
from ..benchmark.registry import validation_benchmark
from ..utils.logging import get_logger

logger = get_logger(__name__)


DISCRETE_POLICY: Dict[str, List[float]] = {
    "policy_power_kW": [11.0, 22.0, 30.0],
    "smoothing_factor": [0.0, 0.5, 1.0],
    "cap_fraction": [0.6, 0.8, 1.0],
    "slot_shift_min": [0.0, 10.0, 20.0],
}


@dataclass(frozen=True)
class Policy:
    policy_power_kW: float
    smoothing_factor: float
    cap_fraction: float
    slot_shift_min: float


@dataclass(frozen=True)
class ObjectiveMinMax:
    T_min: float
    T_max: float
    Q_min: float
    Q_max: float
    E_min: float
    E_max: float
    L_min: float
    L_max: float


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _normalize(x: float, x_min: float, x_max: float) -> float:
    if x_max - x_min <= 1e-12:
        return 0.5
    return (x - x_min) / (x_max - x_min)


def _float_in_allowed(value: float, allowed: List[float], tol: float = 1e-9) -> bool:
    for a in allowed:
        if abs(value - a) <= tol:
            return True
    return False


def _generate_arrival_pattern(pattern: str, base: float, amp: float, buckets: int) -> List[int]:
    vals: List[float] = []
    if pattern == "constant":
        vals = [base] * buckets
    elif pattern == "increase":
        vals = [base + amp * (i / max(1, buckets - 1)) for i in range(buckets)]
    elif pattern == "decrease":
        vals = [base + amp * (1 - i / max(1, buckets - 1)) for i in range(buckets)]
    elif pattern == "alternate":
        vals = [base + (amp if (i % 2 == 0) else 0.0) for i in range(buckets)]
    else:
        vals = [base] * buckets
    return [max(0, int(round(v))) for v in vals]


def _policy_membership(data: Dict[str, Any]) -> Optional[Policy]:
    try:
        p = float(data["policy_power_kW"])  # 11, 22, 30
        s = float(data["smoothing_factor"])  # 0.0, 0.5, 1.0
        c = float(data["cap_fraction"])  # 0.6, 0.8, 1.0
        sh = float(data["slot_shift_min"])  # 0, 10, 20
    except Exception:
        return None
    if not _float_in_allowed(p, DISCRETE_POLICY["policy_power_kW"]):
        return None
    if not _float_in_allowed(s, DISCRETE_POLICY["smoothing_factor"]):
        return None
    if not _float_in_allowed(c, DISCRETE_POLICY["cap_fraction"]):
        return None
    if not _float_in_allowed(sh, DISCRETE_POLICY["slot_shift_min"]):
        return None
    return Policy(policy_power_kW=p, smoothing_factor=s, cap_fraction=c, slot_shift_min=sh)


def _compute_objectives_raw(
    policy: Policy,
    arrivals: List[int],
    target_kwh_per_vehicle: float,
    cap_kw: float,
    horizon_min: int,
    low_tariff_fraction_by_shift: Dict[float, float],
    beta: float,
    gamma: float,
    h_wear: float,
) -> Tuple[float, float, float, float]:
    E_demand = sum(arrivals) * target_kwh_per_vehicle
    E_cap = policy.cap_fraction * cap_kw * (horizon_min / 60.0)
    D = E_demand / max(1e-9, E_cap)

    T_raw = E_cap * min(1.0, D) * (0.8 + 0.2 * policy.smoothing_factor)
    Q_raw = _clamp(0.6 + beta * policy.smoothing_factor - gamma * max(0.0, D - 1.0), 0.0, 1.0)
    E_raw = _clamp(low_tariff_fraction_by_shift.get(policy.slot_shift_min, 0.0), 0.0, 1.0)
    wear_term = (policy.policy_power_kW / 30.0) ** 1.2 * (policy.cap_fraction) ** 1.1 / (1.0 + h_wear * policy.smoothing_factor)
    wear_term = max(1e-9, wear_term)
    L_raw = 1.0 / wear_term
    return T_raw, Q_raw, E_raw, L_raw


def _weighted_utility(
    T_raw: float,
    Q_raw: float,
    E_raw: float,
    L_raw: float,
    minmax: ObjectiveMinMax,
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    Tn = _normalize(T_raw, minmax.T_min, minmax.T_max)
    Qn = _normalize(Q_raw, minmax.Q_min, minmax.Q_max)
    En = _normalize(E_raw, minmax.E_min, minmax.E_max)
    Ln = _normalize(L_raw, minmax.L_min, minmax.L_max)
    U = weights["T"] * Tn + weights["Q"] * Qn + weights["E"] * En + weights["L"] * Ln
    return U, {"T": Tn, "Q": Qn, "E": En, "L": Ln}


@validation_benchmark(
    name="ev_charging_policy",
    agent_types=[AgentType.MODEL_BASED_REFLEX, AgentType.UTILITY_BASED],
    description="Multi-turn EV charging policy optimization with solver-based utility scaling.",
)
class EVChargingPolicyBenchmark(BaseBenchmark):
    COEFFS = {
        "beta": 0.35,   # Q sensitivity to smoothing
        "gamma": 0.25,  # Q penalty for demand over capacity
        "h_wear": 0.50, # Longevity mitigation by smoothing
    }

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)

    def get_tasks(self) -> List[Task]:
        tasks: List[Task] = []
        # Scenarios: fixed 10 buckets (10-minute each => 100 minutes horizon)
        base_scenarios: List[Dict[str, Any]] = [
            {"name": "Constant Low", "pattern": "constant", "base": 2.0, "amp": 0.0, "cap_kw": 60.0, "weights": {"T": 0.5, "Q": 0.2, "E": 0.2, "L": 0.1}, "quality_threshold": 0.65, "low_tariff_frac": {0.0: 0.4, 10.0: 0.5, 20.0: 0.6}},
            {"name": "Linear Increase", "pattern": "increase", "base": 1.0, "amp": 3.0, "cap_kw": 80.0, "weights": {"T": 0.4, "Q": 0.3, "E": 0.2, "L": 0.1}, "quality_threshold": 0.70, "low_tariff_frac": {0.0: 0.35, 10.0: 0.45, 20.0: 0.55}},
            {"name": "Linear Decrease", "pattern": "decrease", "base": 4.0, "amp": -2.0, "cap_kw": 70.0, "weights": {"T": 0.3, "Q": 0.4, "E": 0.2, "L": 0.1}, "quality_threshold": 0.70, "low_tariff_frac": {0.0: 0.30, 10.0: 0.40, 20.0: 0.50}},
            {"name": "Alternating", "pattern": "alternate", "base": 2.0, "amp": 2.0, "cap_kw": 75.0, "weights": {"T": 0.25, "Q": 0.35, "E": 0.25, "L": 0.15}, "quality_threshold": 0.75, "low_tariff_frac": {0.0: 0.4, 10.0: 0.5, 20.0: 0.6}},
        ]

        for i, sc in enumerate(base_scenarios):
            prepared = self._prepare_scenario(sc)
            prompt_turn1 = self._create_turn_prompt(
                turn_index=1,
                total_turns=10,
                scenario=prepared,
                arrivals_history=prepared["arrivals"][:1],
            )
            task = Task(
                task_id=f"ev_charging_policy_{i+1}",
                name=f"EV Charging Policy: {sc['name']}",
                description="Select discrete policy parameters to maximize a weighted utility. Final answer at turn 10.",
                prompt=prompt_turn1,
                expected_output=None,
                metadata={
                    "weights": sc["weights"],
                    "quality_threshold": sc["quality_threshold"],
                    "objective_minmax": prepared["minmax"].__dict__,
                    "U_max": prepared["U_max"],
                    "U_min": prepared["U_min"],
                    "coeffs": self.COEFFS,
                    "cap_kw": sc["cap_kw"],
                    "horizon_min": prepared["horizon_min"],
                    "bucket_min": prepared["bucket_min"],
                    "arrivals": prepared["arrivals"],
                    "target_kwh_per_vehicle": prepared["target_kwh_per_vehicle"],
                    "low_tariff_fraction_by_shift": prepared["low_tariff_fraction_by_shift"],
                    "discrete_values": DISCRETE_POLICY,
                    "total_turns": 10,
                },
            )
            tasks.append(task)
        return tasks

    def _prepare_scenario(self, sc: Dict[str, Any]) -> Dict[str, Any]:
        pattern = sc["pattern"]
        base = float(sc["base"])  # baseline arrivals per bucket
        amp = float(sc["amp"])    # amplitude; may be negative for decrease
        cap_kw = float(sc["cap_kw"])  # station cap
        weights = sc["weights"]
        qt = float(sc["quality_threshold"])
        low_tariff_fraction_by_shift: Dict[float, float] = sc["low_tariff_frac"]

        horizon_min = 100
        bucket_min = 10
        buckets = horizon_min // bucket_min
        arrivals = _generate_arrival_pattern(pattern, base, amp, buckets)
        # Set target energy per vehicle to ensure a non-empty feasible set across discrete policies
        target_kwh_per_vehicle = 6.0

        # Validate low_tariff_fraction_by_shift mapping covers all allowed shifts
        expected_shifts = set(DISCRETE_POLICY["slot_shift_min"]) 
        provided_shifts = set(low_tariff_fraction_by_shift.keys())
        if not expected_shifts.issubset(provided_shifts):
            missing = expected_shifts - provided_shifts
            raise ValueError(
                f"low_tariff_fraction_by_shift is missing keys for shifts: {sorted(missing)}"
            )

        # First pass: find per-objective min/max over feasible policy grid
        beta = self.COEFFS["beta"]
        gamma = self.COEFFS["gamma"]
        h_wear = self.COEFFS["h_wear"]

        T_min = math.inf
        T_max = -math.inf
        Q_min = math.inf
        Q_max = -math.inf
        E_min = math.inf
        E_max = -math.inf
        L_min = math.inf
        L_max = -math.inf

        feasible_count = 0
        for p in DISCRETE_POLICY["policy_power_kW"]:
            for s in DISCRETE_POLICY["smoothing_factor"]:
                for c in DISCRETE_POLICY["cap_fraction"]:
                    for sh in DISCRETE_POLICY["slot_shift_min"]:
                        pol = Policy(p, s, c, sh)
                        T_raw, Q_raw, E_raw, L_raw = _compute_objectives_raw(
                            pol, arrivals, target_kwh_per_vehicle, cap_kw, horizon_min, low_tariff_fraction_by_shift, beta, gamma, h_wear
                        )
                        if Q_raw + 1e-12 < qt:
                            continue
                        feasible_count += 1
                        if T_raw < T_min:
                            T_min = T_raw
                        if T_raw > T_max:
                            T_max = T_raw
                        if Q_raw < Q_min:
                            Q_min = Q_raw
                        if Q_raw > Q_max:
                            Q_max = Q_raw
                        if E_raw < E_min:
                            E_min = E_raw
                        if E_raw > E_max:
                            E_max = E_raw
                        if L_raw < L_min:
                            L_min = L_raw
                        if L_raw > L_max:
                            L_max = L_raw

        if feasible_count == 0:
            raise ValueError(f"No feasible policies for scenario '{sc['name']}'. Adjust thresholds or parameters.")
        if feasible_count < 3:
            logger.warning(
                f"Very small feasible set ({feasible_count}) for scenario '{sc['name']}'. Consider relaxing quality_threshold or adjusting coefficients."
            )

        eps = 1e-12
        if (T_max - T_min) <= eps or (Q_max - Q_min) <= eps or (E_max - E_min) <= eps or (L_max - L_min) <= eps:
            raise ValueError(
                "Degenerate objective range. Adjust coefficients/thresholds so all objectives vary over feasible grid."
            )

        minmax = ObjectiveMinMax(T_min, T_max, Q_min, Q_max, E_min, E_max, L_min, L_max)

        # Second pass: compute U_min/U_max and argmax
        U_max = -math.inf
        U_min = math.inf
        best_policy: Optional[Policy] = None
        for p in DISCRETE_POLICY["policy_power_kW"]:
            for s in DISCRETE_POLICY["smoothing_factor"]:
                for c in DISCRETE_POLICY["cap_fraction"]:
                    for sh in DISCRETE_POLICY["slot_shift_min"]:
                        pol = Policy(p, s, c, sh)
                        T_raw, Q_raw, E_raw, L_raw = _compute_objectives_raw(
                            pol, arrivals, target_kwh_per_vehicle, cap_kw, horizon_min, low_tariff_fraction_by_shift, beta, gamma, h_wear
                        )
                        if Q_raw + 1e-12 < qt:
                            continue
                        U, _ = _weighted_utility(T_raw, Q_raw, E_raw, L_raw, minmax, weights)
                        if U > U_max:
                            U_max = U
                            best_policy = pol
                        if U < U_min:
                            U_min = U

        assert best_policy is not None and U_max != -math.inf

        return {
            "arrivals": arrivals,
            "target_kwh_per_vehicle": target_kwh_per_vehicle,
            "low_tariff_fraction_by_shift": low_tariff_fraction_by_shift,
            "minmax": minmax,
            "U_max": float(U_max),
            "U_min": float(U_min),
            "horizon_min": horizon_min,
            "bucket_min": bucket_min,
        }

    def _create_turn_prompt(
        self,
        turn_index: int,
        total_turns: int,
        scenario: Dict[str, Any],
        arrivals_history: List[int],
    ) -> str:
        weights = scenario.get("weights") or {}
        minmax: ObjectiveMinMax = scenario["minmax"]
        coeffs = self.COEFFS
        horizon_min = scenario["horizon_min"]
        bucket_min = scenario["bucket_min"]
        low_tariff_fraction_by_shift = scenario["low_tariff_fraction_by_shift"]

        def fmt_list(values: List[float]) -> str:
            return ", ".join(f"{v:g}" for v in values)

        parts: List[str] = []
        parts.append("EV Charging Policy Optimization (Multi-Turn)\n\n")
        parts.append(f"Turn {turn_index}/{total_turns}\n\n")
        parts.append("Your goal is to MAXIMIZE a weighted utility of four normalized objectives: Throughput (T), Service Quality (Q), Cost Efficiency (E), Longevity (L).\n")
        parts.append("Final answer is required only on the last turn (Turn 10).\n\n")

        parts.append("Objective Weights (sum to 1.0):\n")
        parts.append(f"- T: {weights.get('T', 0.0)}\n")
        parts.append(f"- Q: {weights.get('Q', 0.0)}\n")
        parts.append(f"- E: {weights.get('E', 0.0)}\n")
        parts.append(f"- L: {weights.get('L', 0.0)}\n\n")

        parts.append("Normalization and Utility (use these EXACT formulas):\n")
        parts.append("- For X ∈ {T,Q,E,L}: X_norm = (X_raw − X_min)/(X_max − X_min); if denominator = 0, X_norm = 0.5.\n")
        parts.append("- Utility = w_T*T_norm + w_Q*Q_norm + w_E*E_norm + w_L*L_norm.\n\n")

        parts.append("Per-objective min/max (precomputed over feasible policy grid for this scenario):\n")
        parts.append(
            f"  T_min={minmax.T_min:.6f}, T_max={minmax.T_max:.6f}; "
            f"Q_min={minmax.Q_min:.6f}, Q_max={minmax.Q_max:.6f}; "
            f"E_min={minmax.E_min:.6f}, E_max={minmax.E_max:.6f}; "
            f"L_min={minmax.L_min:.6f}, L_max={minmax.L_max:.6f}\n\n"
        )

        parts.append("Objective Equations and Constants:\n")
        parts.append("Let E_demand = sum(arrivals) * target_kWh_per_vehicle, E_cap = cap_fraction * Cap_kW * (H/60), D = E_demand / max(1e-9, E_cap).\n")
        parts.append("- T_raw = E_cap * min(1, D) * (0.8 + 0.2 * smoothing_factor)\n")
        parts.append(
            f"- Q_raw = clamp(0.6 + β*smoothing_factor − γ*max(0, D−1), 0, 1), with β={coeffs['beta']:.3f}, γ={coeffs['gamma']:.3f}\n"
        )
        parts.append("- E_raw = low_tariff_fraction_by_shift[slot_shift_min] (share of energy in low-tariff windows)\n")
        parts.append(
            f"- L_raw = 1 / ((policy_power_kW/30)^1.2 * (cap_fraction)^1.1 / (1 + h*smoothing_factor)), with h={coeffs['h_wear']:.3f}\n\n"
        )

        parts.append("Discrete Allowed Values:\n")
        parts.append(f"- policy_power_kW: {{{fmt_list(DISCRETE_POLICY['policy_power_kW'])}}}\n")
        parts.append(f"- smoothing_factor: {{{fmt_list(DISCRETE_POLICY['smoothing_factor'])}}}\n")
        parts.append(f"- cap_fraction: {{{fmt_list(DISCRETE_POLICY['cap_fraction'])}}}\n")
        parts.append(f"- slot_shift_min: {{{fmt_list(DISCRETE_POLICY['slot_shift_min'])}}}\n\n")

        parts.append("Scenario Constants:\n")
        parts.append(f"- Horizon H (minutes): {horizon_min}\n")
        parts.append(f"- Bucket size (minutes): {bucket_min}\n")
        parts.append(f"- target_kWh_per_vehicle: {scenario.get('target_kwh_per_vehicle', 6.0)}\n")
        parts.append("- low_tariff_fraction_by_shift (mapping provided)\n\n")

        parts.append("Arrival History (per bucket):\n")
        parts.append(", ".join(str(x) for x in arrivals_history) + "\n\n")

        parts.append("Output Requirements (Final Turn Only):\n")
        parts.append("- First, write your reasoning and calculations.\n")
        parts.append("- Then, output EXACTLY one line: [FINAL ANSWER: {\"policy_power_kW\":P, \"smoothing_factor\":S, \"cap_fraction\":C, \"slot_shift_min\":SH}]\n")

        return "".join(parts)

    async def evaluate_task(self, task: Task, model: BaseModel) -> TaskResult:
        delays_applied = 0
        total_tokens = 0
        accumulated_call_time = 0.0

        meta = task.metadata
        total_turns = int(meta.get("total_turns", 10))
        arrivals_full: List[int] = list(meta.get("arrivals", []))

        wait_seconds = float(self.config.additional_params.get("wait_seconds", 15.0)) if getattr(self, "config", None) else 15.0

        model_response: Optional[ModelResponse] = None
        last_prompt = task.prompt

        try:
            for turn in range(1, total_turns + 1):
                call_started = time.time()
                response = await model.generate(last_prompt)
                accumulated_call_time += (time.time() - call_started)
                model_response = response
                if response.completion_tokens:
                    total_tokens += response.completion_tokens

                if turn < total_turns:
                    # Build next-turn prompt with extended history
                    history = arrivals_full[: turn + 1]
                    last_prompt = self._create_turn_prompt(turn + 1, total_turns, {
                        **meta,
                        "minmax": ObjectiveMinMax(
                            meta["objective_minmax"]["T_min"],
                            meta["objective_minmax"]["T_max"],
                            meta["objective_minmax"]["Q_min"],
                            meta["objective_minmax"]["Q_max"],
                            meta["objective_minmax"]["E_min"],
                            meta["objective_minmax"]["E_max"],
                            meta["objective_minmax"]["L_min"],
                            meta["objective_minmax"]["L_max"],
                        ),
                    }, history)
                    if wait_seconds > 0:
                        await asyncio.sleep(wait_seconds)
                        delays_applied += 1
                else:
                    # Final turn: parse and score
                    break

            # Parse policy from final response
            parsed, diagnostics = self._parse_final_policy(model_response.text if model_response else "")
            if parsed is None:
                return self._result(task, False, 0.0, accumulated_call_time, {
                    "output_tokens": total_tokens,
                    "parse_diagnostics": diagnostics,
                }, model_response)

            # Solver scoring
            weights = meta["weights"]
            qt = float(meta["quality_threshold"])
            minmax_dict = meta["objective_minmax"]
            minmax = ObjectiveMinMax(
                T_min=minmax_dict["T_min"], T_max=minmax_dict["T_max"],
                Q_min=minmax_dict["Q_min"], Q_max=minmax_dict["Q_max"],
                E_min=minmax_dict["E_min"], E_max=minmax_dict["E_max"],
                L_min=minmax_dict["L_min"], L_max=minmax_dict["L_max"],
            )
            beta = self.COEFFS["beta"]
            gamma = self.COEFFS["gamma"]
            h_wear = self.COEFFS["h_wear"]

            T_raw, Q_raw, E_raw, L_raw = _compute_objectives_raw(
                parsed,
                arrivals_full,
                float(meta["target_kwh_per_vehicle"]),
                float(meta["cap_kw"]),
                int(meta["horizon_min"]),
                {float(k): float(v) for k, v in meta["low_tariff_fraction_by_shift"].items()},
                beta,
                gamma,
                h_wear,
            )

            is_valid = Q_raw + 1e-12 >= qt
            score = 0.0
            if is_valid:
                U_max = float(meta["U_max"])  # from solver
                U_min = float(meta["U_min"])  # from solver
                U, comps = _weighted_utility(T_raw, Q_raw, E_raw, L_raw, minmax, weights)
                denom = max(1e-12, U_max - U_min)
                score = max(0.0, (U - U_min)) / denom
            else:
                comps = {"T": 0.0, "Q": 0.0, "E": 0.0, "L": 0.0}

            metrics = {
                "output_tokens": total_tokens,
                "components": comps,
                "T_raw": T_raw,
                "Q_raw": Q_raw,
                "E_raw": E_raw,
                "L_raw": L_raw,
                "feasibility": is_valid,
            }

            return self._result(task, is_valid and score >= 0.95, max(0.0, min(1.0, score)), accumulated_call_time, metrics, model_response)

        except Exception as e:
            logger.error(f"Error evaluating task {task.task_id}: {e}")
            return self._result(task, False, 0.0, accumulated_call_time, {"error": str(e), "output_tokens": total_tokens}, model_response)

    def _parse_final_policy(self, response_text: str) -> Tuple[Optional[Policy], Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {}
        # Primary: [FINAL ANSWER: {...}] JSON
        try:
            matches = list(re.finditer(r"\[\s*FINAL\s+ANSWER\s*:\s*(\{.*?\})\s*\]", response_text, re.IGNORECASE | re.DOTALL))
            if matches:
                data = json.loads(matches[-1].group(1))
                pol = _policy_membership(data)
                if pol is not None:
                    diagnostics["parse_strategy"] = "final_answer_json"
                    return pol, diagnostics
        except Exception as e:
            diagnostics["final_answer_json_error"] = str(e)

        # Secondary: any JSON object containing required keys
        try:
            json_candidates = list(re.finditer(r"(\{[^{}]*\})", response_text, re.DOTALL))
            for m in reversed(json_candidates):
                try:
                    data = json.loads(m.group(1))
                except Exception:
                    continue
                pol = _policy_membership(data)
                if pol is not None:
                    diagnostics["parse_strategy"] = "raw_json_anywhere"
                    return pol, diagnostics
        except Exception as e:
            diagnostics["raw_json_error"] = str(e)

        diagnostics["parse_strategy"] = "failed"
        return None, diagnostics

    def _result(
        self,
        task: Task,
        success: bool,
        score: float,
        exec_time: float,
        metrics: Dict[str, Any],
        model_response: Optional[ModelResponse],
    ) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            task_name=task.name,
            agent_type=self.agent_type,
            success=success,
            score=score,
            metrics=metrics,
            model_response=model_response,
            execution_time=exec_time,
        )

    def calculate_score(self, task: Task, model_response: ModelResponse) -> float:
        # Scoring handled in evaluate_task
        return 0.0


