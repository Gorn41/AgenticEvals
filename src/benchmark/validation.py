"""
Validation and cross-validation utilities for AgenticEvals.

Supports:
- Within-agent-type leave-one-benchmark-out (LOO) prediction of benchmark scores
- Cross-agent-type LOO using global averages
- External validation benchmarks mapped to one or more agent types
"""

from __future__ import annotations

from typing import Dict, List, Any, Tuple
import math


def _safe_mean(values: List[float]) -> float:
    usable = [v for v in values if v is not None]
    return sum(usable) / len(usable) if usable else 0.0


def _compute_error_metrics(pairs: List[Tuple[float, float]]) -> Dict[str, float]:
    if not pairs:
        return {"mae": 0.0, "rmse": 0.0, "r2": 0.0}

    errors = [pred - true for true, pred in pairs]
    mae = _safe_mean([abs(e) for e in errors])
    rmse = math.sqrt(_safe_mean([e * e for e in errors]))

    # R^2 using total sum of squares; note for constant predictors R^2 may be <= 0
    truths = [t for (t, _) in pairs]
    truth_mean = _safe_mean(truths)
    ss_tot = sum((t - truth_mean) ** 2 for t in truths)
    ss_res = sum((t - p) ** 2 for (t, p) in pairs)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"mae": mae, "rmse": rmse, "r2": r2}


def within_type_loo(all_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    For each agent type, predict each benchmark's average_score using the mean
    of the remaining benchmarks in that agent type, and compute error metrics.
    Returns a dict keyed by agent type.
    """
    # Group by agent type
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for s in all_summaries:
        by_type.setdefault(s["agent_type"], []).append(s)

    results: Dict[str, Any] = {}
    for agent_type, summaries in by_type.items():
        items: List[Dict[str, Any]] = []
        pairs: List[Tuple[float, float]] = []
        for i, s in enumerate(summaries):
            true_score = float(s.get("average_score", 0.0))
            others = [float(x.get("average_score", 0.0)) for j, x in enumerate(summaries) if j != i]
            pred = _safe_mean(others)
            items.append({
                "benchmark": s.get("benchmark_name"),
                "true": true_score,
                "pred": pred,
                "error": pred - true_score,
            })
            pairs.append((true_score, pred))

        metrics = _compute_error_metrics(pairs)
        results[agent_type] = {
            "items": items,
            "metrics": metrics,
        }

    return results


def cross_type_loo(all_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Leave-one-agent-type-out: for each agent type, predict its benchmarks using
    the global mean score across all other agent types. Compute error metrics.
    """
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for s in all_summaries:
        by_type.setdefault(s["agent_type"], []).append(s)

    results: Dict[str, Any] = {}
    # Precompute global pool
    for withheld_type in by_type.keys():
        withheld = by_type[withheld_type]
        others = [x for t, lst in by_type.items() if t != withheld_type for x in lst]
        global_pred = _safe_mean([float(x.get("average_score", 0.0)) for x in others])
        items: List[Dict[str, Any]] = []
        pairs: List[Tuple[float, float]] = []
        for s in withheld:
            true_score = float(s.get("average_score", 0.0))
            items.append({
                "benchmark": s.get("benchmark_name"),
                "true": true_score,
                "pred": global_pred,
                "error": global_pred - true_score,
            })
            pairs.append((true_score, global_pred))

        metrics = _compute_error_metrics(pairs)
        results[withheld_type] = {
            "items": items,
            "metrics": metrics,
        }

    return results


def evaluate_external_validation(
    validation_specs: List[Dict[str, Any]],
    agent_type_aggregated: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Produce prediction proxies for external validation benchmarks based on
    equal-weight averages of listed agent types' mean scores.

    Each spec: {
      name: str,
      agent_types: List[str]
    }

    Returns items with {name, agent_types, pred}.
    """
    items: List[Dict[str, Any]] = []

    for spec in validation_specs:
        types = spec.get("agent_types") or []
        means: List[float] = []
        for t in types:
            agg = agent_type_aggregated.get(t)
            means.append(float(agg.get("mean_score", 0.0)) if agg else 0.0)
        pred = _safe_mean(means)
        items.append({
            "name": spec.get("name"),
            "agent_types": types,
            "pred": pred,
        })

    return {"items": items}


