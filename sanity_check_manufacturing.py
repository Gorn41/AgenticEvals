#!/usr/bin/env python3
"""
Sanity checks for the Manufacturing Line Optimization benchmark.

Runs a one-time grid pass per scenario to:
  - Verify there is at least one feasible configuration (quality >= threshold)
  - Compute dispersion (min/max/mean/std) of normalized objectives T/Q/E/L
    across the feasible grid, and flag if variance collapses.

Usage:
  python3 sanity_check_manufacturing.py [--std-threshold 0.02]

Note: Activate your virtual environment first.
  source .venv/bin/activate
"""

import argparse
import statistics
import sys
from pathlib import Path

# Ensure src is on path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.benchmark.base import AgentType, BenchmarkConfig
from src.benchmarks.manufacturing_optimization import (
    ManufacturingOptimizationBenchmark,
    ObjectiveMinMax,
    _compute_objectives_raw,
    _iter_parameter_grid,
    _weighted_utility,
)


def main(std_threshold: float) -> int:
    config = BenchmarkConfig(
        benchmark_name="manufacturing_line_optimization",
        agent_type=AgentType.UTILITY_BASED,
    )
    bench = ManufacturingOptimizationBenchmark(config)

    print("Running sanity checks for manufacturing_line_optimization...")
    tasks = bench.get_tasks()
    coeffs = bench.COEFFS

    any_collapse = False
    for i, task in enumerate(tasks, start=1):
        name = task.name or task.task_id
        md = task.metadata
        weights = md["weights"]
        q_threshold = float(md["quality_threshold"])
        mm = md["objective_minmax"]
        minmax = ObjectiveMinMax(
            throughput_min=mm["throughput_min"],
            throughput_max=mm["throughput_max"],
            quality_min=mm["quality_min"],
            quality_max=mm["quality_max"],
            efficiency_min=mm["efficiency_min"],
            efficiency_max=mm["efficiency_max"],
            longevity_min=mm["longevity_min"],
            longevity_max=mm["longevity_max"],
        )

        alpha = coeffs["alpha"]
        beta = coeffs["beta"]
        gamma = coeffs["gamma"]
        k_energy = coeffs["k_energy"]
        h_wear = coeffs["h_wear"]

        # Collect normalized components across feasible grid
        t_vals = []
        q_vals = []
        e_vals = []
        l_vals = []
        feasible_count = 0
        for p in _iter_parameter_grid():
            raw = _compute_objectives_raw(p, alpha, beta, gamma, k_energy, h_wear)
            if raw.quality + 1e-12 < q_threshold:
                continue
            feasible_count += 1
            # Get normalized components
            _, comps = _weighted_utility(raw, minmax, weights)
            t_vals.append(comps["T"])  # Already normalized to [0,1]
            q_vals.append(comps["Q"])  # Already normalized to [0,1]
            e_vals.append(comps["E"])  # Already normalized to [0,1]
            l_vals.append(comps["L"])  # Already normalized to [0,1]

        print(f"\nScenario {i}: {name}")
        print(f"  Feasible points: {feasible_count}")
        if feasible_count == 0:
            print("  [ERROR] No feasible points. Consider adjusting coefficients or thresholds.")
            any_collapse = True
            continue

        def stats(xs):
            return (
                min(xs),
                max(xs),
                (sum(xs) / len(xs)) if xs else 0.0,
                (statistics.pstdev(xs) if len(xs) > 1 else 0.0),
            )

        t_min, t_max, t_mean, t_std = stats(t_vals)
        q_min, q_max, q_mean, q_std = stats(q_vals)
        e_min, e_max, e_mean, e_std = stats(e_vals)
        l_min, l_max, l_mean, l_std = stats(l_vals)

        print("  Normalized objective dispersion (min, max, mean, std):")
        print(f"    T: ({t_min:.3f}, {t_max:.3f}, {t_mean:.3f}, {t_std:.3f})")
        print(f"    Q: ({q_min:.3f}, {q_max:.3f}, {q_mean:.3f}, {q_std:.3f})")
        print(f"    E: ({e_min:.3f}, {e_max:.3f}, {e_mean:.3f}, {e_std:.3f})")
        print(f"    L: ({l_min:.3f}, {l_max:.3f}, {l_mean:.3f}, {l_std:.3f})")

        collapsed = []
        if t_std < std_threshold:
            collapsed.append("T")
        if q_std < std_threshold:
            collapsed.append("Q")
        if e_std < std_threshold:
            collapsed.append("E")
        if l_std < std_threshold:
            collapsed.append("L")

        if collapsed:
            any_collapse = True
            joined = ", ".join(collapsed)
            print(f"  [WARN] Low variance detected (std < {std_threshold}): {joined}")

    print("\nSanity checks complete.")
    if any_collapse:
        print("One or more scenarios show collapsed dispersion or infeasibility. Consider tuning coefficients alpha/beta/gamma/k_energy/h_wear and re-run.")
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sanity checks for manufacturing_line_optimization.")
    parser.add_argument("--std-threshold", type=float, default=0.02, help="Threshold for low variance warning (default: 0.02)")
    args = parser.parse_args()
    sys.exit(main(args.std_threshold))


