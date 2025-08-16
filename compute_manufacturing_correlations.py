#!/usr/bin/env python3
"""
Compute Pearson correlations among normalized objectives (T, Q, E, L)
for each scenario in the manufacturing_line_optimization benchmark.

Normalization ranges are taken from the scenario metadata so results
align with evaluation.

Usage:
  source .venv/bin/activate
  python3 compute_manufacturing_correlations.py
"""

import sys
from pathlib import Path
import numpy as np

# Ensure src is on path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.benchmark.base import AgentType, BenchmarkConfig
from src.benchmarks.manufacturing_optimization import (
    ManufacturingOptimizationBenchmark,
    ObjectiveMinMax,
    DISCRETE_PARAMETERS,
    _compute_objectives_raw,
    _weighted_utility,
)


def corrcoef_safe(matrix: np.ndarray) -> np.ndarray:
    """Compute correlation with guard for near-constant columns.

    matrix: shape (n_samples, n_vars)
    Returns: (n_vars, n_vars) matrix with np.nan where std==0.
    """
    std = matrix.std(axis=0, ddof=0)
    # Avoid division by zero: mark zeros and compute corr on standardized data
    safe_std = std.copy()
    safe_std[safe_std == 0] = np.nan
    z = (matrix - matrix.mean(axis=0)) / safe_std
    return np.corrcoef(z.T, rowvar=True)


def main() -> int:
    config = BenchmarkConfig(
        benchmark_name="manufacturing_line_optimization",
        agent_type=AgentType.UTILITY_BASED,
    )
    bench = ManufacturingOptimizationBenchmark(config)
    tasks = bench.get_tasks()

    speeds = DISCRETE_PARAMETERS["line_speed"]
    qcs = DISCRETE_PARAMETERS["qc_strictness"]
    powers = DISCRETE_PARAMETERS["power_setting"]
    maints = DISCRETE_PARAMETERS["maintenance_freq"]

    coeffs = bench.COEFFS
    alpha = coeffs["alpha"]
    beta = coeffs["beta"]
    gamma = coeffs["gamma"]
    k_energy = coeffs["k_energy"]
    h_wear = coeffs["h_wear"]

    for task in tasks:
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

        rows = []  # each row: [T_norm, Q_norm, E_norm, L_norm]
        feasible = 0
        for speed in speeds:
            for qc in qcs:
                for power in powers:
                    for m in maints:
                        raw = _compute_objectives_raw(
                            type("P", (), {"speed": speed, "qc": qc, "power": power, "maintenance": m})(),
                            alpha,
                            beta,
                            gamma,
                            k_energy,
                            h_wear,
                        )
                        if raw.quality + 1e-12 < q_threshold:
                            continue
                        feasible += 1
                        _, comps = _weighted_utility(raw, minmax, weights)
                        rows.append([comps["T"], comps["Q"], comps["E"], comps["L"]])

        print(f"\n{task.name} — feasible points: {feasible}")
        if feasible == 0:
            print("  No feasible points. Correlation: N/A")
            continue

        X = np.array(rows, dtype=float)
        stds = X.std(axis=0, ddof=0)
        labels = ["T", "Q", "E", "L"]
        print("  Std (T,Q,E,L):", ", ".join(f"{s:.4f}" for s in stds))

        C = corrcoef_safe(X)
        # Print compact upper triangle
        for i in range(len(labels)):
            row = []
            for j in range(len(labels)):
                v = C[i, j]
                row.append("nan" if np.isnan(v) else f"{v:.3f}")
            print("  ", labels[i], ":", ", ".join(row))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


