#!/usr/bin/env python3
"""
Heatmap sanity checks for the Manufacturing Line Optimization benchmark.

For each scenario, plots 2x2 heatmaps (T, Q, E, L normalized) over the grid of
speed vs qc_strictness, aggregating over power_setting and maintenance_freq.

This quickly reveals whether any normalized objective collapses into a narrow band.

Usage:
  source .venv/bin/activate
  python3 plot_manufacturing_sanity.py [--aggregate mean|max] [--outdir results/sanity_manufacturing]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Ensure src on path
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


def slugify(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")


def main(aggregate: str, outdir: str) -> int:
    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = BenchmarkConfig(
        benchmark_name="manufacturing_line_optimization",
        agent_type=AgentType.UTILITY_BASED,
    )
    bench = ManufacturingOptimizationBenchmark(config)
    tasks = bench.get_tasks()
    coeffs = bench.COEFFS

    speeds = DISCRETE_PARAMETERS["line_speed"]
    qcs = DISCRETE_PARAMETERS["qc_strictness"]
    powers = DISCRETE_PARAMETERS["power_setting"]
    maints = DISCRETE_PARAMETERS["maintenance_freq"]

    alpha = coeffs["alpha"]
    beta = coeffs["beta"]
    gamma = coeffs["gamma"]
    k_energy = coeffs["k_energy"]
    h_wear = coeffs["h_wear"]

    agg_func = np.mean if aggregate == "mean" else np.max

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

        # Prepare arrays: y=qc index, x=speed index
        shape = (len(qcs), len(speeds))
        heat_T = np.full(shape, np.nan, dtype=float)
        heat_Q = np.full(shape, np.nan, dtype=float)
        heat_E = np.full(shape, np.nan, dtype=float)
        heat_L = np.full(shape, np.nan, dtype=float)

        for ix_s, speed in enumerate(speeds):
            for ix_q, qc in enumerate(qcs):
                # Feasibility depends only on speed and qc
                raw0 = _compute_objectives_raw(
                    params=type("P", (), {"speed": speed, "qc": qc, "power": powers[0], "maintenance": maints[0]})(),
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    k_energy=k_energy,
                    h_wear=h_wear,
                )
                if raw0.quality + 1e-12 < q_threshold:
                    continue  # leave NaNs

                # T and Q don't depend on power/maintenance; compute once
                _, comps_TQ = _weighted_utility(raw0, minmax, weights)
                t_norm = comps_TQ["T"]
                q_norm = comps_TQ["Q"]

                # E and L depend on power/maintenance; aggregate across them
                e_norm_vals = []
                l_norm_vals = []
                for power in powers:
                    for m in maints:
                        raw = _compute_objectives_raw(
                            params=type("P", (), {"speed": speed, "qc": qc, "power": power, "maintenance": m})(),
                            alpha=alpha,
                            beta=beta,
                            gamma=gamma,
                            k_energy=k_energy,
                            h_wear=h_wear,
                        )
                        _, comps = _weighted_utility(raw, minmax, weights)
                        e_norm_vals.append(comps["E"])
                        l_norm_vals.append(comps["L"])

                heat_T[ix_q, ix_s] = t_norm
                heat_Q[ix_q, ix_s] = q_norm
                heat_E[ix_q, ix_s] = agg_func(e_norm_vals)
                heat_L[ix_q, ix_s] = agg_func(l_norm_vals)

        # Plot 2x2 heatmaps
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle(f"{task.name} — normalized objectives (aggregate: {aggregate})")

        def plot(ax, data, title):
            im = ax.imshow(data, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
            ax.set_title(title)
            ax.set_xlabel("speed")
            ax.set_ylabel("qc")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Tick labels (sparse for readability)
            ax.set_xticks(np.linspace(0, len(speeds) - 1, num=6))
            ax.set_xticklabels([f"{speeds[int(i)]:g}" for i in np.linspace(0, len(speeds) - 1, num=6)])
            ax.set_yticks(np.linspace(0, len(qcs) - 1, num=6))
            ax.set_yticklabels([f"{qcs[int(i)]:g}" for i in np.linspace(0, len(qcs) - 1, num=6)])

        plot(axs[0, 0], heat_T, "Throughput (T)")
        plot(axs[0, 1], heat_Q, "Quality (Q)")
        plot(axs[1, 0], heat_E, "Energy Efficiency (E)")
        plot(axs[1, 1], heat_L, "Longevity (L)")

        scenario_dir = out_dir / slugify(task.name or task.task_id)
        scenario_dir.mkdir(parents=True, exist_ok=True)
        out_path = scenario_dir / "heatmap_speed_vs_qc.png"
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"Saved heatmaps: {out_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot heatmap sanity checks for manufacturing_line_optimization.")
    parser.add_argument("--aggregate", choices=["mean", "max"], default="mean", help="Aggregate across power/maintenance (default: mean)")
    parser.add_argument("--outdir", type=str, default="results/sanity_manufacturing", help="Output directory for plots")
    args = parser.parse_args()
    sys.exit(main(args.aggregate, args.outdir))


