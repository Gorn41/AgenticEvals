import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import matplotlib.pyplot as plt
import numpy as np


AGENT_TYPE_ORDER: List[str] = [
    "simple_reflex",
    "model_based_reflex",
    "goal_based",
    "utility_based",
    "learning",
]

DESIRED_MODEL_ORDER: List[str] = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemma-3-4b-it",
    "gemma-3-1b-it",
    "gemma-3n-e4b-it",
]


@dataclass
class AgentTypeMetrics:
    score: Optional[float]
    time: Optional[float]
    tokens: Optional[float]


def find_agent_type_csv(model_dir: Path) -> Optional[Path]:
    for p in model_dir.glob("agent_type_results_*.csv"):
        if p.is_file():
            return p
    return None


def read_agent_type_csv(csv_path: Path) -> Dict[str, AgentTypeMetrics]:
    header_idx: Dict[str, int] = {}
    data: Dict[str, AgentTypeMetrics] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return data
        header_lower = [h.strip().lower() for h in header]

        def idx(name: str) -> int:
            try:
                return header_lower.index(name)
            except ValueError:
                return -1

        idx_agent = idx("agent type")
        idx_score = idx("average score")
        if idx_score == -1:
            idx_score = idx("weighted average score")
        idx_time = idx("average execution time")
        idx_tokens = idx("average output tokens")

        for row in reader:
            if not row:
                continue
            if max(idx_agent, idx_score, idx_time, idx_tokens) >= len(row):
                continue
            agent_type = row[idx_agent].strip()
            try:
                score_val = float(row[idx_score]) if row[idx_score] != "" else None
            except Exception:
                score_val = None
            try:
                time_val = float(row[idx_time]) if row[idx_time] != "" else None
            except Exception:
                time_val = None
            try:
                tokens_val = float(row[idx_tokens]) if row[idx_tokens] != "" else None
            except Exception:
                tokens_val = None
            data[agent_type] = AgentTypeMetrics(score=score_val, time=time_val, tokens=tokens_val)
    return data


def aggregate_results(results_dir: Path) -> Tuple[Dict[str, Dict[str, AgentTypeMetrics]], List[str]]:
    model_to_agent_metrics: Dict[str, Dict[str, AgentTypeMetrics]] = {}
    model_names: List[str] = []
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        csv_path = find_agent_type_csv(sub)
        if not csv_path:
            continue
        model_name = sub.name
        model_names.append(model_name)
        model_to_agent_metrics[model_name] = read_agent_type_csv(csv_path)
    return model_to_agent_metrics, model_names


def build_color_map(model_names: Iterable[str]) -> Dict[str, str]:
    family_to_palette: Dict[str, List[str]] = {
        "gemini": ["#1f77b4", "#2a9df4", "#174a7e", "#4aa3df", "#6bb1e0", "#98c8eb"],
        "gemma": ["#ff7f0e", "#ff9e4a", "#e07b00", "#cc6e00", "#f2a65a", "#d97a00"],
        "other": ["#2ca02c", "#66c266", "#1e7f1e", "#3aa63a", "#4db84d", "#7fd27f"],
    }

    def family_of(model: str) -> str:
        if model.startswith("gemini"):
            return "gemini"
        if model.startswith("gemma"):
            return "gemma"
        return "other"

    colors: Dict[str, str] = {}
    by_family: Dict[str, List[str]] = defaultdict(list)
    for m in model_names:
        by_family[family_of(m)].append(m)

    for family, members in by_family.items():
        palette = family_to_palette.get(family, family_to_palette["other"])
        members_sorted = sorted(members)
        for i, m in enumerate(members_sorted):
            colors[m] = palette[i % len(palette)]
    return colors


def compute_dense_ranks(values: List[Tuple[str, float]], higher_is_better: bool) -> Dict[str, int]:
    filtered = [(m, v) for m, v in values if v is not None and not np.isnan(v)]
    if not filtered:
        return {}
    reverse = higher_is_better
    sorted_pairs = sorted(filtered, key=lambda x: x[1], reverse=reverse)
    ranks: Dict[str, int] = {}
    rank = 0
    last_val: Optional[float] = None
    for m, v in sorted_pairs:
        if last_val is None or v != last_val:
            rank += 1
            last_val = v
        ranks[m] = rank
    return ranks


def ensure_output_dirs(base_output: Path):
    (base_output / "bars").mkdir(parents=True, exist_ok=True)
    (base_output / "heatmaps").mkdir(parents=True, exist_ok=True)
    (base_output / "bumps").mkdir(parents=True, exist_ok=True)
    (base_output / "overall").mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, out_path_base: Path):
    png_path = out_path_base.with_suffix(".png")
    svg_path = out_path_base.with_suffix(".svg")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def pretty_metric_name(metric_key: str) -> str:
    if metric_key == "score":
        return "Average Score"
    if metric_key == "time":
        return "Average Execution Time"
    return "Average Output Tokens"


def compute_nice_step_and_upper(max_val: float, min_extra_increments: int = 2, target_ticks: int = 5) -> Tuple[float, float]:
    if not np.isfinite(max_val) or max_val <= 0:
        return 1.0, 1.0
    raw_step = max_val / max(1, target_ticks)
    exponent = np.floor(np.log10(raw_step))
    base = 10 ** exponent
    step_candidates = [1.0, 2.0, 5.0, 10.0]
    step = step_candidates[-1] * base
    for mult in step_candidates:
        candidate = mult * base
        if raw_step <= candidate:
            step = candidate
            break
    upper = (np.ceil(max_val / step) + min_extra_increments) * step
    return float(step), float(upper)


def _sorted_values_for_agent(
    agent_type: str,
    metric_key: str,
    model_to_agent_metrics: Dict[str, Dict[str, AgentTypeMetrics]],
) -> Tuple[List[str], List[float]]:
    values: List[Tuple[str, Optional[float]]] = []
    for model, per_agent in model_to_agent_metrics.items():
        metrics = per_agent.get(agent_type)
        if not metrics:
            continue
        if metric_key == "score":
            values.append((model, metrics.score))
        elif metric_key == "time":
            values.append((model, metrics.time))
        else:
            values.append((model, metrics.tokens))
    higher_is_better = metric_key == "score"
    filtered = [(m, v) for m, v in values if v is not None]
    if not filtered:
        return [], []
    sorted_vals = sorted(filtered, key=lambda x: x[1], reverse=higher_is_better)
    models_sorted = [m for m, _ in sorted_vals]
    metric_values = [float(v) for _, v in sorted_vals]
    return models_sorted, metric_values


def plot_agent_type_triptych(
    agent_type: str,
    model_to_agent_metrics: Dict[str, Dict[str, AgentTypeMetrics]],
    colors: Dict[str, str],
    out_dir: Path,
):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["score", "time", "tokens"]
    for i, metric_key in enumerate(metrics):
        models_sorted, metric_values = _sorted_values_for_agent(agent_type, metric_key, model_to_agent_metrics)
        ax = axs[i]
        if not models_sorted:
            ax.set_visible(False)
            continue
        y_pos = np.arange(len(models_sorted))
        bar_colors = [colors.get(m, "#999999") for m in models_sorted]
        ax.barh(y_pos, metric_values, color=bar_colors)
        ax.set_yticks(y_pos, labels=models_sorted)
        ax.invert_yaxis()

        pretty = pretty_metric_name(metric_key)

        if metric_key == "score":
            ax.set_xlim(0, 1.4)
            xticks = np.arange(0.0, 1.01, 0.2)
            ax.set_xticks(xticks)
            ax.set_xlabel(pretty)
        elif metric_key == "time":
            max_val = max(metric_values) if metric_values else 1.0
            _, upper = compute_nice_step_and_upper(max_val, min_extra_increments=2, target_ticks=5)
            ax.set_xlim(0, upper)
            ax.set_xlabel(f"{pretty} (s)")
        else:
            max_val = max(metric_values) if metric_values else 1.0
            _, upper = compute_nice_step_and_upper(max_val, min_extra_increments=2, target_ticks=5)
            ax.set_xlim(0, upper)
            ax.set_xlabel(pretty)

        for j, v in enumerate(metric_values):
            ax.text(v + (0.01 if metric_key == "score" else max(metric_values) * 0.01), j + 0.1, f"{v:.3g}")

        ax.set_title(f"{pretty} - {agent_type}")

    fig.suptitle(f"Ranked Performance by Agent Type - {agent_type}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    out_base = out_dir / "bars" / f"{agent_type}"
    save_fig(fig, out_base)


def plot_horizontal_ranked_bars(
    agent_type: str,
    metric_key: str,
    model_to_agent_metrics: Dict[str, Dict[str, AgentTypeMetrics]],
    colors: Dict[str, str],
    out_dir: Path,
):
    values: List[Tuple[str, Optional[float]]] = []
    for model, per_agent in model_to_agent_metrics.items():
        metrics = per_agent.get(agent_type)
        if not metrics:
            continue
        if metric_key == "score":
            values.append((model, metrics.score))
        elif metric_key == "time":
            values.append((model, metrics.time))
        elif metric_key == "tokens":
            values.append((model, metrics.tokens))

    if not values:
        return

    higher_is_better = metric_key == "score"
    filtered = [(m, v) for m, v in values if v is not None]
    if not filtered:
        return

    sorted_vals = sorted(
        filtered,
        key=lambda x: x[1],
        reverse=higher_is_better,
    )

    models_sorted = [m for m, _ in sorted_vals]
    metric_values = [float(v) for _, v in sorted_vals]
    bar_colors = [colors.get(m, "#999999") for m in models_sorted]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(models_sorted))
    ax.barh(y_pos, metric_values, color=bar_colors)
    ax.set_yticks(y_pos, labels=models_sorted)
    ax.invert_yaxis()  # best at top

    if metric_key == "score":
        ax.set_xlim(0, 1.4)
        xticks = np.arange(0.0, 1.01, 0.2)
        ax.set_xticks(xticks)
        ax.set_xlabel("Average Score")
    elif metric_key == "time":
        max_val = max(metric_values) if metric_values else 1.0
        _, upper = compute_nice_step_and_upper(max_val, min_extra_increments=2, target_ticks=5)
        ax.set_xlim(0, upper)
        ax.set_xlabel("Average Execution Time (s)")
    else:
        max_val = max(metric_values) if metric_values else 1.0
        _, upper = compute_nice_step_and_upper(max_val, min_extra_increments=2, target_ticks=5)
        ax.set_xlim(0, upper)
        ax.set_xlabel("Average Output Tokens")

    for i, v in enumerate(metric_values):
        ax.text(v + (0.01 if metric_key == "score" else max(metric_values) * 0.01), i + 0.1, f"{v:.3g}")

    title_metric = pretty_metric_name(metric_key)
    ax.set_title(f"{title_metric} - {agent_type}")

    out_base = out_dir / "bars" / f"{agent_type}_{metric_key}"
    save_fig(fig, out_base)


def plot_rank_heatmap(
    metric_key: str,
    model_to_agent_metrics: Dict[str, Dict[str, AgentTypeMetrics]],
    model_names: List[str],
    out_dir: Path,
):
    higher_is_better = metric_key == "score"
    models_sorted = [m for m in DESIRED_MODEL_ORDER if m in set(model_names)]
    if not models_sorted:
        models_sorted = sorted(model_names)
    num_models = len(models_sorted)
    rank_matrix = np.full((len(AGENT_TYPE_ORDER), num_models), np.nan)

    for r, agent_type in enumerate(AGENT_TYPE_ORDER):
        row_values: List[Tuple[str, Optional[float]]] = []
        for m in models_sorted:
            metrics = model_to_agent_metrics.get(m, {}).get(agent_type)
            if not metrics:
                row_values.append((m, None))
                continue
            v: Optional[float]
            if metric_key == "score":
                v = metrics.score
            elif metric_key == "time":
                v = metrics.time
            else:
                v = metrics.tokens
            row_values.append((m, v))
        dense = compute_dense_ranks([(m, v if v is not None else np.nan) for m, v in row_values], higher_is_better=higher_is_better)
        for c, m in enumerate(models_sorted):
            if m in dense:
                rank_matrix[r, c] = dense[m]

    vmax = np.nanmax(rank_matrix)
    if np.isnan(vmax):
        return

    cmap = plt.get_cmap("viridis_r", int(max(vmax, 2)))
    try:
        cmap = cmap.copy()
    except Exception:
        pass
    try:
        cmap.set_bad(color="#dddddd")
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(max(10, num_models * 0.8), 6))
    im = ax.imshow(rank_matrix, cmap=cmap, aspect="auto", vmin=1, vmax=vmax)

    ax.set_yticks(np.arange(len(AGENT_TYPE_ORDER)), labels=AGENT_TYPE_ORDER)
    ax.set_xticks(np.arange(num_models), labels=models_sorted, rotation=45, ha="right")
    ax.set_title(f"Rank Heatmap by {pretty_metric_name(metric_key)} (1 = best)")

    for r in range(rank_matrix.shape[0]):
        for c in range(rank_matrix.shape[1]):
            val = rank_matrix[r, c]
            text = "-" if np.isnan(val) else f"{int(val)}"
            ax.text(c, r, text, ha="center", va="center", color="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Rank (1 = best)")

    out_base = out_dir / "heatmaps" / f"heatmap_{metric_key}"
    save_fig(fig, out_base)


def plot_rank_heatmaps_triptych(
    model_to_agent_metrics: Dict[str, Dict[str, AgentTypeMetrics]],
    model_names: List[str],
    out_dir: Path,
):
    models_sorted = [m for m in DESIRED_MODEL_ORDER if m in set(model_names)]
    if not models_sorted:
        models_sorted = sorted(model_names)
    num_models = len(models_sorted)

    fig, axs = plt.subplots(1, 3, figsize=(max(18, num_models * 2.4), 6))
    fig.subplots_adjust(right=0.86)
    metrics = ["score", "time", "tokens"]

    last_im = None
    for i, metric_key in enumerate(metrics):
        higher_is_better = metric_key == "score"
        rank_matrix = np.full((len(AGENT_TYPE_ORDER), num_models), np.nan)
        for r, agent_type in enumerate(AGENT_TYPE_ORDER):
            row_values: List[Tuple[str, Optional[float]]] = []
            for m in models_sorted:
                metrics_obj = model_to_agent_metrics.get(m, {}).get(agent_type)
                if not metrics_obj:
                    row_values.append((m, None))
                    continue
                v: Optional[float]
                if metric_key == "score":
                    v = metrics_obj.score
                elif metric_key == "time":
                    v = metrics_obj.time
                else:
                    v = metrics_obj.tokens
                row_values.append((m, v))
            dense = compute_dense_ranks([(m, v if v is not None else np.nan) for m, v in row_values], higher_is_better=higher_is_better)
            for c, m in enumerate(models_sorted):
                if m in dense:
                    rank_matrix[r, c] = dense[m]

        ax = axs[i]
        cmap = plt.get_cmap("viridis_r", max(num_models, 2))
        try:
            cmap = cmap.copy()
        except Exception:
            pass
        try:
            cmap.set_bad(color="#dddddd")
        except Exception:
            pass
        # Normalize all to the same rank range 1..num_models
        im = ax.imshow(rank_matrix, cmap=cmap, aspect="auto", vmin=1, vmax=num_models)
        ax.set_yticks(np.arange(len(AGENT_TYPE_ORDER)), labels=AGENT_TYPE_ORDER)
        ax.set_xticks(np.arange(num_models), labels=models_sorted, rotation=45, ha="right")
        ax.set_title(f"Rank Heatmap by {pretty_metric_name(metric_key)} (1 = best)")
        for r in range(rank_matrix.shape[0]):
            for c in range(rank_matrix.shape[1]):
                val = rank_matrix[r, c]
                text = "-" if np.isnan(val) else f"{int(val)}"
                ax.text(c, r, text, ha="center", va="center", color="black")
        last_im = im

    fig.suptitle("Rank Heatmaps by Metric")
    plt.tight_layout(rect=[0, 0.03, 0.86, 0.95])
    if last_im is not None:
        # Dedicated colorbar axes to avoid overlapping the third subplot
        # [left, bottom, width, height] in figure coordinates
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("Rank (1 = best)")
    out_base = out_dir / "heatmaps" / "heatmaps"
    save_fig(fig, out_base)


def plot_bump_chart(
    metric_key: str,
    model_to_agent_metrics: Dict[str, Dict[str, AgentTypeMetrics]],
    model_names: List[str],
    colors: Dict[str, str],
    out_dir: Path,
):
    higher_is_better = metric_key == "score"
    models_sorted = [m for m in DESIRED_MODEL_ORDER if m in set(model_names)]
    if not models_sorted:
        models_sorted = sorted(model_names)
    num_models = len(models_sorted)

    ranks_per_model: Dict[str, List[Optional[int]]] = {m: [] for m in models_sorted}
    max_rank = 1
    for agent_type in AGENT_TYPE_ORDER:
        values: List[Tuple[str, Optional[float]]] = []
        for m in models_sorted:
            metrics = model_to_agent_metrics.get(m, {}).get(agent_type)
            if not metrics:
                values.append((m, None))
                continue
            v: Optional[float]
            if metric_key == "score":
                v = metrics.score
            elif metric_key == "time":
                v = metrics.time
            else:
                v = metrics.tokens
            values.append((m, v))
        dense = compute_dense_ranks([(m, v if v is not None else np.nan) for m, v in values], higher_is_better=higher_is_better)
        if dense:
            max_rank = max(max_rank, max(dense.values()))
        for m in models_sorted:
            ranks_per_model[m].append(dense.get(m))

    x = np.arange(len(AGENT_TYPE_ORDER))
    fig, ax = plt.subplots(figsize=(max(10, len(AGENT_TYPE_ORDER) * 1.5), 6))

    for m in models_sorted:
        y = [r if r is not None else np.nan for r in ranks_per_model[m]]
        ax.plot(x, y, marker="o", label=m, color=colors.get(m, "#999999"))

    ax.set_xticks(x, labels=AGENT_TYPE_ORDER, rotation=45, ha="right")
    ax.set_yticks(np.arange(1, max_rank + 1))
    ax.invert_yaxis()  # rank 1 at top
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title(f"Ranking Across Agent Types - {pretty_metric_name(metric_key)}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)

    out_base = out_dir / "bumps" / f"bump_{metric_key}"
    save_fig(fig, out_base)


def plot_bump_charts_triptych(
    model_to_agent_metrics: Dict[str, Dict[str, AgentTypeMetrics]],
    model_names: List[str],
    colors: Dict[str, str],
    out_dir: Path,
):
    models_sorted = [m for m in DESIRED_MODEL_ORDER if m in set(model_names)]
    if not models_sorted:
        models_sorted = sorted(model_names)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["score", "time", "tokens"]

    for i, metric_key in enumerate(metrics):
        higher_is_better = metric_key == "score"
        ranks_per_model: Dict[str, List[Optional[int]]] = {m: [] for m in models_sorted}
        max_rank = 1
        for agent_type in AGENT_TYPE_ORDER:
            values: List[Tuple[str, Optional[float]]] = []
            for m in models_sorted:
                metrics_obj = model_to_agent_metrics.get(m, {}).get(agent_type)
                if not metrics_obj:
                    values.append((m, None))
                    continue
                v: Optional[float]
                if metric_key == "score":
                    v = metrics_obj.score
                elif metric_key == "time":
                    v = metrics_obj.time
                else:
                    v = metrics_obj.tokens
                values.append((m, v))
            dense = compute_dense_ranks([(m, v if v is not None else np.nan) for m, v in values], higher_is_better=higher_is_better)
            if dense:
                max_rank = max(max_rank, max(dense.values()))
            for m in models_sorted:
                ranks_per_model[m].append(dense.get(m))

        ax = axs[i]
        x = np.arange(len(AGENT_TYPE_ORDER))
        for m in models_sorted:
            y = [r if r is not None else np.nan for r in ranks_per_model[m]]
            ax.plot(x, y, marker="o", label=m, color=colors.get(m, "#999999"))
        ax.set_xticks(x, labels=AGENT_TYPE_ORDER, rotation=45, ha="right")
        ax.set_yticks(np.arange(1, max_rank + 1))
        ax.invert_yaxis()
        ax.set_ylabel("Rank (1 = best)")
        ax.set_title(f"Ranking Across Agent Types - {pretty_metric_name(metric_key)}")

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)
    fig.suptitle("Bump Charts by Metric")
    plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])
    out_base = out_dir / "bumps" / "bumps"
    save_fig(fig, out_base)


def compute_overall_copeland(
    metric_key: str,
    model_to_agent_metrics: Dict[str, Dict[str, AgentTypeMetrics]],
    model_names: List[str],
) -> Tuple[Dict[str, float], Dict[str, Tuple[int, int, int]], Dict[str, float]]:
    higher_is_better = metric_key == "score"
    models_sorted = sorted(model_names)

    # Rank per agent type for available models
    ranks_by_agent: Dict[str, Dict[str, int]] = {}
    for agent_type in AGENT_TYPE_ORDER:
        values: List[Tuple[str, Optional[float]]] = []
        for m in models_sorted:
            metrics = model_to_agent_metrics.get(m, {}).get(agent_type)
            v = None
            if metrics:
                if metric_key == "score":
                    v = metrics.score
                elif metric_key == "time":
                    v = metrics.time
                else:
                    v = metrics.tokens
            values.append((m, v))
        dense = compute_dense_ranks([(m, v if v is not None else np.nan) for m, v in values], higher_is_better=higher_is_better)
        if dense:
            ranks_by_agent[agent_type] = dense

    wins: Dict[str, int] = {m: 0 for m in models_sorted}
    losses: Dict[str, int] = {m: 0 for m in models_sorted}
    ties: Dict[str, int] = {m: 0 for m in models_sorted}
    copeland: Dict[str, float] = {m: 0.0 for m in models_sorted}
    pair_counts: Dict[str, int] = {m: 0 for m in models_sorted}

    for i in range(len(models_sorted)):
        for j in range(i + 1, len(models_sorted)):
            mi = models_sorted[i]
            mj = models_sorted[j]
            better_i = 0
            better_j = 0
            considered = 0
            for agent_type, ranks in ranks_by_agent.items():
                ri = ranks.get(mi)
                rj = ranks.get(mj)
                if ri is None or rj is None:
                    continue
                considered += 1
                if ri < rj:
                    better_i += 1
                elif rj < ri:
                    better_j += 1
                else:
                    # equal rank in this agent type
                    pass
            if considered == 0:
                continue
            pair_counts[mi] += 1
            pair_counts[mj] += 1
            if better_i > better_j:
                copeland[mi] += 1.0
                wins[mi] += 1
                losses[mj] += 1
            elif better_j > better_i:
                copeland[mj] += 1.0
                wins[mj] += 1
                losses[mi] += 1
            else:
                copeland[mi] += 0.5
                copeland[mj] += 0.5
                ties[mi] += 1
                ties[mj] += 1

    win_rate: Dict[str, float] = {}
    for m in models_sorted:
        denom = pair_counts[m]
        if denom > 0:
            win_rate[m] = (wins[m] + 0.5 * ties[m]) / denom
        else:
            win_rate[m] = 0.0

    return copeland, {m: (wins[m], losses[m], ties[m]) for m in models_sorted}, win_rate


def plot_overall_leaderboard(
    metric_key: str,
    copeland: Dict[str, float],
    win_loss_tie: Dict[str, Tuple[int, int, int]],
    win_rate: Dict[str, float],
    colors: Dict[str, str],
    out_dir: Path,
):
    items = list(copeland.items())
    items.sort(key=lambda x: (x[1], -win_loss_tie[x[0]][1], x[0]), reverse=True)
    models_sorted = [m for m, _ in items]
    scores = [copeland[m] for m in models_sorted]
    bar_colors = [colors.get(m, "#999999") for m in models_sorted]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(models_sorted))
    ax.barh(y_pos, scores, color=bar_colors)
    ax.set_yticks(y_pos, labels=models_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("Copeland Score (pairwise wins)")
    ax.set_title(f"Overall Leaderboard - {pretty_metric_name(metric_key)}")
    # Extend x-axis to 0–9 for visual breathing room
    ax.set_xlim(0, 9)

    for i, m in enumerate(models_sorted):
        w, l, t = win_loss_tie[m]
        wr = win_rate[m]
        ax.text(scores[i] + 0.02, i + 0.1, f"{w}-{l} ({wr:.2f})")

    out_base = out_dir / "overall" / f"overall_{metric_key}"
    save_fig(fig, out_base)


def compute_overall_mean_rank(
    metric_key: str,
    model_to_agent_metrics: Dict[str, Dict[str, AgentTypeMetrics]],
    model_names: List[str],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    higher_is_better = metric_key == "score"
    models_sorted = sorted(model_names)

    # Collect ranks per agent type
    ranks_by_agent: Dict[str, Dict[str, int]] = {}
    for agent_type in AGENT_TYPE_ORDER:
        values: List[Tuple[str, Optional[float]]] = []
        for m in models_sorted:
            metrics = model_to_agent_metrics.get(m, {}).get(agent_type)
            v = None
            if metrics:
                if metric_key == "score":
                    v = metrics.score
                elif metric_key == "time":
                    v = metrics.time
                else:
                    v = metrics.tokens
            values.append((m, v))
        dense = compute_dense_ranks([(m, v if v is not None else np.nan) for m, v in values], higher_is_better=higher_is_better)
        if dense:
            ranks_by_agent[agent_type] = dense

    # Aggregate mean and std of ranks
    mean_rank: Dict[str, float] = {}
    std_rank: Dict[str, float] = {}
    for m in models_sorted:
        ranks: List[int] = []
        for agent_type in AGENT_TYPE_ORDER:
            r = ranks_by_agent.get(agent_type, {}).get(m)
            if r is not None:
                ranks.append(r)
        if ranks:
            mean_rank[m] = float(np.mean(ranks))
            std_rank[m] = float(np.std(ranks))
    return mean_rank, std_rank


def plot_overall_leaderboard_mean_rank(
    metric_key: str,
    mean_rank: Dict[str, float],
    std_rank: Dict[str, float],
    colors: Dict[str, str],
    out_dir: Path,
):
    if not mean_rank:
        return
    items = list(mean_rank.items())
    # Sort ascending (lower mean rank is better), tie-break by smaller std, then name
    items.sort(key=lambda x: (x[1], std_rank.get(x[0], float("inf")), x[0]))
    models_sorted = [m for m, _ in items]
    values = [mean_rank[m] for m in models_sorted]
    bar_colors = [colors.get(m, "#999999") for m in models_sorted]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(models_sorted))
    ax.barh(y_pos, values, color=bar_colors)
    ax.set_yticks(y_pos, labels=models_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Rank (lower is better)")
    ax.set_title(f"Overall Leaderboard - {pretty_metric_name(metric_key)}")
    # Set nice upper bound with at least 2 extra ticks
    max_val = max(values) if values else 1.0
    _, upper = compute_nice_step_and_upper(max_val, min_extra_increments=2, target_ticks=5)
    ax.set_xlim(0, upper)

    out_base = out_dir / "overall" / f"overall_{metric_key}"
    save_fig(fig, out_base)


def plot_overall_leaderboards_mean_rank_triptych(
    model_to_agent_metrics: Dict[str, Dict[str, AgentTypeMetrics]],
    model_names: List[str],
    colors: Dict[str, str],
    out_dir: Path,
):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["score", "time", "tokens"]
    for i, metric_key in enumerate(metrics):
        mean_rank, std_rank = compute_overall_mean_rank(metric_key, model_to_agent_metrics, model_names)
        if not mean_rank:
            continue
        items = list(mean_rank.items())
        items.sort(key=lambda x: (x[1], std_rank.get(x[0], float("inf")), x[0]))
        models_sorted = [m for m, _ in items]
        values = [mean_rank[m] for m in models_sorted]
        bar_colors = [colors.get(m, "#999999") for m in models_sorted]

        ax = axs[i]
        y_pos = np.arange(len(models_sorted))
        ax.barh(y_pos, values, color=bar_colors)
        ax.set_yticks(y_pos, labels=models_sorted)
        ax.invert_yaxis()
        ax.set_xlabel("Mean Rank (lower is better)")
        ax.set_title(f"Overall Leaderboard - {pretty_metric_name(metric_key)}")
        max_val = max(values) if values else 1.0
        _, upper = compute_nice_step_and_upper(max_val, min_extra_increments=2, target_ticks=5)
        ax.set_xlim(0, upper)

    fig.suptitle("Overall Leaderboards by Metric (Mean Rank)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_base = out_dir / "overall" / "overall"
    save_fig(fig, out_base)


def plot_overall_leaderboards_triptych(
    model_to_agent_metrics: Dict[str, Dict[str, AgentTypeMetrics]],
    model_names: List[str],
    colors: Dict[str, str],
    out_dir: Path,
):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["score", "time", "tokens"]
    for i, metric_key in enumerate(metrics):
        copeland, wlt, wr = compute_overall_copeland(metric_key, model_to_agent_metrics, model_names)
        items = list(copeland.items())
        items.sort(key=lambda x: (x[1], -wlt[x[0]][1], x[0]), reverse=True)
        models_sorted = [m for m, _ in items]
        scores = [copeland[m] for m in models_sorted]
        bar_colors = [colors.get(m, "#999999") for m in models_sorted]

        ax = axs[i]
        y_pos = np.arange(len(models_sorted))
        ax.barh(y_pos, scores, color=bar_colors)
        ax.set_yticks(y_pos, labels=models_sorted)
        ax.invert_yaxis()
        ax.set_xlabel("Copeland Score (pairwise wins)")
        ax.set_title(f"Overall Leaderboard - {pretty_metric_name(metric_key)}")
        ax.set_xlim(0, 9)
        # Omit bar-end labels to prevent clipping

    fig.suptitle("Overall Leaderboards by Metric")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_base = out_dir / "overall" / "overall"
    save_fig(fig, out_base)


def run(
    results_dir: Path,
    output_dir: Path,
):
    ensure_output_dirs(output_dir)
    model_to_agent_metrics, model_names = aggregate_results(results_dir)
    if not model_to_agent_metrics:
        print(f"No agent_type_results CSVs found under {results_dir}")
        return

    colors = build_color_map(model_names)

    for agent_type in AGENT_TYPE_ORDER:
        plot_agent_type_triptych(agent_type, model_to_agent_metrics, colors, output_dir)

    # Also create triptych figures for heatmaps, bumps, and overall
    plot_rank_heatmaps_triptych(model_to_agent_metrics, model_names, output_dir)
    plot_bump_charts_triptych(model_to_agent_metrics, model_names, colors, output_dir)
    # Use mean-rank based overall (more directly reflects consistent ranking across agent types)
    plot_overall_leaderboards_mean_rank_triptych(model_to_agent_metrics, model_names, colors, output_dir)

    print(f"Charts written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create agent-type ranking diagrams across models.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Path to the results directory containing per-model subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "agent_type_rankings",
        help="Directory to write generated charts.",
    )
    args = parser.parse_args()

    run(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()


