import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

def plot_benchmark_csv_results(csv_file_path: Path):
    """
    Reads benchmark results from a CSV file, aggregates the data,
    and generates a plot.
    """
    model_name = csv_file_path.stem.replace('benchmark_results_', '')
    
    # Use defaultdict to easily append to lists
    benchmark_data = defaultdict(lambda: {'scores': [], 'times': [], 'tokens': []})

    with open(csv_file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            print(f"Error: Empty CSV file at {csv_file_path}")
            return
        header_lower = [h.strip().lower() for h in header]

        # Detect wrong CSV (agent type results) and fail gracefully
        if header_lower and header_lower[0] == 'agent type':
            print("Error: Provided file appears to be an agent_type_results CSV. Use plot_agent_type_results.py for this file.")
            return

        # Map required columns dynamically
        def col_idx(name: str):
            try:
                return header_lower.index(name)
            except ValueError:
                return -1

        idx_bench = col_idx('benchmark')
        idx_score = col_idx('score')
        idx_time = col_idx('execution time')
        idx_tokens = col_idx('output tokens')

        if min(idx_bench, idx_score, idx_time, idx_tokens) < 0:
            print("Error: CSV header missing required columns: 'Benchmark', 'Score', 'Execution Time', 'Output Tokens'.")
            return

        for row in reader:
            if not row or len(row) <= max(idx_bench, idx_score, idx_time, idx_tokens):
                continue
            benchmark_name = row[idx_bench]
            # Parse floats safely
            try:
                score_val = float(row[idx_score]) if row[idx_score] != '' else 0.0
            except Exception:
                score_val = 0.0
            try:
                time_val = float(row[idx_time]) if row[idx_time] != '' else 0.0
            except Exception:
                time_val = 0.0
            try:
                token_val = float(row[idx_tokens]) if row[idx_tokens] != '' else 0.0
            except Exception:
                token_val = 0.0

            benchmark_data[benchmark_name]['scores'].append(score_val)
            benchmark_data[benchmark_name]['times'].append(time_val)
            benchmark_data[benchmark_name]['tokens'].append(token_val)

    # Calculate averages and standard deviations
    benchmarks = list(benchmark_data.keys())
    avg_scores = [np.mean(benchmark_data[b]['scores']) for b in benchmarks]
    std_scores = [np.std(benchmark_data[b]['scores']) for b in benchmarks]
    avg_times = [np.mean(benchmark_data[b]['times']) if benchmark_data[b]['times'] else 0 for b in benchmarks]
    std_times = [np.std(benchmark_data[b]['times']) if benchmark_data[b]['times'] else 0 for b in benchmarks]
    avg_tokens = [np.mean(benchmark_data[b]['tokens']) if benchmark_data[b]['tokens'] else 0 for b in benchmarks]
    std_tokens = [np.std(benchmark_data[b]['tokens']) if benchmark_data[b]['tokens'] else 0 for b in benchmarks]

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Performance for {model_name} per Task', fontsize=16)

    # Average Score per Task
    axs[0].bar(benchmarks, avg_scores, yerr=std_scores, color='lightgreen', capsize=5)
    axs[0].set_title('Average Score per Task')
    axs[0].set_ylabel('Average Score')
    try:
        score_upper = max((s + e) for s, e in zip(avg_scores, std_scores)) if avg_scores and std_scores else (max(avg_scores) if avg_scores else 1.0)
        axs[0].set_ylim(0, max(1.1, score_upper + 0.1))
    except Exception:
        axs[0].set_ylim(0, 1.0)
    axs[0].tick_params(axis='x', rotation=90)

    # Average Time per Task
    axs[1].bar(benchmarks, avg_times, yerr=std_times, color='salmon', capsize=5)
    axs[1].set_title('Average Time per Task (s)')
    axs[1].set_ylabel('Seconds (log scale)')
    if any(v > 0 for v in avg_times):
        axs[1].set_yscale('log')
    axs[1].tick_params(axis='x', rotation=90)

    # Average Output Tokens
    axs[2].bar(benchmarks, avg_tokens, yerr=std_tokens, color='gold', capsize=5)
    axs[2].set_title('Average Output Tokens per Task')
    axs[2].set_ylabel('Tokens (log scale)')
    axs[2].set_yscale('log')
    axs[2].tick_params(axis='x', rotation=90)

    plt.tight_layout(rect=[0, 0.2, 1, 0.95])
    
    output_filename = f'benchmark_performance_{model_name}.png'
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark performance from a CSV file.")
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to the benchmark_results_*.csv file to plot."
    )
    args = parser.parse_args()

    if not args.csv_file.is_file():
        print(f"Error: File not found at {args.csv_file}")
    else:
        plot_benchmark_csv_results(args.csv_file) 