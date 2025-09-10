import csv
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def plot_csv_results(csv_file_path: Path):
    """
    Reads agent type results from a CSV file and generates a plot.
    """
    agent_types = []
    avg_scores = []
    std_scores = []
    avg_times = []
    std_times = []
    avg_tokens = []
    std_tokens = []

    model_name = csv_file_path.stem.replace('agent_type_results_', '')

    with open(csv_file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header row
        for row in reader:
            if not row or len(row) < 6:
                continue
            # Columns: 0 Agent Type, 1 Weighted Average Score, 2 Std Dev Score,
            #          3 Average Execution Time, 4 Std Dev Time,
            #          5 Average Output Tokens, 6 Std Dev Tokens
            agent_types.append(row[0])
            avg_scores.append(float(row[1]))
            std_scores.append(float(row[2]))
            avg_times.append(float(row[3]))
            std_times.append(float(row[4]))
            avg_tokens.append(float(row[5]))
            std_tokens.append(float(row[6]) if len(row) > 6 and row[6] != '' else 0.0)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Aggregated Performance by Agent Type for {model_name}', fontsize=16)

    # Weighted Average Score by Agent Type
    axs[0].bar(agent_types, avg_scores, yerr=std_scores, color='cornflowerblue', capsize=5)
    axs[0].set_title('Weighted Average Score')
    axs[0].set_ylabel('Score')
    try:
        score_upper = max((s + e) for s, e in zip(avg_scores, std_scores)) if avg_scores and std_scores else (max(avg_scores) if avg_scores else 1.0)
        axs[0].set_ylim(0, max(1.1, score_upper + 0.1))
    except Exception:
        axs[0].set_ylim(0, 1.0)
    axs[0].tick_params(axis='x', rotation=45)

    # Average Execution Time by Agent Type
    axs[1].bar(agent_types, avg_times, yerr=std_times, color='mediumseagreen', capsize=5)
    axs[1].set_title('Average Execution Time (s)')
    axs[1].set_ylabel('Seconds (log scale)')
    if any(v > 0 for v in avg_times):
        axs[1].set_yscale('log')
    axs[1].tick_params(axis='x', rotation=45)

    # Average Output Tokens by Agent Type
    axs[2].bar(agent_types, avg_tokens, yerr=std_tokens, color='lightcoral', capsize=5)
    axs[2].set_title('Average Output Tokens')
    axs[2].set_ylabel('Tokens (log scale)')
    if any(v > 0 for v in avg_tokens):
        axs[2].set_yscale('log')
    axs[2].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_filename = f'agent_type_performance_{model_name}.png'
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot aggregated agent performance from a CSV file.")
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to the agent_type_results_*.csv file to plot."
    )
    args = parser.parse_args()
    
    if not args.csv_file.is_file():
        print(f"Error: File not found at {args.csv_file}")
    else:
        plot_csv_results(args.csv_file) 