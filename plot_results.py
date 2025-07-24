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
    avg_times = []
    avg_tokens = []

    model_name = csv_file_path.stem.replace('agent_type_results_', '')

    with open(csv_file_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header row
        for row in reader:
            agent_types.append(row[0])
            avg_scores.append(float(row[1]))
            avg_times.append(float(row[2]))
            avg_tokens.append(float(row[3]))

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Aggregated Performance by Agent Type for {model_name}', fontsize=16)

    # Weighted Average Score by Agent Type
    axs[0].bar(agent_types, avg_scores, color='cornflowerblue')
    axs[0].set_title('Weighted Average Score')
    axs[0].set_ylabel('Score')
    axs[0].tick_params(axis='x', rotation=45)

    # Average Execution Time by Agent Type
    axs[1].bar(agent_types, avg_times, color='mediumseagreen')
    axs[1].set_title('Average Execution Time (s)')
    axs[1].set_ylabel('Seconds')
    axs[1].tick_params(axis='x', rotation=45)

    # Average Output Tokens by Agent Type
    axs[2].bar(agent_types, avg_tokens, color='lightcoral')
    axs[2].set_title('Average Output Tokens')
    axs[2].set_ylabel('Tokens (log scale)')
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