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
        reader = csv.DictReader(f)
        for row in reader:
            benchmark_name = row['Benchmark']
            benchmark_data[benchmark_name]['scores'].append(float(row['Score']))
            if row['Execution Time']:
                benchmark_data[benchmark_name]['times'].append(float(row['Execution Time']))
            if row['Output Tokens']:
                benchmark_data[benchmark_name]['tokens'].append(float(row['Output Tokens']))

    # Calculate averages
    benchmarks = list(benchmark_data.keys())
    avg_scores = [np.mean(data['scores']) for data in benchmark_data.values()]
    avg_times = [np.mean(data['times']) if data['times'] else 0 for data in benchmark_data.values()]
    avg_tokens = [np.mean(data['tokens']) if data['tokens'] else 0 for data in benchmark_data.values()]

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Benchmark Performance Analysis for {model_name}', fontsize=16)

    # Average Score by Benchmark
    axs[0].bar(benchmarks, avg_scores, color='lightgreen')
    axs[0].set_title('Average Score by Benchmark')
    axs[0].set_ylabel('Average Score')
    axs[0].tick_params(axis='x', rotation=45)

    # Average Time per Task
    axs[1].bar(benchmarks, avg_times, color='salmon')
    axs[1].set_title('Average Time per Task (s)')
    axs[1].set_ylabel('Seconds')
    axs[1].tick_params(axis='x', rotation=45)

    # Average Output Tokens
    axs[2].bar(benchmarks, avg_tokens, color='gold')
    axs[2].set_title('Average Output Tokens per Task')
    axs[2].set_ylabel('Tokens (log scale)')
    axs[2].set_yscale('log')
    axs[2].tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
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