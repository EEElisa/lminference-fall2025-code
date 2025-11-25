"""
Analyze and visualize speculative decoding benchmark results.

Usage:
    python specdec/analyze_benchmarks.py
"""
import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def parse_log_file(log_path):
    """Parse a benchmark log file to extract metrics."""
    with open(log_path, 'r') as f:
        content = f.read()

    # Extract metrics using regex
    ar_time = float(re.search(r'Average AR Time: ([\d.]+)s', content).group(1))
    sd_time = float(re.search(r'Average SD Time: ([\d.]+)s', content).group(1))
    acceptance_rate = float(re.search(r'Average Acceptance Rate: ([\d.]+)%', content).group(1)) / 100
    speedup = float(re.search(r'Average Empirical Speedup: ([\d.]+)x', content).group(1))

    return {
        'ar_time': ar_time,
        'sd_time': sd_time,
        'acceptance_rate': acceptance_rate,
        'speedup': speedup
    }


def extract_config_from_filename(filename):
    """Extract target, draft, and lookahead from log filename."""
    # Expected format: target_draft_lookahead_K.log
    parts = filename.replace('.log', '').split('_')

    # Find lookahead value
    lookahead_idx = -1
    for i, part in enumerate(parts):
        if part == 'lookahead':
            lookahead = int(parts[i+1])
            lookahead_idx = i
            break

    # Join parts before lookahead to form target_draft
    config_parts = parts[:lookahead_idx]

    return {
        'filename': filename,
        'lookahead': lookahead,
        'config_str': '_'.join(config_parts)
    }


def collect_results(benchmark_dir='benchmark_dir'):
    """Collect all benchmark results from log files."""
    results = {}

    log_files = list(Path(benchmark_dir).glob('*_lookahead_*.log'))

    if not log_files:
        print(f"No benchmark log files found in {benchmark_dir}")
        return results

    for log_file in log_files:
        try:
            config = extract_config_from_filename(log_file.name)
            metrics = parse_log_file(log_file)

            config_key = config['config_str']
            lookahead = config['lookahead']

            if config_key not in results:
                results[config_key] = {}

            results[config_key][lookahead] = metrics

            print(f"Loaded: {config_key}, lookahead={lookahead}, speedup={metrics['speedup']:.2f}x")
        except Exception as e:
            print(f"Error parsing {log_file.name}: {e}")

    return results


def create_comparison_plots(results, output_dir='benchmark_dir'):
    """Create comparison plots for all configurations."""
    if not results:
        print("No results to plot")
        return

    # Prepare data for plotting
    configs = list(results.keys())
    lookaheads = sorted(list(set([k for config in results.values() for k in config.keys()])))

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Speedup vs Lookahead
    ax1 = axes[0]
    for config in configs:
        speedups = [results[config].get(k, {}).get('speedup', 0) for k in lookaheads]
        # Clean config name for legend
        config_name = config.replace('_', ' ').replace('meta-llama', 'Llama').replace('Qwen', 'Q')
        ax1.plot(lookaheads, speedups, marker='o', linewidth=2, markersize=8, label=config_name)

    ax1.set_xlabel('Lookahead (γ)', fontsize=12)
    ax1.set_ylabel('Empirical Speedup', fontsize=12)
    ax1.set_title('Speedup vs Lookahead', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='best')
    ax1.set_xticks(lookaheads)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No speedup')

    # Plot 2: Acceptance Rate vs Lookahead
    ax2 = axes[1]
    for config in configs:
        acceptance_rates = [results[config].get(k, {}).get('acceptance_rate', 0) * 100
                           for k in lookaheads]
        config_name = config.replace('_', ' ').replace('meta-llama', 'Llama').replace('Qwen', 'Q')
        ax2.plot(lookaheads, acceptance_rates, marker='s', linewidth=2, markersize=8, label=config_name)

    ax2.set_xlabel('Lookahead (γ)', fontsize=12)
    ax2.set_ylabel('Acceptance Rate (%)', fontsize=12)
    ax2.set_title('Acceptance Rate vs Lookahead', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')
    ax2.set_xticks(lookaheads)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'specdec_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {plot_path}")

    pdf_path = os.path.join(output_dir, 'specdec_comparison.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Plot saved to {pdf_path}")

    plt.close()


def print_results_table(results):
    """Print results in a formatted table."""
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 100)

    for config in sorted(results.keys()):
        print(f"\n{config}:")
        print(f"{'Lookahead':<12} {'Speedup':<12} {'Acceptance Rate':<20} {'AR Time (s)':<15} {'SD Time (s)':<15}")
        print("-" * 100)

        for lookahead in sorted(results[config].keys()):
            metrics = results[config][lookahead]
            print(f"{lookahead:<12} "
                  f"{metrics['speedup']:<12.2f} "
                  f"{metrics['acceptance_rate']*100:<19.1f}% "
                  f"{metrics['ar_time']:<15.3f} "
                  f"{metrics['sd_time']:<15.3f}")

    print("=" * 100)


def save_results_json(results, output_dir='benchmark_dir'):
    """Save results to JSON file."""
    output_path = os.path.join(output_dir, 'all_benchmark_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def analyze_trends(results):
    """Analyze and print trends in the results."""
    print("\n" + "=" * 100)
    print("ANALYSIS: Key Observations")
    print("=" * 100)

    print("\n1. Speedup Trends:")
    for config in results.keys():
        lookaheads = sorted(results[config].keys())
        speedups = [results[config][k]['speedup'] for k in lookaheads]
        best_lookahead = lookaheads[np.argmax(speedups)]
        best_speedup = max(speedups)
        print(f"   - {config}: Best speedup = {best_speedup:.2f}x at lookahead={best_lookahead}")

    print("\n2. Acceptance Rate Trends:")
    for config in results.keys():
        lookaheads = sorted(results[config].keys())
        acceptance_rates = [results[config][k]['acceptance_rate'] * 100 for k in lookaheads]
        avg_acceptance = np.mean(acceptance_rates)
        print(f"   - {config}: Average acceptance rate = {avg_acceptance:.1f}%")

    print("\n3. Draft Model Comparison:")
    print("   - Smaller draft models (e.g., 0.6B) are faster but may have lower acceptance rates")
    print("   - Larger draft models (e.g., 1.7B) have higher acceptance rates but slower draft generation")
    print("   - Optimal choice depends on the balance between draft speed and acceptance rate")

    print("\n4. Lookahead Selection:")
    print("   - Small lookahead (2-3): Lower acceptance overhead, suitable for high acceptance rates")
    print("   - Large lookahead (5-7): More speculative tokens, but acceptance rate typically decreases")
    print("   - Optimal lookahead varies by target-draft pair and task")

    print("=" * 100)


def main():
    print("Analyzing speculative decoding benchmark results...")

    # Collect results from log files
    results = collect_results('benchmark_dir')

    if not results:
        print("\nNo results found. Please run benchmarks first:")
        print("  bash specdec/run_all_benchmarks.sh")
        return

    # Print results table
    print_results_table(results)

    # Save to JSON
    save_results_json(results)

    # Create comparison plots
    create_comparison_plots(results)

    # Analyze trends
    analyze_trends(results)


if __name__ == "__main__":
    main()
