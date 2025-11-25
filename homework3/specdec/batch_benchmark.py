"""
Batch Size Benchmarking Script

Measures forward-pass time for different batch sizes to understand
how forward cost scales with batch size - relevant for speculative decoding.

Usage:
    CUDA_VISIBLE_DEVICES=1 python specdec/batch_benchmark.py --model Qwen/Qwen3-4B
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os


def get_args():
    p = argparse.ArgumentParser(description="Batch Size Forward-Pass Benchmark")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B", help="Model to benchmark")
    p.add_argument("--device", default="cuda:0", help="Device to use")
    p.add_argument("--seq_length", type=int, default=256, help="Sequence length")
    p.add_argument("--batch_sizes", type=int, nargs='+', default=[1, 2, 4, 8, 16],
                   help="Batch sizes to test")
    p.add_argument("--trials", type=int, default=5, help="Number of trials per configuration")
    p.add_argument("--warmup", type=int, default=3, help="Number of warmup iterations")
    p.add_argument("--output_dir", default="benchmark_dir", help="Directory to save results")
    return p.parse_args()


def benchmark_forward_pass(model, input_ids, warmup_iters=3, num_trials=5):
    """
    Benchmark forward pass time using torch.cuda.Event for accurate GPU timing.

    Args:
        model: The model to benchmark
        input_ids: Input tensor [batch_size, seq_length]
        warmup_iters: Number of warmup iterations
        num_trials: Number of trials to average

    Returns:
        Average forward pass time in milliseconds
    """
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(input_ids)
            torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_trials):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = model(input_ids)
            end_event.record()

            torch.cuda.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            times.append(elapsed_time)

    return np.mean(times), np.std(times)


def main():
    args = get_args()

    print(f"=== Batch Size Forward-Pass Benchmark ===")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Sequence Length: {args.seq_length}")
    print(f"Batch Sizes: {args.batch_sizes}")
    print(f"Trials: {args.trials}")
    print(f"Warmup Iterations: {args.warmup}")
    print("=" * 50)

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir='/usr1/data/mingqia2/',
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model loaded successfully.")

    # Results storage
    results = {
        'batch_sizes': args.batch_sizes,
        'seq_length': args.seq_length,
        'avg_times': [],
        'std_times': [],
        'model': args.model,
        'trials': args.trials,
        'warmup': args.warmup
    }

    # Benchmark each batch size
    print(f"\n{'Batch Size':<12} {'Avg Time (ms)':<15} {'Std (ms)':<12} {'Time/Sample (ms)':<18}")
    print("-" * 60)

    for batch_size in args.batch_sizes:
        # Create dummy input of specified batch size and sequence length
        input_ids = torch.randint(
            0, model.config.vocab_size,
            (batch_size, args.seq_length),
            device=args.device
        )

        # Benchmark
        avg_time, std_time = benchmark_forward_pass(
            model, input_ids,
            warmup_iters=args.warmup,
            num_trials=args.trials
        )

        results['avg_times'].append(avg_time)
        results['std_times'].append(std_time)

        time_per_sample = avg_time / batch_size
        print(f"{batch_size:<12} {avg_time:<15.2f} {std_time:<12.2f} {time_per_sample:<18.2f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    model_name_safe = args.model.replace('/', '_')
    results_path = os.path.join(args.output_dir, f"{model_name_safe}_batch_benchmark.json")

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Plot results
    plot_results(results, args.output_dir, model_name_safe)

    # Print analysis
    print_analysis(results)


def plot_results(results, output_dir, model_name_safe):
    """Create visualization of batch size vs forward-pass time."""
    batch_sizes = results['batch_sizes']
    avg_times = results['avg_times']
    std_times = results['std_times']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Batch Size vs Wall-Clock Time
    ax1.errorbar(batch_sizes, avg_times, yerr=std_times,
                 marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Wall-Clock Time (ms)', fontsize=12)
    ax1.set_title(f'Batch Size vs Forward-Pass Time\n({results["model"]})', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(batch_sizes)

    # Plot 2: Time per Sample
    time_per_sample = [avg_times[i] / batch_sizes[i] for i in range(len(batch_sizes))]
    std_per_sample = [std_times[i] / batch_sizes[i] for i in range(len(batch_sizes))]

    ax2.errorbar(batch_sizes, time_per_sample, yerr=std_per_sample,
                 marker='s', capsize=5, capthick=2, linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Time per Sample (ms)', fontsize=12)
    ax2.set_title(f'Batch Size vs Time per Sample\n({results["model"]})', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(batch_sizes)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{model_name_safe}_batch_benchmark.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")

    # Also save as PDF
    pdf_path = os.path.join(output_dir, f"{model_name_safe}_batch_benchmark.pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Plot saved to {pdf_path}")

    plt.close()


def print_analysis(results):
    """Print analysis of how forward cost scales with batch size."""
    batch_sizes = results['batch_sizes']
    avg_times = results['avg_times']

    print("\n" + "=" * 70)
    print("ANALYSIS: Forward Cost Scaling with Batch Size")
    print("=" * 70)

    print("\n1. Absolute Time Scaling:")
    print(f"   - Batch size 1: {avg_times[0]:.2f} ms")
    print(f"   - Batch size {batch_sizes[-1]}: {avg_times[-1]:.2f} ms")
    print(f"   - Ratio: {avg_times[-1]/avg_times[0]:.2f}x")

    print("\n2. Amortized Cost (Time per Sample):")
    for i, bs in enumerate(batch_sizes):
        time_per_sample = avg_times[i] / bs
        efficiency = (avg_times[0] / time_per_sample) * 100
        print(f"   - Batch {bs:2d}: {time_per_sample:6.2f} ms/sample ({efficiency:5.1f}% efficient)")

    # Calculate theoretical speedup
    if len(batch_sizes) >= 2:
        k = 4  # Example lookahead
        single_time = avg_times[0]
        batch_idx = batch_sizes.index(k+1) if (k+1) in batch_sizes else -1
        if batch_idx != -1:
            batch_time = avg_times[batch_idx]
            sequential_time = single_time * (k + 1)
            speedup = sequential_time / batch_time
            print(f"\n5. Example Calculation (K={k}):")
            print(f"   - Sequential verification: {k+1} × {single_time:.2f} = {sequential_time:.2f} ms")
            print(f"   - Batched verification: {batch_time:.2f} ms")
            print(f"   - Speedup from batching: {speedup:.2f}x")

    print("=" * 70)


if __name__ == "__main__":
    main()
