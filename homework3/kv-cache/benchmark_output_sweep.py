"""
Benchmark script for measuring wall-clock time and token throughput
with varying output sequence lengths for Qwen/Qwen3-8B model.
"""

import torch
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import gc


def setup_model(model_name: str = "Qwen/Qwen2.5-7B", device: str = "cuda", cache_dir: Optional[str] = None):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    if cache_dir:
        print(f"Using cache directory: {cache_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        cache_dir=cache_dir,
    )
    model.eval()

    # Print GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"Total VRAM: {total_memory:.2f} GB")

    return model, tokenizer


def warmup_run(model, tokenizer, batch_size: int, input_len: int, output_len: int, num_warmup: int = 3):
    """Perform warm-up iterations."""
    vocab_size = model.config.vocab_size
    print(f"  Warming up with {num_warmup} iterations...")

    for _ in range(num_warmup):
        try:
            input_ids = torch.randint(0, vocab_size, (batch_size, input_len), device=model.device)
            with torch.no_grad():
                _ = model.generate(
                    input_ids,
                    max_new_tokens=output_len,
                    min_new_tokens=output_len,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM during warmup")
                torch.cuda.empty_cache()
                gc.collect()
                return False
            raise e
    return True


def benchmark_single_config(
    model,
    tokenizer,
    batch_size: int,
    input_len: int,
    output_len: int,
    num_iterations: int = 10,
    num_warmup: int = 3
) -> Tuple[float, float, bool]:
    """
    Benchmark a single configuration.

    Returns:
        (avg_time, throughput, success)
        - avg_time: average wall-clock time in seconds
        - throughput: tokens per second
        - success: whether the benchmark completed without OOM
    """
    vocab_size = model.config.vocab_size

    # Warm-up (fewer warmups for shorter sequences)
    if output_len <= 16:
        num_warmup = 5
    elif output_len <= 64:
        num_warmup = 3
    else:
        num_warmup = 2

    warmup_success = warmup_run(model, tokenizer, batch_size, input_len, output_len, num_warmup)
    if not warmup_success:
        return None, None, False

    # Actual benchmark
    times = []
    print(f"  Running {num_iterations} benchmark iterations...")

    for i in range(num_iterations):
        try:
            # Generate random input
            input_ids = torch.randint(0, vocab_size, (batch_size, input_len), device=model.device)

            # Measure time
            torch.cuda.synchronize()
            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=output_len,
                    min_new_tokens=output_len,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            torch.cuda.synchronize()
            end_time = time.time()

            elapsed = end_time - start_time
            times.append(elapsed)

            if (i + 1) % 5 == 0:
                print(f"    Iteration {i+1}/{num_iterations}: {elapsed:.4f}s")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  OOM during benchmark iteration {i+1}")
                torch.cuda.empty_cache()
                gc.collect()
                return None, None, False
            raise e

    # Calculate metrics
    avg_time = np.mean(times)
    std_time = np.std(times)

    # Total tokens generated = batch_size * output_len
    total_tokens = batch_size * output_len
    throughput = total_tokens / avg_time

    print(f"  ✓ Avg time: {avg_time:.4f}s (±{std_time:.4f}s)")
    print(f"  ✓ Throughput: {throughput:.2f} tokens/s")

    return avg_time, throughput, True


def run_sweep(
    model_name: str = "Qwen/Qwen2.5-7B",
    batch_size: int = 8,
    input_len: int = 64,
    output_lengths: List[int] = None,
    num_iterations: int = 10,
    cache_dir: Optional[str] = None
) -> Dict:
    """
    Run the full output length sweep.

    Args:
        model_name: HuggingFace model name
        batch_size: Number of sequences per batch
        input_len: Input sequence length (fixed)
        output_lengths: List of output sequence lengths to test
        num_iterations: Number of iterations to average over
        cache_dir: Directory to cache the model files

    Returns:
        Dictionary with results
    """
    if output_lengths is None:
        # 2^n for n from 0 to 8
        output_lengths = [2**n for n in range(9)]

    print("="*70)
    print("Starting Output Length Sweep Benchmark")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Input length: {input_len} tokens (fixed)")
    print(f"Output lengths: {output_lengths}")
    print(f"Iterations per config: {num_iterations}")
    print("="*70)

    # Setup
    model, tokenizer = setup_model(model_name, cache_dir=cache_dir)

    results = {
        "model_name": model_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "num_iterations": num_iterations,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "measurements": []
    }

    # Run benchmarks
    for output_len in output_lengths:
        print(f"\n{'='*70}")
        print(f"Benchmarking: output_len={output_len}")
        print(f"{'='*70}")

        # Clear cache before each run
        torch.cuda.empty_cache()
        gc.collect()

        avg_time, throughput, success = benchmark_single_config(
            model, tokenizer, batch_size, input_len, output_len, num_iterations
        )

        result = {
            "output_len": output_len,
            "success": success,
            "avg_time": avg_time,
            "throughput": throughput,
        }

        if not success:
            print(f"  ✗ OOM ERROR - Skipping remaining longer sequences")
            result["error"] = "OOM"

        results["measurements"].append(result)

        # If we hit OOM, stop testing longer sequences
        if not success:
            break

    return results


def save_results(results: Dict, filename: str = "benchmark_results.json"):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {filename}")


def plot_results(results: Dict, output_prefix: str = "benchmark"):
    """Create plots for the benchmark results."""
    measurements = results["measurements"]

    # Extract successful measurements
    successful = [m for m in measurements if m["success"]]
    oom_points = [m for m in measurements if not m["success"]]

    if not successful:
        print("No successful measurements to plot!")
        return

    output_lens = [m["output_len"] for m in successful]
    times = [m["avg_time"] for m in successful]
    throughputs = [m["throughput"] for m in successful]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Wall-clock time
    ax1.plot(output_lens, times, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Output Sequence Length (tokens)', fontsize=12)
    ax1.set_ylabel('Wall-clock Time (seconds)', fontsize=12)
    ax1.set_title('Wall-clock Time vs Output Length', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(output_lens)
    ax1.set_xticklabels([str(x) for x in output_lens], rotation=45)

    # Mark OOM points
    if oom_points:
        oom_lens = [m["output_len"] for m in oom_points]
        ax1.axvline(x=min(oom_lens), color='red', linestyle='--', alpha=0.5, label='OOM')
        ax1.legend()

    # Plot 2: Throughput
    ax2.plot(output_lens, throughputs, 's-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Output Sequence Length (tokens)', fontsize=12)
    ax2.set_ylabel('Throughput (tokens/second)', fontsize=12)
    ax2.set_title('Token Throughput vs Output Length', fontsize=14, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(output_lens)
    ax2.set_xticklabels([str(x) for x in output_lens], rotation=45)

    # Mark OOM points
    if oom_points:
        ax2.axvline(x=min(oom_lens), color='red', linestyle='--', alpha=0.5, label='OOM')
        ax2.legend()

    plt.tight_layout()
    filename = f"{output_prefix}_plots.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to {filename}")
    plt.close()


def print_table(results: Dict):
    """Print results in table format."""
    print("\n" + "="*70)
    print("RESULTS TABLE")
    print("="*70)
    print(f"GPU: {results['gpu_name']}")
    print(f"Model: {results['model_name']}")
    print(f"Batch size: {results['batch_size']}, Input length: {results['input_len']} tokens")
    print("="*70)
    print(f"{'Output Length':<15} | {'Time (s)':<12} | {'Throughput (tokens/s)':<25} | {'Status':<10}")
    print("-"*70)

    for m in results["measurements"]:
        output_len = m["output_len"]
        if m["success"]:
            time_str = f"{m['avg_time']:.4f}"
            throughput_str = f"{m['throughput']:.2f}"
            status = "✓"
        else:
            time_str = "OOM"
            throughput_str = "OOM"
            status = "✗ OOM"

        print(f"{output_len:<15} | {time_str:<12} | {throughput_str:<25} | {status:<10}")

    print("="*70)


if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "Qwen/Qwen3-8B"
    BATCH_SIZE = 8
    INPUT_LENGTH = 64
    NUM_ITERATIONS = 10
    CACHE_DIR = "/usr1/data/mingqia2/"

    # Output lengths to test: 1, 4, 16, 64, 256
    OUTPUT_LENGTHS = [1, 4, 16, 64, 256]

    # Run the sweep
    results = run_sweep(
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        input_len=INPUT_LENGTH,
        output_lengths=OUTPUT_LENGTHS,
        num_iterations=NUM_ITERATIONS,
        cache_dir=CACHE_DIR
    )

    # Save and display results
    save_results(results, "output_sweep_results.json")
    print_table(results)
    plot_results(results, "output_sweep")

    print("\n✓ Benchmark complete!")
