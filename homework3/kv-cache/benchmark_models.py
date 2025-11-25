"""
Benchmark script for comparing different models with fixed input/output lengths.
"""

import torch
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional
import gc


def setup_model(model_name: str = "Qwen/Qwen3-8B", device: str = "cuda", cache_dir: Optional[str] = None):
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

    # Print GPU info and model info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"Total VRAM: {total_memory:.2f} GB")

    # Count model parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e9:.2f}B")

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


def benchmark_single_model(
    model,
    tokenizer,
    batch_size: int,
    input_len: int,
    output_len: int,
    num_iterations: int = 10,
    num_warmup: int = 3
) -> Tuple[float, float, bool]:
    """
    Benchmark a single model configuration.

    Returns:
        (avg_time, throughput, success)
        - avg_time: average wall-clock time in seconds
        - throughput: tokens per second
        - success: whether the benchmark completed without OOM
    """
    vocab_size = model.config.vocab_size

    # Warm-up
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


def run_model_comparison(
    model_names: List[str],
    batch_size: int = 8,
    input_len: int = 64,
    output_len: int = 64,
    num_iterations: int = 10,
    cache_dir: Optional[str] = None
) -> Dict:
    """
    Run benchmarks across multiple models.

    Args:
        model_names: List of HuggingFace model names
        batch_size: Number of sequences per batch
        input_len: Input sequence length (fixed)
        output_len: Output sequence length (fixed)
        num_iterations: Number of iterations to average over
        cache_dir: Directory to cache the model files

    Returns:
        Dictionary with results
    """
    print("="*70)
    print("Starting Model Comparison Benchmark")
    print("="*70)
    print(f"Models to test: {model_names}")
    print(f"Batch size: {batch_size}")
    print(f"Input length: {input_len} tokens (fixed)")
    print(f"Output length: {output_len} tokens (fixed)")
    print(f"Iterations per model: {num_iterations}")
    print("="*70)

    results = {
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "num_iterations": num_iterations,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "models": []
    }

    # Run benchmarks for each model
    for model_name in model_names:
        print(f"\n{'='*70}")
        print(f"Benchmarking Model: {model_name}")
        print(f"{'='*70}")

        # Clear cache before loading new model
        torch.cuda.empty_cache()
        gc.collect()

        try:
            # Load model
            model, tokenizer = setup_model(model_name, cache_dir=cache_dir)

            # Run benchmark
            avg_time, throughput, success = benchmark_single_model(
                model, tokenizer, batch_size, input_len, output_len, num_iterations
            )

            result = {
                "model_name": model_name,
                "success": success,
                "avg_time": avg_time,
                "throughput": throughput,
            }

            if not success:
                print(f"  ✗ OOM ERROR for {model_name}")
                result["error"] = "OOM"

            results["models"].append(result)

            # Clean up model to free memory
            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"  ✗ ERROR loading or running {model_name}: {str(e)}")
            results["models"].append({
                "model_name": model_name,
                "success": False,
                "error": str(e),
                "avg_time": None,
                "throughput": None,
            })

    return results


def save_results(results: Dict, filename: str = "model_comparison_results.json"):
    """Save results to JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {filename}")


def print_table(results: Dict):
    """Print results in table format."""
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(f"GPU: {results['gpu_name']}")
    print(f"Configuration: Batch size={results['batch_size']}, "
          f"Input={results['input_len']} tokens, Output={results['output_len']} tokens")
    print("="*80)
    print(f"{'Model':<40} | {'Time (s)':<12} | {'Throughput (tokens/s)':<25} | {'Status':<10}")
    print("-"*80)

    for m in results["models"]:
        model_name = m["model_name"]
        if m["success"]:
            time_str = f"{m['avg_time']:.4f}"
            throughput_str = f"{m['throughput']:.2f}"
            status = "✓"
        else:
            time_str = "ERROR"
            throughput_str = "ERROR"
            status = "✗ " + m.get("error", "FAILED")[:20]

        print(f"{model_name:<40} | {time_str:<12} | {throughput_str:<25} | {status:<10}")

    print("="*80)


if __name__ == "__main__":
    # Configuration
    MODEL_NAMES = [
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-8B",
        "allenai/OLMo-7B-0724-hf"
    ]
    BATCH_SIZE = 8
    INPUT_LENGTH = 64
    OUTPUT_LENGTH = 64
    NUM_ITERATIONS = 10
    CACHE_DIR = "/usr1/data/mingqia2/"

    # Run the comparison
    results = run_model_comparison(
        model_names=MODEL_NAMES,
        batch_size=BATCH_SIZE,
        input_len=INPUT_LENGTH,
        output_len=OUTPUT_LENGTH,
        num_iterations=NUM_ITERATIONS,
        cache_dir=CACHE_DIR
    )

    # Save and display results
    save_results(results, "model_comparison_results.json")
    print_table(results)

    print("\n✓ Model comparison benchmark complete!")
