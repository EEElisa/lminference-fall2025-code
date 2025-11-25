#!/bin/bash
# Script to run all benchmark configurations systematically

# Set your GPU device
GPU=0
RUNS_PER_PROMPT=2

echo "Starting comprehensive speculative decoding benchmarks..."
echo "GPU: $GPU"
echo "Runs per prompt: $RUNS_PER_PROMPT"
echo "=================================================="

# Configuration 1: Qwen-3-8B with Qwen-3-1.7B
echo -e "\n### Configuration 1: Qwen-3-8B + Qwen-3-1.7B ###"
for lookahead in 2 3 5 7; do
    echo "Running with lookahead=$lookahead..."
    CUDA_VISIBLE_DEVICES=$GPU python specdec/benchmark.py \
        --target Qwen/Qwen3-8B \
        --draft Qwen/Qwen3-1.7B \
        --lookahead $lookahead \
        --runs_per_prompt $RUNS_PER_PROMPT
done

# Configuration 2: Qwen-3-8B with Qwen-3-0.6B
echo -e "\n### Configuration 2: Qwen-3-8B + Qwen-3-0.6B ###"
for lookahead in 2 3 5 7; do
    echo "Running with lookahead=$lookahead..."
    CUDA_VISIBLE_DEVICES=$GPU python specdec/benchmark.py \
        --target Qwen/Qwen3-8B \
        --draft Qwen/Qwen3-0.6B \
        --lookahead $lookahead \
        --runs_per_prompt $RUNS_PER_PROMPT
done

# Configuration 3: Llama-3.1-8B with Llama-3.2-1B
echo -e "\n### Configuration 3: Llama-3.1-8B + Llama-3.2-1B ###"
for lookahead in 2 3 5 7; do
    echo "Running with lookahead=$lookahead..."
    CUDA_VISIBLE_DEVICES=$GPU python specdec/benchmark.py \
        --target meta-llama/Llama-3.1-8B \
        --draft meta-llama/Llama-3.2-1B \
        --lookahead $lookahead \
        --runs_per_prompt $RUNS_PER_PROMPT
done

echo -e "\n=================================================="
echo "All benchmarks completed!"
echo "Results saved in benchmark_dir/"
