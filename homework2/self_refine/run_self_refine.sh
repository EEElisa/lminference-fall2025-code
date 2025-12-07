#!/bin/bash
# Self-Refine Experiment Runner
# Run this script to execute self-refine experiments with different configurations

set -e  # Exit on error

# Create results directory
mkdir -p results

echo "================================"
echo "Self-Refine Experiments"
echo "================================"

echo ""
echo "=== PART 1: Temperature Comparison (30 examples) ==="
echo "For HW Question 1 - exploring temperature effects"
echo ""

# Temperature Config 1: Lower temperature (more focused/deterministic)
echo "Running Temp Config 1: MMLU + Qwen3-4B (All temps = 0.3 - focused)"
python src/self_refine.py \
    --dataset mmlu_med \
    --model Qwen/Qwen3-4B \
    --split dev_test \
    --max_examples 30 \
    --temp_draft 0.3 \
    --temp_critique 0.3 \
    --temp_refine 0.3 \
    --output results/temp_comparison_low.json

# Temperature Config 2: Default/moderate temperature (balanced)
echo ""
echo "Running Temp Config 2: MMLU + Qwen3-4B (All temps = 0.7 - balanced)"
python src/self_refine.py \
    --dataset mmlu_med \
    --model Qwen/Qwen3-4B \
    --split dev_test \
    --max_examples 30 \
    --temp_draft 0.7 \
    --temp_critique 0.7 \
    --temp_refine 0.7 \
    --output results/temp_comparison_default.json

echo ""
echo "=== PART 2: Main Evaluation (Full 100 examples) ==="
echo "For HW Questions 2-5 - accuracy plots, conditional probs, model comparison"
echo ""

# Configuration 1: MMLU with Qwen3-4B (full evaluation)
echo "Running Main Config 1: MMLU + Qwen3-4B (temp=0.7)"
python src/self_refine.py \
    --dataset mmlu_med \
    --model Qwen/Qwen3-4B \
    --split dev_test \
    --temp_draft 0.7 \
    --temp_critique 0.7 \
    --temp_refine 0.7 \
    --output results/mmlu_qwen3-4b.json

# Configuration 2: MMLU with Qwen3-0.6B
echo ""
echo "Running Main Config 2: MMLU + Qwen3-0.6B (temp=0.7)"
python src/self_refine.py \
    --dataset mmlu_med \
    --model Qwen/Qwen3-0.6B \
    --split dev_test \
    --temp_draft 0.7 \
    --temp_critique 0.7 \
    --temp_refine 0.7 \
    --output results/mmlu_qwen3-0.6b.json

# Configuration 3: Graph with Qwen3-4B
echo ""
echo "Running Main Config 3: Graph + Qwen3-4B (temp=0.7)"
python src/self_refine.py \
    --dataset graph \
    --model Qwen/Qwen3-4B \
    --split dev_test \
    --temp_draft 0.7 \
    --temp_critique 0.7 \
    --temp_refine 0.7 \
    --output results/graph_qwen3-4b.json

# Configuration 4: Graph with Qwen3-0.6B
echo ""
echo "Running Main Config 4: Graph + Qwen3-0.6B (temp=0.7)"
python src/self_refine.py \
    --dataset graph \
    --model Qwen/Qwen3-0.6B \
    --split dev_test \
    --temp_draft 0.7 \
    --temp_critique 0.7 \
    --temp_refine 0.7 \
    --output results/graph_qwen3-0.6b.json

echo ""
echo "================================"
echo "All experiments completed!"
echo ""
echo "Temperature comparison results (30 examples):"
echo "  - results/temp_comparison_low.json (temp=0.3)"
echo "  - results/temp_comparison_default.json (temp=0.7)"
echo ""
echo "Main evaluation results (100 examples):"
echo "  - results/mmlu_qwen3-4b.json"
echo "  - results/mmlu_qwen3-0.6b.json"
echo "  - results/graph_qwen3-4b.json"
echo "  - results/graph_qwen3-0.6b.json"
echo "================================"
