# Self-Refine Implementation

This directory contains the implementation of the Self-Refine algorithm for the Graph and MMLU Medical tasks.

## Structure

```
self_refine/
├── src/
│   ├── dataset.py       # Dataset handlers for Graph and MMLU
│   └── self_refine.py   # Main self-refine implementation
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── run_self_refine.sh  # Example run scripts
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run self-refine on MMLU Med dataset:
```bash
python src/self_refine.py \
    --dataset mmlu_med \
    --model Qwen/Qwen3-4B \
    --split dev_test \
    --output results_mmlu_qwen3-4b.json
```

Run self-refine on Graph dataset:
```bash
python src/self_refine.py \
    --dataset graph \
    --model Qwen/Qwen3-0.6B \
    --split dev_test \
    --output results_graph_qwen3-0.6b.json
```

### Temperature Settings

You can specify different temperatures for draft, critique, and refine stages:

```bash
python src/self_refine.py \
    --dataset mmlu_med \
    --model Qwen/Qwen3-4B \
    --temp_draft 0.8 \
    --temp_critique 0.2 \
    --temp_refine 0.7 \
    --output results_custom_temp.json
```

### Testing on Subset

Test on a small number of examples:
```bash
python src/self_refine.py \
    --dataset mmlu_med \
    --model Qwen/Qwen3-4B \
    --max_examples 10 \
    --output test_results.json
```

## Arguments

- `--dataset`: Dataset to use (`graph` or `mmlu_med`)
- `--model`: HuggingFace model path (e.g., `Qwen/Qwen3-4B`)
- `--split`: Dataset split (default: `dev_test`)
- `--max_examples`: Limit number of examples for testing
- `--output`: Output JSON file path
- `--temp_draft`: Temperature for draft generation (default: 0.7)
- `--temp_critique`: Temperature for critique generation (default: 0.3)
- `--temp_refine`: Temperature for refinement (default: 0.7)

## Output Format

The output JSON contains:
```json
[
  {
    "example_id": 0,
    "question": "...",
    "ground_truth": "...",
    "iterations": [
      {
        "iteration": 1,
        "type": "draft",
        "raw_response": "...",
        "parsed_answer": "...",
        "is_correct": true
      },
      {
        "iteration": 2,
        "type": "refine",
        "feedback": "...",
        "raw_response": "...",
        "parsed_answer": "...",
        "is_correct": true
      },
      ...
    ]
  },
  ...
]
```

## Algorithm

Self-Refine follows this process for each example:

1. **Draft (Iteration 1)**: Generate initial answer
2. **Critique & Refine (Iterations 2-4)**:
   - Generate feedback on current answer
   - Refine answer based on feedback
   - Repeat for 3 refinement iterations

Total: 4 iterations per example (1 draft + 3 refinements)

## Analysis

After running, the script prints:
- Accuracy at each iteration
- Best-so-far accuracy (if correct at any iteration)

### Detailed Analysis

Use the analysis script to compute detailed metrics and generate plots:

```bash
python src/analyze_results.py \
    --input results/mmlu_qwen3-4b_default.json \
    --output_dir analysis/mmlu_qwen3-4b
```

This generates:
- **accuracy_plot.png**: Shows accuracy at each iteration and best-so-far accuracy
- **conditional_probs.png**: Visualizes P(correct_{i+1} | correct_i) and P(correct_{i+1} | incorrect_i)
- Console output with:
  - Iteration-by-iteration accuracy
  - Conditional probabilities
  - Example transitions (incorrect→correct and correct→incorrect)

### Analyzing Multiple Configurations

```bash
# Analyze all result files
for result_file in results/*.json; do
    basename=$(basename "$result_file" .json)
    python src/analyze_results.py \
        --input "$result_file" \
        --output_dir "analysis/$basename"
done
```
