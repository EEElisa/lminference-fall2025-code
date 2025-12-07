#!/usr/bin/env python3
"""
Analysis script for Self-Refine results.
Computes conditional probabilities and generates plots.
"""

import json
import argparse
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt


def load_results(filepath: str) -> List[Dict]:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compute_iteration_accuracy(results: List[Dict]) -> Dict[int, float]:
    """Compute accuracy at each iteration."""
    accuracies = {}
    num_iterations = len(results[0]['iterations']) if results else 0

    for i in range(num_iterations):
        correct = sum(1 for r in results if r['iterations'][i]['is_correct'])
        accuracies[i+1] = correct / len(results) if results else 0

    return accuracies


def compute_best_so_far_accuracy(results: List[Dict]) -> Dict[int, float]:
    """Compute best accuracy up to each iteration."""
    best_accuracies = {}
    num_iterations = len(results[0]['iterations']) if results else 0

    for i in range(num_iterations):
        # Count examples correct in at least one iteration up to i
        correct = sum(1 for r in results
                     if any(r['iterations'][j]['is_correct'] for j in range(i+1)))
        best_accuracies[i+1] = correct / len(results) if results else 0

    return best_accuracies


def compute_conditional_probabilities(results: List[Dict]) -> Dict[str, List[float]]:
    """
    Compute P(correct_{i+1} | correct_i) and P(correct_{i+1} | incorrect_i).

    Returns:
        Dictionary with 'p_correct_given_correct' and 'p_correct_given_incorrect' lists
    """
    num_iterations = len(results[0]['iterations']) if results else 0

    p_correct_given_correct = []
    p_correct_given_incorrect = []

    for i in range(num_iterations - 1):
        # Count transitions
        correct_to_correct = 0
        correct_to_incorrect = 0
        incorrect_to_correct = 0
        incorrect_to_incorrect = 0

        for r in results:
            curr_correct = r['iterations'][i]['is_correct']
            next_correct = r['iterations'][i+1]['is_correct']

            if curr_correct and next_correct:
                correct_to_correct += 1
            elif curr_correct and not next_correct:
                correct_to_incorrect += 1
            elif not curr_correct and next_correct:
                incorrect_to_correct += 1
            else:  # not curr_correct and not next_correct
                incorrect_to_incorrect += 1

        # Compute probabilities
        total_correct = correct_to_correct + correct_to_incorrect
        total_incorrect = incorrect_to_correct + incorrect_to_incorrect

        p_cc = correct_to_correct / total_correct if total_correct > 0 else 0
        p_ic = incorrect_to_correct / total_incorrect if total_incorrect > 0 else 0

        p_correct_given_correct.append(p_cc)
        p_correct_given_incorrect.append(p_ic)

    return {
        'p_correct_given_correct': p_correct_given_correct,
        'p_correct_given_incorrect': p_correct_given_incorrect
    }


def find_transition_examples(results: List[Dict],
                             from_state: bool,
                             to_state: bool,
                             iteration: int,
                             limit: int = 3) -> List[Dict]:
    """
    Find examples of specific transitions.

    Args:
        from_state: True for correct, False for incorrect at iteration i
        to_state: True for correct, False for incorrect at iteration i+1
        iteration: The iteration number (0-indexed) to check
        limit: Maximum number of examples to return
    """
    examples = []

    for r in results:
        if iteration >= len(r['iterations']) - 1:
            continue

        curr_correct = r['iterations'][iteration]['is_correct']
        next_correct = r['iterations'][iteration+1]['is_correct']

        if curr_correct == from_state and next_correct == to_state:
            examples.append({
                'example_id': r['example_id'],
                'question': r['question'][:200] + '...' if len(r['question']) > 200 else r['question'],
                'iteration_i': r['iterations'][iteration],
                'iteration_i_plus_1': r['iterations'][iteration+1]
            })

            if len(examples) >= limit:
                break

    return examples


def plot_accuracy_curves(accuracies: Dict[int, float],
                         best_accuracies: Dict[int, float],
                         output_path: str = 'accuracy_plot.png'):
    """Plot accuracy and best-so-far accuracy curves."""
    iterations = sorted(accuracies.keys())
    acc_values = [accuracies[i] for i in iterations]
    best_acc_values = [best_accuracies[i] for i in iterations]

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, acc_values, 'o-', label='Accuracy at iteration i', linewidth=2, markersize=8)
    plt.plot(iterations, best_acc_values, 's--', label='Best accuracy so far', linewidth=2, markersize=8)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Self-Refine: Accuracy over Iterations', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(iterations)
    plt.ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy plot to {output_path}")


def plot_conditional_probabilities(cond_probs: Dict[str, List[float]],
                                   output_path: str = 'conditional_probs.png'):
    """Plot conditional probability curves."""
    iterations = list(range(1, len(cond_probs['p_correct_given_correct']) + 1))
    p_cc = cond_probs['p_correct_given_correct']
    p_ic = cond_probs['p_correct_given_incorrect']

    plt.figure(figsize=(10, 6))
    x_labels = [f'{i}→{i+1}' for i in iterations]
    x_pos = np.arange(len(x_labels))

    width = 0.35
    plt.bar(x_pos - width/2, p_cc, width, label='P(correct_{i+1} | correct_i)', alpha=0.8)
    plt.bar(x_pos + width/2, p_ic, width, label='P(correct_{i+1} | incorrect_i)', alpha=0.8)

    plt.xlabel('Iteration Transition', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Conditional Probabilities of Correctness', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, x_labels)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved conditional probability plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Self-Refine results")
    parser.add_argument('--input', type=str, required=True, help='Input results JSON file')
    parser.add_argument('--output_dir', type=str, default='analysis', help='Output directory for plots')

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.input}")
    results = load_results(args.input)
    print(f"Loaded {len(results)} examples")

    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # Compute metrics
    print("\n" + "="*60)
    print("ITERATION ACCURACY")
    print("="*60)
    accuracies = compute_iteration_accuracy(results)
    for i, acc in accuracies.items():
        print(f"Iteration {i}: {acc:.2%}")

    print("\n" + "="*60)
    print("BEST-SO-FAR ACCURACY")
    print("="*60)
    best_accuracies = compute_best_so_far_accuracy(results)
    for i, acc in best_accuracies.items():
        print(f"Up to Iteration {i}: {acc:.2%}")

    print("\n" + "="*60)
    print("CONDITIONAL PROBABILITIES")
    print("="*60)
    cond_probs = compute_conditional_probabilities(results)

    for i, (p_cc, p_ic) in enumerate(zip(cond_probs['p_correct_given_correct'],
                                         cond_probs['p_correct_given_incorrect'])):
        print(f"Iteration {i+1} → {i+2}:")
        print(f"  P(correct_{i+2} | correct_{i+1}) = {p_cc:.3f}")
        print(f"  P(correct_{i+2} | incorrect_{i+1}) = {p_ic:.3f}")

    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    acc_plot = os.path.join(args.output_dir, 'accuracy_plot.png')
    plot_accuracy_curves(accuracies, best_accuracies, acc_plot)

    cond_plot = os.path.join(args.output_dir, 'conditional_probs.png')
    plot_conditional_probabilities(cond_probs, cond_plot)

    # Find example transitions
    print("\n" + "="*60)
    print("EXAMPLE TRANSITIONS")
    print("="*60)

    # Find incorrect → correct transitions
    print("\nExamples of INCORRECT → CORRECT (refinement helps):")
    ic_examples = find_transition_examples(results, False, True, 0, limit=2)
    for ex in ic_examples:
        print(f"\n  Example {ex['example_id']}:")
        print(f"  Question: {ex['question']}")
        print(f"  Iteration 1 (incorrect): {ex['iteration_i']['parsed_answer']}")
        print(f"  Iteration 2 (correct): {ex['iteration_i_plus_1']['parsed_answer']}")

    # Find correct → incorrect transitions
    print("\n\nExamples of CORRECT → INCORRECT (refinement hurts):")
    ci_examples = find_transition_examples(results, True, False, 0, limit=2)
    for ex in ci_examples:
        print(f"\n  Example {ex['example_id']}:")
        print(f"  Question: {ex['question']}")
        print(f"  Iteration 1 (correct): {ex['iteration_i']['parsed_answer']}")
        print(f"  Iteration 2 (incorrect): {ex['iteration_i_plus_1']['parsed_answer']}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
