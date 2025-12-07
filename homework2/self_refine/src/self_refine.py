# self_refine.py
import os, json, time, argparse, random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import dataset
from typing import List, Dict, Any, Tuple
from vllm import LLM, SamplingParams
from datasets import load_dataset

random.seed(42)
np.random.seed(42)

GraphHandler = dataset.GraphHandler
MMLUMedHandler = dataset.MMLUMedHandler

os.environ["HF_HOME"] = "/usr1/data/mingqia2"

"""NOTE: The repo includes a bare-bones scaffolds. 
It exists to help you start quickly. 
Please feel free to change your structure. 
Any clean, reproducible solution is acceptable.
"""

@dataclass
class RefineConfig:
    """Configuration for self-refine process."""
    # Adjust as needed
    model_path: str = None
    dtype: str = "bfloat16"
    # chat_template_kwargs: Dict[str, Any] = None


def chat(role: str, content: str) -> Dict[str, str]:
    """Format chat messages - adjust for your model's chat template"""
    return {"role": role, "content": content}

# These are for abstractions you can make them dataset specific or agnostic based on your design
def draft_prompt(question: str, handler_type: str) -> str:
    # NOTE: This model is the solver
    # You might wanna experiment with prompts and instruction for each dataset
    if handler_type == "graph":
        # Remove the submit_paths instruction and replace with parseable format
        # Cut off everything after "Return your answer by calling the submit_paths"
        if "Return your answer by calling the submit_paths" in question:
            question = question.split("Return your answer by calling the submit_paths")[0].strip()

        return f"""{question}

Solve this problem. Keep your reasoning concise and provide your final answer in this EXACT format:

Path 1: [0, 7, 15] (weight: 123)
Path 2: [0, 3, 5, 13, 1, 10, 15] (weight: 252)

IMPORTANT: List each path on its own line with format: Path N: [comma-separated nodes] (weight: number)"""
    elif handler_type == "mmlu_med":
        return f"""{question}

Think through this medical question carefully. After your reasoning, provide your final answer in this EXACT format:

Final Answer: A

(or B, C, or D as appropriate)"""
    else:
        return question

def feedback_prompt(question: str, attempt: str, handler_type: str) -> str:
    # Give the feedback on the attempt
    if handler_type == "graph":
        return f"""Question:
{question}

Your previous answer:
{attempt}

Please review your answer and provide constructive feedback. Consider:
1. Are all paths valid (no cycles, all edges exist)?
2. Are the path weights calculated correctly?
3. Are these actually the shortest paths?
4. Did you find all required paths?

Provide specific feedback on what might be wrong or could be improved:"""
    elif handler_type == "mmlu_med":
        return f"""Question:
{question}

Your previous answer:
{attempt}

Please critically evaluate your answer. Consider:
1. Is your medical reasoning sound?
2. Did you consider all relevant medical knowledge?
3. Are there any errors in your logic?
4. Could any of the other options be more correct?

Provide specific feedback on your answer:"""
    else:
        return f"""Question: {question}

Your answer: {attempt}

Please critique your answer. What could be improved?"""

def refine_prompt(question: str, attempt: str, feedback: str, handler_type: str) -> str:
    # Refine the attempt based on feedback
    if handler_type == "graph":
        # Remove the confusing submit_paths instruction
        if "Return your answer by calling the submit_paths" in question:
            question = question.split("Return your answer by calling the submit_paths")[0].strip()

        return f"""Question:
{question}

Your previous answer:
{attempt}

Feedback on your answer:
{feedback}

Based on this feedback, provide an improved answer. After your reasoning, you MUST provide your final answer using this EXACT format:

Path 1: [0, 7, 15] (weight: 123)
Path 2: [0, 3, 5, 13, 1, 10, 15] (weight: 252)

Each path must be on its own line with format: Path N: [comma-separated nodes] (weight: number)"""
    elif handler_type == "mmlu_med":
        return f"""Question:
{question}

Your previous answer:
{attempt}

Feedback on your answer:
{feedback}

Based on this feedback, provide your refined answer in this EXACT format:

Final Answer: A

(or B, C, or D as appropriate)"""
    else:
        return f"""Question: {question}

Previous answer: {attempt}

Feedback: {feedback}

Please provide a refined answer based on the feedback:"""


class Generator:
    "LLM Engine for generation, feedback, and refinement"
    # You can use transformers, hf piepeline, vllm, etc.
    def __init__(self, cfg: RefineConfig, handler_type: str, temp_draft: float = 0.7, temp_critique: float = 0.3, temp_refine: float = 0.7):
        self.cfg = cfg
        self.handler_type = handler_type
        self.temp_draft = temp_draft
        self.temp_critique = temp_critique
        self.temp_refine = temp_refine

        print(f"Loading model with vLLM: {cfg.model_path}")
        self.model = LLM(
            model=cfg.model_path,
            dtype=cfg.dtype,
            trust_remote_code=True,
            tensor_parallel_size=1, 
            seed=42  
        )
        self.tokenizer = self.model.get_tokenizer()


    def _gen(self, prompts: List[str], temperature: float = 0.7, max_new_tokens: int = 2048) -> List[str]:
        """generic generate function to do inference over a list of prompts"""
        # Format prompts with chat template
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                formatted = prompt
            formatted_prompts.append(formatted)

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            max_tokens=max_new_tokens,
            top_p=0.95,
            seed=42  
        )

        outputs = self.model.generate(formatted_prompts, sampling_params)

        # Extract responses
        responses = [output.outputs[0].text for output in outputs]
        return responses

    def draft(self, qs: List[str]) -> List[str]:
        """Generate initial drafts for questions"""
        prompts = [draft_prompt(q, self.handler_type) for q in qs]
        return self._gen(prompts, temperature=self.temp_draft)

    def feedback(self, qs_attempts: List[Tuple[str, str]]) -> List[str]:
        """Generate feedback for question-attempt pairs"""
        prompts = [feedback_prompt(q, a, self.handler_type) for q, a in qs_attempts]
        return self._gen(prompts, temperature=self.temp_critique)

    def refine(self, qs_attempts_feedback: List[Tuple[str, str, str]]) -> List[str]:
        """Generate refinements for the attempts based on feedback"""
        prompts = [refine_prompt(q, a, f, self.handler_type) for q, a, f in qs_attempts_feedback]
        return self._gen(prompts, temperature=self.temp_refine)


def run_self_refine(
    examples: List[Dict[str, Any]],
    handler: dataset.DatasetHandler,
    generator: Generator,
    config: RefineConfig,
    max_iterations: int = 4,
    output_file: str = None,
) -> List[Dict[str, Any]]:
    """
    Implement the self-refinement algorithm.

    Args:
        examples: List of dataset examples
        handler: Dataset handler
        generator: Generator instance for model inference
        config: Your configuration
        max_iterations: Total steps (1 draft + 3 refinements = 4)
        output_file: If provided, save results incrementally after each example

    Returns:
        - You might want to keep track of outputs at different stages so you can do interesting analysis later
    """
    results = []

    for idx, example in enumerate(examples):
        print(f"\n{'='*60}")
        print(f"Processing example {idx+1}/{len(examples)}")
        print(f"{'='*60}")

        # Format the question
        question = handler.format_question(example)
        ground_truth = handler.get_ground_truth(example)

        # Track all iterations
        iteration_history = []

        # Iteration 1: Initial draft
        print(f"\nIteration 1 (Draft)...")
        drafts = generator.draft([question])
        current_answer = drafts[0]

        # Parse and evaluate
        parsed = handler.parse_answer(current_answer)
        is_correct = handler.verify_answer(parsed, ground_truth)

        iteration_history.append({
            'iteration': 1,
            'type': 'draft',
            'raw_response': current_answer,
            'parsed_answer': parsed,
            'is_correct': is_correct
        })

        print(f"  Correct: {is_correct}")

        # Iterations 2-4: Refinement loop
        for iter_num in range(2, max_iterations + 1):
            print(f"\nIteration {iter_num} (Refine)...")

            # Get feedback
            feedbacks = generator.feedback([(question, current_answer)])
            feedback = feedbacks[0]

            # Refine based on feedback
            refined = generator.refine([(question, current_answer, feedback)])
            current_answer = refined[0]

            # Parse and evaluate
            parsed = handler.parse_answer(current_answer)
            is_correct = handler.verify_answer(parsed, ground_truth)

            iteration_history.append({
                'iteration': iter_num,
                'type': 'refine',
                'feedback': feedback,
                'raw_response': current_answer,
                'parsed_answer': parsed,
                'is_correct': is_correct
            })

            print(f"  Correct: {is_correct}")

        result_item = {
            'example_id': idx,
            'question': question,
            'ground_truth': ground_truth,
            'iterations': iteration_history
        }
        results.append(result_item)

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n  → Saved progress: {len(results)}/{len(examples)} examples")

    return results


def main():
    parser = argparse.ArgumentParser(description="Self-Refine Implementation")
    parser.add_argument("--dataset", type=str, required=True, choices=["graph", "mmlu_med"],
                        help="Dataset to use (graph or mmlu_med)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model path (e.g., Qwen/Qwen3-4B)")
    parser.add_argument("--split", type=str, default="dev_test",
                        help="Dataset split to use")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Maximum number of examples to process")
    parser.add_argument("--output", type=str, default="results.json",
                        help="Output file path")
    parser.add_argument("--temp_draft", type=float, default=0.7,
                        help="Temperature for draft generation")
    parser.add_argument("--temp_critique", type=float, default=0.3,
                        help="Temperature for critique generation")
    parser.add_argument("--temp_refine", type=float, default=0.7,
                        help="Temperature for refinement generation")

    args = parser.parse_args()
    
    HANDLERS = {
        "graph": GraphHandler,
        "mmlu_med": MMLUMedHandler,
    }

    # Initialize handler
    handler = HANDLERS[args.dataset]()

    # Initialize config
    config = RefineConfig(
        model_path=args.model,
        dtype="bfloat16"
    )

    # Initialize generator with temperature settings
    generator = Generator(
        config,
        handler_type=args.dataset,
        temp_draft=args.temp_draft,
        temp_critique=args.temp_critique,
        temp_refine=args.temp_refine
    )

    print(f"Loading dataset: {args.dataset}, split: {args.split}")

    dataset_name_map = {
        "graph": "graph_dev",
        "mmlu_med": "mmlu_med"
    }

    ds = load_dataset("vashistht/11763_datasets", dataset_name_map[args.dataset], split=args.split)
    examples = list(ds)

    if args.max_examples:
        examples = examples[:args.max_examples]

    print(f"Loaded {len(examples)} examples")

    # Run self-refine (with incremental saving)
    print("\nStarting self-refinement...")
    print(f"Results will be saved incrementally to: {args.output}\n")
    results = run_self_refine(examples, handler, generator, config, output_file=args.output)

    # Final save (already done incrementally, but this ensures completion)
    print(f"\n✓ All results saved to {args.output}")

    # Analyze the results
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    for iteration in range(1, 5):
        correct = sum(1 for r in results if r['iterations'][iteration-1]['is_correct'])
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        print(f"Iteration {iteration}: Accuracy = {correct}/{total} = {accuracy:.2%}")

    # Best accuracy so far (at least one correct)
    best_correct = sum(1 for r in results if any(it['is_correct'] for it in r['iterations']))
    best_acc = best_correct / len(results) if len(results) > 0 else 0
    print(f"\nBest-so-far Accuracy: {best_correct}/{len(results)} = {best_acc:.2%}")

    print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()