import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import llm_blender
from evaluate import load 
from tqdm import tqdm

os.environ["HF_HOME"] = "/usr1/data/mingqia2"
'''
Compute logprobs: python rerank_outputs.py --scores qwen3_4b qwen3_13b
'''
def compute_and_save_scores(input_file, output_file, scores_to_compute=['qwen3_4b', 'qwen3_13b']):
    """Compute scores for all candidates and save to output file.

    Args:
        input_file (str): Path to input JSON file (e.g., 'all_results_processed.json')
        output_file (str): Path to output JSON file
        scores_to_compute (list): List of score types to compute. Options:
            - 'qwen3_4b': Log probability using Qwen3-4B
            - 'qwen3_13b': Log probability using Qwen3-14B
            - 'r_scalar': Reward model scalar score (to be implemented)
            - 'r_pairwise': Reward model pairwise score (to be implemented)
            - 'mbr_bleu': MBR with BLEU (to be implemented)
            - 'mbr_bert': MBR with BERTScore (to be implemented)
    """

    # Load the data
    with open(input_file, 'r') as f:
        all_results = json.load(f)

    print(f"Computing scores for {len(all_results['results'])} queries...")
    print(f"Scores to compute: {scores_to_compute}")

    # Compute Qwen3-4B log probabilities
    if 'qwen3_4b' in scores_to_compute:
        print("\nLoading Qwen3-4B...")
        model_4b = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to('cuda:0')
        model_4b.eval()

        print("Processing with Qwen3-4B...")
        for result in tqdm(all_results['results'], desc="Qwen3-4B"):
            # Get prompt tokens and candidates
            prompt_tokens = result['prompt']['prompt_tokens']
            candidates = result['candidates']

            # Compute log probs using pre-generated token IDs
            log_probs = compute_model_prob(candidates, prompt_tokens, model_4b)

            # Save scores
            for i, candidate in enumerate(result['candidates']):
                candidate['scores']['qwen3_4b'] = log_probs[i]

        # Free memory
        del model_4b
        torch.cuda.empty_cache()

    # Compute Qwen3-14B log probabilities
    if 'qwen3_13b' in scores_to_compute:
        print("\nLoading Qwen3-14B...")
        model_14b = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B", dtype=torch.bfloat16).to('cuda:0')
        model_14b.eval()

        print("Processing with Qwen3-14B...")
        for result in tqdm(all_results['results'], desc="Qwen3-14B"):
            # Get prompt tokens and candidates
            prompt_tokens = result['prompt']['prompt_tokens']
            candidates = result['candidates']

            # Compute log probs using pre-generated token IDs
            log_probs = compute_model_prob(candidates, prompt_tokens, model_14b)

            # Save scores
            for i, candidate in enumerate(result['candidates']):
                candidate['scores']['qwen3_13b'] = log_probs[i]

        # Free memory
        del model_14b
        torch.cuda.empty_cache()

    # Compute scalar reward model scores
    if 'r_scalar' in scores_to_compute:
        print("\nLoading Skywork reward model...")
        reward_tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
            dtype=torch.bfloat16,
            device_map='cuda:0'
        )
        reward_model.eval()

        print("Processing with r_scalar...")
        for result in tqdm(all_results['results'], desc="r_scalar"):
            # Get prompt text and candidate outputs
            prompt = result['prompt']['intermediate_prompt']
            outputs = [candidate['generated_text'] for candidate in result['candidates']]

            # Compute scalar rewards
            rewards = compute_scalar_reward(outputs, prompt, reward_model, reward_tokenizer)

            # Save scores
            for i, candidate in enumerate(result['candidates']):
                candidate['scores']['r_scalar'] = rewards[i]

        # Free memory
        del reward_model
        del reward_tokenizer
        torch.cuda.empty_cache()

    # Compute pairwise reward model scores
    if 'r_pairwise' in scores_to_compute:
        print("\nLoading PairRM model...")

        blender = llm_blender.Blender()
        blender.loadranker("llm-blender/PairRM")

        # Calculate total number of comparisons
        num_queries = len(all_results['results'])
        comparisons_per_query = 50 * 49 // 2  # 1225
        total_comparisons = num_queries * comparisons_per_query

        print(f"Processing with r_pairwise ({comparisons_per_query} comparisons per query, {total_comparisons} total)...")

        # Create overall progress bar
        with tqdm(total=total_comparisons, desc="r_pairwise comparisons") as pbar:
            for result in all_results['results']:
                # Get prompt text and candidate outputs
                prompt = result['prompt']['intermediate_prompt']
                outputs = [candidate['generated_text'] for candidate in result['candidates']]

                # Compute pairwise scores
                pairwise_scores = compute_pairwise_reward(outputs, prompt, blender, pbar)

                # Save scores
                for i, candidate in enumerate(result['candidates']):
                    candidate['scores']['r_pairwise'] = pairwise_scores[i]

        # Free memory
        del blender
        torch.cuda.empty_cache()

    # Compute MBR with BLEU scores
    if 'mbr_bleu' in scores_to_compute:
        print("\nLoading BLEU metric...")
        bleu_metric = load("bleu")

        print("Computing mbr_bleu scores...")
        for result in tqdm(all_results['results'], desc="mbr_bleu"):
            # Get candidate outputs
            outputs = [candidate['generated_text'] for candidate in result['candidates']]

            # Compute MBR BLEU scores
            mbr_scores = compute_mbr_bleu(outputs, bleu_metric)

            # Save scores
            for i, candidate in enumerate(result['candidates']):
                candidate['scores']['mbr_bleu'] = mbr_scores[i]

    # Compute MBR with BERTScore
    if 'mbr_bert' in scores_to_compute:
        print("\nLoading BERTScorer...")
        from bert_score import BERTScorer
        bert_scorer = BERTScorer(lang='en', rescale_with_baseline=True, model_type='bert-base-uncased')

        print("Computing mbr_bert scores...")
        for result in tqdm(all_results['results'], desc="mbr_bert"):
            # Get candidate outputs
            outputs = [candidate['generated_text'] for candidate in result['candidates']]

            # Compute MBR BERTScore
            mbr_scores = compute_mbr_bertscore(outputs, bert_scorer)

            # Save scores
            for i, candidate in enumerate(result['candidates']):
                candidate['scores']['mbr_bert'] = mbr_scores[i]

        # Free memory
        del bert_scorer
        torch.cuda.empty_cache()

    # Save to output file
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Also save a compact scores-only file
    scores_only_file = output_file.replace('.json', '_scores_only.json')
    print(f"Saving compact scores to {scores_only_file}...")

    compact_scores = []
    for result in all_results['results']:
        query_scores = {
            'question_id': result['question_id'],
            'candidates': [candidate['scores'] for candidate in result['candidates']]
        }
        compact_scores.append(query_scores)

    with open(scores_only_file, 'w') as f:
        json.dump(compact_scores, f, indent=2)

    print("Done!")

def compute_model_prob(candidates, prompt_tokens, model):
    """Compute the log probability of each candidate output given the prompt.

    Args:
        candidates (list[dict]): List of candidate dictionaries with 'token_ids' and 'full_chat' keys
        prompt_tokens (list): The prompt token IDs
        model: The loaded model to use for scoring

    Returns:
        list[float]: Log probability for each candidate output
    """
    log_probs_list = []

    # Process each candidate
    for candidate in candidates:
        in_toks = prompt_tokens
        out_toks = candidate['token_ids']
        all_toks = candidate['full_chat']['full_chat_tokens']

        # Create model input
        model_input = torch.tensor([all_toks]).to('cuda:0')

        # Get logits and compute log probabilities
        with torch.no_grad():
            logits = model(model_input).logits
            log_probs = F.log_softmax(logits[0], dim=-1)

        # Accumulate log probs for output tokens
        cumulative_logprobs = 0.0
        for i in range(len(in_toks) - 1, len(in_toks) - 1 + len(out_toks)):
            # Check bounds to avoid index error
            if i + 1 >= len(all_toks):
                break
            actual_token = all_toks[i + 1]
            cumulative_logprobs += log_probs[i, actual_token].item()

        log_probs_list.append(cumulative_logprobs)

    return log_probs_list

def load_and_merge_scores(input_file, scores_file):
    """Load main data and merge in scores from a separate scores file.

    Args:
        input_file (str): Path to main JSON file
        scores_file (str): Path to scores-only JSON file

    Returns:
        dict: Merged data with scores
    """
    with open(input_file, 'r') as f:
        all_results = json.load(f)

    with open(scores_file, 'r') as f:
        compact_scores = json.load(f)

    print(f"Loaded {len(all_results['results'])} queries from {input_file}")
    print(f"Loaded {len(compact_scores)} queries from {scores_file}")

    # Merge scores back into main data
    for idx, (result, scores_data) in enumerate(zip(all_results['results'], compact_scores)):
        if result['question_id'] != scores_data['question_id']:
            print(f"Warning: Question ID mismatch at index {idx}: {result['question_id']} != {scores_data['question_id']}")
            continue

        if len(result['candidates']) != len(scores_data['candidates']):
            print(f"Warning: Candidate count mismatch at question {result['question_id']}: {len(result['candidates'])} != {len(scores_data['candidates'])}")
            continue

        for candidate, scores in zip(result['candidates'], scores_data['candidates']):
            # Only update scores that are not None
            for score_name, score_value in scores.items():
                if score_value is not None:
                    candidate['scores'][score_name] = score_value

    print("Merge complete!")
    return all_results

def compute_mbr_bertscore(outputs, bert_scorer):
    """Compute MBR scores using BERTScore metric.

    Args:
        outputs (list[str]): List of candidate outputs
        bert_scorer: Pre-loaded BERTScorer object

    Returns:
        list[float]: MBR score for each output (average BERTScore F1 against all others)
    """
    n = len(outputs)
    mbr_scores = []

    # For each candidate, compute average BERTScore against all other candidates
    for i in range(n):
        total_bertscore = 0.0

        # Compare candidate i against all other candidates as references
        for j in range(n):
            if i != j:
                # Compute BERTScore with candidate j as reference
                # scorer.score returns (precision, recall, F1) tensors
                P, R, F1 = bert_scorer.score([outputs[i]], [outputs[j]])
                # Use F1 score
                total_bertscore += F1.item()

        # Average BERTScore F1 against all other candidates
        avg_bertscore = total_bertscore / (n - 1) if n > 1 else 0.0
        mbr_scores.append(avg_bertscore)

    return mbr_scores

def compute_mbr_bleu(outputs, bleu_metric):
    """Compute MBR scores using BLEU metric.

    Args:
        outputs (list[str]): List of candidate outputs
        bleu_metric: Pre-loaded BLEU metric from evaluate library

    Returns:
        list[float]: MBR score for each output (average BLEU against all others)
    """
    n = len(outputs)
    mbr_scores = []

    # For each candidate, compute average BLEU score against all other candidates
    for i in range(n):
        total_bleu = 0.0

        # Compare candidate i against all other candidates as references
        for j in range(n):
            if i != j:
                # Compute BLEU score with candidate j as reference
                result = bleu_metric.compute(
                    predictions=[outputs[i]],
                    references=[[outputs[j]]]
                )
                total_bleu += result['bleu']

        # Average BLEU score against all other candidates
        avg_bleu = total_bleu / (n - 1) if n > 1 else 0.0
        mbr_scores.append(avg_bleu)

    return mbr_scores

def compute_pairwise_reward(outputs, prompt, blender, pbar=None, batch_size=128):
    """Compute pairwise rewards for outputs using PairRM with batching.

    Args:
        outputs (list[str]): List of output texts to score
        prompt (str): The input prompt text
        blender: Pre-loaded llm_blender.Blender with PairRM loaded
        pbar: Optional tqdm progress bar to update
        batch_size (int): Number of comparisons to batch together

    Returns:
        list[int]: Number of wins for each output (how many others it beats)
    """
    n = len(outputs)
    wins = [0] * n

    # Collect all pairs to compare
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))

    # Process pairs in batches for efficiency
    for batch_start in range(0, len(pairs), batch_size):
        batch_end = min(batch_start + batch_size, len(pairs))
        batch_pairs = pairs[batch_start:batch_end]

        # Prepare batch inputs
        batch_prompts = [prompt] * len(batch_pairs)
        batch_candidates_A = [outputs[i] for i, j in batch_pairs]
        batch_candidates_B = [outputs[j] for i, j in batch_pairs]

        # Batch compare - much faster than one at a time
        comparison_results = blender.compare(
            batch_prompts,
            batch_candidates_A,
            batch_candidates_B
        )

        # Update wins based on results
        for (i, j), result in zip(batch_pairs, comparison_results):
            if result:
                wins[i] += 1
            else:
                wins[j] += 1

        # Update progress bar
        if pbar is not None:
            pbar.update(len(batch_pairs))

    return wins

def compute_scalar_reward(outputs, prompt, model, tokenizer):
    """Compute scalar rewards for outputs using Skywork reward model.

    Args:
        outputs (list[str]): List of output texts to score
        prompt (str): The input prompt text
        model: Pre-loaded reward model
        tokenizer: Pre-loaded tokenizer

    Returns:
        list[float]: Scalar reward for each output
    """
    rewards = []

    # Process each output
    for output in outputs:
        # Format as conversation
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output}
        ]

        # Apply chat template
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to('cuda:0')

        # Get reward score
        with torch.no_grad():
            outputs_model = model(**inputs)
            # The reward model outputs a scalar score
            reward = outputs_model.logits[0][0].item()

        rewards.append(reward)

    return rewards

def subsample_candidates(all_results, n_samples, seed=42):
    """
    Subsample n candidates from each query.

    Args:
        all_results: Full results dictionary with 'metadata' and 'results'
        n_samples: Number of candidates to subsample per query
        seed: Random seed for reproducibility

    Returns:
        Subsampled results dictionary with same structure
    """
    np.random.seed(seed)

    subsampled_results = {
        'metadata': all_results['metadata'].copy(),
        'results': []
    }

    for result in all_results['results']:
        # Random subsample of candidate indices
        all_indices = np.arange(len(result['candidates']))
        sampled_indices = np.random.choice(all_indices, size=n_samples, replace=False)

        # Create new result with subsampled candidates
        new_result = {
            'question_id': result['question_id'],
            'prompt': result['prompt'],
            'question_data': result['question_data'],
            'candidates': [result['candidates'][i] for i in sampled_indices]
        }

        subsampled_results['results'].append(new_result)

    return subsampled_results

def recompute_dependent_scores(results_dict, methods_to_compute=['r_pairwise', 'mbr_bleu', 'mbr_bert']):
    """
    Recompute scores that depend on the candidate set (e.g., after subsampling).

    These methods need recomputation because their scores depend on comparisons
    between all candidates in the set:
    - r_pairwise: Pairwise comparisons between all candidates
    - mbr_bleu: Average BLEU score against all other candidates
    - mbr_bert: Average BERTScore against all other candidates

    Args:
        results_dict: Results dictionary to update
        methods_to_compute: List of method names to recompute
                           Default: ['r_pairwise', 'mbr_bleu', 'mbr_bert']

    Returns:
        Updated results dictionary
    """
    if 'r_pairwise' in methods_to_compute:
        print("\nRecomputing pairwise rewards...")
        blender = llm_blender.Blender()
        blender.loadranker("llm-blender/PairRM")

        for result in tqdm(results_dict['results'], desc="r_pairwise"):
            prompt = result['prompt']['intermediate_prompt']
            outputs = [candidate['generated_text'] for candidate in result['candidates']]

            pairwise_scores = compute_pairwise_reward(outputs, prompt, blender)

            for i, candidate in enumerate(result['candidates']):
                candidate['scores']['r_pairwise'] = pairwise_scores[i]

        del blender
        torch.cuda.empty_cache()

    if 'mbr_bleu' in methods_to_compute:
        print("\nRecomputing MBR BLEU scores...")
        bleu_metric = load("bleu")

        for result in tqdm(results_dict['results'], desc="mbr_bleu"):
            outputs = [candidate['generated_text'] for candidate in result['candidates']]
            mbr_scores = compute_mbr_bleu(outputs, bleu_metric)

            for i, candidate in enumerate(result['candidates']):
                candidate['scores']['mbr_bleu'] = mbr_scores[i]

    if 'mbr_bert' in methods_to_compute:
        print("\nRecomputing MBR BERTScore...")
        from bert_score import BERTScorer
        bert_scorer = BERTScorer(lang='en', rescale_with_baseline=True, model_type='bert-base-uncased')

        for result in tqdm(results_dict['results'], desc="mbr_bert"):
            outputs = [candidate['generated_text'] for candidate in result['candidates']]
            mbr_scores = compute_mbr_bertscore(outputs, bert_scorer)

            for i, candidate in enumerate(result['candidates']):
                candidate['scores']['mbr_bert'] = mbr_scores[i]

        del bert_scorer
        torch.cuda.empty_cache()

    return results_dict

def subsample_and_recompute_cli(input_file, output_file, n_samples, methods_to_compute=['r_pairwise', 'mbr_bleu', 'mbr_bert'], seed=42):
    """
    CLI function to subsample candidates and recompute dependent scores.

    Args:
        input_file: Path to input JSON file with full results
        output_file: Path to save subsampled results
        n_samples: Number of candidates to subsample per query
        methods_to_compute: List of methods to recompute
                           Default: ['r_pairwise', 'mbr_bleu', 'mbr_bert']
        seed: Random seed for reproducibility
    """
    print(f"Loading results from {input_file}...")
    with open(input_file, 'r') as f:
        all_results = json.load(f)

    print(f"Subsampling to {n_samples} candidates per query...")
    subsampled = subsample_candidates(all_results, n_samples, seed=seed)

    print(f"Recomputing dependent scores: {methods_to_compute}")
    subsampled = recompute_dependent_scores(subsampled, methods_to_compute=methods_to_compute)

    print(f"Saving subsampled results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(subsampled, f, indent=2)

    print("Done!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compute reranking scores for candidate outputs')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Original compute_and_save_scores command
    compute_parser = subparsers.add_parser('compute', help='Compute scores for all candidates')
    compute_parser.add_argument('--input', type=str, default='all_results_processed.json',
                        help='Input JSON file path')
    compute_parser.add_argument('--output', type=str, default='all_results_with_scores.json',
                        help='Output JSON file path')
    compute_parser.add_argument('--scores', nargs='+',
                        default=['qwen3_4b', 'qwen3_13b'],
                        choices=['qwen3_4b', 'qwen3_13b', 'r_scalar', 'r_pairwise', 'mbr_bleu', 'mbr_bert'],
                        help='Scores to compute (space-separated)')

    # New subsample command
    subsample_parser = subparsers.add_parser('subsample', help='Subsample candidates and recompute dependent scores')
    subsample_parser.add_argument('--input', type=str, required=True,
                        help='Input JSON file path')
    subsample_parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')
    subsample_parser.add_argument('--n', type=int, required=True,
                        help='Number of candidates to subsample per query')
    subsample_parser.add_argument('--recompute', nargs='+',
                        default=['r_pairwise', 'mbr_bleu'],
                        choices=['r_pairwise', 'mbr_bleu', 'mbr_bert'],
                        help='Dependent scores to recompute (space-separated)')
    subsample_parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.command == 'compute':
        compute_and_save_scores(
            input_file=args.input,
            output_file=args.output,
            scores_to_compute=args.scores
        )
    elif args.command == 'subsample':
        subsample_and_recompute_cli(
            input_file=args.input,
            output_file=args.output,
            n_samples=args.n,
            methods_to_compute=args.recompute,
            seed=args.seed
        )
    else:
        parser.print_help()

    # python -m pip install "https://github.com/yuchenlin/LLM-Blender/archive/refs/heads/main.zip"