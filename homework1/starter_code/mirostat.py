import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import os
import pandas as pd

def mirostat(model, tokenizer, prompt, max_length=50, device='cpu', temperature=1.0, target_ce=3.0, learning_rate=0.1):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    mu = 2 * target_ce  # Initial mu value / "maximal surprisal"

    # TODO: YOUR CODE HERE -- additional variable init
    # We will not be checking this section for correctness,
    # But you will probably eventually want to set up some 
    # extra variables here for plotting metrics.
    # Our advice is to fill out the other sections first!
    
    N = len(tokenizer.get_vocab())  # Vocabulary size
    tracking_data = {
        'k_values': [],
        's_hat_values': [],
        'mu_values': [],
        'surprisal_errors': [],
        'per_token_perplexity': [],
        'logit_distributions': defaultdict(list),  # Store at steps 1, 10, 100
        'generated_tokens': [],
        'step_numbers': []
    }
    for step in range(max_length):
        with torch.no_grad():
            logits = model(input_ids).logits[:, -1, :]
            adjusted_logits = logits / temperature
            adjusted_probs = torch.softmax(adjusted_logits, dim=-1)
            
            sorted_logits, sorted_inds = torch.sort(adjusted_logits, descending = True)
        
        # TODO: YOUR CODE HERE -- Estimate Zipf's exponent
        # Following Basu et al, use m=100 (i.e. use only the top 100 tokens(' diffs) to estimate the exponent)
        # Refer to Equation 30 https://arxiv.org/pdf/2007.14966#equation.C.30 for pointers
        
        # Store logit distributions at specific steps
        if step + 1 in [1, 10, 100]:
            # Store top 1000 logits for visualization
            top_logits = sorted_logits[0, :1000].cpu().numpy()
            tracking_data['logit_distributions'][step + 1] = top_logits
        
        m = 100 
        
        # Compute the ratios for MMSE estimation
        # We need log ratios of consecutive probabilities: b_i = log(p_i / p_{i+1})
        numerator = 0.0
        denominator = 0.0
        
        for i in range(m - 1):
            t_i = math.log((i + 2) / (i + 1))  # log((i+2)/(i+1)) since we're 0-indexed
            b_i = float(sorted_logits[0, i] - sorted_logits[0, i + 1])  # log(p_i) - log(p_{i+1}) = log(p_i/p_{i+1})
            
            numerator += t_i * b_i
            denominator += t_i * t_i

        s_hat = numerator / denominator
        # Add bounds checking to prevent math domain errors
        if abs(s_hat) < 1e-8:
            s_hat = 1e-8 if s_hat >= 0 else -1e-8
            
        # TODO: YOUR CODE HERE -- Compute k using Zipf exponent
        epsilon = s_hat - 1
        n_to_neg_eps = math.pow(N, -epsilon)
        denominator_k = 1 - n_to_neg_eps
        
        numerator_k = epsilon * math.pow(2, target_ce)

        try:
            k_raw = math.pow(numerator_k / denominator_k, 1.0 / s_hat)
            k = max(1, int(round(k_raw)))  
        except (ValueError, OverflowError, ZeroDivisionError):
            k = 1  # Fallback on numerical errors

        # Ensure k doesn't exceed the number of available tokens to prevent indexing errors
        vocab_size = sorted_logits.size(1)
        k = min(k, vocab_size)

        # top k sampling
        topk_logits = sorted_logits[:,0:k]
        topk_inds = sorted_inds[:,0:k]
        topk_probs = torch.softmax(topk_logits, dim=1)
        next_tok = topk_inds[0, torch.multinomial(topk_probs, num_samples=1)]
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_token_id:
            break

        # TODO: YOUR CODE HERE -- Compute surprisal error and adjust mu accordingly
        # Compute surprisal error and adjust mu accordingly
        # Surprisal of the chosen token
        log_probs = torch.log_softmax(adjusted_logits, dim=-1)
        chosen_token_log_prob = log_probs[0, next_tok.item()]
        surprisal = -float(chosen_token_log_prob)  # -log(p) = surprisal
        per_token_perplexity = math.exp(surprisal)
        
        err = surprisal - target_ce
        mu = mu - learning_rate * err
        
        # Store tracking data
        tracking_data['k_values'].append(k)
        tracking_data['s_hat_values'].append(s_hat)
        tracking_data['mu_values'].append(mu)
        tracking_data['surprisal_errors'].append(err)
        tracking_data['per_token_perplexity'].append(per_token_perplexity)
        tracking_data['generated_tokens'].append(tokenizer.decode([next_tok.item()]))
        tracking_data['step_numbers'].append(step + 1)
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    sequence_perplexity = math.exp(np.mean([math.log(p) for p in tracking_data['per_token_perplexity']]))
    
    return generated_text, tracking_data, sequence_perplexity

def run_experiments():
    """
    Run the comprehensive Mirostat experiments as specified
    """
    # Experiment parameters
    tau_values = [2.0, 3.0, 4.0]  # Three different tau values
    prompts = [
        "Once upon a time,",
        "The capital of France is,"
    ]
    model_names = [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.1-8B"
    ]
    
    print("Running Mirostat experiments...")
    print("=" * 50)
    
    results = []
    
    for model_name in model_names:
        print(f"\nLoading model: {model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                      cache_dir="/usr1/data/models_cache")
            model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                         cache_dir="/usr1/data/models_cache",
                                                         torch_dtype=torch.float16)
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            
            for prompt in prompts:
                for tau in tau_values:
                    print(f"\nExperiment: τ={tau}, prompt='{prompt[:20]}...'")
                    
                    generated_text, tracking_data, seq_perplexity = mirostat(
                        model, tokenizer, prompt, 
                        max_length=128, device=device, 
                        temperature=0.9, target_ce=tau, learning_rate=0.1
                    )
                    
                    # Calculate statistics
                    per_token_stats = {
                        'mean': np.mean(tracking_data['per_token_perplexity']),
                        'median': np.median(tracking_data['per_token_perplexity']),
                        'std': np.std(tracking_data['per_token_perplexity'])
                    }
                    
                    result = {
                        'model': model_name,
                        'prompt': prompt,
                        'tau': tau,
                        'generated_text': generated_text,
                        'sequence_perplexity': seq_perplexity,
                        'per_token_stats': per_token_stats,
                        'tracking_data': tracking_data
                    }
                    
                    results.append(result)
                    
                    # Print results
                    print(f"Generated text: {generated_text}")
                    print(f"Sequence-level perplexity: {seq_perplexity:.3f}")
                    print(f"Per-token perplexity - Mean: {per_token_stats['mean']:.3f}, "
                          f"Median: {per_token_stats['median']:.3f}, Std: {per_token_stats['std']:.3f}")
                    
        except Exception as e:
            print(f"Error with model {model_name}: {str(e)}")
            print("Continuing with available models...")
            continue
    
    return results

def save_results_to_csv(results, output_file="mirostat_results.csv"):
    """
    Save the experimental results to a CSV file
    """
    csv_data = []

    for result in results:
        row = {
            'Model': result['model'],
            'Prompt': result['prompt'],
            'Tau': result['tau'],
            'Generated_Text': result['generated_text'],
            'Sequence_Perplexity': result['sequence_perplexity'],
            'Mean_Per_Token_Perplexity': result['per_token_stats']['mean'],
            'Median_Per_Token_Perplexity': result['per_token_stats']['median'],
            'Std_Per_Token_Perplexity': result['per_token_stats']['std']
        }
        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    return df

def create_visualizations(results, save_plots=True, output_dir="mirostat_plots"):
    """
    Create comprehensive visualizations of the Mirostat experiments
    """
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    # Group results by model
    model_results = {}
    for result in results:
        model_name = result['model']
        if model_name not in model_results:
            model_results[model_name] = []
        model_results[model_name].append(result)

    # Create separate plots for each model
    for model_name, model_result_list in model_results.items():
        model_display_name = model_name.split('/')[-1]

        # Create a figure with the four required subplots for this model
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Mirostat Analysis: {model_display_name} - k, ŝ, μ, and Surprisal Error vs Generation Step', fontsize=16)
    
        # Plot 1: k values over time
        ax1 = axes[0, 0]
        for result in model_result_list:
            steps = result['tracking_data']['step_numbers']
            k_vals = result['tracking_data']['k_values']
            label = f"τ={result['tau']}"
            ax1.plot(steps, k_vals, marker='o', markersize=2, label=label, alpha=0.7)
        ax1.set_xlabel('Generation Step')
        ax1.set_ylabel('k value')
        ax1.set_title('k values over generation')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
    
        # Plot 2: s_hat values over time
        ax2 = axes[0, 1]
        for result in model_result_list:
            steps = result['tracking_data']['step_numbers']
            s_hat_vals = result['tracking_data']['s_hat_values']
            label = f"τ={result['tau']}"
            ax2.plot(steps, s_hat_vals, marker='o', markersize=2, label=label, alpha=0.7)
        ax2.set_xlabel('Generation Step')
        ax2.set_ylabel('ŝ value')
        ax2.set_title('Zipf exponent (ŝ) over generation')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        # Plot 3: mu values over time
        ax3 = axes[1, 0]
        for result in model_result_list:
            steps = result['tracking_data']['step_numbers']
            mu_vals = result['tracking_data']['mu_values']
            label = f"τ={result['tau']}"
            ax3.plot(steps, mu_vals, marker='o', markersize=2, label=label, alpha=0.7)
        ax3.set_xlabel('Generation Step')
        ax3.set_ylabel('μ value')
        ax3.set_title('Maximum surprisal (μ) over generation')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Surprisal error over time
        ax4 = axes[1, 1]
        for result in model_result_list:
            steps = result['tracking_data']['step_numbers']
            error_vals = result['tracking_data']['surprisal_errors']
            label = f"τ={result['tau']}"
            ax4.plot(steps, error_vals, marker='o', markersize=2, label=label, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Generation Step')
        ax4.set_ylabel('Surprisal Error')
        ax4.set_title('Surprisal error over generation')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            # Create model-specific filenames
            safe_model_name = model_display_name.replace(".", "_")
            plt.savefig(f"{output_dir}/mirostat_analysis_main_{safe_model_name}.pdf", dpi=300, bbox_inches='tight', format='pdf')
            plt.savefig(f"{output_dir}/mirostat_analysis_main_{safe_model_name}.png", dpi=300, bbox_inches='tight', format='png')
            print(f"Main analysis plot for {model_display_name} saved to {output_dir}/mirostat_analysis_main_{safe_model_name}.pdf and .png")

        plt.show()
    
    # Create logit distribution plots for a single selected combination
    # Pick the first available result (you can modify this selection criteria)
    if results:
        selected_result = results[0]  # Select first result
        print(f"\nLogit distribution analysis:")
        print(f"Selected model: {selected_result['model']}")
        print(f"Selected prompt: '{selected_result['prompt']}'")
        print(f"Selected τ: {selected_result['tau']}")
        create_logit_distribution_plots(selected_result)

def create_logit_distribution_plots(result, save_plots=True, output_dir="mirostat_plots"):
    """
    Create logit distribution plots for steps 1, 10, and 100
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    model_name = result["model"].split("/")[-1]
    prompt_preview = result["prompt"][:30] + "..." if len(result["prompt"]) > 30 else result["prompt"]
    fig.suptitle(f'Logit Distributions - Model: {model_name}, τ={result["tau"]}, Prompt: "{prompt_preview}"', fontsize=14)
    
    steps_to_plot = [1, 10, 100]
    
    for i, step in enumerate(steps_to_plot):
        ax = axes[i]
        if step in result['tracking_data']['logit_distributions']:
            logits = result['tracking_data']['logit_distributions'][step]
            ranks = np.arange(1, len(logits) + 1)
            
            ax.plot(ranks, logits, 'b-', alpha=0.7, linewidth=1)
            ax.set_xlabel('Token Rank')
            ax.set_ylabel('Log Probability')
            ax.set_title(f'Step {step}')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'Step {step}\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    if save_plots:
        # Create a safe filename
        model_name = result["model"].split("/")[-1].replace(".", "_")
        tau_str = str(result["tau"]).replace(".", "_")
        filename = f"logit_distributions_tau{tau_str}_{model_name}"
        
        plt.savefig(f"{output_dir}/{filename}.pdf", dpi=300, bbox_inches='tight', format='pdf')
        plt.savefig(f"{output_dir}/{filename}.png", dpi=300, bbox_inches='tight', format='png')
        print(f"Logit distribution plots saved to {output_dir}/{filename}.pdf and .png")
    
    plt.show()


if __name__ == "__main__":
    '''
    model_name = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = "Once upon a time,"
    result = mirostat(model, tokenizer, prompt, max_length=256, device=device, temperature=1.0, target_ce=3.0, learning_rate=0.1)
    print(result)
    '''

    results = run_experiments()

    # Save results to CSV
    df = save_results_to_csv(results, "mirostat_results.csv")

    # Create visualizations
    create_visualizations(results, save_plots=True, output_dir="mirostat_plots")

    print("\n1. Per-token Perplexity Statistics:")
    for result in results:
        stats = result['per_token_stats']
        print(f"   τ={result['tau']}, {result['model'].split('/')[-1]}: "
                f"Mean={stats['mean']:.2f}, Median={stats['median']:.2f}, Std={stats['std']:.2f}")

    print("\n2. Sequence-level Perplexity:")
    for result in results:
        print(f"   τ={result['tau']}, {result['model'].split('/')[-1]}: "
                f"{result['sequence_perplexity']:.3f}")

    print(f"\n3. CSV Summary:")
    print("All results have been saved to mirostat_results.csv with the following columns:")
    print("- Model, Prompt, Tau, Generated_Text, Sequence_Perplexity")
    print("- Mean_Per_Token_Perplexity, Median_Per_Token_Perplexity, Std_Per_Token_Perplexity")
        