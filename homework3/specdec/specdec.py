"""
Speculative Decoding Implementation

Reference Papers:
1. Fast Inference from Transformers via Speculative Decoding (https://arxiv.org/pdf/2211.17192)
2. Accelerating Large Language Model Decoding with Speculative Sampling (https://arxiv.org/pdf/2302.01318)

This implementation follows Algorithm 2 from Paper 2 (DeepMind).
See Theorem 1 for why the rejection sampling preserves the target distribution.
"""
import torch
import transformers
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from termcolor import colored

torch.manual_seed(42)
transformers.set_seed(42)


class SamplingConfig:
    def __init__(self,
                 max_new_tokens: int=50,
                 temperature: float=1.0,
                 lookahead_K: int=3,
                 device: str = "cuda:0",
                 debug: bool = False):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.lookahead_K = lookahead_K
        self.debug = debug
        self.dtype = torch.bfloat16


class SpecDecSamplingConfig(SamplingConfig):
    def __init__(self,
                 target_name: str,
                 draft_name: str):
        super().__init__()
        self.target_name = target_name
        self.draft_name = draft_name


class SpeculativeDecoder:
    def __init__(self, config: SpecDecSamplingConfig):
        """
        Initialize target model, draft model, and tokenizer.
        Set models to eval mode.
        """
        self.config = config
        self.device = config.device
        self.temperature = config.temperature

        # TODO: Load models and tokenizer
        print(f"Loading target model: {config.target_name}")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            config.target_name,
            torch_dtype=config.dtype,
            device_map=self.device
        )
        self.target_model.eval()

        print(f"Loading draft model: {config.draft_name}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            config.draft_name,
            torch_dtype=config.dtype,
            device_map=self.device
        )
        self.draft_model.eval()

        # Load tokenizer (use target model's tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(config.target_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def max_fn(self, x):
        """Max function from paper 2 (f)_+"""
        # TODO: Max function from paper 2 (f)_+
        return torch.clamp(x, min=0.0)

    def get_distribution(self, logits, temperature, epsilon=1e-8):
        """Get probability distribution from logits"""
        # TODO: Softmax with temperature
        if temperature <= epsilon:
            # Greedy decoding
            probs = torch.zeros_like(logits)
            probs[..., logits.argmax(dim=-1)] = 1.0
            return probs
        # temperature scaling
        logits = logits / temperature
        # normalize
        probs = F.softmax(logits, dim=-1)
        return probs

    @torch.inference_mode()
    def ar_sample(self, model, tokenized_prompt, max_new_tokens, temperature=1.0):
        """
        Standard autoregressive sampling.
        Returns generated sequence and temperature temp-normalized probs."""
        # TODO: Implement autoregressive generation
        input_ids = tokenized_prompt.clone()
        all_logits = []

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            all_logits.append(logits)

            # Sample from distribution
            probs = self.get_distribution(logits, temperature)
            next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        generated = input_ids[:, tokenized_prompt.shape[1]:]  # Only return generated tokens
        return generated, torch.stack(all_logits, dim=1)  # [batch_size, max_new_tokens, vocab_size]

    @torch.inference_mode()
    # debug flags are left as is for easier debugging / seeing where outputs diverge
    def sd_sample(self, tokenized_prompt, max_new_tokens, lookahead, temperature):
        """
        Speculative decoding (Algorithm 2 from Paper 2).

        Args:
            tokenized_prompt: [batch_size, seq_len]
            max_new_tokens: Total tokens to generate
            lookahead: Number of speculative tokens (K)
            temperature: Sampling temperature

        Returns:
            generated_tokens: [batch_size, max_new_tokens]
            acceptance_rate: Fraction of draft tokens accepted
        """
        debug = self.config.debug
        bsz, n = tokenized_prompt.shape
        assert bsz == 1, 'Batch size should be 1'
        target_len = n + max_new_tokens

        # Metrics
        accepted_count = 0
        draft_token_num = 0
        n_orig = n

        # Start with the prompt
        prefix = tokenized_prompt.clone()

        while n < target_len:
            # TODO: HINT: you dont want to overshoot on max_new_tokens
            corrected_lookahead = min(lookahead, target_len - n)

            # TODO: Generate K draft tokens
            draft_outputs, draft_logits = self.ar_sample(
                self.draft_model,
                prefix,
                max_new_tokens=corrected_lookahead,
                temperature=temperature
            )
            draft_token_num += corrected_lookahead

            if debug:
                drafted_text = self.tokenizer.decode(draft_outputs[0],
                                                     skip_special_tokens=False)
                print(colored(f"Possible continuations: {drafted_text}", 'blue', 'on_black'))


            # TODO: Run target model on draft sequence to verify
            # Concatenate prefix with draft tokens
            prefix_with_draft = torch.cat([prefix, draft_outputs], dim=-1)  # [1, n + K]

            # Get target model predictions for all positions
            target_outputs = self.target_model(prefix_with_draft)
            target_logits = target_outputs.logits[:, n-1:, :]  # [1, K+1, vocab_size]
            # We need logits for positions n-1 to n+K-1 to get distributions for tokens at positions n to n+K

            # TODO: For each draft token, compute acceptance probability and accept/reject
            for t in range(corrected_lookahead):
                # Get draft token at position t
                draft_token = draft_outputs[0, t].unsqueeze(0)  # [1]

                # Get probability distributions
                p_target = self.get_distribution(target_logits[0, t], temperature)  # [vocab_size]
                p_draft = self.get_distribution(draft_logits[0, t], temperature)  # [vocab_size]

                # Compute acceptance probability: min(1, p_target(x) / p_draft(x))
                accept_prob = (p_target[draft_token] / (p_draft[draft_token] + 1e-10)).item()
                accept_prob = min(1.0, accept_prob)

                # Sample uniform random number
                r = torch.rand(1).item()

                # accept loop
                if r < accept_prob:
                    # Accept the draft token
                    prefix = torch.cat([prefix, draft_token.unsqueeze(0)], dim=-1)
                    n += 1
                    accepted_count += 1

                    if debug:
                        accepted_token = self.tokenizer.decode(draft_token)
                        print(f"Accepted token: '{accepted_token}'")

                # reject loop
                else:
                    # TODO: Reject and resample from adjusted distribution
                    # Resample from adjusted distribution: max(0, p_target - p_draft) / sum(max(0, p_target - p_draft))
                    adjusted_probs = self.max_fn(p_target - p_draft)
                    adjusted_probs = adjusted_probs / (adjusted_probs.sum() + 1e-10)

                    new_token = torch.multinomial(adjusted_probs, num_samples=1)  # [1]
                    prefix = torch.cat([prefix, new_token.unsqueeze(0)], dim=-1)
                    n += 1

                    if debug:
                        rejected_token = self.tokenizer.decode(draft_token)
                        new_token_text = self.tokenizer.decode(new_token[0])
                        print(colored(f"Rejected: {rejected_token}", 'red', 'on_black'))
                        print(colored(f"Replaced with: {new_token_text}", 'green', 'on_black'))
                    break
            else:
                # TODO: Sample bonus token if all accepted
                # All K tokens were accepted, sample one more from target model
                if n < target_len:
                    # Use the last target logit (position K) for bonus token
                    bonus_probs = self.get_distribution(target_logits[0, corrected_lookahead], temperature)
                    bonus_token = torch.multinomial(bonus_probs, num_samples=1)  # [1]
                    prefix = torch.cat([prefix, bonus_token.unsqueeze(0)], dim=-1)
                    n += 1
                    accepted_count += 1

                    if debug:
                        bonus_text = self.tokenizer.decode(bonus_token[0])
                        print(colored(f"Bonus token: {bonus_text}", 'green', 'on_black'))

        # Calculate acceptance rate
        acceptance_rate = accepted_count / draft_token_num if draft_token_num > 0 else 0.0
        generated = prefix[:, n_orig:]  # Only return generated tokens

        return generated, acceptance_rate
