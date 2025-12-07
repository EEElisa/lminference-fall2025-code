#!/usr/bin/env python3
"""
Lightweight speculative decoding helper adapted from homework3/specdec.

Key changes for service use:
- Accept injected tokenizer/target/draft models (no internal loading)
- Focus on single-sequence decoding (batch size 1) for simplicity
- Returns generated token ids and acceptance rate

This can be extended to batched speculative decoding later; the current
implementation keeps the interface small to drop into the Modal service.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


@dataclass
class SpecDecConfig:
    lookahead_k: int = 3
    temperature: float = 0.7
    max_new_tokens: int = 256
    device: Optional[str] = None
    dtype: torch.dtype = torch.bfloat16
    debug: bool = False


class SpeculativeDecoder:
    """
    Single-example speculative decoder (Algorithm 2 style).

    This version is intentionally minimal: it assumes batch size 1 and
    reuses pre-loaded models/tokenizer from the calling service.
    """

    def __init__(self, tokenizer, target_model, draft_model, config: SpecDecConfig):
        self.tokenizer = tokenizer
        self.target_model = target_model
        self.draft_model = draft_model
        self.config = config

        # Derive device if not specified
        if self.config.device is None:
            # Try to read from target model
            try:
                self.config.device = str(next(target_model.parameters()).device)
            except Exception:
                self.config.device = "cpu"

        # Track acceptance rates for adaptive lookahead
        self.recent_acceptance_rates = []
        self.max_history = 100  # Keep last 100 acceptance rates

    def _update_acceptance_rate(self, rate: float):
        """Track acceptance rate for adaptive lookahead."""
        self.recent_acceptance_rates.append(rate)
        if len(self.recent_acceptance_rates) > self.max_history:
            self.recent_acceptance_rates.pop(0)

    def get_avg_acceptance_rate(self) -> float:
        """Get average acceptance rate from recent history."""
        if not self.recent_acceptance_rates:
            return 0.5  # Default neutral value
        return sum(self.recent_acceptance_rates) / len(self.recent_acceptance_rates)

    def get_adaptive_lookahead(self, requested_k: int) -> int:
        """
        Adjust lookahead dynamically based on recent acceptance rates.

        Strategy:
        - High acceptance (>0.7): Increase lookahead (more drafts accepted)
        - Medium acceptance (0.4-0.7): Use requested lookahead
        - Low acceptance (<0.4): Decrease lookahead (drafts often rejected)

        Benefits:
        - Saves computation when draft quality is poor
        - Maximizes throughput when draft quality is good
        - Adapts to different prompt types automatically
        """
        avg_rate = self.get_avg_acceptance_rate()

        if avg_rate > 0.75:
            # High acceptance - increase lookahead by 1-2
            adjusted_k = min(requested_k + 2, 8)
        elif avg_rate > 0.60:
            # Good acceptance - increase lookahead by 1
            adjusted_k = min(requested_k + 1, 8)
        elif avg_rate > 0.40:
            # Medium acceptance - use requested
            adjusted_k = requested_k
        elif avg_rate > 0.25:
            # Low acceptance - decrease by 1
            adjusted_k = max(requested_k - 1, 2)
        else:
            # Very low acceptance - decrease by 2
            adjusted_k = max(requested_k - 2, 1)

        return adjusted_k

    def _max_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp negatives to zero ( (f)_+ in the paper )."""
        return torch.clamp(x, min=0.0)

    def _get_distribution(self, logits: torch.Tensor, temperature: float, epsilon: float = 1e-8) -> torch.Tensor:
        """
        Convert logits to probabilities with temperature.
        """
        if temperature <= epsilon:
            probs = torch.zeros_like(logits)
            probs[..., logits.argmax(dim=-1)] = 1.0
            return probs
        logits = logits / temperature
        return F.softmax(logits, dim=-1)

    @torch.inference_mode()
    def _ar_sample(self, model, tokenized_prompt: torch.Tensor, max_new_tokens: int, temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard autoregressive sampling (batch size 1).

        Returns:
            generated_ids: [1, max_new_tokens]
            logits_history: [1, max_new_tokens, vocab]
        """
        input_ids = tokenized_prompt.clone()
        all_logits = []

        for _ in range(max_new_tokens):
            attn = torch.ones_like(input_ids, device=self.config.device)
            outputs = model(input_ids, attention_mask=attn)
            logits = outputs.logits[:, -1, :]  # [1, vocab]
            all_logits.append(logits)

            probs = self._get_distribution(logits, temperature)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        generated = input_ids[:, tokenized_prompt.shape[1]:]
        return generated, torch.stack(all_logits, dim=1)

    @torch.inference_mode()
    def _ar_sample_batch(
        self,
        model,
        sequences: List[torch.Tensor],
        max_new_tokens: int,
        temperature: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched autoregressive sampling for variable-length inputs.

        Args:
            model: HF causal LM
            sequences: list of 1D LongTensor token ids per example
            max_new_tokens: number of tokens to draft
            temperature: sampling temperature

        Returns:
            draft_tokens: [B, max_new_tokens]
            draft_logits: [B, max_new_tokens, vocab]
        """
        # Pad to batch
        padded = pad_sequence(sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attn = (padded != self.tokenizer.pad_token_id).long()
        device = self.config.device
        padded = padded.to(device)
        attn = attn.to(device)

        draft_tokens = []
        draft_logits = []

        for _ in range(max_new_tokens):
            outputs = model(padded, attention_mask=attn)
            # Logits at each sequence's last valid position
            last_idx = attn.sum(dim=1) - 1  # [B]
            logits = outputs.logits[torch.arange(padded.size(0)), last_idx]  # [B, vocab]
            draft_logits.append(logits)

            probs = self._get_distribution(logits, temperature)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]
            draft_tokens.append(next_tok)

            # Append to padded sequences
            padded = torch.cat([padded, next_tok.unsqueeze(1)], dim=1)
            attn = torch.cat([attn, torch.ones_like(next_tok).unsqueeze(1)], dim=1)

        draft_tokens = torch.stack(draft_tokens, dim=1)  # [B, K]
        draft_logits = torch.stack(draft_logits, dim=1)  # [B, K, vocab]
        return draft_tokens, draft_logits

    @torch.inference_mode()
    def generate_one(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        lookahead: Optional[int] = None,
        use_adaptive_lookahead: bool = True,
    ) -> Tuple[torch.Tensor, float, int, torch.Tensor]:
        """
        Run speculative decoding for a single prompt (string).

        Args:
            prompt: The input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            lookahead: Base lookahead value (will be adjusted if adaptive)
            use_adaptive_lookahead: Whether to use adaptive lookahead based on acceptance rates

        Returns:
            generated_ids: tensor of shape [1, generated_len]
            acceptance_rate: float
            prompt_length: number of prompt tokens (for accounting)
            prompt_ids: tensor of shape [1, prompt_len]
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        base_lookahead = lookahead or self.config.lookahead_k

        # Use adaptive lookahead if enabled
        if use_adaptive_lookahead:
            lookahead = self.get_adaptive_lookahead(base_lookahead)
            if self.config.debug and lookahead != base_lookahead:
                avg_rate = self.get_avg_acceptance_rate()
                print(f"[adaptive] Adjusted lookahead: {base_lookahead} → {lookahead} (avg_acc={avg_rate:.3f})")
        else:
            lookahead = base_lookahead

        # Tokenize
        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.config.device)
        input_ids = tokenized.input_ids
        prompt_len = input_ids.shape[1]

        bsz, cur_len = input_ids.shape
        if bsz != 1:
            # This helper currently assumes single-example decoding
            raise ValueError("SpeculativeDecoder.generate_one only supports batch size 1.")

        target_len = cur_len + max_new_tokens
        accepted_count = 0
        draft_token_num = 0
        prefix = input_ids

        while cur_len < target_len:
            corrected_lookahead = min(lookahead, target_len - cur_len)

            # Draft K tokens
            draft_outputs, draft_logits = self._ar_sample(
                self.draft_model,
                prefix,
                max_new_tokens=corrected_lookahead,
                temperature=temperature,
            )
            draft_token_num += corrected_lookahead

            # Target verification on prefix + draft
            prefix_with_draft = torch.cat([prefix, draft_outputs], dim=-1)
            attn_mask = torch.ones_like(prefix_with_draft, device=self.config.device)
            target_outputs = self.target_model(prefix_with_draft, attention_mask=attn_mask)
            # logits for positions cur_len-1 .. cur_len+K-1
            target_logits = target_outputs.logits[:, cur_len - 1 :, :]

            for t in range(corrected_lookahead):
                draft_token = draft_outputs[0, t].unsqueeze(0)  # [1]

                p_target = self._get_distribution(target_logits[0, t], temperature)
                p_draft = self._get_distribution(draft_logits[0, t], temperature)

                accept_prob = (p_target[draft_token] / (p_draft[draft_token] + 1e-10)).item()
                accept_prob = min(1.0, accept_prob)
                r = torch.rand(1).item()

                if r < accept_prob:
                    # Accept
                    prefix = torch.cat([prefix, draft_token.unsqueeze(0)], dim=-1)
                    cur_len += 1
                    accepted_count += 1
                else:
                    # Reject and resample from adjusted distribution
                    adjusted = self._max_fn(p_target - p_draft)
                    adjusted = adjusted / (adjusted.sum() + 1e-10)
                    new_token = torch.multinomial(adjusted, num_samples=1)  # [1]
                    prefix = torch.cat([prefix, new_token.unsqueeze(0)], dim=-1)
                    cur_len += 1
                    break
            else:
                # All draft tokens accepted, sample bonus token if room
                if cur_len < target_len:
                    bonus_probs = self._get_distribution(target_logits[0, corrected_lookahead], temperature)
                    bonus_token = torch.multinomial(bonus_probs, num_samples=1)  # [1]
                    prefix = torch.cat([prefix, bonus_token.unsqueeze(0)], dim=-1)
                    cur_len += 1
                    accepted_count += 1

        acceptance_rate = accepted_count / draft_token_num if draft_token_num > 0 else 0.0

        # Track acceptance rate for adaptive lookahead
        self._update_acceptance_rate(acceptance_rate)

        generated = prefix[:, prompt_len:]
        return generated, acceptance_rate, prompt_len, input_ids

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        lookahead: Optional[int] = None,
        use_adaptive_lookahead: bool = True,
    ) -> Tuple[List[torch.Tensor], List[float], List[int]]:
        """
        Batched speculative decoding for a list of prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            lookahead: Base lookahead value (will be adjusted if adaptive)
            use_adaptive_lookahead: Whether to use adaptive lookahead based on acceptance rates

        Returns:
            generated_ids_list: list of tensors (1, gen_len) per prompt
            acceptance_rates: list of floats
            prompt_lengths: list of ints
        """
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature if temperature is not None else self.config.temperature
        base_lookahead = lookahead or self.config.lookahead_k

        # Use adaptive lookahead if enabled
        if use_adaptive_lookahead:
            lookahead = self.get_adaptive_lookahead(base_lookahead)
            if self.config.debug and lookahead != base_lookahead:
                avg_rate = self.get_avg_acceptance_rate()
                print(f"[adaptive batch] Adjusted lookahead: {base_lookahead} → {lookahead} (avg_acc={avg_rate:.3f})")
        else:
            lookahead = base_lookahead

        # Tokenize with padding, then store per-example trimmed tensors
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        input_ids = tokenized.input_ids
        attn = tokenized.attention_mask
        prompt_lens = attn.sum(dim=1).tolist()

        sequences = [
            input_ids[i, :prompt_lens[i]].to(self.config.device)
            for i in range(len(prompts))
        ]

        remaining = [max_new_tokens for _ in prompts]
        accepted_counts = [0 for _ in prompts]
        draft_counts = [0 for _ in prompts]

        # Main loop while any sequence still needs tokens
        while any(r > 0 for r in remaining):
            active_indices = [i for i, r in enumerate(remaining) if r > 0]
            if not active_indices:
                break

            # Use a single lookahead for active batch, capped by smallest remaining
            global_k = min(lookahead, min(remaining[i] for i in active_indices))

            active_sequences = [sequences[i] for i in active_indices]
            draft_tokens, draft_logits = self._ar_sample_batch(
                self.draft_model,
                active_sequences,
                max_new_tokens=global_k,
                temperature=temperature,
            )

            # Build prefix+draft for target verification
            concat_sequences = []
            for idx, seq in enumerate(active_sequences):
                concat_sequences.append(torch.cat([seq, draft_tokens[idx]], dim=0))
            padded_concat = pad_sequence(concat_sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            concat_attn = (padded_concat != self.tokenizer.pad_token_id).long().to(self.config.device)
            padded_concat = padded_concat.to(self.config.device)

            target_outputs = self.target_model(padded_concat, attention_mask=concat_attn)
            target_logits = target_outputs.logits  # [B_active, L_max, vocab]

            # Per-sample acceptance
            for local_idx, orig_idx in enumerate(active_indices):
                start_len = active_sequences[local_idx].shape[0]
                remaining_tokens = remaining[orig_idx]
                local_k = min(global_k, remaining_tokens)

                for t in range(local_k):
                    if remaining[orig_idx] == 0:
                        break

                    draft_tok = draft_tokens[local_idx, t]
                    p_target = self._get_distribution(
                        target_logits[local_idx, start_len - 1 + t],
                        temperature,
                    )
                    p_draft = self._get_distribution(draft_logits[local_idx, t], temperature)

                    accept_prob = (p_target[draft_tok] / (p_draft[draft_tok] + 1e-10)).item()
                    accept_prob = min(1.0, accept_prob)
                    r = torch.rand(1).item()

                    if r < accept_prob:
                        sequences[orig_idx] = torch.cat(
                            [sequences[orig_idx], draft_tok.view(1).to(self.config.device)],
                            dim=0,
                        )
                        accepted_counts[orig_idx] += 1
                        remaining[orig_idx] -= 1
                    else:
                        adjusted = self._max_fn(p_target - p_draft)
                        adjusted = adjusted / (adjusted.sum() + 1e-10)
                        new_tok = torch.multinomial(adjusted, num_samples=1)
                        sequences[orig_idx] = torch.cat(
                            [sequences[orig_idx], new_tok.view(1).to(self.config.device)],
                            dim=0,
                        )
                        remaining[orig_idx] -= 1
                        break
                else:
                    # All local_k tokens accepted; add bonus if room
                    if remaining[orig_idx] > 0:
                        bonus_probs = self._get_distribution(
                            target_logits[local_idx, start_len - 1 + local_k],
                            temperature,
                        )
                        bonus_tok = torch.multinomial(bonus_probs, num_samples=1)
                        sequences[orig_idx] = torch.cat(
                            [sequences[orig_idx], bonus_tok.view(1).to(self.config.device)],
                            dim=0,
                        )
                        accepted_counts[orig_idx] += 1
                        remaining[orig_idx] -= 1

                draft_counts[orig_idx] += local_k

        generated_ids = []
        acceptance_rates = []
        for i, seq in enumerate(sequences):
            prompt_len = prompt_lens[i]
            gen = seq[prompt_len:].unsqueeze(0)
            generated_ids.append(gen)
            rate = accepted_counts[i] / draft_counts[i] if draft_counts[i] > 0 else 0.0
            acceptance_rates.append(rate)

            # Track each acceptance rate for adaptive lookahead
            self._update_acceptance_rate(rate)

        return generated_ids, acceptance_rates, prompt_lens
