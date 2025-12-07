#!/usr/bin/env python3
"""
Enhanced Modal deployment with integrated task router.

This system:
1. Routes incoming prompts to their task type (graph/mmlu/infobench)
2. Applies task-specific optimizations
3. Preserves original batch indices for correct result ordering
"""
import json
import os
import sys
from pathlib import Path
import threading
from concurrent.futures import Future

import modal
import torch

# Ensure local modules are importable when run by Modal
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

CONFIG_PATH = BASE_DIR / "config.json"
with open(CONFIG_PATH, "r") as _f:
    GLOBAL_CONFIG = json.load(_f)
DEFAULT_MAX_CONCURRENT = int(GLOBAL_CONFIG.get("deployment", {}).get("max_concurrent", 300))

from specdec import SpeculativeDecoder, SpecDecConfig
from infobench_batcher import ContinuousInfoBenchBatcher

app = modal.App("mingqia2-1")

# Define the image with required dependencies and bake in local code
image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers",
        "torch",
        "accelerate",
        "fastapi[standard]",
    )
    .add_local_dir(str(BASE_DIR), remote_path="/root", copy=True)
)


@app.cls(
    image=image,
    gpu="A100-80GB:2",
    scaledown_window=600,  # allow 10 min after the last request
    timeout=600,  # 10 minute timeout as required
)
@modal.concurrent(max_inputs=DEFAULT_MAX_CONCURRENT)  # configurable via config.json
class Model:
    @modal.enter()
    def load_model(self):
        """Load both the main model and the task router."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        print("Loading models...")

        # Prefer flash/mem-efficient attention where available
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)
            print("Flash/mem-efficient attention enabled via torch.backends.cuda")
        except Exception as e:
            print(f"Could not enable flash attention optimizations: {e}")

        cfg = GLOBAL_CONFIG
        model_cfg = cfg.get("models", {})
        spec_cfg = cfg.get("specdec", {})
        tasks_cfg = cfg.get("tasks", {})

        target_name = model_cfg.get("target_name", "Qwen/Qwen3-14B")
        draft_name = model_cfg.get("draft_name", "Qwen/Qwen3-0.6B")

        self.tokenizer = AutoTokenizer.from_pretrained(target_name, padding_side="left")
        # Use left padding/truncation for decoder-only generation
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        # Ensure pad token is set (required for batching)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            target_name,
            device_map="auto",
            dtype=torch.bfloat16,
        )

        # Draft model for speculative decoding
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_name,
            device_map="auto",
            dtype=torch.bfloat16,
        )

        # Initialize task router with the SAME model to save memory
        # The router will use the already-loaded model for classification
        from task_router import TaskRouter
        self.router = TaskRouter(
            model=draft_name,
            tokenizer=self.tokenizer,
            model_instance=self.draft_model
        )

        # Speculative decoding helper (single-example for now)
        self.specdec = SpeculativeDecoder(
            tokenizer=self.tokenizer,
            target_model=self.model,
            draft_model=self.draft_model,
            config=SpecDecConfig(
                lookahead_k=int(spec_cfg.get("lookahead", 3)),
                temperature=float(spec_cfg.get("default_temp", 0.7)),
                max_new_tokens=int(spec_cfg.get("max_new_tokens", 256)),
                device=str(self.model.device),
                debug=bool(spec_cfg.get("debug", False)),
            ),
        )

        # Task-specific knobs (enable spec dec where it helps long generations)
        self.task_settings = tasks_cfg
        # Backfill lookahead overrides for tasks missing them
        self.task_settings.setdefault("mmlu", {})
        self.task_settings["mmlu"].setdefault("lookahead", 1)
        self.task_settings.setdefault("graph", {})
        self.task_settings["graph"].setdefault("lookahead", int(spec_cfg.get("lookahead_graph", 3)))
        self.task_settings.setdefault("infobench", {})
        self.task_settings["infobench"].setdefault("lookahead", int(spec_cfg.get("lookahead_info", 4)))

        # Continuous batcher for InfoBench (disaggregated prefill/decode conceptually)
        self.infobench_batcher = ContinuousInfoBenchBatcher(
            process_fn=self._process_by_task,
            max_batch_size=int(os.getenv("INFOBATCH_MAX_BATCH", "16")),
            max_wait_ms=int(os.getenv("INFOBATCH_MAX_WAIT_MS", "25")),
        )

        print(f"Models loaded on device: {self.model.device}")

    def _process_by_task(self, prompts, max_tokens, temperature, task_type, top_p=None, top_k=None):
        """
        Apply task-specific processing optimizations.

        Args:
            prompts: List of prompts for this task
            max_tokens: Max tokens to generate
            temperature: Temperature setting
            top_p: nucleus sampling param (optional)
            top_k: top-k sampling param (optional)
            task_type: One of 'graph', 'mmlu', or 'infobench'

        Returns:
            Tuple of (generated_texts, prompt_lengths, outputs)
        """
        import torch

        settings = self.task_settings.get(task_type, {})

        effective_max_tokens = min(max_tokens, settings.get("max_tokens", max_tokens))
        effective_temperature = settings.get("temperature", temperature)
        do_sample = settings.get("do_sample", temperature > 0)
        use_specdec = settings.get("use_specdec", False)
        lookahead = settings.get("lookahead", self.specdec.config.lookahead_k)
        best_of_n = int(settings.get("best_of_n", 1))

        # If using best-of-n sampling, disable spec dec for simplicity
        if best_of_n > 1:
            use_specdec = False

        if task_type == "mmlu":
            # Short answers with small best-of-N voting
            do_sample = True
            effective_temperature = settings.get("temperature", 0.3)
            best_of_n = max(best_of_n, 3)
            effective_max_tokens = min(effective_max_tokens, 8)
            use_specdec = False
            # Append strict instruction for letter-only answer
            prompts = [
                f"{p}\n\nAnswer with a single letter (A, B, C, or D). Output only the letter."
                for p in prompts
            ]

        if task_type == "graph":
            do_sample = False  # greedy initial decode

        if use_specdec:
            generated_texts = []
            prompt_lengths = []
            outputs = []

            if len(prompts) == 1:
                gen_ids, acc_rate, prompt_len, prompt_ids = self.specdec.generate_one(
                    prompts[0],
                    max_new_tokens=effective_max_tokens,
                    temperature=effective_temperature,
                    lookahead=lookahead,
                )
                full_seq = torch.cat([prompt_ids.to(self.model.device), gen_ids.to(self.model.device)], dim=1).squeeze(0)
                outputs.append(full_seq)
                prompt_lengths.append(torch.tensor(prompt_len, device=self.model.device))
                generated_texts.append(self.tokenizer.decode(gen_ids[0], skip_special_tokens=True))
                print(f"[specdec] task={task_type} acc_rate={acc_rate:.2f} len={gen_ids.shape[1]}")
            else:
                gen_list, acc_rates, prompt_lens = self.specdec.generate_batch(
                    prompts,
                    max_new_tokens=effective_max_tokens,
                    temperature=effective_temperature,
                    lookahead=lookahead,
                )
                for gen_ids, acc_rate, prompt_len, prompt_text in zip(gen_list, acc_rates, prompt_lens, prompts):
                    # rebuild full sequence: prompt tokens + generated tokens
                    prompt_tokens = self.tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=2048,
                    ).input_ids[:, :prompt_len].to(self.model.device)
                    full_seq = torch.cat([prompt_tokens, gen_ids.to(self.model.device)], dim=1).squeeze(0)
                    outputs.append(full_seq)
                    prompt_lengths.append(torch.tensor(prompt_len, device=self.model.device))
                    generated_texts.append(self.tokenizer.decode(gen_ids[0], skip_special_tokens=True))
                    print(f"[specdec] task={task_type} acc_rate={acc_rate:.2f} len={gen_ids.shape[1]}")
        else:
            # Graph uses self-refinement (draft -> feedback -> refine)
            if task_type == "graph":
                generated_texts = []
                prompt_lengths = []
                outputs = []
                refine_max = settings.get("refine_max_tokens", 256)
                refine_temp = settings.get("refine_temperature", 0.0)

                for prompt in prompts:
                    # Tokenize prompt
                    prompt_tok = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=2048
                    )
                    prompt_ids = prompt_tok.input_ids.to(self.model.device)
                    prompt_attn = prompt_tok.attention_mask.to(self.model.device)
                    prompt_len = int(prompt_attn.sum().item())

                    # Draft (greedy)
                    with torch.no_grad():
                        draft_out = self.model.generate(
                            input_ids=prompt_ids,
                            attention_mask=prompt_attn,
                            max_new_tokens=effective_max_tokens,
                            temperature=1.0,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                    draft_gen = draft_out[0, prompt_len:]
                    draft_text = self.tokenizer.decode(draft_gen, skip_special_tokens=True)

                    # Single refinement using the model's own feedback implicitly in prompt
                    refine_prompt = (
                        "You are fixing the graph answer. Keep the same format and correct mistakes.\n\n"
                        f"Original prompt:\n{prompt}\n\n"
                        f"Previous answer:\n{draft_text}\n\n"
                        "Provide an improved answer. Format:\n"
                        "Path 1: [nodes] (weight: number)\n"
                        "Path 2: ...\n"
                        "Each path on its own line."
                    )
                    refine_tok = self.tokenizer(
                        refine_prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=2048
                    )
                    refine_input_ids = refine_tok.input_ids.to(self.model.device)
                    refine_attention = refine_tok.attention_mask.to(self.model.device)
                    with torch.no_grad():
                        refined = self.model.generate(
                            input_ids=refine_input_ids,
                            attention_mask=refine_attention,
                            max_new_tokens=refine_max,
                            temperature=refine_temp if refine_temp > 0 else 1.0,
                            do_sample=refine_temp > 0,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                    refine_prompt_len = int(refine_attention.sum().item())
                    refine_gen = refined[0, refine_prompt_len:]
                    refined_text = self._format_graph_answer(self.tokenizer.decode(refine_gen, skip_special_tokens=True))

                    generated_texts.append(refined_text)
                    prompt_lengths.append(torch.tensor(prompt_len, device=self.model.device))
                    full_seq = torch.cat([prompt_ids[0, :prompt_len], refine_gen.to(self.model.device)], dim=0)
                    outputs.append(full_seq)

                prompt_lengths = torch.stack(prompt_lengths)

            else:
                # Batch tokenization
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.model.device)

                # Track actual (non-padded) length of each prompt
                prompt_lengths = (inputs.input_ids != self.tokenizer.pad_token_id).sum(dim=1)

                generated_texts = []
                outputs = []

                for idx, prompt in enumerate(prompts):
                    prompt_len = prompt_lengths[idx]
                    prompt_ids = inputs.input_ids[idx:idx+1, :prompt_len]

                generation_kwargs = {
                    "max_new_tokens": effective_max_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "do_sample": do_sample,
                }
                if do_sample:
                    # Only include sampling args when sampling to avoid warnings
                    generation_kwargs["temperature"] = effective_temperature
                    if top_p is not None:
                        generation_kwargs["top_p"] = top_p
                    if top_k is not None:
                        generation_kwargs["top_k"] = top_k
                    if best_of_n > 1:
                        generation_kwargs["num_return_sequences"] = best_of_n

                    with torch.no_grad():
                        out = self.model.generate(
                            prompt_ids,
                            **generation_kwargs,
                            attention_mask=torch.ones_like(prompt_ids),
                        )

                    if best_of_n > 1 and do_sample and task_type == "infobench":
                        # out shape: [best_of_n, prompt_len + gen_len]; select via pairwise judge
                        candidates = [seq[prompt_len:] for seq in out]
                        cand_texts = [self.tokenizer.decode(c, skip_special_tokens=True) for c in candidates]
                        chosen_idx = self._pairwise_select(cand_texts, prompts[idx])
                        chosen = candidates[chosen_idx]
                        generated_texts.append(cand_texts[chosen_idx])
                        full_seq = torch.cat([prompt_ids.squeeze(0), chosen.to(self.model.device)], dim=0)
                        outputs.append(full_seq)
                    elif best_of_n > 1 and do_sample and task_type == "mmlu":
                        # Best-of-N short samples with majority vote over A/B/C/D
                        candidates = [seq[prompt_len:] for seq in out]
                        cand_texts = [self.tokenizer.decode(c, skip_special_tokens=True) for c in candidates]
                        letters = []
                        for t in cand_texts:
                            letter = self._extract_letter(t)
                            letters.append(letter)
                        # Majority vote
                        counts = {}
                        for idx_l, letter in enumerate(letters):
                            if letter:
                                counts.setdefault(letter, []).append(idx_l)
                        if counts:
                            # pick letter with most votes; tie -> first encountered
                            best_letter = max(counts.keys(), key=lambda k: len(counts[k]))
                            chosen_idx = counts[best_letter][0]
                        else:
                            chosen_idx = 0
                        chosen = candidates[chosen_idx]
                        chosen_letter = self._extract_letter(cand_texts[chosen_idx]) or ""
                        generated_texts.append(chosen_letter)
                        full_seq = torch.cat([prompt_ids.squeeze(0), chosen.to(self.model.device)], dim=0)
                        outputs.append(full_seq)
                    else:
                        seq = out[0]
                        gen_only = seq[prompt_len:]
                        decoded = self.tokenizer.decode(gen_only, skip_special_tokens=True)
                        if task_type == "mmlu":
                            decoded = self._extract_letter(decoded) or ""
                        generated_texts.append(decoded)
                        outputs.append(seq)

                prompt_lengths = prompt_lengths.to(self.model.device)

        return generated_texts, prompt_lengths, outputs

    def _pairwise_select(self, candidates, prompt_text):
        """
        Pairwise reranking using the draft model as a lightweight reward model.
        Returns index of best candidate.
        """
        wins = [0] * len(candidates)

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                judge_prompt = (
                    "Instruction and question:\n"
                    f"{prompt_text}\n\n"
                    f"Candidate A:\n{candidates[i]}\n\n"
                    f"Candidate B:\n{candidates[j]}\n\n"
                    "Which answer is better? Reply with 'A' or 'B' only."
                )
                judge_ids = self.tokenizer(
                    judge_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                )
                judge_input_ids = judge_ids.input_ids.to(self.model.device)
                judge_attn = judge_ids.attention_mask.to(self.model.device)
                with torch.no_grad():
                    judge_out = self.draft_model.generate(
                        input_ids=judge_input_ids,
                        attention_mask=judge_attn,
                        max_new_tokens=4,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                judge_prompt_len = int(judge_attn.sum().item())
                judge_gen = judge_out[0, judge_prompt_len:]
                judge_text = self.tokenizer.decode(judge_gen, skip_special_tokens=True).lower()
                if "a" in judge_text and "b" not in judge_text:
                    wins[i] += 1
                elif "b" in judge_text and "a" not in judge_text:
                    wins[j] += 1
                else:
                    # tie-break: favor first
                    wins[i] += 1

        best_idx = max(range(len(candidates)), key=lambda k: wins[k])
        return best_idx

    def _extract_letter(self, text):
        """Extract first occurrence of A/B/C/D (uppercase) from text."""
        for ch in text:
            c = ch.upper()
            if c in ("A", "B", "C", "D"):
                return c
        return None

    def _format_graph_answer(self, text):
        """
        Extract lines that look like Path N: [...] (weight: X).
        If none found, return original text.
        """
        import re
        pattern = re.compile(r"Path\s*\d+:\s*\[.*?\]\s*\(weight:\s*[\d\.]+\)", re.IGNORECASE)
        matches = pattern.findall(text)
        if matches:
            return "\n".join(matches)
        return text.strip()

    @modal.fastapi_endpoint(method="POST")
    def completions(self, request: dict):
        """
        OpenAI-compatible completions endpoint with task routing.

        The system automatically:
        1. Detects task type for each prompt in the batch
        2. Groups prompts by task type
        3. Applies task-specific optimizations
        4. Merges results back in original order
        """
        # Extract OpenAI-style parameters
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 2048)
        temperature = request.get("temperature", 0.7)
        top_p = request.get("top_p")
        top_k = request.get("top_k")

        # Handle both single prompt (string) and batch (list)
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        # Route prompts by task type
        grouped = self.router.route_batch_grouped(prompts)

        # Process each task group separately with task-specific optimizations
        results_by_task = {}
        all_prompt_lengths = {}
        all_outputs = {}

        for task_type, data in grouped.items():
            task_prompts = data['prompts']
            task_indices = data['indices']

            print(f"Processing {len(task_prompts)} {task_type} prompts at indices {task_indices}")

            # InfoBench uses continuous batcher (cross-request batching)
            if task_type == "infobench":
                futures = []
                for prompt in task_prompts:
                    fut = self.infobench_batcher.submit(prompt, max_tokens, temperature)
                    futures.append(fut)
                # Wait for results
                generated_texts = [f.result() for f in futures]
                # Token counts for usage accounting
                inputs = self.tokenizer(
                    task_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.model.device)
                prompt_lengths = (inputs.input_ids != self.tokenizer.pad_token_id).sum(dim=1)

                outputs = []
                for gen_text, prompt_len, input_ids_row in zip(generated_texts, prompt_lengths, inputs.input_ids):
                    comp_ids = self.tokenizer(
                        gen_text,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).input_ids.squeeze(0)
                    # Build a synthetic sequence for length/finish_reason accounting
                    seq_len = prompt_len.item() + comp_ids.numel()
                    seq = torch.full(
                        (seq_len + 1,),
                        fill_value=self.tokenizer.pad_token_id,
                        device=self.model.device,
                        dtype=torch.long,
                    )
                    seq[:prompt_len] = input_ids_row[:prompt_len]
                    seq[prompt_len:prompt_len + comp_ids.numel()] = comp_ids.to(self.model.device)
                    seq[-1] = self.tokenizer.eos_token_id
                    outputs.append(seq)
            else:
                # Apply task-specific processing
                generated_texts, prompt_lengths, outputs = self._process_by_task(
                    task_prompts, max_tokens, temperature, task_type, top_p=top_p, top_k=top_k
                )

            # Store results with original indices
            for i, orig_idx in enumerate(task_indices):
                results_by_task[orig_idx] = generated_texts[i]
                all_prompt_lengths[orig_idx] = prompt_lengths[i]
                all_outputs[orig_idx] = outputs[i]

            if task_type == "mmlu":
                print(f"[mmlu debug] outputs for indices {task_indices}: {generated_texts}")

        # Merge results back to original order
        ordered_texts = []
        ordered_prompt_lengths = []
        ordered_outputs = []

        for i in range(len(prompts)):
            ordered_texts.append(results_by_task[i])
            ordered_prompt_lengths.append(all_prompt_lengths[i])
            ordered_outputs.append(all_outputs[i])

        # Create choices for each completion in original order
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for i, (generated_text, prompt_len, output) in enumerate(
            zip(ordered_texts, ordered_prompt_lengths, ordered_outputs)
        ):
            prompt_tokens = prompt_len.item()
            completion_tokens = len(output) - prompt_tokens

            if output[-1].item() == self.tokenizer.eos_token_id:
                finish_reason = "stop"
            else:
                finish_reason = "length"

            choices.append({
                "text": generated_text,
                "index": i,
                "finish_reason": finish_reason
            })

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

        # Return OpenAI-style response
        return {
            "choices": choices,
            "model": "andrewID-system-1",
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens
            }
        }
