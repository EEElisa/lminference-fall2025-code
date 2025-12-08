#!/usr/bin/env python3
"""
System with task routing, speculative decoding, and task-specific optimizations.

This system:
1. Routes incoming prompts to their task type (graph/mmlu/infobench)
2. Applies task-specific handling:
   - Graph: Direct computation (Dijkstra's algorithm) - NO LLM needed!
   - MMLU: Batched best-of-N sampling with majority voting
   - InfoBench: Batched speculative decoding (optional self-refinement toggle)
3. Preserves original batch order

Task-specific optimizations:
- Graph: Direct computation using Dijkstra's algorithm (no tokens used!)
- MMLU: Batched best-of-5 with majority vote, temperature=0.3, max_tokens=50
- InfoBench: Batched speculative decoding only by default, temperature=0.7, max_tokens=1024

Batching strategy:
- All tasks benefit from batching when multiple prompts of the same type arrive
- Graph: No GPU batching (instant algorithmic computation)
- MMLU: Batches all N*M candidates together (M prompts × N samples each)
- InfoBench: Batches specdec generation across requests (single pass by default; refinement helper available)
"""
import concurrent.futures
import itertools
import time
import threading
import modal
from pathlib import Path

app = modal.App("mingqia2-system-3")

# Define the image with required dependencies
# Copy local modules to make them available in the Modal environment
BASE_DIR = Path(__file__).resolve().parent
image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers",
        "torch",
        "accelerate",
        "fastapi[standard]",
        "nvidia-ml-py3",  # NVML bindings for GPU monitoringt
    )
    .add_local_dir(str(BASE_DIR), remote_path="/root", copy=True)
)


@app.cls(
    image=image,
    gpu="A100-80GB:2",
    startup_timeout=300,
    scaledown_window=600,  # allow 10 min after the last request
    timeout=600,
)
@modal.concurrent(max_inputs=150)
class Model:
    @modal.enter()
    def load_model(self):
        """Load target model, draft model, task router, and speculative decoder."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        import sys
        from pathlib import Path

        print("Loading models...")

        # Model configuration
        target_model_name = "Qwen/Qwen3-4B"
        draft_model_name = "Qwen/Qwen3-0.6B"

        # Load shared tokenizer for router/graph/MMLU string handling
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = 'left'
        # Per-worker tokenizers (one per GPU) to avoid contention in InfoBench workers
        self.worker_tokenizers = []
        # Shared tokenizer lock (only for shared tokenizer uses)
        self._tok_lock = threading.Lock()

        # Initialize task router
        from task_router import TaskRouter
        self.router = TaskRouter(
            model=draft_model_name,
            tokenizer=self.tokenizer,
            model_instance=None
        )
        print("Task router initialized")

        # Initialize dual model pairs (one target/draft/specdec per GPU) with round-robin selection
        self._init_model_pairs(target_model_name, draft_model_name)
        print(f"Initialized {len(self.model_pairs)} model pairs (one per GPU) with round-robin routing")

        # Task-specific settings (MUST be set BEFORE starting batch workers)
        self.task_specdec_enabled = {
            "infobench": False,  # ❌ Disabled - low acceptance rate (0.02) makes it slower
            "graph": False,      # ❌ No specdec (uses direct computation instead)
            "mmlu": False,
        }

        self.task_settings = {
            "graph": {
                # Graph uses direct computation - no LLM settings needed
            },
            "mmlu": {
                "max_tokens": 256,
                "temperature": 0.3,
                "best_of_n": 3,  # reduced to cut per-request latency
            },
            "infobench": {
                "max_tokens": 512,
                "add_concise_prompt": True,
                "concise_token_limit": 150, # thinking token limit
            },
        }

        # Micro-batching config for server-side batching (per task)
        self._microbatch_settings = {
            # InfoBench (Long-sequence, no speculative decoding)
            "infobench_gpu0": {"flush_window": 10, "max_batch_size": 8},
            "infobench_gpu1": {"flush_window": 10, "max_batch_size": 8},
            # MMLU (Short-sequence, high-throughput)
            "mmlu_gpu0": {"flush_window": 3, "max_batch_size": 8},
            "mmlu_gpu1": {"flush_window": 3, "max_batch_size": 8},
        }
        
        # Separate queues: MMLU (one per GPU), InfoBench (one per GPU for parallel processing)
        self._batch_queues = {
            "infobench_gpu0": {"queue": [], "cond": threading.Condition(), "running": True},
            "infobench_gpu1": {"queue": [], "cond": threading.Condition(), "running": True},
            "mmlu_gpu0": {"queue": [], "cond": threading.Condition(), "running": True},
            "mmlu_gpu1": {"queue": [], "cond": threading.Condition(), "running": True},
        }

        # Per-GPU locks to prevent concurrent CUDA operations on the same GPU
        # Critical: Prevents CUBLAS_STATUS_EXECUTION_FAILED errors from thread contention
        self._gpu_locks = {
            0: threading.Lock(),
            1: threading.Lock(),
        }
        
        # Start background batchers
        # GPU 0 Workers
        threading.Thread(
            target=self._batch_worker, args=("infobench_gpu0", self._batch_queues["infobench_gpu0"], self.model_pairs[0]), daemon=True
        ).start()
        threading.Thread(
            target=self._batch_worker, args=("mmlu_gpu0", self._batch_queues["mmlu_gpu0"], self.model_pairs[0]), daemon=True
        ).start()
        
        # GPU 1 Workers
        threading.Thread(
            target=self._batch_worker, args=("infobench_gpu1", self._batch_queues["infobench_gpu1"], self.model_pairs[1]), daemon=True
        ).start()
        threading.Thread(
            target=self._batch_worker, args=("mmlu_gpu1", self._batch_queues["mmlu_gpu1"], self.model_pairs[1]), daemon=True
        ).start()

        # Round-robin counter and lock for InfoBench queue assignment removed, now using shortest-queue routing.

        # NOTE: Graph tasks use direct computation (no LLM needed!)
        # NOTE: InfoBench uses batched speculative decoding directly (specdec.generate_batch)
        # No separate batcher needed - specdec handles batching internally

        # Initialize GPU monitor
        self._init_gpu_monitor()
        self._log_gpu_stats(context="startup")

    def _init_model_pairs(self, target_model_name, draft_model_name):
        """Load one target/draft pair per GPU with per-pair tokenizer/specdec; round-robin selection."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from specdec import SpeculativeDecoder, SpecDecConfig
        import torch

        devices = [0, 1]
        self.model_pairs = []

        for device_id in devices:
            print(f"Loading target model on cuda:{device_id}")
            target_model = AutoModelForCausalLM.from_pretrained(
                target_model_name,
                device_map={"": device_id},
                dtype=torch.bfloat16,
            )
            print(f"Target model loaded on device: {target_model.device}")

            print(f"Loading draft model on cuda:{device_id}")
            draft_model = AutoModelForCausalLM.from_pretrained(
                draft_model_name,
                device_map={"": device_id},
                dtype=torch.bfloat16,
            )
            print(f"Draft model loaded on device: {draft_model.device}")

            # Per-pair tokenizer to avoid cross-worker contention
            pair_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
            if pair_tokenizer.pad_token_id is None:
                pair_tokenizer.pad_token_id = pair_tokenizer.eos_token_id
            pair_tokenizer.padding_side = 'left'
            self.worker_tokenizers.append(pair_tokenizer)

            specdec = SpeculativeDecoder(
                tokenizer=pair_tokenizer,
                target_model=target_model,
                draft_model=draft_model,
                config=SpecDecConfig(
                    lookahead_k=5,
                    temperature=0.7,
                    max_new_tokens=512,
                    device=str(target_model.device),
                    debug=True,
                ),
            )
            print(f"Speculative decoder initialized on cuda:{device_id}")

            self.model_pairs.append({
                "device": f"cuda:{device_id}",
                "target": target_model,
                "draft": draft_model,
                "specdec": specdec,
                "tokenizer": pair_tokenizer,
            })

        # Round-robin iterator for batch assignment
        self._pair_cycle = itertools.cycle(self.model_pairs)
        self._pair_lock = threading.Lock()

    def _batch_worker(self, task_type, qinfo, pair=None):
        """Background micro-batcher for a given task."""
        settings = self._microbatch_settings[task_type]
        queue = qinfo["queue"]
        cond = qinfo["cond"]

        while True:
            with cond:
                while not queue and qinfo["running"]:
                    cond.wait()
                if not qinfo["running"] and not queue:
                    break
                # Wait for flush window or size
                queue_size_before_wait = len(queue)
                print(f"[{task_type}] Queue size: {queue_size_before_wait}, waiting for flush window or max batch size...")
                while len(queue) < settings["max_batch_size"]:
                    if not queue:
                        break
                    remaining = settings["flush_window"] - (time.time() - queue[0]["arrival"])
                    if remaining <= 0:
                        break
                    print(f"[{task_type}] Waiting {remaining:.1f}s more (queue: {len(queue)}/{settings['max_batch_size']})")
                    cond.wait(timeout=remaining)
                if not queue:
                    continue
                batch = queue[:settings["max_batch_size"]]
                del queue[:settings["max_batch_size"]]
                print(f"[{task_type}] Processing batch of {len(batch)} prompts (queue remaining: {len(queue)})")

            # Process batch outside queue lock, but WITH GPU lock to prevent CUDA context corruption
            if not batch:
                continue
            prompts = [item["prompt"] for item in batch]
            max_tokens = batch[0]["max_tokens"]
            temperature = batch[0]["temperature"]
            pair = pair or self._next_pair()

            # Determine GPU ID from task_type (e.g., "infobench_gpu0" -> 0)
            gpu_id = int(task_type.split("_gpu")[-1]) if "_gpu" in task_type else 0

            # Acquire GPU lock before any CUDA operations
            with self._gpu_locks[gpu_id]:
                start_time = time.time()
                try:
                    generated_texts, prompt_lengths, outputs = self._process_by_task(
                        pair, prompts, max_tokens, temperature, task_type
                    )
                    elapsed = time.time() - start_time
                    print(f"[{task_type}] Batch of {len(batch)} completed in {elapsed:.1f}s")
                    for item, gen, plen, out in zip(batch, generated_texts, prompt_lengths, outputs):
                        item["result"] = (gen, plen, out)
                        item["event"].set()
                except Exception as e:
                    elapsed = time.time() - start_time
                    print(f"[{task_type}] Batch of {len(batch)} FAILED after {elapsed:.1f}s: {e}")
                    for item in batch:
                        item["result"] = e
                        item["event"].set()

    def _next_pair(self):
        """Round-robin select the next model pair."""
        with self._pair_lock:
            pair = next(self._pair_cycle)
        return pair

    def _enqueue_for_batch(self, task_type, prompt, max_tokens, temperature):
        """Add a single prompt to the micro-batch queue and wait for result."""
        if task_type not in self._microbatch_settings:
            raise ValueError(f"Unsupported task for batching: {task_type}")
        entry = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "arrival": time.time(),
            "event": threading.Event(),
            "result": None,
        }
        qinfo = self._batch_queues[task_type]

        with qinfo["cond"]:
            qinfo["queue"].append(entry)
            queue_size = len(qinfo["queue"])
            print(f"[{task_type}] Enqueued prompt (queue now: {queue_size})")
            qinfo["cond"].notify()
        entry["event"].wait()
        if isinstance(entry["result"], Exception):
            raise entry["result"]
        return entry["result"]

    def _init_gpu_monitor(self):
        """Set up NVML handles for lightweight GPU monitoring."""
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._gpu_handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(pynvml.nvmlDeviceGetCount())
        ]

    def _log_gpu_stats(self, context="request"):
        """Log utilization and memory for each GPU; safe to call frequently."""
        if not hasattr(self, "_nvml"):
            return
        stats = self._gpu_stats()
        msg_parts = []
        for s in stats:
            msg_parts.append(
                f"{s['name']} util={s['gpu_util_pct']}% mem={s['mem_used_gb']:.1f}/{s['mem_total_gb']:.1f}GB ({s['mem_pct']}%)"
            )
        pretty = " | ".join(msg_parts)
        print(f"[gpu][{context}] {pretty}")
        return stats

    def _gpu_stats(self):
        """Return per-GPU stats as list of dicts (util %, mem used/total)."""
        nvml = self._nvml
        stats = []
        for handle in self._gpu_handles:
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            mem = nvml.nvmlDeviceGetMemoryInfo(handle)
            stats.append({
                "name": nvml.nvmlDeviceGetName(handle).decode("utf-8"),
                "gpu_util_pct": util.gpu,
                "mem_util_pct": util.memory,
                "mem_used_mb": round(mem.used / (1024 ** 2), 1),
                "mem_total_mb": round(mem.total / (1024 ** 2), 1),
                "mem_used_gb": round(mem.used / (1024 ** 3), 2),
                "mem_total_gb": round(mem.total / (1024 ** 3), 2),
                "mem_pct": round((mem.used / mem.total) * 100, 1) if mem.total else 0.0,
            })
        return stats

    def _generate_standard(self, model, tokenizer, prompt, max_tokens, temperature):
        """Standard generation for a single prompt."""
        import torch

        # Apply chat template with thinking mode disabled
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Disable Qwen thinking mode
        )

        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model.device)

        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )

        generated_tokens = output[0, prompt_len:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text, prompt_len, output[0]

    def _generate_draft_batch(self, model, tokenizer, prompts, max_tokens, temperature, task_type):
        """
        Generate drafts for a batch of prompts (used by continuous batcher).
        Returns texts, empty lists for prompt_lengths and outputs (unused by batcher).
        """
        import torch

        # Apply chat template to each prompt with thinking mode disabled
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Disable Qwen thinking mode
            )
            formatted_prompts.append(formatted_prompt)

        # Batch tokenization
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        # Track actual (non-padded) length of each prompt
        prompt_lengths = (inputs.input_ids != self.tokenizer.pad_token_id).sum(dim=1)

        # Generate for all prompts in batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )

        # Decode only the generated text (without input prompt) for each example
        generated_texts = []
        for i, prompt_len in enumerate(prompt_lengths):
            generated_only = outputs[i, prompt_len:]
            generated_text = tokenizer.decode(generated_only, skip_special_tokens=True)
            generated_texts.append(generated_text)

        # Return empty lists for compatibility with batcher signature
        return generated_texts, [], []

    def _apply_infobench_concise_prompt(self, prompt, token_limit):
        """Add a brevity hint to keep generations short for InfoBench tasks."""
        return (
            f"{prompt}\n"
            f"Provide a direct, concise answer. Do not show your reasoning or thought process."
        )

    def _process_by_task(self, pair, prompts, max_tokens, temperature, task_type):
        """Process prompts with task-specific handling."""
        import torch

        model = pair["target"]
        specdec = pair["specdec"]

       # Normalize task_type: infobench_gpu0/gpu1/mmlu_gpu0/mmlu_gpu1 → infobench/mmlu
        normalized_task_type = task_type.split('_')[0]
        # print(f"[DEBUG] task_type='{task_type}' → normalized_task_type='{normalized_task_type}'")

        settings = self.task_settings.get(normalized_task_type, {})
        max_tokens = min(max_tokens, settings.get("max_tokens", max_tokens))
        temperature = settings.get("temperature", temperature)
        add_concise_prompt = settings.get("add_concise_prompt", False)
        concise_limit = settings.get("concise_token_limit", 0)
        best_of_n = settings.get("best_of_n", 1)
        use_specdec = self.task_specdec_enabled.get(normalized_task_type, False)
        # print(f"[DEBUG] normalized_task_type='{normalized_task_type}', use_specdec={use_specdec}, best_of_n={best_of_n}")

        # Task-specific prompt modifications
        if normalized_task_type == "graph":
            temperature = 0.0
        elif normalized_task_type == "infobench" and add_concise_prompt and concise_limit > 0:
            prompts = [
                self._apply_infobench_concise_prompt(p, concise_limit)
                for p in prompts
            ]

        # Instead of using LLM, directly compute shortest paths
        if normalized_task_type == "graph":
            print(f"[graph] Using direct computation for {len(prompts)} prompts")
            from graph_solver import solve_graph_problem

            generated_texts = []
            prompt_lengths = []
            outputs = []

            for prompt in prompts:
                # Directly solve the graph problem
                solution = solve_graph_problem(prompt)
                generated_texts.append(solution)

                # Create dummy tensors
                prompt_lengths.append(torch.tensor(0, device=model.device))
                outputs.append(torch.tensor([0], device=model.device))

            prompt_lengths = torch.stack(prompt_lengths)
            return generated_texts, prompt_lengths, outputs

        # MMLU BATCHING: Batch best-of-N sampling
        if normalized_task_type == "mmlu" and len(prompts) > 1:
            print(f"[mmlu] Using BATCHED best-of-{best_of_n} with majority vote for {len(prompts)} prompts")

            # Step 1: Create N copies of each prompt for best-of-N sampling
            expanded_prompts = []
            for prompt in prompts:
                expanded_prompts.extend([prompt] * best_of_n)

            # Step 2: Batch generate all candidates at once
            all_candidates, _, _ = self._generate_draft_batch(
                model,
                pair.get("tokenizer", self.tokenizer),
                expanded_prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                task_type="mmlu",
            )

            # Step 3: Group candidates back by original prompt and apply majority vote
            generated_texts = []
            prompt_lengths = []
            outputs = []

            for i in range(len(prompts)):
                # Extract N candidates for this prompt
                start_idx = i * best_of_n
                end_idx = start_idx + best_of_n
                candidates = all_candidates[start_idx:end_idx]

                # Majority vote on extracted letters
                letters = [self._extract_letter(c) or "" for c in candidates]
                from collections import Counter
                if letters:
                    most_common = Counter(letters).most_common(1)[0][0]
                    final_text = self._format_mmlu_answer(most_common)
                else:
                    fallback = candidates[0] if candidates else ""
                    final_text = self._format_mmlu_answer(self._extract_letter(fallback))

                print(f"[mmlu] Prompt {i}: Candidates: {letters}, Chose: {final_text}")

                generated_texts.append(final_text)
                # Create dummy tensors (we don't track individual token counts for batched MMLU)
                prompt_lengths.append(torch.tensor(0, device=model.device))
                outputs.append(torch.tensor([0], device=model.device))

            prompt_lengths = torch.stack(prompt_lengths)
            return generated_texts, prompt_lengths, outputs

        # INFOBENCH BATCHING: Uses BATCHED SPECULATIVE DECODING (if enabled)
        # Maximizes specdec usage for ALL InfoBench requests (single + multiple)
        if normalized_task_type == "infobench" and len(prompts) > 1 and use_specdec:
            print(f"[infobench] Using BATCHED speculative decoding for {len(prompts)} prompts")

            # Step 1: Batch speculative decoding (draft: 0.6B, target: 4B)
            gen_ids_list, acc_rates, prompt_lens = specdec.generate_batch(
                prompts,
                max_new_tokens=max_tokens,
                temperature=temperature,
                lookahead=3,
                use_adaptive_lookahead=True,
            )

            # Decode generated tokens (gen_ids is [1, gen_len], need to squeeze to [gen_len])
            draft_texts = [
                pair["tokenizer"].decode(gen_ids.squeeze(0), skip_special_tokens=True)
                for gen_ids in gen_ids_list
            ]

            avg_acc_rate = sum(acc_rates) / len(acc_rates)
            print(f"[infobench] Batch specdec avg acceptance rate: {avg_acc_rate:.2f}")

            # Use draft texts directly (no refinement)
            generated_texts = draft_texts
            prompt_lengths = []
            outputs = []

            # Create output tensors
            for prompt_len in prompt_lens:
                prompt_lengths.append(torch.tensor(prompt_len, device=model.device))
                outputs.append(torch.tensor([0], device=model.device))  # Dummy

            prompt_lengths = torch.stack(prompt_lengths)
            return generated_texts, prompt_lengths, outputs

        # INFOBENCH BATCH WITHOUT SPECULATIVE DECODING
        # Use standard batch generation when specdec is disabled
        if normalized_task_type == "infobench" and len(prompts) > 1 and not use_specdec:
            print(f"[infobench] Using standard BATCH generation for {len(prompts)} prompts (specdec disabled)")
            generated_texts, prompt_lengths, outputs = self._generate_draft_batch(
                model, pair["tokenizer"], prompts, max_tokens, temperature, task_type
            )
            # Convert empty lists to proper tensors
            if not prompt_lengths:
                prompt_lengths = [torch.tensor(0, device=model.device) for _ in prompts]
            if not outputs:
                outputs = [torch.tensor([0], device=model.device) for _ in prompts]

            prompt_lengths = torch.stack(prompt_lengths) if isinstance(prompt_lengths[0], torch.Tensor) else torch.tensor(prompt_lengths, device=model.device)
            return generated_texts, prompt_lengths, outputs

        # STANDARD PATH (single request, or MMLU tasks)
        generated_texts = []
        prompt_lengths = []
        outputs = []

        for prompt in prompts:
            # Step 1: Generate initial answer
            if use_specdec:
                # Use speculative decoding (InfoBench only)
                print(f"[{task_type}] Using speculative decoding")
                gen_ids, acc_rate, prompt_len, prompt_ids = specdec.generate_one(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    lookahead=3,
                    use_adaptive_lookahead=True,
                )
                full_seq = torch.cat([prompt_ids.to(model.device), gen_ids.to(model.device)], dim=1).squeeze(0)
                initial_text = pair["tokenizer"].decode(gen_ids[0], skip_special_tokens=True)
                print(f"[{task_type}] Specdec acceptance rate: {acc_rate:.2f}")
            elif best_of_n > 1:
                # Best-of-N sampling (MMLU only)
                print(f"[{task_type}] Using best-of-{best_of_n} with majority vote")
                candidates = []
                for _ in range(best_of_n):
                    cand_text, prompt_len, full_seq = self._generate_standard(model, pair.get("tokenizer", self.tokenizer), prompt, max_tokens, temperature)
                    candidates.append(cand_text)

                # Majority vote on extracted letters
                letters = [self._extract_letter(c) or "" for c in candidates]
                from collections import Counter
                if letters:
                    most_common = Counter(letters).most_common(1)[0][0]
                    initial_text = self._format_mmlu_answer(most_common)
                else:
                    initial_text = self._format_mmlu_answer(self._extract_letter(candidates[0]))

                print(f"[{task_type}] Candidates: {letters}, Chose: {initial_text}")
            else:
                # Standard generation
                print(f"[{task_type}] Using standard generation")
                initial_text, prompt_len, full_seq = self._generate_standard(model, pair.get("tokenizer", self.tokenizer), prompt, max_tokens, temperature)

            final_text = initial_text

            # Step 3: Post-processing
            if normalized_task_type == "mmlu":
                # Skip re-formatting if already formatted; otherwise format extracted letter
                if not final_text.startswith("The answer is "):
                    final_text = self._format_mmlu_answer(self._extract_letter(final_text))

            generated_texts.append(final_text)
            prompt_lengths.append(torch.tensor(prompt_len, device=model.device))
            outputs.append(full_seq)

        prompt_lengths = torch.stack(prompt_lengths) if len(prompt_lengths) > 1 else prompt_lengths[0].unsqueeze(0)
        return generated_texts, prompt_lengths, outputs

    def _extract_letter(self, text):
        """Extract first occurrence of A/B/C/D from common answer formats."""
        import re
        # Prefer explicit "The answer is X" pattern
        m = re.search(r"The answer is\s*\(?([A-D])\)?", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # Fallback: first standalone letter
        for ch in text:
            c = ch.upper()
            if c in ("A", "B", "C", "D"):
                return c
        return None

    def _format_mmlu_answer(self, letter):
        """Format MMLU answers consistently."""
        return f"The answer is {letter}" if letter else ""

    @modal.fastapi_endpoint(method="POST")
    def completions(self, request: dict):
        """OpenAI-compatible completions endpoint with task routing."""
        import torch
        try:
            self._log_gpu_stats(context="pre-route")
            prompt = request.get("prompt", "")
            max_tokens = request.get("max_tokens", 100)
            temperature = request.get("temperature", 0.7)

            if isinstance(prompt, str):
                prompts = [prompt]
            elif isinstance(prompt, list):
                prompts = prompt
            else:
                raise ValueError("prompt must be a string or list of strings")

            # Route prompts by task type
            grouped = self.router.route_batch_grouped(prompts)
            print(f"Routing results: {list(grouped.keys())}")

            results_by_task = {}
            all_prompt_lengths = {}
            all_outputs = {}

            def _run_group(task_type, data):
                task_prompts = data['prompts']
                task_indices = data['indices']
                
                # --- IMPLEMENT WORKLOAD-AWARE SHORTEST QUEUE ROUTING ---
                
                # 1. GRAPH: Direct processing (no change, no batching needed)
                if task_type == "graph":
                    pair = self._next_pair()
                    print(f"Processing {len(task_prompts)} {task_type} prompts at indices {task_indices} on {pair['device']}")
                    generated_texts, prompt_lengths, outputs = self._process_by_task(
                        pair, task_prompts, max_tokens, temperature, task_type
                    )
                    return task_type, task_indices, generated_texts, prompt_lengths, outputs

                # 2. MMLU: Route to the MMLU queue with the fewest pending requests (Shortest Queue)
                elif task_type == "mmlu":
                    # Accessing queue length for routing outside the lock is safe here
                    queue0 = self._batch_queues["mmlu_gpu0"]["queue"]
                    queue1 = self._batch_queues["mmlu_gpu1"]["queue"]
                    queue_name = "mmlu_gpu0" if len(queue0) <= len(queue1) else "mmlu_gpu1"

                # 3. INFOBENCH: Route to the InfoBench queue with the fewest pending requests (Workload-Aware)
                elif task_type == "infobench":
                    # Accessing queue length for routing outside the lock is safe here
                    queue0 = self._batch_queues["infobench_gpu0"]["queue"]
                    queue1 = self._batch_queues["infobench_gpu1"]["queue"]
                    
                    q0_size = len(queue0)
                    q1_size = len(queue1)
                    
                    if q0_size == 0 and q1_size > 0:
                        queue_name = "infobench_gpu0"
                    elif q1_size == 0 and q0_size > 0:
                        queue_name = "infobench_gpu1"
                    else:
                        # Fall back to standard shortest queue if both are busy or both are empty
                        queue_name = "infobench_gpu0" if q0_size <= q1_size else "infobench_gpu1"
                
                # UNHANDLED TASK (Error check)
                else:
                    raise ValueError(f"Unsupported or unrouted task type: {task_type}")

                # ENQUEUE & WAIT (Shared Logic for MMLU and INFOBENCH)
                print(f"Enqueuing {len(task_prompts)} {task_type} prompts at indices {task_indices} to {queue_name}")
                # Fire all enqueues concurrently so prompts from the same request can batch together
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(task_prompts)) as pool:
                    futures = [
                        pool.submit(self._enqueue_for_batch, queue_name, p, max_tokens, temperature)
                        for p in task_prompts
                    ]
                    batch_results = [f.result() for f in futures]
                generated_texts = [r[0] for r in batch_results]
                prompt_lengths = [r[1] for r in batch_results]
                outputs = [r[2] for r in batch_results]
                
                return task_type, task_indices, generated_texts, prompt_lengths, outputs

            max_workers = min(len(grouped), len(self.model_pairs)) if grouped else 1
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_run_group, task_type, data)
                    for task_type, data in grouped.items()
                ]
                for future in concurrent.futures.as_completed(futures):
                    task_type, task_indices, generated_texts, prompt_lengths, outputs = future.result()
                    for i, orig_idx in enumerate(task_indices):
                        results_by_task[orig_idx] = generated_texts[i]
                        all_prompt_lengths[orig_idx] = prompt_lengths[i]
                        all_outputs[orig_idx] = outputs[i]

            # Merge results back to original order
            ordered_texts = []
            ordered_prompts = []
            ordered_prompt_lengths = []
            ordered_outputs = []

            for i in range(len(prompts)):
                ordered_texts.append(results_by_task[i])
                ordered_prompts.append(prompts[i])
                ordered_prompt_lengths.append(all_prompt_lengths[i])
                ordered_outputs.append(all_outputs[i])

            # Create choices
            choices = []
            total_prompt_tokens = 0
            total_completion_tokens = 0

            for i, (generated_text, prompt_text, prompt_len, output) in enumerate(
                zip(ordered_texts, ordered_prompts, ordered_prompt_lengths, ordered_outputs)
            ):
                prompt_tokens = prompt_len.item()

                # Handle dummy outputs from batched paths (Graph, MMLU, InfoBench)
                if len(output) <= 1:
                    # Dummy output - approximate tokens without using tokenizer to avoid borrow conflicts
                    completion_tokens = max(1, len(generated_text.split()))
                    finish_reason = "length"
                else:
                    completion_tokens = len(output) - prompt_tokens
                    if output[-1].item() == self.tokenizer.eos_token_id:
                        finish_reason = "stop"
                    else:
                        finish_reason = "length"

                choices.append({
                    "text": generated_text,
                    "prompt": prompt_text,
                    "index": i,
                    "finish_reason": finish_reason
                })

                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

            return {
                "choices": choices,
                "model": "mingqia2-system-3",
                "usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_prompt_tokens + total_completion_tokens
                }
            }
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[system_with_router error] {e}\n{tb}")
            return {"error": str(e), "traceback": tb}

    @modal.fastapi_endpoint(method="GET")
    def gpu(self):
        """FastAPI endpoint to pull live GPU stats."""
        stats = self._gpu_stats() if hasattr(self, "_nvml") else []
        return {"gpus": stats}
