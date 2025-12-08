# InfoBench Batching Review & Optimization Recommendations

## Current Architecture Analysis

### 1. Current InfoBench Batching Setup

#### Micro-batching Configuration
```python
"infobench": {
    "flush_window": 5,      # 5 seconds wait time
    "max_batch_size": 10    # Maximum 10 prompts per batch
}
```

#### Worker Architecture
- **2 InfoBench workers** (one per GPU)
- **Shared queue** with condition variable synchronization
- Each worker processes batches independently using assigned GPU

---

## 🔴 CRITICAL ISSUES IDENTIFIED

### Issue 1: **5-Second Flush Window is TOO LONG**

**Problem:**
- Each request waits up to 5 seconds before batch processing starts
- In a 20-minute evaluation (1200 seconds), this causes massive latency overhead
- Single InfoBench requests suffer 5-second minimum latency penalty

**Impact:**
- If you receive InfoBench requests sequentially, each one waits 5 seconds
- For 100 InfoBench requests: 500 seconds wasted just waiting!
- This is likely why InfoBench takes the longest time

**Recommended Fix:**
```python
"infobench": {
    "flush_window": 0.1,     # Change from 5 → 0.1 seconds
    "max_batch_size": 10
}
```

**Rationale:**
- 100ms is enough to capture concurrent requests
- Reduces per-request latency from 5s → 0.1s
- Still allows batching when requests arrive in bursts
- MMLU already uses 0.5s successfully

---

### Issue 2: **Two Workers Competing for Same Queue**

**Problem:**
```python
# Both workers share ONE queue
for pair in self.model_pairs:  # Creates 2 workers
    threading.Thread(
        target=self._batch_worker,
        args=("infobench", self._batch_queues["infobench"], pair),
        daemon=True
    ).start()
```

**Current Behavior:**
- Worker 1 (GPU 0) and Worker 2 (GPU 1) both pull from same queue
- When batch of 10 arrives, only ONE worker processes it
- Other GPU sits idle
- No actual parallelization!

**Impact:**
- Only 1 GPU active for InfoBench at any time
- 50% GPU utilization for InfoBench
- Doubling latency unnecessarily

**Recommended Fix Option A: Split Queues (Best)**
```python
# Separate queue per GPU
self._batch_queues = {
    "mmlu": {"queue": [], "cond": threading.Condition(), "running": True},
    "infobench_gpu0": {"queue": [], "cond": threading.Condition(), "running": True},
    "infobench_gpu1": {"queue": [], "cond": threading.Condition(), "running": True},
}

# Round-robin assignment to GPU-specific queues
def _enqueue_for_batch_infobench(self, prompt, max_tokens, temperature):
    # Alternate between gpu0 and gpu1 queues
    gpu_id = hash(prompt) % 2  # Or use a counter
    queue_name = f"infobench_gpu{gpu_id}"
    # ... enqueue to specific queue
```

**Recommended Fix Option B: Direct Processing (Simpler)**
```python
# Remove InfoBench from micro-batching entirely
# Process batches directly in _run_group
if task_type == "infobench":
    # Batch size already determined by router
    pair = self._next_pair()  # Round-robin GPU selection
    generated_texts, prompt_lengths, outputs = self._process_by_task(
        pair, task_prompts, max_tokens, temperature, task_type
    )
```

This is simpler and leverages the fact that router already groups InfoBench requests.

---

### Issue 3: **Inefficient Enqueuing Pattern**

**Current Code:**
```python
# Lines 688-693: Each prompt spawns a thread that waits
with concurrent.futures.ThreadPoolExecutor(max_workers=len(task_prompts)) as pool:
    futures = [
        pool.submit(self._enqueue_for_batch, task_type, p, max_tokens, temperature)
        for p in task_prompts
    ]
    batch_results = [f.result() for f in futures]
```

**Problem:**
- If you have 10 InfoBench prompts in a request:
  1. Spawn 10 threads
  2. Each thread adds to queue and BLOCKS waiting for result
  3. Worker pulls all 10, processes as batch
  4. All 10 threads wake up

**Inefficiency:**
- Thread overhead (10 threads created/destroyed)
- Unnecessary synchronization complexity
- Each request already batched by router!

**Why is this happening?**
- Router groups prompts: `['ib1', 'ib2', 'ib3', ...]`
- Then you split them again into individual queue entries
- Worker recombines them into batch
- **Double batching!**

---

### Issue 4: **Single-Prompt InfoBench Uses Sequential Path**

**Code:**
```python
# Line 527: Batch path only for len(prompts) > 1
if task_type == "infobench" and len(prompts) > 1:
    # Batched speculative decoding
else:
    # STANDARD PATH - sequential processing
    for prompt in prompts:
        if use_specdec:
            gen_ids, acc_rate, prompt_len, prompt_ids = specdec.generate_one(...)
```

**Problem:**
- Single InfoBench requests don't benefit from batch optimization
- Falls back to slow sequential path
- In production, many requests arrive as single prompts

**Impact:**
- Inconsistent performance: batch of 1 is slower than being part of batch of 10
- Micro-batching could fix this, but current setup is broken (Issue #2)

---

## 📊 Performance Impact Analysis

### Current System (Estimated):

**Scenario: 100 InfoBench requests over 20 minutes**

```
Sequential arrival (worst case):
- Request 1: Wait 5s + Process 8s = 13s
- Request 2: Wait 5s + Process 8s = 13s
- ...
- Total: 100 × 13s = 1300s = 21.7 minutes → TIMEOUT!

Batched arrival (10 at a time):
- Batch 1: Wait 5s + Process 8s = 13s
- Batch 2: Wait 5s + Process 8s = 13s
- ...
- Total: 10 batches × 13s = 130s
```

### With Optimizations (Estimated):

```
With 0.1s flush + direct batching:
Sequential arrival:
- Request 1: Wait 0.1s + Process 8s = 8.1s
- Request 2: Wait 0.1s + Process 8s = 8.1s
- ...
- Total: 100 × 8.1s = 810s = 13.5 minutes ✓

Batched arrival (10 at a time, parallel GPUs):
- GPU 0 processes batch 1,3,5,7,9: 5 × 8s = 40s
- GPU 1 processes batch 2,4,6,8,10: 5 × 8s = 40s
- Total: 40s (parallel execution!)
```

**Expected Improvement: 3.25x faster**

---

## 🎯 RECOMMENDED OPTIMIZATIONS (Priority Order)

### Priority 1: CRITICAL - Reduce Flush Window ⚡

**Change:**
```python
"infobench": {
    "flush_window": 0.1,  # From 5 → 0.1 seconds
    "max_batch_size": 10
}
```

**Expected Impact:**
- **~60x reduction in wait time** per request
- Minimum 4.9 second latency improvement per request
- Still allows batching for concurrent requests

**Implementation:** 1-line change, zero risk

---

### Priority 2: HIGH - Remove Micro-batching for InfoBench ⚡⚡

**Rationale:**
- Router already batches requests by task type
- Micro-batching adds overhead without benefit
- Simplifies architecture

**Change:**
```python
# Remove InfoBench from micro-batching settings
self._microbatch_settings = {
    "mmlu": {"flush_window": 0.5, "max_batch_size": 5},
    # Remove infobench entry
}

# Don't start InfoBench workers
threading.Thread(
    target=self._batch_worker, args=("mmlu", self._batch_queues["mmlu"], None), daemon=True
).start()
# Remove the loop that creates InfoBench workers
```

**Update _run_group:**
```python
def _run_group(task_type, data):
    task_prompts = data['prompts']
    task_indices = data['indices']

    # All tasks use direct processing now
    pair = self._next_pair()  # Round-robin GPU assignment
    print(f"Processing {len(task_prompts)} {task_type} prompts at indices {task_indices} on {pair['device']}")
    generated_texts, prompt_lengths, outputs = self._process_by_task(
        pair, task_prompts, max_tokens, temperature, task_type
    )
    return task_type, task_indices, generated_texts, prompt_lengths, outputs
```

**Expected Impact:**
- Better GPU utilization (round-robin instead of competing workers)
- Simpler, more maintainable code
- Eliminates thread spawning overhead
- Consistent batching behavior

---

### Priority 3: MEDIUM - Optimize Speculative Decoding Batch Size

**Current:**
```python
max_batch_size": 10
```

**Analysis:**
- Speculative decoding has diminishing returns with large batches
- Padding overhead increases with batch size
- Variable-length sequences cause inefficiency

**Recommended Experiments:**
```python
# Try smaller batches for better throughput
"infobench": {"max_batch_size": 4}  # Test 4, 6, 8
```

**Why smaller might be better:**
- Less padding waste
- Faster completion of smaller batches
- Better pipelining (start next batch sooner)
- 2 GPUs × 4-item batches = 8 parallel requests

---

### Priority 4: MEDIUM - Increase Lookahead for InfoBench

**Current:**
```python
lookahead=3  # In specdec.generate_batch call
```

**Observation:**
- InfoBench generates long outputs (512 tokens)
- Adaptive lookahead adjusts based on acceptance rate
- Starting lookahead of 3 might be conservative

**Recommended:**
```python
gen_ids_list, acc_rates, prompt_lens = specdec.generate_batch(
    prompts,
    max_new_tokens=max_tokens,
    temperature=temperature,
    lookahead=5,  # Increase from 3 → 5
    use_adaptive_lookahead=True,
)
```

**Expected Impact:**
- More draft tokens per iteration
- Fewer target model calls
- 10-20% speedup if acceptance rate is high (>0.6)

---

### Priority 5: LOW - Optimize Token Limit

**Current:**
```python
"max_tokens": 512
```

**Analysis:**
- InfoBench questions vary in complexity
- Some may only need 256 tokens
- Generating extra tokens wastes compute

**Options:**

**Option A: Reduce default**
```python
"max_tokens": 256  # Start lower
```

**Option B: Dynamic allocation**
```python
# Parse prompt to estimate required length
def _estimate_infobench_length(self, prompt):
    if "explain in detail" in prompt.lower():
        return 512
    elif "list" in prompt.lower() or "summarize" in prompt.lower():
        return 256
    else:
        return 384
```

**Expected Impact:**
- 2x speedup for shorter responses
- Risk: might truncate some answers

---

## 🔧 IMPLEMENTATION GUIDE

### Step 1: Quick Win (5 minutes)

Change flush window:
```python
"infobench": {"flush_window": 0.1, "max_batch_size": 10}
```

**Expected improvement: 50-60% latency reduction**

---

### Step 2: Architectural Fix (30 minutes)

Remove InfoBench from micro-batching:

1. Update `_microbatch_settings` to only include MMLU
2. Remove InfoBench worker creation loop
3. Remove InfoBench from `_batch_queues`
4. Update `_run_group` to always use direct processing for InfoBench

**Expected improvement: Additional 20-30% throughput increase**

---

### Step 3: Fine-tuning (1 hour)

Experiment with:
- Batch sizes: 4, 6, 8, 10
- Lookahead: 4, 5, 6
- Max tokens: 256, 384, 512

Run benchmarks and pick best configuration.

**Expected improvement: Additional 10-20% optimization**

---

## 📈 PREDICTED PERFORMANCE

### Before Optimizations:
- InfoBench avg latency: 13s (5s wait + 8s process)
- Throughput: ~7 requests/minute/GPU
- 20-minute test: ~280 requests total

### After Priority 1 (Flush Window):
- InfoBench avg latency: 8.1s (0.1s wait + 8s process)
- Throughput: ~11 requests/minute/GPU
- 20-minute test: ~440 requests total (+57%)

### After Priority 1 + 2 (Remove Micro-batch):
- Better GPU utilization: 2 GPUs working in parallel
- Throughput: ~22 requests/minute (both GPUs)
- 20-minute test: ~440 requests total (same, but more consistent)

### After All Optimizations:
- InfoBench avg latency: 6-7s
- Throughput: ~17 requests/minute/GPU
- 20-minute test: ~680 requests total (+143%)

---

## ⚠️ POTENTIAL RISKS

### Risk 1: Batch Size Too Small
- If max_batch_size < number of prompts in request, might split batches
- **Mitigation:** Router already handles grouping; shouldn't be an issue

### Risk 2: Race Conditions
- Removing micro-batching changes concurrency patterns
- **Mitigation:** Round-robin GPU selection is thread-safe with existing lock

### Risk 3: Uneven Load
- Some GPU might get harder prompts
- **Mitigation:** Round-robin should balance over time

---

## 🧪 TESTING RECOMMENDATIONS

1. **Test with varied batch sizes:**
   - Single request (1 InfoBench prompt)
   - Small batch (3-5 prompts)
   - Large batch (10+ prompts)

2. **Monitor GPU utilization:**
   - Use the existing `_log_gpu_stats()` function
   - Ensure both GPUs are active during InfoBench processing

3. **Measure acceptance rates:**
   - Current specdec already logs this
   - Verify lookahead adjustments are helping

4. **Benchmark throughput:**
   - Count requests completed in 20 minutes
   - Compare before/after each optimization

---

## 📝 SUMMARY

**Critical Issue:** 5-second flush window is killing performance ❌

**Quick Fix:** Change to 0.1 seconds → 50%+ improvement ✅

**Better Fix:** Remove micro-batching entirely → 100%+ improvement ✅

**Best Fix:** All optimizations combined → 140%+ improvement ✅
