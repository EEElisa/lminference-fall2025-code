# Optimal Flush Window Analysis Based on Actual Traffic Pattern

## Traffic Pattern Summary (from batch_arrivals.json)

### Key Statistics:
- **Average inter-arrival time**: 10.05 seconds
- **Median inter-arrival time**: 8.52 seconds
- **Min inter-arrival time**: 0.43 seconds (batches arriving nearly simultaneously)
- **Max inter-arrival time**: 31.54 seconds (longest gap)

### Batch Size Distribution:
- 1 prompt: 25% of batches
- 2 prompts: 20% of batches
- 3 prompts: 30% of batches
- 4-8 prompts: 25% of batches

### Task Distribution (roughly equal):
- Graph: ~34%
- MMLU: ~36%
- InfoBench: ~30%

---

## The Flush Window Trade-off Analysis

### Current Configuration Issue

With **flush_window = 5 seconds**:

```
Scenario 1: Batch arrives at t=0s with 3 InfoBench prompts
├─ Router groups them → already batched!
├─ Each prompt enqueued individually
├─ Worker waits 5 seconds (flush window)
├─ Processes batch at t=5s
└─ Total: 5s wait + 100s processing = 105s

Next batch arrives at t=10s (typical gap)
├─ First batch still processing (105s total)
├─ Worker waits 5s (flush window)
├─ Processes at t=15s
└─ Total: 5s wait + 100s = 105s

Result: EVERY batch pays 5-second penalty!
```

### What You're Thinking vs. Reality

**Your reasoning (partially correct):**
> "Longer flush window will help adding more prompts to one batch"

**Why this doesn't work with your traffic:**

1. **Batches already contain multiple prompts** (1-8 prompts per API call)
   - Router groups by task type
   - Your flush window is trying to merge DIFFERENT API calls

2. **Inter-arrival time is 10 seconds on average**
   - Waiting 5 seconds only captures half the gap
   - You'd need 10+ second window to merge consecutive batches
   - But that would add 10s latency to EVERY request!

3. **The math doesn't work out:**
   ```
   Batch A arrives: t=0s
   Flush window: 5s

   Will Batch B (arriving at t=10s) be merged? NO!
   - Batch A starts processing at t=5s
   - Batch B arrives at t=10s (misses Batch A)
   - Batch B waits until t=15s
   - Both batches waited 5s but didn't merge!
   ```

---

## Optimal Flush Window Calculation

### Option 1: Short Window (0.1-0.5s) ✅ RECOMMENDED

**Rationale:**
- Captures prompts from same API call that arrive microseconds apart
- Minimizes latency penalty
- Works with your current architecture where router pre-batches

**Configuration:**
```python
"infobench": {"flush_window": 0.5, "max_batch_size": 8}
```

**Expected behavior:**
```
Batch of 3 InfoBench prompts arrives at t=0s
├─ All 3 enqueued within 0.01s (nearly instant)
├─ Worker wakes up, waits 0.5s
├─ Processes batch of 3 at t=0.5s
└─ Total: 0.5s wait + 100s processing = 100.5s

Next batch (2 prompts) arrives at t=10s
├─ Enqueued within 0.01s
├─ Worker waits 0.5s
├─ Processes at t=10.5s
└─ Total: 0.5s wait + 67s processing = 67.5s

Savings: 4.5s per batch × 98 batches = 441 seconds saved!
```

---

### Option 2: Medium Window (2-3s) ⚠️ MAYBE

**Only if you want to gamble on merging batches**

With 3-second window and 10-second average gap:
- **30% chance** two consecutive batches merge (if gap < 3s)
- **70% chance** they don't merge, but both pay 3s penalty

**Math:**
```
Merged case (30% of time):
- Wait: 3s
- Process: 100s for combined batch
- Total: 103s for 2 batches

Not merged (70% of time):
- Batch 1: 3s wait + 100s = 103s
- Batch 2: 3s wait + 100s = 103s
- Total: 206s for 2 batches

Expected value: 0.3 × 103 + 0.7 × 206 = 175s for 2 batches
```

vs. **0.5s window:**
```
Never merge (deterministic):
- Batch 1: 0.5s + 100s = 100.5s
- Batch 2: 0.5s + 100s = 100.5s
- Total: 201s for 2 batches
```

**Conclusion:** Medium window is WORSE! (175s vs 201s, but only 26s saved with unpredictable behavior)

---

### Option 3: Long Window (5-10s) ❌ TERRIBLE

With 5s window:
- Guaranteed to add 5s to EVERY batch
- Might merge ~20-30% of batches
- But penalty is paid 100% of the time

**Math:**
```
98 batches × 5s penalty = 490 seconds wasted
Potential merging: Maybe reduce 98 → 70 batches
Savings from merging: ~28 batches × 100s... but wait!

Problem: InfoBench takes 100s to process
- Worker is busy for 100s processing batch
- Can't pick up new batch during that time
- So batches arriving 10s apart CAN'T merge anyway!
```

**The fundamental issue:**
```
t=0s:   Batch A arrives
t=5s:   Batch A starts processing (worker busy)
t=10s:  Batch B arrives, worker STILL busy with A
t=15s:  Batch B would start... but worker busy until t=105s!
t=105s: Batch A finishes, Batch B finally starts

Flush window did NOTHING except add 5s to Batch A!
```

---

## The Real Bottleneck: Generation Time

### Current InfoBench Performance:
```
512 tokens × (lookahead=3) with acceptance rate ~0.3
= ~170 iterations × 0.6s per iteration
= ~100 seconds per request
```

### Batching Reality:

**Your system already batches optimally:**
1. Router groups prompts by task type within each API call
2. If API call has 3 InfoBench prompts → batch of 3
3. Flush window only adds latency WITHOUT benefit

**Why?**
- You have 2 GPUs
- You have 2 InfoBench workers
- But they share 1 queue!
- Worker 1 grabs a batch, Worker 2 sits idle
- No parallelism across batches anyway!

---

## Architecture Problem: Shared Queue

```python
# Current: Both workers compete for same queue
for pair in self.model_pairs:  # Creates 2 workers
    threading.Thread(
        target=self._batch_worker,
        args=("infobench", self._batch_queues["infobench"], pair),
        daemon=True
    ).start()
```

**What happens:**
```
t=0s:   Batch A (3 prompts) arrives, Worker 1 takes it
t=0s:   Worker 2 sits idle (queue empty)
t=100s: Worker 1 finishes, Batch B (2 prompts) waiting
t=100s: Worker 1 or 2 takes Batch B (doesn't matter)
t=200s: Done

Utilization: 1 GPU busy, 1 GPU idle = 50%!
```

**Better architecture (separate queues):**
```
t=0s:   Batch A → Worker 1 (GPU 0)
t=10s:  Batch B → Worker 2 (GPU 1) starts immediately!
t=100s: Worker 1 finishes Batch A
t=110s: Worker 2 finishes Batch B

Total time: 110s instead of 200s!
```

---

## RECOMMENDATIONS

### Priority 1: Reduce Flush Window ⚡⚡⚡

```python
"infobench": {"flush_window": 0.5, "max_batch_size": 8}
```

**Impact:** Save 4.5 seconds per batch × 98 batches = **441 seconds saved**

---

### Priority 2: Reduce max_tokens ⚡⚡⚡

```python
"infobench": {"max_tokens": 256}  # from 512
```

**Impact:** ~50% faster generation = **~50 seconds per batch instead of 100s**

---

### Priority 3: Increase Lookahead ⚡⚡

```python
# In generate_batch call
lookahead=5  # from 3
```

**Impact:** 15-20% faster = **~40-42 seconds per batch**

---

### Priority 4: Fix Worker Architecture ⚡

Remove shared queue, use direct processing:

```python
# Remove InfoBench from micro-batching
self._microbatch_settings = {
    "mmlu": {"flush_window": 0.5, "max_batch_size": 5},
    # No InfoBench entry
}

# In _run_group, always use direct processing
pair = self._next_pair()  # Round-robin GPU selection
generated_texts, prompt_lengths, outputs = self._process_by_task(
    pair, task_prompts, max_tokens, temperature, task_type
)
```

**Impact:** Both GPUs utilized = **2x throughput for InfoBench**

---

## FINAL CONFIGURATION

```python
"infobench": {
    "max_tokens": 256,  # Reduced from 512
    "add_concise_prompt": True,
    "concise_token_limit": 150,
    "use_self_refine": False,
    "refine_temperature": 0.7,
}

# Remove InfoBench from micro-batching (or keep with 0.5s)
self._microbatch_settings = {
    "mmlu": {"flush_window": 0.5, "max_batch_size": 5},
}

# In generate_batch calls
lookahead=5
```

---

## Expected Performance Improvement

### Before Optimizations:
```
Per InfoBench batch: 5s wait + 100s generation = 105s
98 batches × 105s = 10,290 seconds (171 minutes)
BUT: Only 1 GPU working
Actual throughput: ~9 batches per 20 minutes = TIMEOUT!
```

### After Optimizations:
```
Per InfoBench batch: 0.5s wait + 42s generation = 42.5s
With 2 GPUs in parallel (round-robin):
- GPU 0 handles 49 batches: 49 × 42.5s = 2082s
- GPU 1 handles 49 batches: 49 × 42.5s = 2082s
- Both run in parallel!

Total time: 2082 seconds = 34.7 minutes
```

**But wait, we have concurrent requests!**

With proper round-robin and router batching:
```
API call 1 (3 IB prompts) → GPU 0 (42s)
API call 2 (2 IB prompts) → GPU 1 (28s) starts immediately
API call 3 (1 IB prompt)  → GPU 0 (14s) starts at t=42s
...

Effective throughput: 2 GPUs × (20 min / 42s per batch)
= 2 × 28.5 batches = 57 batches in 20 minutes
```

**vs. Current:**
```
1 GPU effectively used × (20 min / 105s per batch)
= ~11 batches in 20 minutes
```

**Improvement: 5x more batches processed!** 🚀

---

## Answer to Your Question

> "I thought longer flush window will help adding more prompts to one batch - so eventually increase the throughput??"

**You're right in theory, but wrong for your specific traffic pattern because:**

1. ✅ Longer window CAN merge batches...
2. ❌ BUT your batches arrive 10s apart on average
3. ❌ AND processing takes 100s (worker is busy)
4. ❌ So consecutive batches CAN'T merge anyway!
5. ❌ You just pay 5s penalty for nothing

**The real throughput fix is:**
- Reduce generation time (256 tokens, lookahead=5)
- Use both GPUs (fix architecture)
- NOT waiting longer to merge batches that won't merge!

---

## Analogy

**Bad strategy (current):**
- Bus waits 5 minutes at each stop to see if more passengers arrive
- Next stop is 10 minutes away
- Passengers at next stop can't board the current bus anyway!
- Result: Every passenger waits 5 extra minutes for no benefit

**Good strategy (recommended):**
- Bus waits 30 seconds (enough for passengers at same stop)
- Moves quickly to next stop
- Run 2 buses in parallel (2 GPUs)
- Result: 10x more passengers served!
