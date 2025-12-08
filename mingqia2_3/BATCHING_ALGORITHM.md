# Batching Logic and Algorithm Analysis for `system_with_router.py`

## Overview
This system implements a sophisticated multi-level batching architecture that routes requests to task-specific queues (Graph, MMLU, InfoBench) and processes them using GPU-specific batch workers with dynamic batching strategies.

---

## 1. Queue Architecture

### 1.1 Queue Topology
The system maintains **4 separate queues** (lines 129-134):
- `infobench_gpu0`: InfoBench tasks for GPU 0
- `infobench_gpu1`: InfoBench tasks for GPU 1
- `mmlu_gpu0`: MMLU tasks for GPU 0
- `mmlu_gpu1`: MMLU tasks for GPU 1

**Note**: Graph tasks do NOT use queues - they are processed directly using algorithmic computation (Dijkstra's algorithm).

### 1.2 Queue Data Structure
Each queue entry contains (lines 298-305):
```python
{
    "prompt": str,           # The input prompt
    "max_tokens": int,       # Maximum tokens to generate
    "temperature": float,    # Sampling temperature
    "arrival": float,        # Timestamp of arrival (time.time())
    "event": threading.Event(), # Synchronization primitive for blocking/waking
    "result": None,          # Will hold (generated_text, prompt_len, output) tuple
}
```

---

## 2. Batching Parameters (Micro-batch Settings)

### 2.1 Flush Window and Max Batch Size (lines 119-126)
Different tasks have different batching thresholds:

| Queue | Flush Window | Max Batch Size | Rationale |
|-------|--------------|----------------|-----------|
| `infobench_gpu0/1` | 10 seconds | 8 prompts | Long sequences need more wait time to accumulate requests |
| `mmlu_gpu0/1` | 3 seconds | 8 prompts | Short sequences can batch quickly with shorter window |

**Flush Window Logic**: Maximum time to wait for the batch to fill before processing
**Max Batch Size**: Process batch immediately when this size is reached

---

## 3. Batch Worker Algorithm (The Core Batching Loop)

### 3.1 Worker Thread Lifecycle (lines 229-286)
Each of the 4 queues has a dedicated background thread running `_batch_worker()`:

```
Thread 1: infobench_gpu0 worker (GPU 0)
Thread 2: mmlu_gpu0 worker (GPU 0)
Thread 3: infobench_gpu1 worker (GPU 1)
Thread 4: mmlu_gpu1 worker (GPU 1)
```

### 3.2 Batch Collection Algorithm (lines 236-256)

**Step-by-step batching logic**:

1. **Wait for Requests** (lines 236-240):
   - Thread blocks on condition variable until queue is non-empty
   - If shutdown signal received and queue empty, exit

2. **Dynamic Wait Strategy** (lines 241-251):
   - Record initial queue size
   - Enter wait loop with two exit conditions:
     - **Condition A**: Queue reaches `max_batch_size` → process immediately
     - **Condition B**: `flush_window` seconds elapsed since first item arrival → process what we have

   ```python
   while len(queue) < max_batch_size:
       remaining = flush_window - (time.time() - queue[0]["arrival"])
       if remaining <= 0:
           break  # Flush window expired
       cond.wait(timeout=remaining)  # Wait for more requests or timeout
   ```

3. **Extract Batch** (lines 254-256):
   - Take up to `max_batch_size` items from queue front
   - Remove them from queue (other threads can now add to queue)
   - Print batch size and remaining queue depth

### 3.3 Batch Processing (lines 258-286)

4. **GPU Lock Acquisition** (lines 266-270):
   - Parse GPU ID from queue name (e.g., `"infobench_gpu0"` → GPU 0)
   - Acquire per-GPU lock to prevent concurrent CUDA operations
   - **Critical**: Prevents `CUBLAS_STATUS_EXECUTION_FAILED` errors from thread contention

5. **Task-Specific Processing** (lines 273-275):
   - Call `_process_by_task()` with the batch
   - Extract prompts, settings from batch items
   - Generate responses in batch

6. **Result Distribution** (lines 278-286):
   - On success: Store result tuple in each item's `result` field
   - On error: Store exception object
   - Set each item's `event` to wake waiting threads
   - All original requesters are unblocked simultaneously

---

## 4. Request Routing Algorithm

### 4.1 Task-Type Routing (lines 699-701)
Incoming requests first go through task classification:
- Uses `TaskRouter` to classify prompts as `graph`, `mmlu`, or `infobench`
- Groups prompts by task type for batch processing

### 4.2 Queue Selection Strategy (lines 707-763)

**A. Graph Tasks** (lines 714-720):
- NO QUEUING - processed immediately via round-robin GPU selection
- Direct computation using Dijkstra's algorithm (no LLM)

**B. MMLU Tasks** (lines 722-728):
- **Shortest Queue Routing**: Compare `len(queue0)` vs `len(queue1)`
- Route to GPU with fewer pending MMLU requests
- Goal: Balance load across GPUs

**C. InfoBench Tasks** (lines 729-744):
- **Workload-Aware Shortest Queue Routing** with idle GPU preference:

  Priority rules:
  1. If GPU 0 queue empty and GPU 1 has work → route to GPU 0
  2. If GPU 1 queue empty and GPU 0 has work → route to GPU 1
  3. Otherwise → route to queue with fewer items

  Rationale: Prefer idle GPUs to minimize queueing delay

### 4.3 Concurrent Enqueueing (lines 752-758)
For MMLU/InfoBench prompts:
- Launch ThreadPoolExecutor with `max_workers=len(task_prompts)`
- Submit all prompts to chosen queue **in parallel**
- Ensures prompts from the same request can batch together
- Each thread blocks until its result is ready

---

## 5. Task-Specific Batch Processing Logic

### 5.1 Graph Processing (lines 484-502)
```python
# No batching - direct computation
for prompt in prompts:
    solution = solve_graph_problem(prompt)  # Dijkstra's algorithm
    generated_texts.append(solution)
```
- **No GPU usage**
- **No batching benefits** (algorithmic computation is instant)

### 5.2 MMLU Batching (lines 504-552)

**Algorithm**: Batched Best-of-N with Majority Voting

```
Input: N prompts, best_of_n=3
Step 1: Expand prompts → [p1, p1, p1, p2, p2, p2, ..., pN, pN, pN]
Step 2: Batch generate ALL candidates (N × 3 generations in ONE forward pass)
Step 3: Group candidates back by original prompt (every 3)
Step 4: Extract answer letters from each candidate
Step 5: Majority vote per prompt → final answer
Step 6: Format as "The answer is X"
```

**Example**:
```
Batch: ["What is 2+2?", "What is the capital of France?"]
best_of_n: 3

Expanded batch (6 generations):
- "What is 2+2?" → ["4", "four", "4"]
- "What is the capital of France?" → ["Paris", "Paris", "Lyon"]

Majority vote:
- Prompt 1: "4" (appears 2/3 times)
- Prompt 2: "Paris" (appears 2/3 times)
```

**Key optimization**: Single batched forward pass instead of N × best_of_n sequential calls

### 5.3 InfoBench Batching (lines 554-604)

**Two Modes**:

**Mode A: With Speculative Decoding** (lines 556-588, currently disabled):
```
Step 1: Batch speculative decoding (draft model proposes, target model verifies)
Step 2: Decode generated token IDs to text
Step 3: Return draft texts directly (no refinement)
```

**Mode B: Standard Batch Generation** (lines 590-604, currently active):
```
Step 1: Apply chat template to all prompts
Step 2: Batch tokenization with padding
Step 3: Single forward pass through target model
Step 4: Decode only generated tokens (strip prompt)
```

**Current configuration**: Speculative decoding is **disabled** (line 97) due to low acceptance rate (0.02), making it slower than standard generation.

---

## 6. Synchronization Mechanisms

### 6.1 Queue-Level Synchronization
- **Condition Variable** (`threading.Condition()`): Coordinates producer (enqueue) and consumer (batch worker)
- **Notify on Enqueue** (line 312): Wakes batch worker when new item arrives
- **Wait with Timeout** (line 251): Batch worker sleeps until timeout or notification

### 6.2 Request-Level Synchronization
- **Threading Event** (line 303): Each request gets an event object
- **Blocking Wait** (line 313): Request thread blocks until event is set
- **Set on Completion** (line 280/285): Batch worker sets event after processing

### 6.3 GPU-Level Synchronization
- **Per-GPU Locks** (lines 138-141): Prevent concurrent CUDA operations on same GPU
- **Critical for stability**: Avoids CUDA context corruption from multi-threaded access

---

## 7. Batching Flow Example

### Scenario: 5 MMLU requests arrive within 1 second

```
t=0.0s: Request 1 arrives → enqueued to mmlu_gpu0 (queue size: 1)
        Worker wakes, sees size < 8, waits (flush_window=3s remaining)

t=0.1s: Request 2 arrives → enqueued to mmlu_gpu1 (queue size: 1)
        Worker wakes, sees size < 8, waits

t=0.2s: Request 3 arrives → routes to mmlu_gpu0 (shorter queue)
        Queue size: 2, worker still waiting

t=0.3s: Request 4 arrives → routes to mmlu_gpu0
        Queue size: 3, worker still waiting

t=0.4s: Request 5 arrives → routes to mmlu_gpu1
        Queue size: 2 for gpu1, worker still waiting

t=3.0s: GPU0 worker's flush_window expires
        Processes batch of 3 prompts [1, 3, 4]
        Expands to 9 generations (3 × best_of_n=3)
        Single forward pass generates all 9 candidates
        Majority vote per prompt → 3 final answers
        Sets events for requests 1, 3, 4

t=3.1s: GPU1 worker's flush_window expires
        Processes batch of 2 prompts [2, 5]
        Similar process
```

---

## 8. Key Design Decisions

### 8.1 Why Separate Queues per Task and GPU?
- **Task isolation**: Different tasks have different characteristics (MMLU is short, InfoBench is long)
- **GPU parallelism**: Allows concurrent processing of different task types on different GPUs
- **Independent tuning**: Each queue can have custom `flush_window` and `max_batch_size`

### 8.2 Why Shortest Queue Routing?
- **Load balancing**: Prevents one GPU from becoming a bottleneck
- **Latency reduction**: New requests go to less-busy GPU
- **Adaptive**: Automatically adjusts to workload skew

### 8.3 Why Different Flush Windows?
- **Task characteristics**:
  - MMLU: Short responses (256 tokens) → fast generation → short window (3s) acceptable
  - InfoBench: Long responses (512 tokens) → slow generation → longer window (10s) needed to amortize batch overhead

### 8.4 Why GPU Locks?
- **CUDA limitation**: PyTorch models are NOT thread-safe for concurrent forward passes
- **Without locks**: `CUBLAS_STATUS_EXECUTION_FAILED` errors from GPU context corruption
- **With locks**: Serialized GPU access ensures stability

---

## 9. Performance Characteristics

### Latency Components
For a request arriving at time `t`:

1. **Queueing delay**: Wait until batch is ready (0 to `flush_window` seconds)
2. **Processing delay**: Forward pass through model (depends on batch size and sequence length)
3. **Total latency**: `queueing_delay + processing_delay`

### Throughput Optimization
- **Batching efficiency**: Larger batches → better GPU utilization → higher throughput
- **Trade-off**: Larger batches → longer queueing delays → higher latency for individual requests

### Current Tuning
- MMLU: Optimized for **low latency** (3s window, quick batches)
- InfoBench: Optimized for **throughput** (10s window, accumulate more requests)

---

## 10. Code References

### Key Functions
- [_batch_worker (lines 229-286)](system_with_router.py#L229-L286): Core batching loop
- [_enqueue_for_batch (lines 294-316)](system_with_router.py#L294-L316): Request enqueueing
- [_process_by_task (lines 454-662)](system_with_router.py#L454-L662): Task-specific batch processing
- [completions (lines 682-835)](system_with_router.py#L682-L835): Main endpoint with routing logic

### Configuration
- [_microbatch_settings (lines 119-126)](system_with_router.py#L119-L126): Flush windows and batch sizes
- [_batch_queues (lines 129-134)](system_with_router.py#L129-L134): Queue initialization
- [task_settings (lines 102-116)](system_with_router.py#L102-L116): Task-specific parameters

---

## Summary

This is a **dynamic micro-batching system** with:
- **4 independent queues** (2 tasks × 2 GPUs)
- **Time-based + size-based flushing** (dual threshold)
- **Shortest-queue load balancing**
- **Task-specific optimizations** (Graph = direct compute, MMLU = batched best-of-N, InfoBench = standard batching)
- **Thread-safe GPU access** via per-GPU locks
- **Concurrent enqueueing** to maximize batching opportunities

The batching algorithm balances latency (how fast individual requests complete) and throughput (how many requests/second the system handles) through careful tuning of flush windows and batch sizes per task type.
