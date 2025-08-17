# LCM Batch Aggregation Strategy

## Background & Objectives

In RLHF / GRPO training pipelines, different stages (such as **rollout**, **ref logps**, **old logps**) may have different batch requirements. To maximize throughput, reduce HBM pressure, and ensure computational alignment, we introduce the **Least Common Multiple (LCM) Aggregation Strategy**. This strategy allows aggregating multiple training microbatches first, then splitting them when entering the compute stage, avoiding waste and misalignment issues.

## Core Concepts

1. **Input Data**: Sequentially fetch **training microbatches** from iterator (sample counts may be inconsistent)
2. **Aggregation Strategy**: Temporarily place multiple microbatches into buffer until total sample count in buffer ≥ LCM(rollout, ref_logps, old_logps)
3. **Data Processing**
   - Aggregate → repeat (sample-level) → execute generate+advantage in one shot → split back to original microbatch boundaries
4. **Output Data**: Split batches maintain original `TrainExample` data volume, placed in queue for subsequent training

## Parameter Description

- **rollout_micro_batch_size**: Microbatch size for rollout stage
- **ref_logps_micro_batch_size**: Microbatch size for reference logps computation
- **old_logps_micro_batch_size**: Microbatch size for old logps computation
- **service_target_bs**: LCM of the three above, determines trigger point for `_aggregate_and_compute_advantage_from_buffer`
- **sample_repeat**: Repeat batch in sequence dimension (expand sample count)
- **batch_repeat**: Repeat `TrainExample` in data_queue (repeat training for multiple rounds)
- **proceed_num_steps**: Maximum number of microbatches to consume, preventing infinite loops

## Module Design

### 1. Buffer Management

- **buf**: Store temporary microbatches (dict format)
- **buf_sizes**: Record sample count of each microbatch for subsequent splitting
- **buf_B**: Total aggregated sample count in current buffer

### 2. `_aggregate_and_compute_advantage_from_buffer`

#### Trigger Conditions

- `force=True`: Force execution (e.g., reached step limit / iteration end)
- `buf_B ≥ service_target_bs`: Buffer reached LCM threshold

#### Execution Steps

1. **Merge microbatches**: Concatenate dicts in buf by key
2. **Repeat**: Apply one-time repeat to merged batch (sample-level)
3. **Invoke computation**: Execute `_generate_and_compute_advantage`
4. **Split back to original microbatches**: Split back to multiple `TrainExample` based on buf_sizes (multiplied by sample_repeat)
5. **Return results**: Hand over to main loop for processing (unified batch_repeat + enqueue)
6. **Clear buffer**: Prepare for next aggregation round

### 3. Main Loop

1. Fetch one microbatch from iterator
2. Place in buffer, update counters
3. Call `_aggregate_and_compute_advantage_from_buffer(force=False)` to check if production needed
4. Increment step count, check if exceed proceed_num_steps
5. If ending:
   - Call `_aggregate_and_compute_advantage_from_buffer(force=True)` to produce remaining items
   - Uniformly execute batch_repeat and enqueue here, ensuring order **a a b b …**
   - return

## Flow Diagram

### Main Flow

```
while True:
    example = next(iterator)  ← One training microbatch
       ↓
    Place in buffer
       ↓
    if buf_B ≥ service_target_bs → _aggregate_and_compute_advantage_from_buffer()
       ↓
    else continue collecting
```

### `_aggregate_and_compute_advantage_from_buffer` Internal Flow

```
Merge buffer → repeat → one-shot computation → split back to microbatch → return TrainExample[]
```

## Key Implementation

### 1. Initialization Phase

```python
service_target_bs = _lcm3(
    self.rollout_micro_batch_size,
    self.ref_logps_micro_batch_size,
    self.old_logps_micro_batch_size,
)

buf, buf_sizes, buf_B = [], [], 0
consumed_steps = 0
pending_examples: list[TrainExample] = []
```

### 2. `_aggregate_and_compute_advantage_from_buffer(force)`

```python
def _aggregate_and_compute_advantage_from_buffer(force: bool = False) -> list[TrainExample]:
    if not buf: return []
    if (not force) and (buf_B < service_target_bs): return []

    merged = concat_along_batch(buf)
    merged_repeated = jax.tree.map(lambda x: np.repeat(x, sample_repeat, axis=0), merged)

    big_example = self._generate_and_compute_advantage(merged_repeated, mode)

    produced = []
    offset = 0
    for n in buf_sizes:
        token_sl = slice(offset * sample_repeat, (offset + n) * sample_repeat)
        produced.append(TrainExample(...token_sl...))
        offset += n

    reset_buffer()
    return produced
```

### 3. Main Loop

```python
while True:
    example = next(iterator)
    buf.append(example)
    buf_sizes.append(len(example["prompts"]))
    buf_B += len(example["prompts"])
    consumed_steps += 1

    produced_now = _aggregate_and_compute_advantage_from_buffer(False)
    if produced_now: pending_examples.extend(produced_now)

    if proceed_num_steps > 0 and consumed_steps >= proceed_num_steps:
        tail = _aggregate_and_compute_advantage_from_buffer(True)
        if tail: pending_examples.extend(tail)

        # Execute batch_repeat and enqueue here, ensuring order a a b b …
        enqueue_repeated(pending_examples, batch_repeat)
        return
```

### 4. `_rollout_by_micro`

```python
def _rollout_by_micro(self, prompts: list[str], micro: int):
    outs_tokens = []
    outs_text = []
    outs_left_padded = []
    for slc in _chunk_slices_by_size(len(prompts), micro):
        sub_prompts = prompts[slc]
        out = self.rl_cluster.generate(prompts=sub_prompts)
        outs_tokens.append(out.tokens)                         # [b, T_out]
        outs_text.extend(out.text)
        outs_left_padded.append(out.left_padded_prompt_tokens) # [b, T_in]
    completion_ids = jnp.concatenate(outs_tokens, axis=0)
    left_padded = jnp.concatenate(outs_left_padded, axis=0)
    return completion_ids, left_padded, outs_text
```

**Functionality:**

- Split prompts into small batches by `micro` size
- Call `rl_cluster.generate` (model inference) for each small batch
- Collect generated tokens, original text, and left-padded prompt tokens
- Finally concatenate back to complete batch, ensuring external interface remains end-to-end

**Purpose:** Solve HBM explosion problem in generation stage, generate in chunks then merge.

### 5. `_ref_logps_by_micro`

```python
def _ref_logps_by_micro(self, prompt_ids: jnp.ndarray, completion_ids: jnp.ndarray, micro: int):
    pad_id = self.rl_cluster.rollout.pad_id()
    eos_id = self.rl_cluster.rollout.eos_id()
    outs = []
    B = prompt_ids.shape[0]
    for slc in _chunk_slices_by_size(B, micro):
        outs.append(self.rl_cluster.get_ref_per_token_logps(
            prompt_tokens=prompt_ids[slc],
            completion_tokens=completion_ids[slc],
            pad_id=pad_id, eos_id=eos_id
        ))
    return jnp.concatenate(outs, axis=0)
```

**Functionality:**

- Compute **reference model** per-token logp
- Split by `micro` to avoid feeding large batch to ref model at once
- Concatenate results from each segment, maintaining output dimension `[B, T]`

**Purpose:** Reduce ref model computation memory pressure.

### 6. `_old_logps_by_micro`

```python
def _old_logps_by_micro(self, prompt_ids: jnp.ndarray, completion_ids: jnp.ndarray, micro: int):
    outs = []
    B = prompt_ids.shape[0]
    for slc in _chunk_slices_by_size(B, micro):
        outs.append(self.rl_cluster.get_old_per_token_logps(
            prompt_tokens=prompt_ids[slc],
            completion_tokens=completion_ids[slc]
        ))
    return jnp.concatenate(outs, axis=0)
```

**Functionality:**

- Compute **old policy** per-token logp
- Similar to `_ref_logps_by_micro`, use `micro` for chunked processing
- Finally concatenate to `[B, T]`

**Purpose:** Control memory peak for old policy logp computation.

### 7. `_lcm3`

```python
def _lcm3(a: int, b: int, c: int) -> int:
    return (a * b) // math.gcd(a, b) * c // math.gcd(((a * b) // math.gcd(a, b)), c)
```

**Functionality:**

- Calculate LCM of `a, b, c`
- Used to align `rollout_micro / ref_logps_micro / old_logps_micro`

**Purpose:** Ensure aggregated batch can be evenly divided by chunks in each stage.



## Order & Shape Invariants

### Order Guarantee

**Order:** Enqueue order strictly follows **a a b b …** (same micro appears μ consecutive times), consistent with original implementation



## Performance Improvement Analysis

### 1. Host Scheduling & Kernel Launch Overhead Amortization

- **Direct Processing:** Two independent executions, each bearing Python/JAX scheduling, data transfer, RNG/state initialization, kernel launch and other fixed overheads
- **Aggregate→Split:** Completed within one "large service batch" context, sharing one-time setup overhead, device-side submits chunks continuously, fewer fixed overhead hits, launch/sync tail latency amortized

### 2. Compilation & Shape Stability (XLA / PJIT Friendly)

- **Direct Processing:** Two small batches may have different shapes, easily triggering multiple compilations/shape polymorphism
- **Aggregate→Split:** Can unify pad_to_multiple, unified length bucketing, stable shapes after chunking, higher compilation cache hit rate

### 3. Communication/Parallel Topology More Efficient

- **Direct Processing:** Two small batches each trigger one collective (NCCL), high startup/sync overhead ratio for small messages
- **Aggregate→Split:** Sequential chunk submission within same "large logical batch", more coherent collectives, better overall bandwidth utilization

### 4. Better KV/Activation Cache Reuse

- **Direct Processing:** Cache not easily reused between two small batches, decode kernel pipeline hard to warm up
- **Aggregate→Split:** Continuous chunk progression within one service batch, sustained pipeline, KV/activation/attention mask fast-path more likely to hit

### 5. Better Load Balancing (by token budget)

- **Direct Processing:** Length distribution of two small batches may vary greatly, causing large step jitter
- **Aggregate→Split:** Aggregate first then repack by token budget, evenly split chunks, each chunk more uniform

### 6. Logging & Metrics Consistency

- **Direct Processing:** Two independent statistics, small granularity for mean/variance statistics
- **Aggregate→Split:** Unified reward/length distribution/KL metrics on "large logical batch", more stable

### 7. CPU-side Vectorization (tokenize / reward / post-processing)

- **Direct Processing:** High overhead ratio for Python loops and object operations under small batches
- **Aggregate→Split:** One-shot string concatenation, regex, reward evaluation and other CPU-bound work on large batch

### 8. Smoother Async Producer-Consumer

- **Direct Processing:** Consumer (training) easily "sawtoothed" by producer's small batch rhythm
- **Aggregate→Split:** Producer prepares one "large logical batch" at a time, better decoupled rhythm between both ends

## Performance Comparison Example

### Time Model

Device execution time for one submission: `T ≈ T_overhead + T_compute`

- `T_overhead`: Fixed overhead (launch/sync/preparation)
- `T_compute`: Approximately linear with sample count and token count

### Approach Comparison

#### Approach A (Direct Processing): Two small batches

```
T_A = 2 * (T_overhead + T_compute(64))
```

#### Approach B (Aggregate→Split): Aggregate to 128, submit two 64-chunks within one context

```
T_B = T_overhead_once + T_compute(64) + T_compute(64) + small_stage_switching_overhead
    < 2 * T_overhead + 2 * T_compute(64)
```

#### Conclusion

```
T_B < T_A`, because `T_overhead_once` is significantly smaller than `2*T_overhead
```