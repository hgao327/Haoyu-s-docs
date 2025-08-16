# Tunix GRPO Data Flow Analysis

## Data Flow Timeline: From Production to Consumption

Tunix is essentially a producer-consumer pattern that interacts with data through a data queue. This is consistent with the analysis in multi-turn: RLHF can essentially be divided into frontend and backend. The frontend is broadly trajectory generation, while the backend takes the generated training data for actual training.

### 1. Dataset Creation

```python
dataset = get_dataset(TRAIN_DATA_DIR, "train").batch(BATCH_SIZE)[:NUM_BATCHES]
train_dataset = dataset.repeat(NUM_EPOCHS)
```

- Create dataset where `BATCH_SIZE` is actually **microbatch size** (batch size for single forward pass)
- Each iteration returns a microbatch, all subsequent data processing and training use this batch as the basic unit

### 2. gradient_accumulation_steps Configuration

```python
# Training config
cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: mesh,
        rl_cluster_lib.Role.REFERENCE: mesh,
        rl_cluster_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine='vanilla',
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        gradient_accumulation_steps=4,  # Gradient accumulation steps
        # metrics logging
        metrics_logging_options=metrics_logging_options,
        # checkpoint saving
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=TOTAL_GENERATION_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    ),
)
```

**Key Configuration**:

- `gradient_accumulation_steps`: Controls how many microbatches accumulate for one gradient update
- In fact, `gradient_accumulation_steps` microbatches form a mini batch, and one minibatch performs one actual parameter update, which is the main logic of frameworks like veRL

### 3. Start Training

```python
with mesh:
  grpo_trainer.train(dataset)
```

### 4. Create Iterator

```python
train_iterator = iter(train_ds)
```

### 5. Start Asynchronous Data Production Thread

```python
future = self.executor.submit(
    self._prepare_data,
    iterator=train_iterator,
    proceed_num_steps=self.grad_acc_steps,  # K microbatches
    sample_repeat=self.grpo_config.num_generations,  # G expansions
    batch_repeat=self.grpo_config.num_iterations,  # μ repetitions
    data_queue=train_data_queue,
    async_loading=self.can_enable_async_rollout,
    mode=metrics_logger.Mode.TRAIN,
)
```

### 6. Data Production Process (`_prepare_data`)

#### 6.1 Skip Already Trained Steps (if resuming from checkpoint)

```python
while mode==TRAIN and self._train_steps < self._last_train_step:
    next(iterator)
    self._train_steps += 1
```

#### 6.2 Loop Processing K Microbatches

```python
example_list = []
for _ in range(proceed_num_steps):  # proceed_num_steps = grad_acc_steps = K
```

#### 6.3 Get One Microbatch

```python
example = next(iterator)  # Get one microbatch, size is BATCH_SIZE
```

#### 6.4 Expand G Times (GRPO multi-candidate generation)

```python
example = jax.tree.map(lambda x: np.repeat(x, sample_repeat, axis=0), example)
# [B] -> [B * G], B is microbatch size, G is num_generations
```

#### 6.5 Execute Rollout Generation and Advantage Calculation

```python
with jax.profiler.StepTraceAnnotation("sampler", step_num=self._train_steps):
    advantage = self._generate_and_compute_advantage(example, mode)
```

This includes:

- Generate text using policy: `self.rl_cluster.generate(prompts=training_input["prompts"])`
- Calculate reference model logps (if beta != 0): `self.rl_cluster.get_ref_per_token_logps(...)`
- Calculate old policy logps (if num_iterations > 1): `self.rl_cluster.get_old_per_token_logps(...)`
- Calculate rewards: `self._compute_rewards(prompts, completions, ...)`
- Calculate advantages: `grpo_helpers.compute_advantages(rewards, self.grpo_config.num_generations)`

#### 6.6 Immediately Put into Queue in Async Mode

```python
if async_loading:
    data_queue.put([advantage])  # Wrapped as single-element list
```

#### 6.7 Accumulate to example_list

```python
example_list.append(advantage)
self._train_steps += 1
```

#### 6.8 Package After Collecting K Microbatches

```python
if len(example_list) == proceed_num_steps:  # Collected K
    if not async_loading:
        # Sync mode: Put RepeatIterable at once, repeat μ times
        data_queue.put(common.RepeatIterable(example_list, batch_repeat))
    else:
        # Async mode: Already put once, put μ-1 more times
        if batch_repeat > 1:
            data_queue.put(common.RepeatIterable(example_list, batch_repeat - 1))
    return
```

#### 6.9 End Signal

```python
finally:
    data_queue.put(None)  # Tell consumer no more data
```

### 7. Main Thread Consumes Data

#### 7.1 Get Data from Queue

```python
while True:
    curr_train_ds = train_data_queue.get(block=True)  # Blocking wait
    if curr_train_ds is None:  # Received end signal
        break
```

- `curr_train_ds` is `RepeatIterable(example_list, batch_repeat)`
- Contains K processed microbatches, will repeat μ times

#### 7.2 Prepare Evaluation Data (First Time)

```python
if eval_ds and not curr_eval_ds:
    self._prepare_data(
        iterator=iter(eval_ds),
        proceed_num_steps=-1,  # Process all evaluation data
        sample_repeat=self.grpo_config.num_generations,
        batch_repeat=1,  # Evaluation doesn't need repetition
        data_queue=eval_data_queue,
        async_loading=False,  # Synchronous processing
        mode=metrics_logger.Mode.EVAL,
    )
    curr_eval_ds = eval_data_queue.get(block=True)
```

#### 7.3 Update Actor Model

```python
self.rl_cluster.update_actor(
    curr_train_ds,
    curr_eval_ds,
    skip_jit,
)
```

### 8. Actor Update Internal Process

#### 8.1 Call trainer.train

```python
def update_actor(self, train_ds, eval_ds, skip_jit=False):
    self.actor_trainer.train(train_ds, eval_ds, skip_jit)
```

#### 8.2 Create Training Iterator

```python
train_iterator = iter(train_ds)  # train_ds = RepeatIterable([K microbatches], μ)
```

#### 8.3 Training Loop (Repeat μ Times)

```python
for iteration in range(batch_repeat):  # μ times
    for i in range(grad_acc_steps):  # K microbatches
```

#### 8.4 Process Each Microbatch

```python
train_example = next(train_iterator)  # Get one microbatch (already contains advantage)
```

#### 8.5 Execute Training Step

```python
train_loss, aux = train_step(self.model, self.optimizer, train_example)
```

#### 8.6 Gradient Calculation and Accumulation

```python
def _train_step(self, model, optimizer, inputs):
    inputs = self.gen_model_input_fn(inputs)
    
    # Calculate loss and gradients
    grad_fn = nnx.value_and_grad(self.loss_fn, ...)
    out, grads = grad_fn(model, **inputs)
    
    # Update optimizer (internally accumulates gradients)
    optimizer.update(grads)
```

#### 8.7 MultiSteps Optimizer Gradient Accumulation

```python
# If gradient_accumulation_steps is configured
optimizer = optax.MultiSteps(base_optimizer, gradient_accumulation_steps)

# Internal mechanism:
# Step 1: microbatch_1 → Calculate gradients → Accumulate (no parameter update)
# Step 2: microbatch_2 → Calculate gradients → Accumulate (no parameter update)
# ...
# Step K: microbatch_K → Calculate gradients → Accumulate + Update parameters + Reset
```

#### 8.8 Complete One Round of Gradient Accumulation Update

- After processing K microbatches, complete one parameter update
- This process repeats μ times (due to RepeatIterable)

### 9. Synchronize Weights (If Needed)

```python
if self.should_sync_weights:
    with jax.profiler.StepTraceAnnotation("sync_sampler_weights", step_num=initial_train_steps):
        self.rl_cluster.sync_weights()
```

### 10. Check if Maximum Steps Reached

```python
if self._train_steps >= self.rl_cluster.cluster_config.training_config.max_steps:
    break
```

### 11. Wait for Data Production Thread to Complete

```python
future.result()  # Wait for _prepare_data to complete or raise StopIteration
```

### 12. Update Training Steps

```python
self._train_steps = self.rl_cluster.actor_trainer.train_steps
```

### 13. Continue or End Loop

- If there's more data, go back to step 5 to continue
- If data is exhausted (StopIteration), end training

## Data Volume Relationship Summary

A complete data processing cycle:

1. **Input**: K original microbatches (each of size BATCH_SIZE)
2. **Expansion**: Each microbatch expanded G times → K batches of size (BATCH_SIZE × G)
3. **Generation**: Execute rollout and advantage calculation for each expanded batch
4. **Packaging**: K processed batches packaged into RepeatIterable, repeated μ times
5. **Training**: Execute μ rounds, each round processes K microbatches, updates parameters once after accumulating gradients
6. **Result**: Complete μ parameter updates

**Effective Batch Size** = BATCH_SIZE × gradient_accumulation_steps × num_generations

This design achieves large batch training effects with limited memory through asynchronous data preparation, gradient accumulation, and batch repetition.

## Key Issues

1. **Microbatch values that work for Train will definitely work for rollout and ref**
   - Training phase microbatch size is the strictest constraint (requires gradients and optimizer states)
   - Rollout and ref logps calculation can use larger batches
2. **Current logic is opposite to veRL**
   - In veRL, all batch sizes are user-defined, with different parameters for ref model rollout and training process, which better utilizes resources
