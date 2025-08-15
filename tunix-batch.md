# Tunix Data Flow Analysis

`examples/grpo_demo.ipynb`

```python
# ====== Training ======
BATCH_SIZE = 1
# Increase `NUM_BATCHES` and `MAX_STEPS` for better results.
NUM_BATCHES = 3738
# Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly. It can be
# increased to a max. of 330 (if batch size is 4).
NUM_TEST_BATCHES = 100

EVAL_EVERY_N_STEPS = 10  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
NUM_EPOCHS = 1  # can potentially train for more epochs

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)
```

```python
dataset = get_dataset(TRAIN_DATA_DIR, "train").batch(BATCH_SIZE)[:NUM_BATCHES]

if TRAIN_FRACTION == 1.0:
  train_dataset = dataset.repeat(NUM_EPOCHS)
  val_dataset = None
else:
  train_dataset = dataset[: int(len(dataset) * TRAIN_FRACTION)]
  train_dataset = train_dataset.repeat(NUM_EPOCHS)

  val_dataset = dataset[int(len(dataset) * TRAIN_FRACTION) :].repeat(NUM_EPOCHS)

test_dataset = get_dataset(TEST_DATA_DIR, "test").batch(BATCH_SIZE)[
    :NUM_TEST_BATCHES
]

len(train_dataset), len(val_dataset) if val_dataset is not None else 0, len(
    test_dataset
)
```

1. Not using all data
2. Start from micro-batch

Lance solved this

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
        gradient_accumulation_steps=1,
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

gradient_accumulation_steps should be related to microbatch

```python
with mesh:
  grpo_trainer.train(dataset)
```

```
def train(
      self,
      train_ds: Iterable[_TrainingInputT],
      eval_ds: Iterable[_TrainingInputT] | None = None,
      skip_jit: bool = False,
  ) -> None:
```

```python
train_iterator = iter(train_ds)
```

## Start background thread to prepare training data

```
future = self.executor.submit(
    self._prepare_data,
    iterator=train_iterator,
    proceed_num_steps=self.grad_acc_steps,
    sample_repeat=self.grpo_config.num_generations,
    batch_repeat=self.grpo_config.num_iterations,
    data_queue=train_data_queue,
    async_loading=self.can_enable_async_rollout,
    mode=metrics_logger.Mode.TRAIN,
)
```

- Start an **asynchronous task** through `self.executor.submit`, running `_prepare_data`:
  - `proceed_num_steps=self.grad_acc_steps`: Prepare **K** micro-batches at once (gradient accumulation steps).
  - `sample_repeat=num_generations (G)`: Expand each sample into G candidates (required for GRPO).
  - `batch_repeat=num_iterations (μ)`: Train the same batch of samples for μ rounds.
  - Put generated data into `train_data_queue`.

## Main thread consumes training data

```
while True:
    curr_train_ds = train_data_queue.get(block=True)
    if curr_train_ds is None:
        break
```

- Get data from `train_data_queue` (blocking wait).
- `None` is the "end of round" signal put by producer `_prepare_data`.

## First time getting eval data

```python
if eval_ds and not curr_eval_ds:
    self._prepare_data(
        iterator=iter(eval_ds),
        proceed_num_steps=-1,
        sample_repeat=self.grpo_config.num_generations,
        batch_repeat=1,
        data_queue=eval_data_queue,
        async_loading=False,
        mode=metrics_logger.Mode.EVAL,
    )
    curr_eval_ds = eval_data_queue.get(block=True)
```

- If there's eval dataset and haven't prepared yet:
  - Synchronously call `_prepare_data` once (`proceed_num_steps=-1` means take complete eval dataset).
  - Eval doesn't need gradient accumulation, nor repeated training, so `batch_repeat=1`.
  - Get `curr_eval_ds` from eval queue.

## Call trainer to update parameters

```python
self.rl_cluster.update_actor(
    curr_train_ds,
    curr_eval_ds,
    skip_jit,
)  # loop over μ
```

- `curr_train_ds` contains **K micro-batches** (data for one gradient accumulation).
- `update_actor` will:
  - Iterate through micro-batches, compute gradients, accumulate to optimizer.
  - If using `optax.MultiSteps`: only accumulate for first K-1 times, update on the Kth time.
  - μ iterations: same batch of data will be repeatedly optimized for μ rounds (GRPO feature).

```
if self._train_steps >= max_steps:
    break
```

This method is the **data producer**, which takes a certain number of batches (proceed_num_steps) from `iterator` (training or validation set iterator), performs **rollout + advantage calculation**, then puts into `data_queue` for training thread consumption.

It runs asynchronously through

```python
future = self.executor.submit(self._prepare_data, ...)
```

started in the main loop, so it's **asynchronous** data production.

---

## Key parameters (focus on gradient accumulation related)

- `iterator`: Data iterator, each `next(iterator)` gets a **micro-batch** (size = DataLoader's batch_size).
- `proceed_num_steps`: How many micro-batches to take. **Usually equals `grad_acc_steps`**, i.e., gradient accumulation steps.
- `sample_repeat`: How many times to repeat each sample (G = num_generations, used for GRPO multi-candidate generation).
- `batch_repeat`: How many times to reuse the same group of micro-batches (μ = num_iterations).
- `data_queue`: Queue, put for training end consumption.
- `async_loading`: Whether to put data asynchronously; True means generate and put simultaneously, False means wait to collect a group then put all at once.

---

## 3. Flow breakdown

```python
example_list = []
```

Store micro-batches (with advantage) collected in this round.

### (1) Checkpoint resume fast forward

```python
while mode==TRAIN and self._train_steps < self._last_train_step:
    next(iterator)
    self._train_steps += 1
```

Skip already trained batches.

---

### (2) Take one micro-batch and expand G times

```python
example = next(iterator)
example = jax.tree.map(lambda x: np.repeat(x, sample_repeat, axis=0), example)
# [B] -> [B * G]
```

- Here B is micro-batch size (DataLoader batch_size).
- `sample_repeat = G`, expand each sample into G candidates, used for GRPO group relative advantage calculation.
- Number of micro-batches unchanged, only computation load per batch increased.

---

### (3) rollout + advantage

```python
advantage = self._generate_and_compute_advantage(example, mode)
if async_loading:
    data_queue.put([advantage])
```

- Generate policy output (rollout), compute reward, baseline, then get advantage.
- Async mode: immediately put to queue after generating each micro-batch (wrapped as single-element list).

---

### (4) Step counting

```python
if mode == TRAIN:
    self._train_steps += 1
else:
    self._eval_steps += 1
```

---

### (5) Deliver when collected proceed_num_steps

```python
example_list.append(advantage)
if proceed_num_steps > 0 and len(example_list) == proceed_num_steps:
    _put_list_of_examples_to_data_queue()
    return
```

- Here `proceed_num_steps` is generally `grad_acc_steps = K`.
- Means taking K micro-batches (with advantage) at once, this group of data is given to training end for one **gradient accumulation**.
- `_put_list_of_examples_to_data_queue()`:
  - **Sync mode**: Put entire `RepeatIterable(example_list, μ)` at once, μ is batch_repeat;
  - **Async mode**: Already put 1 round before, here only put remaining μ-1 rounds.

Repeating μ times means **same batch of data** will trigger μ rounds of "K micro-batches → accumulate → update" cycles.

---

### (6) Data exhausted handling

```python
except StopIteration as e:
    if proceed_num_steps > 0:
        raise e  # Insufficient data for a group of K micro-batches → end directly
    else:
        _put_list_of_examples_to_data_queue()
        return
```

---

### (7) End signal

```python
finally:
    data_queue.put(None)
```

Tell consumer "no more data".

---

## 4. Gradient accumulation relationship

- **micro-batch** = one batch from each `next(iterator)` (size determined by DataLoader `BATCH_SIZE`).
- **Gradient accumulation steps K** = `proceed_num_steps` (usually = `grad_acc_steps`).
- `_prepare_data` collects K micro-batches at once → put to queue → training end uses these K batches to accumulate gradients and update parameters once.
- μ (`batch_repeat`) in GRPO is repeatedly training μ rounds on the same group of micro-batches.

Microbatch values that work for Train will definitely work for rollout and ref

Seems like microbatch isn't needed?

```python
def update_actor(self, train_ds, eval_ds, skip_jit=False):
    with self.cluster_config.role_to_mesh[Role.ACTOR]:
      self._maybe_load_model_from_cpu(self.actor_trainer.model, Role.ACTOR)
      self.actor_trainer.train(train_ds, eval_ds, skip_jit)
      self._maybe_offload_model_to_cpu(self.actor_trainer.model, Role.ACTOR)
```

```
self._actor_trainer = rl_trainer.Trainer(
        model=self.train_actor,
        optimizer=self.cluster_config.training_config.actor_optimizer,
        training_config=self.cluster_config.training_config,
    )
```

```
def update_actor(self, train_ds, eval_ds, skip_jit=False):
    with self.cluster_config.role_to_mesh[Role.ACTOR]:
      self._maybe_load_model_from_cpu(self.actor_trainer.model, Role.ACTOR)
      self.actor_trainer.train(train_ds, eval_ds, skip_jit)
      self._maybe_offload_model_to_cpu(self.actor_trainer.model, Role.ACTOR)
```

## 1. Outer RL training loop

```python
while True:
    curr_train_ds = train_data_queue.get(block=True)  # Get data
    if curr_train_ds is None:
        break
    self.rl_cluster.update_actor(curr_train_ds, curr_eval_ds, skip_jit)
```

- **`curr_train_ds`** is generated by `_prepare_data()`, type is:
  - `RepeatIterable(example_list, batch_repeat)`
     or directly a list composed of `[advantage1, advantage2, ...]`
- **Features**:
  - Each element in `example_list` is a **microbatch** (size = `BATCH_SIZE`)
  - `grad_acc_steps` controls how many microbatches one update will consume

---

## 2. `_prepare_data()` produces microbatch

```python
example = next(iterator)                   # Take one batch (actually microbatch)
example = np.repeat(example, sample_repeat, axis=0)  # Multiply by num_generations
advantage = self._generate_and_compute_advantage(example)
example_list.append(advantage)
if len(example_list) == proceed_num_steps: # proceed_num_steps = grad_acc_steps
    data_queue.put(RepeatIterable(example_list, batch_repeat))
```

- **Key points**:
  1. `iterator` comes from `get_dataset(...).batch(BATCH_SIZE)`
      → The `BATCH_SIZE` here is actually microbatch size
  2. `proceed_num_steps = grad_acc_steps`
      → Means it will take `grad_acc_steps` microbatches and put into `example_list`
  3. `RepeatIterable(example_list, batch_repeat)`
      → Will repeat this group of microbatches `batch_repeat` times (num_iterations)

---

## 3. `update_actor()` internally just calls `train()`

```python
train_iterator = iter(train_ds)  # train_ds = curr_train_ds = RepeatIterable([...])
while True:
    train_example = next(train_iterator)   # Take one microbatch
    train_loss, aux = train_step(...)
    self._train_steps += 1
```

- Each `next(train_iterator)` takes a microbatch
- `grad_acc_steps` determines how many microbatches accumulate for one gradient update
- If `batch_repeat > 1`, `RepeatIterable` will make the same batch of microbatches be reused multiple times

---

```
train_loss, aux = train_step(
              self.model, self.optimizer, train_example
          )
```

```
def _train_step(
      self, model: nnx.Module, optimizer: nnx.ModelAndOptimizer, inputs: Any
  ) -> ArrayLike | Tuple[ArrayLike, Any]:
    """Main body for one train step.

    Args:
      model: The model to train.
      optimizer: The optimizer to use.
      inputs: The training input.

    Returns:
      The loss and auxiliary data if has_aux is True, otherwise the loss.
    """
    inputs = self.gen_model_input_fn(inputs)

    grad_fn = nnx.value_and_grad(
        self.loss_fn,
        argnums=nnx.DiffState(0, nnx.LoRAParam) if self._lora_enabled else 0,
        has_aux=self._has_aux,
    )
    out, grads = grad_fn(model, **inputs)
    optimizer.update(grads)
    if self._has_aux:
      loss, aux = out
      return loss, aux
    else:
      return out, None
```

```
if training_config.gradient_accumulation_steps is not None:
      optimizer = optax.MultiSteps(
          optimizer, training_config.gradient_accumulation_steps
      )
```

```
optax.MultiSteps
```

When using `optax.MultiSteps`, the input is **microbatch**, not minibatch.

- **Microbatch**: Small batch used in single forward pass (limited by memory)
- **Minibatch**: Logical training batch (effect after accumulating multiple microbatches)

## Usage:

```python
import optax

# 1. Create base optimizer
base_optimizer = optax.adam(learning_rate=1e-3)

# 2. Wrap with MultiSteps, set accumulation steps
gradient_accumulation_steps = 4
optimizer = optax.MultiSteps(base_optimizer, gradient_accumulation_steps)

# 3. Initialize optimizer state
opt_state = optimizer.init(params)

# 4. Training loop
for microbatch in dataloader:  # Note: this is microbatch
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(params, microbatch)
    
    # Update parameters (internally accumulates gradients)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
```

## Internal working mechanism:

```python
# Assume gradient_accumulation_steps = 4
# Effective batch size = microbatch_size × 4

Step 1: microbatch_1 → compute gradients → accumulate (no parameter update)
Step 2: microbatch_2 → compute gradients → accumulate (no parameter update)  
Step 3: microbatch_3 → compute gradients → accumulate (no parameter update)
Step 4: microbatch_4 → compute gradients → accumulate + update parameters + reset
```

## Actual effect:

If your memory can only support microbatch of size 8, but you want training effect of batch size 32:

```python
microbatch_size = 8
gradient_accumulation_steps = 4  # 32 / 8 = 4
effective_batch_size = microbatch_size × gradient_accumulation_steps  # = 32
```

This way we can get large batch training stability and effect with limited memory.

Solution 1: Maintain current approach, add batch accumulation operations at logps calculation location

Solution 2: Align with verl logic, no longer have users specify accumulation steps, but calculate through mini batch and micro batch

mini batch/micro batch gets accumulation steps

The current meaning of batch becomes mini batch, at logps calculation location, use our own microbatch. During training phase, use our own microbatch.
