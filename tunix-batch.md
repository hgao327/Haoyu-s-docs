## Tunix data flow analysis

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

1. 没有使用全部数据
2. start from micro-batch

Lance解决了



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

gradient_accumulation_steps应该 和 microbatch有关系

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



## 启动后台线程准备训练数据

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

- 通过 `self.executor.submit` 启动一个**异步任务**，运行 `_prepare_data`：
  - `proceed_num_steps=self.grad_acc_steps`：一次准备 **K** 个 micro-batch（梯度累积步数）。
  - `sample_repeat=num_generations (G)`：每个样本扩成 G 个候选（GRPO 需要）。
  - `batch_repeat=num_iterations (μ)`：同一批样本训练 μ 轮。
  - 生成好的数据放进 `train_data_queue`。





## 主线程消费训练数据

```
while True:
    curr_train_ds = train_data_queue.get(block=True)
    if curr_train_ds is None:
        break
```

- 从 `train_data_queue` 取数据（阻塞等待）。
- `None` 是生产者 `_prepare_data` 放进去的“本轮结束”信号。



## 首次取 eval 数据

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

- 如果有 eval 数据集，并且还没准备过：
  - 同步调用 `_prepare_data` 一次（`proceed_num_steps=-1` 表示取完整 eval 数据集）。
  - eval 不需要梯度累积，也不重复训练，所以 `batch_repeat=1`。
  - 从 eval 队列取到 `curr_eval_ds`。



## 调用训练器更新参数

```python
self.rl_cluster.update_actor(
    curr_train_ds,
    curr_eval_ds,
    skip_jit,
)  # loop over μ
```

- `curr_train_ds` 里包含 **K 个 micro-batch**（一次梯度累积用的数据）。
- `update_actor` 会：
  - 遍历 micro-batch，计算梯度，累积到 optimizer。
  - 如果用 `optax.MultiSteps`：前 K-1 次只累加，第 K 次才更新。
  - μ 次迭代：同一批数据会重复优化 μ 轮（GRPO 特性）。



```
if self._train_steps >= max_steps:
    break
```





这个方法是**数据生产者**，会从 `iterator`（训练集或验证集迭代器）里取出一定数量的批次（proceed_num_steps 个），做 **rollout + advantage 计算**，然后放入 `data_queue` 给训练线程用。

它的运行是在主循环里用

```python
future = self.executor.submit(self._prepare_data, ...)
```

启动的，所以是**异步**生产数据。

------

##  关键参数（和梯度累积相关的重点）

- `iterator`：数据迭代器，每次 `next(iterator)` 拿到的就是**一个 micro-batch**（大小 = DataLoader 的 batch_size）。
- `proceed_num_steps`：要取多少个 micro-batch。**通常等于 `grad_acc_steps`**，即梯度累积步数。
- `sample_repeat`：每个样本重复多少次（G = num_generations，用于 GRPO 多候选生成）。
- `batch_repeat`：同一组 micro-batch 重复使用多少次（μ = num_iterations）。
- `data_queue`：队列，放给训练端消费。
- `async_loading`：是否异步放数据；True 表示边生成边放，False 表示等攒够一组后一次性放。

------

## 3. 流程拆解

```python
example_list = []
```

存放本轮攒到的 micro-batch（含 advantage）。

### (1) 断点续训快进

```python
while mode==TRAIN and self._train_steps < self._last_train_step:
    next(iterator)
    self._train_steps += 1
```

跳过已训练过的 batch。

------

### (2) 取一个 micro-batch 并扩 G 倍

```python
example = next(iterator)
example = jax.tree.map(lambda x: np.repeat(x, sample_repeat, axis=0), example)
# [B] -> [B * G]
```

- 这里的 B 是 micro-batch 大小（DataLoader batch_size）。
- `sample_repeat = G`，把每条样本扩成 G 个候选，用于 GRPO 计算 group relative advantage。
- micro-batch 个数没变，只是单个 batch 的计算量变大了。

------

### (3) rollout + advantage

```python
advantage = self._generate_and_compute_advantage(example, mode)
if async_loading:
    data_queue.put([advantage])
```

- 生成策略输出（rollout），计算奖励、基线，再得到 advantage。
- 异步模式：每生成一个 micro-batch 就立即放到队列（包装成单元素 list）。

------

### (4) 步数计数

```python
if mode == TRAIN:
    self._train_steps += 1
else:
    self._eval_steps += 1
```

------

### (5) 攒够 proceed_num_steps 就交付

```python
example_list.append(advantage)
if proceed_num_steps > 0 and len(example_list) == proceed_num_steps:
    _put_list_of_examples_to_data_queue()
    return
```

- 这里 `proceed_num_steps` 一般就是 `grad_acc_steps = K`。
- 意味着一次要取 K 个 micro-batch（带 advantage），这组数据交给训练端做一次**梯度累积**。
- `_put_list_of_examples_to_data_queue()`：
  - **同步模式**：一次性放整个 `RepeatIterable(example_list, μ)`，μ 是 batch_repeat；
  - **异步模式**：前面已放过 1 轮，这里只放剩下的 μ-1 轮。

重复 μ 次意味着**同一批数据**会触发 μ 轮 “K 个 micro-batch → 累积 → 更新” 循环。

------

### (6) 数据用完处理

```python
except StopIteration as e:
    if proceed_num_steps > 0:
        raise e  # 数据不足一组 K 个 micro-batch → 直接结束
    else:
        _put_list_of_examples_to_data_queue()
        return
```

------

### (7) 结束信号

```python
finally:
    data_queue.put(None)
```

告诉消费者“没有更多数据了”。

------

## 4. 梯度累积关系

- **micro-batch** = 每次 `next(iterator)` 拿到的一个 batch（大小由 DataLoader `BATCH_SIZE` 决定）。
- **梯度累积步数 K** = `proceed_num_steps`（通常 = `grad_acc_steps`）。
- `_prepare_data` 一次攒满 K 个 micro-batch → 放到队列 → 训练端用这 K 个 batch 累积梯度一次更新参数。
- GRPO 里的 μ（`batch_repeat`）是在同一组 micro-batch 上重复训练 μ 轮。



Train都ok的microbatch数值，rollout和ref中一定更能成功

似乎不需要microbatch？



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















## 1. 外层 RL 训练循环

```python
while True:
    curr_train_ds = train_data_queue.get(block=True)  # 取数据
    if curr_train_ds is None:
        break
    self.rl_cluster.update_actor(curr_train_ds, curr_eval_ds, skip_jit)
```

- **`curr_train_ds`** 是 `_prepare_data()` 生成的，类型是：
  - `RepeatIterable(example_list, batch_repeat)`
     或者是直接 `[advantage1, advantage2, ...]` 组成的 list
- **特点**：
  - 这里的 `example_list` 里每个元素就是一个 **microbatch**（大小 = `BATCH_SIZE`）
  - `grad_acc_steps` 控制一次 update 会消费多少个 microbatch

------

## 2. `_prepare_data()` 生产 microbatch

```python
example = next(iterator)                   # 取一个 batch（其实是 microbatch）
example = np.repeat(example, sample_repeat, axis=0)  # 乘 num_generations
advantage = self._generate_and_compute_advantage(example)
example_list.append(advantage)
if len(example_list) == proceed_num_steps: # proceed_num_steps = grad_acc_steps
    data_queue.put(RepeatIterable(example_list, batch_repeat))
```

- **关键点**：
  1. `iterator` 来源于 `get_dataset(...).batch(BATCH_SIZE)`
      → 这里的 `BATCH_SIZE` 实际上就是 microbatch size
  2. `proceed_num_steps = grad_acc_steps`
      → 意味着会取 `grad_acc_steps` 个 microbatch 放进 `example_list`
  3. `RepeatIterable(example_list, batch_repeat)`
      → 会把这组 microbatch 重复 `batch_repeat` 次（num_iterations）

------

## 3. `update_actor()` 内部实际就是调用 `train()`

```python
train_iterator = iter(train_ds)  # train_ds = curr_train_ds = RepeatIterable([...])
while True:
    train_example = next(train_iterator)   # 取一个 microbatch
    train_loss, aux = train_step(...)
    self._train_steps += 1
```

- 每一次 `next(train_iterator)` 取的就是一个 microbatch
- `grad_acc_steps` 决定了多少个 microbatch 累积一次梯度更新
- 如果 `batch_repeat > 1`，`RepeatIterable` 会让同一批 microbatch 被重复使用多次

------

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



使用 `optax.MultiSteps` 时，输入的是 **microbatch**，而不是 minibatch。

- **Microbatch**: 单次前向传播使用的小批次（受显存限制）
- **Minibatch**: 逻辑上的训练批次（多个 microbatch 累积后的效果）

## 使用方式：

```python
import optax

# 1. 创建基础优化器
base_optimizer = optax.adam(learning_rate=1e-3)

# 2. 用 MultiSteps 包装，设置累积步数
gradient_accumulation_steps = 4
optimizer = optax.MultiSteps(base_optimizer, gradient_accumulation_steps)

# 3. 初始化优化器状态
opt_state = optimizer.init(params)

# 4. 训练循环
for microbatch in dataloader:  # 注意：这里是 microbatch
    # 计算损失和梯度
    loss, grads = jax.value_and_grad(loss_fn)(params, microbatch)
    
    # 更新参数（内部会累积梯度）
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
```

## 内部工作机制：

```python
# 假设 gradient_accumulation_steps = 4
# 有效批次大小 = microbatch_size × 4

步骤1: microbatch_1 → 计算梯度 → 累积（不更新参数）
步骤2: microbatch_2 → 计算梯度 → 累积（不更新参数）  
步骤3: microbatch_3 → 计算梯度 → 累积（不更新参数）
步骤4: microbatch_4 → 计算梯度 → 累积 + 更新参数 + 清零
```

## 实际效果：

如果你的显存只能支持批次大小为 8 的 microbatch，但你想要批次大小为 32 的训练效果：

```python
microbatch_size = 8
gradient_accumulation_steps = 4  # 32 / 8 = 4
effective_batch_size = microbatch_size × gradient_accumulation_steps  # = 32
```

这样就能在有限显存下获得大批次训练的稳定性和效果。









方案1: 维持现在的，在logps计算位置，加入batch累加的操作

方案2: 与verl逻辑对其，不再又用户指定accumulation steps，而是通过mini batch和micro batch进行计算

mini batch/micro batch 得到 accumulation steps

batch现在的含义变成了mini batch，logps计算位置，用自己的microbatch。training阶段，用自己的microbatch。

