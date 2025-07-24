# Google Tunix 源码分析

### 1. `tunix/rl/grpo/grpo_learner.py`

####  `train()`：主训练循环

- 整个 GRPO 的训练流程入口。
- 对应 GRPO 论文中的 outer loop（每轮 iteration）。
- 核心包括：
  1. **prepare_dataset**：采样多个 completions，打分，计算优势。
  2. **trainer.train**：进行 μ 次参数更新（内层 RL 更新）。
  3. **sync_sampler_weights**：将 trainer 的模型参数同步给 sampler。

------

#### 🟦 `prepare_dataset()`：准备训练数据

- 接收 prompt dataset，生成训练数据 `TrainExample`。
- 每条 prompt 采样 G 个 completions → 用 reward model 打分。
- 调用 `_generate_and_compute_advantage` 来处理采样和 advantage 计算。

------

#### 🟨 `_generate_and_compute_advantage()`：生成 + 打分 + Advantage 计算

- 调用 `rollout_worker.generate()` 生成多个回复。
- 使用 `reward_fn(prompts, completions)` 对每个回复打分。
- 调用 `compute_advantages()` 进行归一化。

------

#### 🟩 `compute_advantages()`：

```python
(rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
```

- 实现论文中的 group relative advantage。
- 对每组 G 个回答，进行组内标准化。

------

#### 🟧 `_compute_rewards()`：

- 支持多个 reward function，对每个回复调用 `reward_fn(prompts, completions)`。
- 返回 `[num_prompts * G]` 长度的奖励分数。

------

#### 🟥 `grpo_loss_fn()`：

- 采用 PPO-style clipped loss：

```python
- min(r_t * A, clip(r_t, 1 - ε, 1 + ε) * A)
```

- 同时加入 KL 约束项：

```python
+ β * KL(trained || reference)
```

- 这里不使用 value function，而是直接使用优势（A）来自 reward。

------

#### 🔁 `trainer.train()`：

- 实际执行梯度反向传播，参数更新。
- 会执行 μ 次（即 `GrpoConfig.num_iterations`）每次使用相同的样本。

------

#### 🔄 `sync_sampler_weights()`：

- 将 `trainer` 更新后的参数同步给 `rollout_worker`。
- 如果模型使用了 LoRA，会仅同步 LoRA 权重。









### `2. tunix/rl/inference/inference_worker.py`

这段代码位于 `tunix.rl` 模块，定义了一个名为 `InferenceWorker` 的类，用于托管在强化学习中推理阶段会用到的模型，例如：

- `critic`：用来估算状态-动作值（未实现）
- `reference`：参考模型，用于计算 KL 散度
- `reward`：奖励模型，用来对生成的回答评分（未实现）

------

#### 引入的模块

```python
from flax import nnx
import jax
from tunix.rl import common
```

- `nnx` 是 Flax 中的模块构建系统
- `jax` 提供数组和设备加速
- `common` 模块中包含一些常用函数，比如计算 logp 的方法

------

#### 类定义：`InferenceWorker`

```python
class InferenceWorker:
  """Inference worker hosting critic, reference and reward models."""
```

这是一个通用的推理工作类，负责“托管”一些 RL 相关的模型。

------

#### 构造函数

```python
def __init__(self, models: dict[str, nnx.Module]):
```

接受一个 `dict` 类型的模型容器。它检查：

```python
for k in models.keys():
  if k not in ["critic", "reference", "reward"]:
    raise ValueError(...)
```

确保只包含合法的模型角色。

接着它将 `models` 存入实例变量：

```python
self._models = models
```

------

#### 方法：`get_ref_per_token_logps`

```python
def get_ref_per_token_logps(self, prompt_tokens, completion_tokens, pad_id, eos_id):
```

此函数使用参考模型（`reference`）来计算：

- 输入为 prompt 和生成的 completion（两者都为 token 数组）
- 调用 `common.compute_per_token_logps` 函数进行处理

```python
return common.compute_per_token_logps(...)
```

这个函数最终返回一个 `jax.Array`，表示参考模型在每个生成 token 上的 log 概率，常用于计算 RL 中的 KL 散度约束项。

------

#### 方法：`compute_rewards`

```python
def compute_rewards(self):
  raise NotImplementedError()
```

保留接口未实现，未来用于使用 reward 模型打分。

------

#### 方法：`compute_values`

```python
def compute_values(self):
  raise NotImplementedError()
```



这个模型**不是用来生成文本的**，而是用来对生成结果进行**评分（reward）**、**参考对比（reference KL）和估值（critic）**，用于强化学习训练中的评估阶段。生成文本是由 `actor` 模型完成的。









### 3. `tunix/rl/queue/data_queue.py`

------

#### 👇 在 `GrpoLearner.train()` 中会看到这样一段代码：

```python
train_data_queue = queue_lib.SimpleDataQueue(maxsize=self.grad_acc_steps + 2)
```

然后异步调用：

```python
self.executor.submit(
    self.prepare_dataset,
    ...,
    data_queue=train_data_queue,
)
```

再在主线程中消费：

```python
while True:
  curr_train_ds = train_data_queue.get(block=True)
  if curr_train_ds is None:
    break
  self.rl_cluster.update_actor(curr_train_ds, ...)
```

------

#### 🔄 这背后的流程结合 `SimpleDataQueue` 是这样的：

| 操作         | 方法                   | 作用                                           |
| ------------ | ---------------------- | ---------------------------------------------- |
| 数据准备线程 | `data_queue.put(...)`  | 把 `TrainExample` 或 `RepeatIterable` 放入队列 |
| 主训练线程   | `data_queue.get()`     | 从队列中取出数据并进行训练                     |
| 训练完成     | `data_queue.put(None)` | 发送“结束信号”                                 |
| 清理资源     | `data_queue.close()`   | 清空队列中残留的数据                           |

------

#### 🧠 关键特性

- `SimpleDataQueue` 是 `AbstractDataQueue` 的具体实现，底层使用了 Python 标准库的 `queue.Queue`
- 使用了泛型 `_T`，可以传递任意类型数据（如 `TrainExample`, `List[TrainExample]`）
- `close()` 方法是为了优雅终止：将队列清空，防止资源泄漏或阻塞

------

#### ✅ 总结

这段代码是为了**在训练过程中异步传输数据**，让数据准备和模型训练解耦，提升并发性能。其中 `SimpleDataQueue` 是一个轻量级的线程安全队列，用于在 `prepare_dataset()` 和 `update_actor()` 之间传递训练样本。







### 4. `tunix/rl/rollout/vanilla_rollout.py`

这段代码实现了 Tunix 中的 **Vanilla Rollout Worker**，用于**从 actor 模型生成文本**（也就是推理 / rollout）并支持 KV Cache 管理和 token logp 的计算。是整个 GRPO / PPO 强化学习中的**核心生成模块**。

------

#### 🧠 类职责概览

| 类 / 方法               | 功能                                       |
| ----------------------- | ------------------------------------------ |
| `VanillaRollout`        | 托管 rollout 过程，使用 `Sampler` 完成生成 |
| `generate()`            | 从模型生成文本（核心的 rollout 步骤）      |
| `get_per_token_logps()` | 获取生成 token 的逐 token log 概率         |
| `update_params()`       | 在 RL 训练中接收新参数并更新模型           |
| `pad_id()` / `eos_id()` | 获取 tokenizer 中的特殊 token ID           |

------

#### 🔍 关键组件逐一解释

##### 1. `CacheConfig`

```python
@dataclasses.dataclass(frozen=True)
class CacheConfig:
  cache_size: int
  num_layers: int
  num_kv_heads: int
  head_dim: int
```

定义 KV Cache 的配置参数，用于初始化 `Sampler` 时的缓存空间大小。

------

##### 2. `VanillaRollout.__init__`

```python
self._sampler = sampler.Sampler(...)
```

- 实例化 `Sampler`，用于执行高效的 token 生成
- 传入模型、tokenizer 和 KV cache 配置

------

##### 3. `generate(prompts, rollout_config, **kwargs)`

```python
output = self._sampler(
    input_strings=prompts,
    ...
)
return base_rollout.RolloutOutput(...)
```

- 核心推理函数
- 调用 `Sampler.__call__` 执行批量文本生成
- `RolloutConfig` 控制生成长度、温度、top-p、top-k 等 sampling 参数
- 输出包括文本、logits、tokens 以及 padded 的 prompt tokens

------

##### 4. `get_per_token_logps(prompt_tokens, completion_tokens)`

```python
return common.compute_per_token_logps(...)
```

- 用于计算策略在生成 completion 上的 log 概率
- 用于训练时 reward 加权或 KL 散度计算

------

##### 5. `update_params(params)`

```python
flat_new_params = utils.to_flat_dict(params)
...
merged_params = jax.tree.unflatten(...)
self._sampler.transformer_state = ...
```

- 用于 RL 训练中将新权重参数更新到 sampler 的模型中
- 支持原地更新 transformer 权重

------

##### 6. 其他辅助方法

```python
def pad_id(self) -> int:
def eos_id(self) -> int:
def model(self) -> nnx.Module:
```

- `pad_id()` 和 `eos_id()` 返回 tokenizer 的特殊 token 编号
- `model()` 返回当前的 transformer 模块实例

------

#### ✅ 小结

你可以理解 `VanillaRollout` 是 **actor 模型的生成接口包装器**，它：

- 使用 `Sampler` 生成文本
- 支持从外部加载参数（用于 policy 更新）
- 提供 token-level logp，用于 loss 计算
- 支持 GRPO/PPO 中 rollout 的所有需求





### 5. `tunix/rl/common.py`

这段代码是 Tunix 中 RL 训练所用的**通用辅助函数集合**，用于处理：

1. 模型 log probability 的计算
2. mask 和 attention mask 构造
3. padding 与位置编码
4. 为 GRPO / PPO 中 KL loss 和 advantage 计算等提供底层支持

------

#### 🔍 关键函数解释

#### ✅ 1. `selective_log_softmax(logits, input_ids)`

从模型 logits 中提取每个 token 的 log probability：

```python
logps = jax.nn.log_softmax(logits, axis=-1)
per_token_logps = jnp.take_along_axis(logps, input_ids[..., None], axis=-1)
return per_token_logps[..., 0]
```

------

#### ✅ 2. `get_per_token_logps(model, input_tokens, positions, attn_mask, logits_to_keep)`

执行模型前向，得到 logits，选取最后 `logits_to_keep` 个 token：

```python
logits, _ = model(...)
logits = logits[:, -logits_to_keep - 1 : -1, :]
input_tokens = input_tokens[:, -logits_to_keep:]
return selective_log_softmax(logits, input_tokens)
```

用于 reward、KL 或 RL loss 的 token logp 计算。

------

#### ✅ 3. `compute_per_token_logps(model, prompt_tokens, completion_tokens, pad_id, eos_id)`

完整包装函数，接收 `prompt + completion`：

- 拼接后构造 `prompt_completion_mask`
- 生成 RoPE 所需的位置编码 + causal attention mask
- 调用 `get_per_token_logps`

可选 `stop_gradient` 以避免反向传播。

------

#### ✅ 4. `make_completion_mask(completion_ids, eos_tok)`

为每条 `completion` 生成 mask，遇到 `eos_id` 后 padding：

```python
completion_mask = (sequence_indices <= eos_idx[:, None]).astype(jnp.int32)
```

例如：
 输入：`[42, 17, 9, <eos>, 0, 0]`
 输出：`[1, 1, 1, 1, 0, 0]`

------

#### ✅ 5. `make_causal_attn_mask(input_mask)`

构造 causal attention mask（每个 token 只能看自己和前面）：

```python
causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
attn_mask = input_mask[..., None, :] * causal_mask[None, ...]
```

------

#### ✅ 6. `pad_to_length(x, target_length, pad_value, left=False, axis=0)`

通用的 padding 工具函数，可以指定：

- 左右 pad
- 哪个轴 pad

例如：

```python
pad_to_length([1, 2], 4, pad_value=0) → [1, 2, 0, 0]
```

------

#### ✅ 7. `build_positions_from_mask(input_mask)`

根据 mask 构建位置编码索引：

```python
positions = jnp.cumsum(input_mask, axis=-1) - 1
```

- 如果 mask 为 `[1, 1, 0, 1]`
   → 位置为 `[0, 1, 0, 2]` （跳过 pad）










### 6. `tunix/rl/reshard.py`

这段代码提供了 **JAX 中用于跨 mesh 或跨 sharding 的参数迁移（resharding）功能**，是 Tunix 框架中支持多设备训练（尤其是 TPUs）时的重要组件。

------

#### 🔧 功能简述

- `reshard_pytree(...)`：将一个 `PyTree`（通常是模型参数）从一种设备分布方式（sharding）迁移到另一种
- 支持使用 Google 内部的 Pathways 工具（如果可用），否则降级为 `jax.device_put`
- 提供异步 ready 回调 `callback_on_ready`，用于记录迁移成功或失败的耗时

------

#### 📦 逐步讲解

#### 1. `callback_on_ready`

```python
def callback_on_ready(x, success, failure):
```

- 用于在 resharding 结束后，**异步回调 success 或 failure**
- 会开一个新线程执行 `jax.block_until_ready(x)` 来阻塞等待数组完成，然后触发 callback

------

#### 2. `reshard_pytree(...)`

核心函数，参数如下：

```python
reshard_pytree(source, target, cache_plan=True, donate_input=False, use_experimental_pre_reshard=True)
```

- `source`：原始 tensor 或 PyTree（包含分布策略）
- `target`：目标 tensor 或 PyTree，用来提供目标 sharding 信息
- `cache_plan`：是否缓存 reshard 计划，加速后续重复迁移
- `donate_input`：是否把输入数组所有权“让渡”出去（减少拷贝）
- `use_experimental_pre_reshard`：是否尝试使用谷歌内部优化路径

------

#### 3. `_get_dst_sharding`

用于从 `target` 树中提取或构建目标 `sharding` 对象：

```python
if isinstance(x, NamedSharding | SingleDeviceSharding): return x
else: return NamedSharding(x.sharding.mesh, x.sharding.spec)
```

这个函数用于处理 JAX 的 Sharding 对象。

------

#### 4. 动态选择 reshard 实现

```python
if reshardfn is None:
  try:
    import pathwaysutils
    from pathwaysutils.experimental import reshard
    reshardfn = functools.partial(experimental_reshard.reshard, x=source)
  except ImportError:
    logging.error("Can't import PathwaysUtils...")
```

- 如果能用 Google 内部的 `pathwaysutils.experimental.reshard`，就用它（可能更快）
- 否则 fallback 到：

```python
jax.device_put(source, dst_shardings)
```

这是普通的设备迁移函数。

------

#### 5. 异步日志记录

```python
callback_on_ready(resharded_array, on_success, on_failure)
```

迁移完成后记录日志，包括用时或失败信息。







### 7. `tunix/rl/rl_cluster.py`

这段代码定义了 `RLCluster` 类，是 Tunix 框架中面向用户的**核心接口类**，用于管理和协调 RLHF / GRPO 强化学习训练中涉及的多个模型与子组件。

------

#### 🧠 简洁理解：RLCluster 做什么？

`RLCluster` 封装了：

| 模块               | 功能                                             |
| ------------------ | ------------------------------------------------ |
| `train_actor`      | RL 训练使用的 actor 模型                         |
| `rollout_actor`    | rollout 用于生成 response 的 actor 模型          |
| `critic`           | 值函数模型                                       |
| `reference`        | KL 散度参考模型                                  |
| `reward`           | 奖励模型                                         |
| `trainer`          | 用于训练模型的优化器和训练逻辑                   |
| `inference_worker` | 包装 reference / reward / critic 用于评估        |
| `rollout`          | 调用 `Sampler` 实际生成文本的接口                |
| `sync_weights()`   | 训练和采样模型之间同步参数（支持 LoRA 和全参数） |

------

#### 🔍 核心逻辑拆解

#### 1. `__init__`

构造函数中主要做了 3 件事：

- 加载所有模型到指定 `Mesh` 上（可能会自动 reshard）
- 构造 rollout 模型（使用 `VanillaRollout` 或将来支持 `vLLM`）
- 构造 inference_worker + trainer

```python
self.train_actor = self._load_model(actor, mesh)
self._rollout = VanillaRollout(...)
self._inference_worker = InferenceWorker(...)
self._actor_trainer = Trainer(...)
```

------

#### 2. `_load_model(...)`

```python
def _load_model(self, model_or_path: nnx.Module | str, mesh: Mesh)
```

- 如果是 `nnx.Module` 实例，会检查模型当前 mesh 是否匹配目标 mesh
- 如果不匹配，则调用 `reshard.reshard_pytree()` 迁移权重
- 暂不支持从路径加载模型（`NotImplementedError`）

------

#### 3. `generate(prompts)`

```python
return self.rollout.generate(prompts, rollout_config)
```

在 `ROLLOUT` 设备上执行，调用 `Sampler` 生成 response。

------

#### 4. `update_actor()` / `update_critic()`

```python
self.actor_trainer.train(train_ds, eval_ds, skip_jit)
```

调用 `Trainer.train()` 执行优化器更新。

------

#### 5. `get_ref_per_token_logps()` / `get_old_per_token_logps()`

分别用于：

- KL 散度计算（由 reference 模型推理）
- 旧策略 logp 获取（由 rollout actor 推理）

------

#### 6. `sync_weights()`

```python
if is_lora_enabled(model):
    src = LoRAParam(actor_trainer)
    dst = LoRAParam(rollout)
else:
    src = Param(actor_trainer)
    dst = Param(rollout)
reshard_pytree(src, dst)
rollout.update_params(...)
```

- 将训练得到的新权重同步到 rollout actor，用于下一轮采样
- 支持 LoRA 与全参数两种同步方式
- 使用 `reshard_pytree` 保障 mesh / sharding 对齐

------

#### ✅ 总结

可以把 `RLCluster` 理解为 Tunix 中 RLHF 的“训练协调中心”，它屏蔽了多设备部署、模型迁移、采样、训练、评估等所有复杂细节，统一对外提供：

- `generate()` → rollout
- `update_actor()` / `update_critic()` → 训练
- `sync_weights()` → rollout <-> trainer 权重同步





### 8. `tunix/rl/trainer.py`

这段代码定义了 `Trainer` 类，是强化学习中对 `PeftTrainer` 的一个扩展版本，核心目的是为 **RL 训练过程添加日志记录和进度条显示功能**。

它是整个 Tunix RLHF / GRPO 框架中用于训练 actor / critic 的通用训练器。

------

#### 🧠 类结构说明：`Trainer`

继承自：

```python
from tunix.sft import peft_trainer
class Trainer(peft_trainer.PeftTrainer)
```

主要职责是：

1. 添加和记录 RL 中自定义指标（如 KL）
2. 显示自定义进度条指标（比如 reward, kl 等）
3. 保留原始 SFT 训练的能力（来自 PeftTrainer）

------

#### 🔍 方法逐项解释

#### `__init__(...)`

调用 `super()` 初始化模型、优化器和训练配置。然后初始化两个字段：

```python
self.rl_metrics_to_log = {}  # eg: {"kl": "kl"}
self.tqdm_metrics_to_display = []  # eg: ["kl", lambda: "rewards/overall"]
```

------

#### `with_rl_metrics_to_log(...)`

```python
def with_rl_metrics_to_log(self, rl_metrics_to_log: dict[str, str])
```

注册你希望记录到 `metrics_logger` 中的指标。

示例：

```python
trainer.with_rl_metrics_to_log({"kl": "kl"})
```

表示从 `train_step` 的 `aux["kl"]` 中取出值并记录为 `"kl"`。

------

#### `with_tqdm_metrics_to_display(...)`

```python
def with_tqdm_metrics_to_display(self, tqdm_metrics_to_display: list[str | Callable[[], str]])
```

控制哪些指标会显示在训练进度条上。可以是静态字符串，也可以是函数返回的 key。

------

#### `_post_process_train_step` / `_post_process_eval_step`

```python
self.metrics_logger.log(metric_name, aux[metric_key], mode, steps)
```

- 被 `train_step` 和 `eval_step` 调用后执行
- 会把 `aux` 中的值记录下来（如 KL 散度、奖励等）
- 用于 `wandb` 等系统统一日志记录

------

#### `_get_additional_tqdm_metrics`

- 返回 tqdm 中要显示的额外指标列表（防止重复）
- 支持动态和静态 key 混用

------

#### `_tqdm_train_metrics` / `_tqdm_eval_metrics`

重写父类属性，追加 RL 中设置的显示项。

------

#### ✅ 总结作用（简洁）

这个 `Trainer` 是 `PeftTrainer` 的 RL 加强版，专门为 **GRPO/PPO 强化学习训练过程**增加：

- 自定义 metric 记录（如 KL/reward）
- 自定义进度条输出
- 保留 optax 优化器训练流程

整个 Tunix 中，`GrpoLearner` 就是通过：

```python
rl_cluster.actor_trainer.with_loss_fn(...)
rl_cluster.actor_trainer.with_rl_metrics_to_log(...)
```

把 loss 函数、指标接入进来的。

是否需要我讲它如何调用 `train_step()`、或结合完整训练 loop 展示调用路径？









### 9. `tunix/rl/utils.py`

这段代码是 GRPO（Group Relative Policy Optimization）中的辅助工具，用于处理：

1. **参数结构的扁平化**（`to_flat_dict`）
2. **模型参数的 mesh 分布信息提取**（`get_pytree_mesh_info`）

它主要用于 reshard、模型同步、参数检查等步骤，确保在多设备训练（TPU/GPU）中参数的一致性。

------

#### 📦 函数解释

------

#### ✅ `to_flat_dict(tree: PyTree)`

```python
def to_flat_dict(tree) -> tuple[dict[tuple[str, ...], Array], PyTreeDef]
```

**功能：**
 将一个模型的 `PyTree`（如 `nnx.state(model)` 返回的状态）展开成：

- 一个扁平的 `dict`，key 是路径元组（如 `("layer1", "dense", "weight")`）
- 一个 `PyTreeDef`，用于后续重建结构

**用途：**
 在 `update_params()` 中用于合并新旧参数：

```python
flat_new_params, _ = to_flat_dict(params)
flat_old_params, tree_def = to_flat_dict(self._sampler.transformer_state)
```

------

#### ✅ `get_pytree_mesh_info(tree: PyTree) -> Mesh | None`

**功能：**
 遍历一个 `PyTree`，提取所有 `jax.Array` 的 `.sharding.mesh` 信息

```python
if isinstance(sharding, NamedSharding):
    mesh_info.add(sharding.mesh)
```

如果发现多种 mesh，会报错；否则返回唯一 mesh。

**用途：**
 在 `RLCluster._load_model()` 中用于判断模型是否已经位于目标 mesh 上：

```python
model_mesh = get_pytree_mesh_info(nnx.state(model))
if model_mesh != mesh:
    reshard(...)
```

------

#### ✅ 总结

| 函数                   | 用途                                             |
| ---------------------- | ------------------------------------------------ |
| `to_flat_dict`         | 把模型参数展平为 dict，方便修改或合并            |
| `get_pytree_mesh_info` | 获取参数在哪个 mesh 上，辅助判断是否需要 reshard |

这些函数在 Tunix 的权重同步、参数迁移、模型加载中非常关键，特别是在多设备并行场景下。
