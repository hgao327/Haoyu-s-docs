---

# 1) 工具函数：LCM 与切片

```python
# 放到文件顶部 imports 附近
import math

def _lcm3(a: int, b: int, c: int) -> int:
  return (a * b) // math.gcd(a, b) * c // math.gcd(((a * b) // math.gcd(a, b)), c)

def _chunk_slices_by_size(n: int, micro: int):
  """按样本数 micro 返回 [slice(...), ...] 切片列表。最后一块允许 < micro。"""
  i = 0
  out = []
  while i < n:
    out.append(slice(i, min(i + micro, n)))
    i += micro
  return out
```

---

# 2) 分阶段分块：rollout / ref / old

```python
# 放在 GrpoLearner 类里（与之前给你的 _rollout_in_chunks / _per_token_logps_in_chunks 类似）
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

> 以上三个方法都按“**样本数 micro**”切分；不涉及 token 预算或对齐。

---

# 3) 改 `_generate_and_compute_advantage`：内部按各自 micro 分块执行

把你原先“一把梭”的 generate / ref\_logps / old\_logps 换成“按 micro 分块”：

```python
def _generate_and_compute_advantage(
    self,
    training_input: _TrainingInputT,
    mode: metrics_logger.Mode = metrics_logger.Mode.TRAIN,
) -> TrainExample:
  pad_value = self.rl_cluster.rollout.pad_id()
  eos_value = self.rl_cluster.rollout.eos_id()

  prompts: List[str] = training_input["prompts"]

  # === 1) rollout：按 rollout_micro 分块 ===
  completion_ids, prompt_ids, completion_text = self._rollout_by_micro(
      prompts, self.batch_config.rollout_micro_batch_size
  )

  # === 2) 组 mask ===
  prompt_mask = (prompt_ids != pad_value).astype("int32")
  completion_padding_mask = jnp.not_equal(completion_ids, pad_value).astype("int32")
  completion_mask = common.make_completion_mask(completion_ids, eos_tok=eos_value)
  completion_mask = completion_mask * completion_padding_mask

  # === 3) ref / old logps：按各自 micro 分块 ===
  if self.grpo_config.beta != 0.0:
    ref_per_token_logps = self._ref_logps_by_micro(
        prompt_ids, completion_ids, self.batch_config.ref_logps_micro_batch_size
    )
  else:
    ref_per_token_logps = None

  if self.grpo_config.num_iterations > 1:
    old_per_token_logps = self._old_logps_by_micro(
        prompt_ids, completion_ids, self.batch_config.old_logps_micro_batch_size
    )
  else:
    old_per_token_logps = None

  # === 4) reward 与 advantage（与原逻辑一致）===
  rewards = self._compute_rewards(
      prompts=prompts,
      completions=completion_text,
      mode=mode,
      **{k: v for k, v in training_input.items() if k != "prompts"},
  )
  advantages = grpo_helpers.compute_advantages(
      rewards, self.grpo_config.num_generations
  )

  # === 5) 记录长度指标（原样）===
  agg_completion_mask = completion_mask.sum(axis=-1)
  steps = self._get_metric_logging_steps(mode)
  self._metrics_logger.log("completions/mean_length", agg_completion_mask.mean(), mode, steps)
  self._metrics_logger.log("completions/max_length", agg_completion_mask.max(), mode, steps)
  self._metrics_logger.log("completions/min_length", agg_completion_mask.min(), mode, steps)

  # === 6) 返回 TrainExample ===
  return TrainExample(
      prompt_ids=prompt_ids,
      prompt_mask=prompt_mask,
      completion_ids=completion_ids,
      completion_mask=completion_mask,
      ref_per_token_logps=ref_per_token_logps,
      advantages=advantages,
      old_per_token_logps=old_per_token_logps,
  )
```

---

# 4) 改 `_prepare_data`：聚合大小=LCM(rollout, ref, old)

核心变化：把原先的 `service_target_bs` 改成 LCM；达到 LCM（或数据尾部）就 flush。其余保持你已有语义（G 倍 repeat、μ 次 RepeatIterable、async/sync 两种入队方式）。

```python
def _prepare_data(
    self,
    iterator: Iterator[_TrainingInputT],
    proceed_num_steps: int,
    sample_repeat: int,
    batch_repeat: int,
    data_queue: queue_lib.AbstractDataQueue[
        list[TrainExample] | common.RepeatIterable | None
    ],
    async_loading: bool = False,
    mode: metrics_logger.Mode = metrics_logger.Mode.TRAIN,
) -> None:

  # === LCM 作为服务批大小（不考虑对齐与上下限）===
  service_target_bs = _lcm3(
      self.batch_config.rollout_micro_batch_size,
      self.batch_config.ref_logps_micro_batch_size,
      self.batch_config.old_logps_micro_batch_size,
  )

  buf: list[_TrainingInputT] = []
  buf_sizes: list[int] = []   # 每个训练 micro 的样本数
  buf_B = 0                   # 聚合的总样本数（未 repeat）
  example_list: list[TrainExample] = []
  consumed_steps = 0          # 按“训练 micro 个数”计数（不是样本条数）

  def _flush(force: bool = False):
    nonlocal buf, buf_sizes, buf_B, example_list
    if not buf:
      return
    if (not force) and (buf_B < service_target_bs):
      return

    # 1) 合并多个训练 micro 成 single batch（未 repeat）
    merged: dict = {}
    keys = buf[0].keys()
    for k in keys:
      merged[k] = buf[0][k]
    for i in range(1, len(buf)):
      for k in keys:
        a, b = merged[k], buf[i][k]
        if isinstance(a, list):
          merged[k] = a + b
        else:
          merged[k] = jnp.concatenate([jnp.asarray(a), jnp.asarray(b)], axis=0)

    # 2) 一次性 repeat G（等价于逐 micro repeat 再拼）
    merged_repeated = jax.tree.map(
        lambda x: np.repeat(x, sample_repeat, axis=0),
        merged,
    )  # [sum(B_i)] -> [sum(B_i) * G]

    # 3) 大批执行（内部按各阶段 micro 重新切分）
    with jax.profiler.StepTraceAnnotation(
        "sampler",
        step_num=self._train_steps if mode == metrics_logger.Mode.TRAIN else self._eval_steps,
    ):
      big_example = self._generate_and_compute_advantage(merged_repeated, mode)

    # 4) 按原训练 micro 边界（×G）切回，构造多个 TrainExample
    offset = 0
    for n in buf_sizes:
      token_sl = slice(offset * sample_repeat, (offset + n) * sample_repeat)
      te_small = TrainExample(
          prompt_ids=big_example.prompt_ids[token_sl],
          prompt_mask=big_example.prompt_mask[token_sl],
          completion_ids=big_example.completion_ids[token_sl],
          completion_mask=big_example.completion_mask[token_sl],
          ref_per_token_logps=None if big_example.ref_per_token_logps is None else big_example.ref_per_token_logps[token_sl],
          advantages=big_example.advantages[token_sl],  # 每序列一个 advantage → 跟随 repeat 后的序列切
          old_per_token_logps=None if big_example.old_per_token_logps is None else big_example.old_per_token_logps[token_sl],
      )
      example_list.append(te_small)
      offset += n

    # 5) 入队（与原逻辑一致）
    if not async_loading:
      data_queue.put(common.RepeatIterable(example_list, batch_repeat))
      example_list = []
    else:
      for te_small in example_list:
        data_queue.put([te_small])
      if batch_repeat > 1:
        data_queue.put(common.RepeatIterable(example_list, batch_repeat - 1))
      example_list = []

    # 6) 清空缓冲
    buf.clear()
    buf_sizes.clear()
    buf_B = 0

  try:
    while True:
      # 从 checkpoint 恢复时快进（原逻辑保留）
      while (mode == metrics_logger.Mode.TRAIN and self._train_steps < self._last_train_step):
        next(iterator)
        self._train_steps += 1

      # 取一个“训练 microbatch”
      example = next(iterator)
      B = len(example["prompts"])
      buf.append(example)
      buf_sizes.append(B)
      buf_B += B
      consumed_steps += 1

      # 达到 LCM 就 flush 一次
      _flush(force=False)

      # 步数计数（原逻辑保留）
      if mode == metrics_logger.Mode.TRAIN:
        self._train_steps += 1
      else:
        self._eval_steps += 1

      # 只推进固定个数的 micro 时，达到上限强制 flush 并返回
      if proceed_num_steps > 0 and consumed_steps >= proceed_num_steps:
        _flush(force=True)
        return

  except StopIteration as e:
    if proceed_num_steps > 0:
      raise e
    else:
      _flush(force=True)
      return
  finally:
    data_queue.put(None)
```

---

## 使用说明 / 约束

* **聚合大小** = `lcm(rollout_micro, ref_micro, old_micro)`；可能会比单一 micro 大不少（比如三者互素时），这是你要求的“无上下限、不考虑对齐”的直接实现。
* 进入 `_generate_and_compute_advantage` 后，各阶段会**按各自 micro** 切分执行。
* 训练侧仍按原训练 micro 粒度入队；μ 次通过 `RepeatIterable` 复用同一批。
* 本实现**不做 token 预算兜底**，如果你的样本长度波动很大、ref/old 是全序列前向，建议后续加个简单兜底（否则仍可能 OOM）。

---

