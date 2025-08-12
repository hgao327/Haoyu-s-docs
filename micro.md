好的，我给你一套**最小侵入**、可逐步演进的 micro-batch 设计，专门覆盖你现在两条“打分前向”路径：

* rollout：`rollout.get_per_token_logps(...)`
* ref：`inference_worker.get_ref_per_token_logps(...)` → `common.compute_per_token_logps(...)`

思路是**在调用入口外层切 batch**，不改 `common.compute_per_token_logps`，这样风险小、回滚容易。

---

# 目标

* 在不改 `common.*` 的前提下，为 **rollout/ref 打分**增加 **推理 micro-batch**。
* 支持不满尾块；尽量**避免反复编译**（动态 batch 导致 JIT 抖动）。
* 不影响现有 **actor 训练端的梯度累积**（MultiSteps）。

---

# 配置项（建议）

在 `ClusterConfig` 或其子配置里加两个字段（按你喜欢的地方存放即可）：

```python
# for reference打分
ref_logprob_mbs_per_gpu: int = 16

# for rollout打分（old）
rollout_logprob_mbs_per_gpu: int = 32
```

> 你也可以放在更细粒度的 config（例如 `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`），只要能在 RLCluster/InferenceWorker/rollout 入口读到就行。

---

# 方案 A：最小改动（Python 循环切块）

### 1) 给 `InferenceWorker` 加一个带切批的包装

```python
# inference_worker.py
def _batched_compute_per_token_logps(self, model, prompts, completions, pad_id, eos_id, mbs: int):
    B = prompts.shape[0]
    outs = []
    for start in range(0, B, mbs):
        end = min(start + mbs, B)
        logps = common.compute_per_token_logps(
            model,
            prompt_tokens=prompts[start:end],
            completion_tokens=completions[start:end],
            pad_id=pad_id,
            eos_id=eos_id,
        )
        outs.append(logps)
    return jnp.concatenate(outs, axis=0)

def get_ref_per_token_logps(self, prompt_tokens, completion_tokens, pad_id, eos_id, mbs: int | None = None):
    ref_model = self._models.get("reference")
    if ref_model is None:
        raise ValueError("Reference model is not available.")
    if not mbs:  # 兼容旧调用
        return common.compute_per_token_logps(ref_model, prompt_tokens, completion_tokens, pad_id, eos_id)
    return self._batched_compute_per_token_logps(ref_model, prompt_tokens, completion_tokens, pad_id, eos_id, mbs)
```

### 2) 在 `RLCluster.get_ref_per_token_logps` 里把 mbs 传进去

```python
def get_ref_per_token_logps(...):
  with self.cluster_config.role_to_mesh[Role.REFERENCE]:
    self._maybe_load_model_from_cpu(self.inference_worker.get_model("reference"), Role.REFERENCE)
    mbs = getattr(self.cluster_config, "ref_logprob_mbs_per_gpu", None)
    ref_per_token_logps = self.inference_worker.get_ref_per_token_logps(
        prompt_tokens, completion_tokens, pad_id, eos_id, mbs=mbs
    )
    self._maybe_offload_model_to_cpu(self.inference_worker.get_model("reference"), Role.REFERENCE)
    return ref_per_token_logps
```

### 3) 给 rollout 的 `get_per_token_logps` 也包一层

```python
# rollout/base_rollout.py or vanilla_rollout.py 里
def get_per_token_logps(self, prompt_tokens, completion_tokens, mbs: int | None = None):
    model = self.model()
    if not mbs:
        return common.compute_per_token_logps(
            model, prompt_tokens, completion_tokens, pad_id=self.pad_id(), eos_id=self.eos_id()
        )

    B = prompt_tokens.shape[0]
    outs = []
    for start in range(0, B, mbs):
        end = min(start + mbs, B)
        outs.append(common.compute_per_token_logps(
            model,
            prompt_tokens=prompt_tokens[start:end],
            completion_tokens=completion_tokens[start:end],
            pad_id=self.pad_id(),
            eos_id=self.eos_id(),
        ))
    return jnp.concatenate(outs, axis=0)
```

并在 `RLCluster.get_old_per_token_logps` 里传值：

```python
def get_old_per_token_logps(...):
  with self.cluster_config.role_to_mesh[Role.ROLLOUT]:
    model = self.rollout.model()
    self._maybe_load_model_from_cpu(model, Role.ROLLOUT)
    if self.cluster_config.offload_to_cpu:
      self.rollout.update_params(nnx.state(model))
    mbs = getattr(self.cluster_config, "rollout_logprob_mbs_per_gpu", None)
    per_token_logps = self.rollout.get_per_token_logps(prompt_tokens, completion_tokens, mbs=mbs)
    ...
    return per_token_logps
```

**优点**：实现量小、直观好测。
**缺点**：不同大小的 batch 会触发多次编译（一般问题不大）。

---

# 方案 B：稳定编译（固定块大小 + `lax.scan`）

为减少 JIT 抖动，可以把 batch **pad 到 mbs 的整数倍**，再 reshape 成 `[num_chunks, mbs, ...]`，用 `lax.scan`（或 `vmap`）一次编译。思路如下：

```python
def _scan_compute_per_token_logps(self, model, prompts, completions, pad_id, eos_id, mbs: int):
    B = prompts.shape[0]
    # 1) pad 到上取整的 chunks * mbs
    chunks = (B + mbs - 1) // mbs
    pad_n = chunks * mbs - B
    if pad_n > 0:
        prompts_pad = common.pad_to_length(prompts, B + pad_n, pad_value=self._pad_id, left=False, axis=0)
        completions_pad = common.pad_to_length(completions, B + pad_n, pad_value=self._pad_id, left=False, axis=0)
    else:
        prompts_pad, completions_pad = prompts, completions

    # 2) reshape 成 [chunks, mbs, ...]
    p = prompts_pad.reshape(chunks, mbs, *prompts.shape[1:])
    c = completions_pad.reshape(chunks, mbs, *completions.shape[1:])

    def body(carry, pc):
        pt, ct = pc  # [mbs, ...]
        logps = common.compute_per_token_logps(model, pt, ct, pad_id, eos_id)
        return carry, logps  # [mbs, T]
    _, outs = jax.lax.scan(body, None, (p, c))  # [chunks, mbs, T]

    outs = outs.reshape(chunks * mbs, *outs.shape[2:])  # [B_pad, T]
    return outs[:B, ...]  # 去掉pad
```

把这个函数替换方案 A 中的循环实现即可。
**优点**：只编译一次，长跑更稳定。
**缺点**：代码稍复杂。

---

# 注意点 & 踩坑清单

1. **mask/positions 一致**：我们不动 `common.compute_per_token_logps`，保证和 new/ref/old 的对齐逻辑完全一致。
2. **dtype 稳定**：内部 `log_softmax` 用 fp32（你现在就是这样风格），避免 NaN。
3. **EOS 截断**：仍使用 `make_completion_mask`，不影响 micro-batch。
4. **mesh**：切批前后仍在 `with role_to_mesh[...]` 上下文里；不要跨 mesh 传大张量。
5. **offload**：保持你现在的 load→compute→offload 包裹不变；切批只在 compute 内部生效。
6. **mbs 固定**：建议配置后固定不变，避免反复编译。
7. **尾块**：方案 A 直接裁切；方案 B 通过 pad→scan→裁掉，实现一次编译。
8. **并发/队列**：不改变你当前的队列与异步流水（`SimpleDataQueue` / `ThreadPoolExecutor`）；micro-batch 是每次“打分调用”的内部细化。
9. **兼容 vLLM rollout**：如果 rollout 引擎是 vLLM，自身可能就有 batching 行为；在你封装的 `rollout.get_per_token_logps` 里优先判断是否能直接批内处理，不能再 fallback 到上面 micro-batch。

---

# 验证 & 指标

* **数值一致性**：开/关 micro-batch 计算的 `ref/old logprob` 应 bitwise（或 1e-6 级别）一致。
* **吞吐与显存**：记录 `HBM peak`、`throughput (tokens/s)`；调不同 mbs 找平衡点。
* **编译次数**：用 `jax.profiler` 看 XLA 编译是否因动态 batch 重复触发；如频繁，迁移到方案 B。

---

# 渐进式落地顺序

1. 先上 **方案 A**（入口层 Python 循环），加两个配置项，验证数值与显存。
2. 再把 ref 路径换成 **方案 B**（scan + pad），观察编译次数下降。
3. rollout 路径按需替换为方案 B（若 vLLM 不需要可跳过）。
4. 加 metrics：`ref/old mbs 实际值`、`ref/old forward latency`、`num_chunks`，便于排查。

---

要不要我直接给你做两份最小 PR diff（“只改 InferenceWorker”和“只改 VanillaRollout”）？
你贴出配置对象在哪儿取值，我就把 `mbs` 的读取也一并接好。
