##### current questions

inference worker为什么叫这个名字

sampler的兼容性需要考虑吗

异步传输 offpolicy问题

数据集处理时，prompt写法是固定的，如何确定准确性

pretrain model本身没有tool calling的机制，训完也只是针对某几种有了，而不是提升了tooling calling的能力

框架层需要管这些？



#### `examples/data_preprocess/gsm8k_multiturn_w_tool.py`

数据准备阶段，直接注入了extra info等信息，通过prompt engineering的方式，告诉了模型可用的模型，调用的初始逻辑，工具的含义和使用方式，thinking step by step. 

目前的逻辑是

1. thinking step by step
2. call tool 头
3. verify
4. refine

tool在其中作为了验证性质的工具，而且tool的调用逻辑是强制的，模型并没有学会什么场景下需要调用

```text
"You are a math expert. You are given a question and you need to solve it step by step."
"Reasoning step by step before any tool call. "
"You should use the `calc_gsm8k_reward` tool after step by step solving the question,"
"before generate final answer at least once and refine your answer if necessary."
"Put your final answer in the format of `#### <answer>`."
```

```python
{
"data_source": data_source,
"prompt": [
    {
        "role": "system",
        "content": (
            ###content
        ),
    },
    {
        "role": "user",
        "content": question,
    },
],
"ability": "math",
"reward_model": {"style": "rule", "ground_truth": solution},
"extra_info": {
    "split": split,
    "index": idx,
    "answer": answer_raw,
    "question": question_raw,
    "need_tools_kwargs": True,
    "tools_kwargs": {
        "calc_gsm8k_reward": {
            "create_kwargs": {"ground_truth": solution},
            # "execute_kwargs": {},
            # "calc_reward_kwargs": {},
            # "release_kwargs": {},
        },
    },
    "interaction_kwargs": {
        "query": question,
        "ground_truth": solution,
    },
},
}
```

tool config结构：

```python
tools_kwargs = {
    "tool_name": {
        "create_kwargs": {...},      # 工具创建时的参数
        "execute_kwargs": {...},     # 工具执行时的参数
        "calc_reward_kwargs": {...}, # 计算奖励时的参数
        "release_kwargs": {...},     # 释放资源时的参数
    }
}
```



1. Async RL 代表的是在 training rollout 分离的系统上，rollout 只在 update weights 的时候被打断，其余时刻永远 rollout，哪怕 target policy 正在被 training engine 更新。
2. Async Rollout 这个词是特指在 rollout 的时候，把一个 batch requests 拆为单个 request，然后逐个调用 `SGLangEngine.generate()`。



异步：一整个 batch 进去，要求一整个 batch 出来。这些 batch 里面的 requests 同时返回，同时被 paser 解析查看是否有 tool call 的 parameter，然后发送请求给 tool。

整个 tool 的调用大概率会拥堵，未来加入多个 tool的话，用一个状态机去管理每个 request 的 tool call 状态会相对困难（有的 requests 会在多轮里面多次调用 tool）。为了方便管理每个 request tool call 的状态机和让 tool 被调度的更加均匀，SGLang 采取了 Async Rollout 策略，把一个 batch 的 requests 拆为单个 request，然后逐个异步调用 `SGLangEngine.generate()`，每个 reqeuest 自己管理自己的状态机。



##### `generate_sequences`

```python
def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    if self.config.multi_turn.enable:
        return self._req_level_generate_sequences(prompts, **kwargs)
    return self._batch_level_generate_sequences(prompts, **kwargs)
```



##### `_req_level_generate_sequences`

`_req_level_generate_sequences` 在 rollout 阶段，它接收一批经过数据处理封装的 `prompts`（含多轮对话内容、工具信息、ground truth 等），根据配置判断是否开启采样模式，并由 TP rank 0 节点统一构造出多个 `AsyncRolloutRequest`，每个请求携带完整的多轮上下文、`tools_kwargs` 与 `tool_schema`，指明模型可以调用哪些工具。接着，系统通过 `asyncio.gather` 并发调度所有请求进行异步生成，模型在生成中会依据 prompt 引导执行工具调用（如 `openai_tool_call`），从而实现多轮、可调用工具的 Agent 行为。所有生成结果会按原始 batch 顺序整理，最终返回用于 reward 评估或 policy 训练。这个流程实现了具备工具增强能力的多轮对话 rollout。

```python
@GPUMemoryLogger(role="sglang rollout", logger=logger)
@torch.no_grad()
def _req_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    # Async rollout with tools support
    do_sample = prompts.meta_info.get("do_sample", True)
    is_validate = prompts.meta_info.get("validate", False)
    tgt_device = prompts.batch["input_ids"].device
    if self._tp_rank == 0:
        req_list = self._preprocess_prompt_to_async_rollout_requests(
            prompts,
            n=1 if is_validate else self.config.n,
        )
        loop = asyncio.get_event_loop()
        output_req_list = loop.run_until_complete(
            asyncio.gather(
                *[self._async_rollout_a_request(req, do_sample, is_validate, **kwargs) for req in req_list],
            )
        )
        sorted_output_req_list = sorted(output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset))
    else:
        sorted_output_req_list = None
```



##### Step1 `_preprocess_prompt_to_async_rollout_requests`

1. 将 prompts 展开，首先拆开 batch 中的每个 prompt，内层循环为每个 prompt 生成 `n` 个不同的序列。每个生成的请求都有唯一的 `batch_data_id` 和 `rollout_offset` 标识。
2. 当配置了工具时，`_input_ids` 和 `_attention_mask` 被设为 `None`，因为工具调用需要动态构建输入。而没有配置工具的话，使用 `_pre_process_inputs` 函数处理预处理的 token IDs，去除左填充。
3. 每个请求对象包含状态管理、工具配置、序列长度限制、tokenizer 配置等元数据，为后续的异步处理提供完整信息。

实质上就是初始化prompt，附加状态

##### Step2 `_async_rollout_a_request` 

根据 `AsyncRolloutRequest` 进行状态控制，rollout ->实际生成训练数据





SGLang全流程

A：`Parquet` 文件

```python
data_files = "~/data/rlhf/gsm8k/train.parquet"
```

B：RLHFDataset

```python
dataset = RLHFDataset(
    data_files=data_files,
    tokenizer=tokenizer,
    config=config,
    processor=processor
)
```

C：DataLoader + collate_fn

```python
dataloader = DataLoader(
    dataset=dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_fn
)
```

D：`DataProto` 原始数据

```python
batch_dict = next(iter(dataloader))  # 返回 dict
batch: DataProto = DataProto.from_single_dict(batch_dict)
```

E：`pop` 提取生成数据

```python
gen_batch = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
```

F：`Rollout` 生成

```python
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
```

G：`union` 合并数据

```python
batch = batch.union(gen_batch_output)
```

H：奖励计算

```python
rewards = self.reward_fn(batch)
batch.batch["token_level_rewards"] = rewards
```

I：优势计算

```python
batch = compute_advantage(batch, adv_estimator=self.config.algorithm.adv_estimator)
```

J：重新计算 `log_probs`

```python
old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
batch = batch.union(old_log_prob)
```

K：计算 reference model 的 `log_probs`

```python
if self.use_reference_policy:
    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
    batch = batch.union(ref_log_prob)
```

L：计算 value function

```python
if self.use_critic:
    values = self.critic_wg.compute_values(batch)
    batch = batch.union(values)
```

M1：更新 critic

```python
if self.use_critic:
    critic_output = self.critic_wg.update_critic(batch)
    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
    metrics.update(critic_output_metrics)
```

M2：更新 actor

```python
actor_output = self.actor_rollout_wg.update_actor(batch)
```

N：返回训练指标

```python
actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
metrics.update(actor_output_metrics)
logger.log(data=metrics, step=self.global_steps)
```



My design：

数据流: `tunix/rl/queue/data_queue.py`

Tools calling: 

SGLang: tool本身融合了env的抽象功能

```python
class MyCustomTool(BaseTool):
def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
    super().__init__(config, tool_schema)
    self._instance_dict = {}

def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
    return self.tool_schema

async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
    if instance_id is None:
        instance_id = str(uuid4())
    self._instance_dict[instance_id] = {
        "history": [],
        "reward": 0.0,
    }
    return instance_id

async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
    query = parameters.get("query", "")
    
    # return response
    result = f"Processed: {query}"
    self._instance_dict[instance_id]["history"].append(result)
    
    reward = 1.0 if "correct" in query else 0.0

    return result, reward, {"query_len": len(query)}

async def calc_reward(self, instance_id: str, **kwargs) -> float:
    return self._instance_dict[instance_id]["reward"]

async def release(self, instance_id: str, **kwargs) -> None:
    self._instance_dict.pop(instance_id, None)
```

```
rl_task/
│
├── tools/
│   ├── base_tool.py          # 抽象接口
│   └── gsm8k_tool.py         # 具体 Tool 实现
│
├── envs/
│   ├── base_env.py           # 抽象 Env
│   └── gsm8k_env.py          # 针对 tool 的 Env 实现

```

```python
from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseTool(ABC):
    def __init__(self, config: dict, tool_schema: Any):
        self.config = config
        self.tool_schema = tool_schema

    @abstractmethod
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        pass

    @abstractmethod
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[str, float, dict]:
        pass

    @abstractmethod
    async def release(self, instance_id: str, **kwargs) -> None:
        pass

```



```python
from abc import ABC, abstractmethod
from typing import Tuple

class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> str:
        """初始化并返回 observation"""
        pass

    @abstractmethod
    async def step(self, action: str) -> Tuple[str, float, bool, dict]:
        """
        接收模型输出，返回 (next_obs, reward, done, info)
        """
        pass

```



```python
from envs.base_env import BaseEnv
from tools.gsm8k_tool import Gsm8kTool

class Gsm8kEnv(BaseEnv):
    def __init__(self, tool: Gsm8kTool, sample: dict):
        """
        sample: 包含 question 和 answer
        """
        self.tool = tool
        self.sample = sample
        self.instance_id = None
        self.turns = []
        self.done = False
        
async def reset(self) -> str:
    self.turns = [("user", self.sample["question"])]
    self.instance_id = await self.tool.create(ground_truth=self.sample["answer"])
    self.done = False
    return self._get_obs()

async def step(self, action: str):
    self.turns.append(("assistant", action))

    # 执行 tool，获得 reward
    result_str, reward, info = await self.tool.execute(
        self.instance_id,
        parameters={"answer": action}
    )

    self.turns.append(("tool", result_str))
    self.done = True
    return "", reward, True, info

def _get_obs(self) -> str:
    return "\n".join(f"{r}: {m}" for r, m in self.turns)
```



#### Rollout

```python
import asyncio

tool_schema = OpenAIFunctionToolSchema(...)  # 如上结构
tool = Gsm8kTool(config={}, tool_schema=tool_schema)

sample = {
    "question": "abcdefg#############",
    "answer": "8"
}

env = Gsm8kEnv(tool, sample)

async def test():
    obs = await env.reset()
    print("Observation:", obs)

    action = "8"
    obs, reward, done, _ = await env.step(action)
    print("Final Reward:", reward)

asyncio.run(test())

```



design async_rollout_mode

```python
if true: _async_rollout_a_request
```

```python
if _req.state == PENDING:
    ...
    await self._handle_pending_state(_req)
		_req.state = RUNNING

elif _req.state == TOOL_CALLING:
    ...
    tool_call_results = await asyncio.gather(...)
elif _req.state == RUNNING:
    ...

if finish_reason == STOP or LENGTH or TOOL_CALL then break

#call each tool's calc_reward and release 

# _req.finalize(...)，change state, release tool
```







```yaml
{
  "prompt": [...],                   
  "extra_info": {
    "tools_kwargs": {
      "get_weather": {
        "execute_kwargs": {
          "city": "Beijing"          
        },
        "calc_reward_kwargs": {     
          ...
        },
        "release_kwargs": {            
          ...
        }
      }
    },
    "need_tools_kwargs": true,       
    "index": 123,                     
    "data_source": "my_dataset"       
  }
}
```





```yaml
tools_kwargs = {
    "tool_name": {
        "create_kwargs": {...},      
        "execute_kwargs": {...},     
        "calc_reward_kwargs": {...},
        "release_kwargs": {...},     
    }
}
```



```python
tool_schemas = [WhateverTool(...).get_openai_tool_schema()]
```



```python
if _req.state == PENDING:
    ...
    await self._handle_pending_state(_req)
		_req.state = RUNNING

elif _req.state == TOOL_CALLING:
    ...
    tool_call_results = await asyncio.gather(...)
elif _req.state == RUNNING:
    ...

if finish_reason == STOP or LENGTH or TOOL_CALL then break

#call each tool's calc_reward and release 

# _req.finalize(...)，change state, release tool
```
