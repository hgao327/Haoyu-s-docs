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
