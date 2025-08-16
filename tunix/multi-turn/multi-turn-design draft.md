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





# Tunix Multi-turn Architecture Design v1

## 0. Design Philosophy

For multi-turn/agentic RL, it can be divided into frontend and backend

- Frontend: trajectory collect
- Backend: train

```python
for task in tasks:                        # Each task
    agent = ToolAgent()                  # Create Agent
    env = ToolEnv(task)                  # Create Env
    trajectory = run_rollout(agent, env)  # Interaction sampling
    trajectories.append(trajectory)

learner.train(trajectories)             # Train with this data
```

In **post-training / reinforcement learning fine-tuning**, there are two common approaches

| Mode        | Trajectory Generation                                        | When to Start Training                                       | Pros and Cons                                                |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Offline** | First use "old model" or human annotation to **generate/collect** large amounts of trajectories at once, written as fixed dataset | Start Learner after trajectories are collected, no environment interaction during training | • But model can only learn from **old policy distribution** experience, limited improvement |
| **Online**  | Frontend **continuously** dialogues with environment, sending new trajectories to Replay Buffer constantly | Learner updates parameters while receiving new data; pushes new weights to frontend every N steps | • Training and sampling proceed simultaneously, model can gradually "bootstrap" to stronger policy |

We adopt the second approach, **online loop**:

```text
      (collect)                            (train)
Actor ──► Trajectory ──► Buffer ──► Learner ─► New params ─► Actor … (loop)
```

- **Actor/Frontend** is always interacting with the environment—batch by batch "appending" new trajectories.
- **Learner/Backend** performs gradient update each time it gets a small batch of latest data, instead of waiting for all to be collected.

## 1. Overall Architecture

```
tunix/rl/multi_turn/
├── agents/                      # 1. Agent layer
├── environments/                # 2. Environment layer  
├── parser/tool_parser/          # 3. Tool parsing layer
├── tools/                       # 4. Tool execution layer
├── rewards/                     # 5. Reward system
├── TrajectoryCollector/         # 6. Trajectory collection layer
└── prompts/                     # 7. Prompt management layer
```

## 2. Core Data Structures

### 2.1 Basic Data Types

```python
@dataclass
class Step:
    chat_completions: list[dict[str, str]]  # OpenAI format messages
    thought: str                            # Reasoning process
    action: Any                            # Structured action
    observation: Any                       # Environment observation
    model_response: str                    # LLM raw response
    reward: float                          # Immediate reward
    done: bool                            # Termination flag
    mc_return: float                      # Monte Carlo return

@dataclass 
class Trajectory:
    task: Any                             # Task description
    steps: list[Step]                     # Step sequence
    reward: float                         # Total reward

@dataclass
class ToolCall:
    name: str                             # Tool name
    arguments: dict[str, Any]             # Call parameters

@dataclass  
class ToolOutput:
    name: str                             # Tool name
    output: str | list | dict             # Execution result
    error: str                            # Error message
    metadata: dict                        # Metadata
```

## 3. Module Architecture Design

### 3.1 Agent Layer (agents)

#### 3.1.1 BaseAgent Abstract Interface

```python
class BaseAgent(ABC):
    # Property interface
    @property
    def chat_completions(self) -> list[dict[str, str]]  # LLM input messages
    @property 
    def trajectory(self) -> Trajectory                   # Trajectory object
    
    # Core methods
    @abstractmethod
    def update_from_env(observation, reward, done, info)  # Environment feedback processing
    @abstractmethod
    def update_from_model(response: str) -> Action        # Model output parsing
    @abstractmethod
    def reset()                                          # State reset
```

#### 3.1.2 ToolAgent Tool Calling Implementation

```python
class ToolAgent(BaseAgent):
    def __init__(self, system_prompt, parser_name, tool_map):
        self.tool_manager = ToolManager(tool_map)         # Tool routing
        self.tool_parser = get_tool_parser(parser_name)   # Parser
        self._messages = []                               # Dialogue history
        self._trajectory = Trajectory()                   # Trajectory recording
```

### 3.2 Environment Layer (environments)

#### 3.2.1 BaseEnv Abstract Interface

```python
class BaseEnv(ABC):
    @abstractmethod
    def reset() -> tuple[dict, dict]                    # Reset environment
    @abstractmethod 
    def step(action) -> tuple[Any, float, bool, dict]   # Execute action
    @staticmethod
    @abstractmethod
    def from_dict(env_args: dict) -> "BaseEnv"          # Configuration creation
```

#### 3.2.2 ToolEnvironment Tool Execution Environment

```python
class ToolEnvironment(BaseEnv):
    def __init__(self, task, tool_map, reward_fn, max_steps=10):
        self.tool_manager = ToolManager(tool_map)       # Tool manager
        self.reward_fn = reward_fn                      # Reward function
        self.step_count = 0                            # Step counter
```

### 3.3 Tool Parsing Layer (parser/tool_parser)

#### 3.3.1 ToolParser Abstract Interface

```python
class ToolParser(ABC):
    @abstractmethod
    def parse(model_response: str) -> list[ToolCall]     # Parse tool calls
    @abstractmethod  
    def get_tool_prompt(tools_schema: str) -> str        # Generate tool prompt
    def parse_tool_outputs(model_response: str) -> dict  # Parse tool outputs (optional)
```

#### 3.3.2 Parser Registration Mechanism

```python
_PARSER_REGISTRY = {
    "qwen": QwenToolParser,
    # "openai": OpenAIFunctionToolParser,
}

def get_tool_parser(parser_name: str) -> type[ToolParser]:
    # Dynamically get parser class
```

### 3.4 Tool Execution Layer (tools)

#### 3.4.1 BaseTool Abstract Interface

```python
class BaseTool(ABC):
    def __init__(self, name: str, description: str):
        # Tool basic information
    
    @property
    @abstractmethod
    def json(self) -> dict       # OpenAI compatible tool schema
    
    def forward(self, **kwargs) -> ToolOutput           # Synchronous execution
    async def async_forward(self, **kwargs) -> ToolOutput  # Asynchronous execution
```

#### 3.4.2 ToolManager Tool Manager

```python
class ToolManager:
    def __init__(self, tool_map: Dict[str, Type[BaseTool]]):
        # Tool class instantiation and registration
    
    @property
    def json(self) -> List[dict]  # Schema list of all tools
    
    def run(self, tool_name: str, **kwargs) -> ToolOutput
    def execute_calls(self, calls: List[ToolCall], parallel=True) -> Dict[str, str]
```

### 3.5 Reward System (rewards)

#### 3.5.1 Reward Data Structure

```python
@dataclass
class RewardOutput:
    reward: float                    # Scalar reward value
    metadata: Dict[str, Any]         # Debug info and detailed metrics
```

#### 3.5.2 Reward Function Registration Mechanism

```python
_REGISTRY: Dict[str, Callable] = {}

@register("reward_name")
def reward_function(task: Dict, action: str) -> RewardOutput:
    # Reward calculation logic
```

### 3.6 Trajectory Collection Engine (execution)

#### 3.6.1 TrajectoryCollectEngine Core Class

```python
class TrajectoryCollectEngine:
    def __init__(self, agent, env, model_call, final_reward_fn, 
                 max_steps=10, gamma=1.0, timeout=30.0):
        # Component dependency injection
        
    async def collect(self) -> Trajectory:
        # Complete rollout execution flow
```

## 5. Complete System Execution Flow

### 5.1 System Startup and Initialization

```
1. Component Creation Phase
   ├── ToolAgent(system_prompt, parser_name, tool_map)
   │   ├── ToolManager instantiation → Tool class registration
   │   ├── ToolParser acquisition → Parser loading  
   │   └── System prompt building → Tool schema injection
   │
   ├── ToolEnvironment(task, tool_map, reward_fn, max_steps)
   │   ├── Task configuration loading
   │   ├── Tool mapping setup
   │   └── Reward function binding
   │
   └── TrajectoryCollectEngine(agent, env, model_call, final_reward_fn)
       └── Dependency injection completed

2. System Ready State
   └── All components initialized, waiting to execute collect()
```

### 5.2 Single Episode Complete Execution Flow

```
engine.collect() call
    ↓
5.2.1 Reset Phase (_reset)
    ├── env.reset() → Return initial task observation
    ├── agent.reset() → Clear trajectory and message history
    ├── agent.update_from_env(obs) → Load task into message list
    └── Start timing

    ↓
5.2.2 Loop Interaction Phase (_one_step * max_steps)
    ├── LLM Inference Subprocess
    │   ├── agent.chat_completions → Get message list
    │   ├── model_call(messages) → Async LLM call
    │   └── Return response text
    │
    ├── Response Parsing Subprocess  
    │   ├── agent.update_from_model(response)
    │   ├── ├── tool_parser.parse(response) → ToolCall list
    │   ├── ├── Construct Action object
    │   ├── ├── Record Step to trajectory
    │   ├── └── Return Action
    │   └── Pass Action to environment
    │
    ├── Environment Execution Subprocess
    │   ├── env.step(action)
    │   ├── ├── Check termination conditions (finish function/max_steps)
    │   ├── ├── If terminate → Calculate reward and return
    │   ├── ├── If continue → tool_manager.execute_calls()
    │   ├── └── Return (obs, reward, done, info)
    │   └── agent.update_from_env() → Update trajectory and messages
    │
    ├── Timeout Check
    │   └── If timeout → Mark done=True
    │
    └── If done=True → Break loop

    ↓
5.2.3 Cleanup Phase
    ├── _append_final_reward() → Add final reward
    ├── _fill_returns() → Calculate Monte Carlo returns
    ├── _close() → Resource cleanup
    └── Return complete Trajectory object
```

### 5.3 Tool Call Detailed Execution Flow

```
ToolManager.execute_calls(calls, parallel=True)
    ↓
5.3.1 Tool Call Preprocessing
    ├── Iterate through ToolCall list
    ├── Assign unique call_id for each call
    └── Prepare thread execution queue

    ↓
5.3.2 Parallel Execution Phase
    ├── Each ToolCall starts independent thread
    ├── ├── tool_manager.run(tool_name, **arguments)
    ├── ├── ├── Tool instance acquisition
    ├── ├── ├── tool.forward(**kwargs) → ToolOutput
    ├── ├── └── Exception catching → Error wrapping
    ├── └── Write result to queue (call_id, output_string)

    ↓
5.3.3 Result Aggregation
    ├── Wait for all threads to complete
    ├── Collect all results from queue
    └── Return {call_id: output_string} dictionary
```

### 5.4 Message Format Conversion Flow

```
5.4.1 Environment Observation → Chat Message Conversion
    ├── Task observation: {"question": "1+1=?"} 
    │   → {"role": "user", "content": "1+1=?"}
    │
    └── Tool output: {"tool_outputs": {call_id: result}}
        → {"role": "user", "tool_call_id": call_id, "content": "Tool returned result: ..."}

5.4.2 Chat Message History Maintenance
    ├── System message: system_prompt + tools_prompt
    ├── User message: task question + tool return results
    ├── Assistant message: LLM generated response
    └── Complete dialogue context preservation
```



可以的！你有两种主流做法，推荐第 1 种（最稳）：在 **宿主机路径和容器做绑定挂载**，然后用 VS Code 远程到 TPU VM，直接改宿主机代码；容器里立刻生效，notebook 里 `%autoreload 2` 实时同步。

---

# 绑定挂载 + VS Code Remote-SSH

**思路**：把 `tunix` 代码目录挂到容器里，VS Code 通过 SSH 登到 TPU VM，改宿主机上的代码即可。

1. 先停掉当前的 colab 容器（如果在跑）

```bash
docker stop colab || true
```

2. 用绑定挂载重启容器（把路径换成你的实际项目路径）

```bash
PROJECT=/home/youruser/tunix  # ← 你的 tunix 项目目录
docker run --rm --privileged --net=host \
  -v /dev:/dev -v /run:/run -v /var/lib/cloud:/var/lib/cloud \
  -v $PROJECT:/workspace/tunix \
  -e TPU_NAME="local" --name colab \
  -w /workspace \
  us-docker.pkg.dev/colab-images/public/runtime
```

> 说明：
>
> * `-v $PROJECT:/workspace/tunix` 把宿主机代码挂进去；
> * `-w /workspace` 把工作目录设到容器里；
> * 你原命令里如果用 `-p 127.0.0.1:9000:8080` 也可以保留；`--net=host` 则不需要再 `-p`。

3. VS Code 这边

* 安装 **Remote-SSH** 扩展。
* 用 SSH 连接到 TPU VM（Command Palette → *Remote-SSH: Connect to Host*）。
* 在远程 VS Code 里打开 **宿主机** 的项目目录（比如 `/home/youruser/tunix`）。
* 在 notebook 里加：

  ```python
  %load_ext autoreload
  %autoreload 2
  ```

  之后在 VS Code 改代码，notebook 里重跑 cell 就能看到变化。


`-w` 只是设置 **容器的工作目录**（working directory），
并不限制容器只能看到 `tunix` 这个目录。

我在例子里写 `-w /workspace/tunix` 是因为假设你的项目叫 `tunix`，而且我们把它挂载到容器的 `/workspace/tunix` 目录里，这样容器启动的时候直接落到项目根目录，方便你在 notebook 里 `pip install -e .` 或运行代码。

如果你想启动后落在上一级，比如 `/workspace`，完全可以改成：

```bash
-w /workspace
```

这样进入容器默认就在 `/workspace`，然后里面能看到 `tunix` 子目录。

---

**区别总结：**

| 配置                    | 效果                        |
| --------------------- | ------------------------- |
| `-w /workspace/tunix` | 容器启动直接进入 `tunix` 项目目录     |
| `-w /workspace`       | 启动进入上一级目录，需要手动 `cd tunix` |
| 不加 `-w`               | 进入容器的默认目录（取决于镜像设定）        |

---

