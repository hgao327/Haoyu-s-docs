



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



Rollout

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

