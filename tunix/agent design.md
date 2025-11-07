## 1. What is this architecture actually for?

An agentic RL system ultimately wants to do this:

> **Use an LLM as the Agent, let it act in various “environments” (answer questions, call tools, play games), then give rewards based on performance for RL training or evaluation.**

There are a few core challenges here:

1. **Environments are very different**
   - Some tasks are single-turn Q&A;
   - Some are multi-turn tool use (search/db/tools);
   - Some are games/simulations (multi-step interaction, complex state).
2. **Agent and Env must be decoupled**
   - The Agent only needs to know:
      “I receive an observation, I produce messages for the LLM, I get a response, and turn that into an Action.”
   - The Env only needs to know:
      “I receive an Action, I update my state, compute a reward, and output the next observation.”
3. **It must be easy to extend**
   - When we add new Env types or new Agents (e.g., with memory, planners), we don’t want to rewrite the orchestrator/trainer.

So the current design splits the system into three clear layers:

- **Env layer**: defines the “world rules”, handles `reset / step`, state transitions, and rewards;
- **Agent layer**: built on top of LLMs, mapping observation ↔ prompt and response ↔ action;
- **Data layer (Step / Trajectory)**: records the entire episode for training and analysis.

Then we use two “mid-level abstractions” to absorb most of the boilerplate:

- For Env: `BaseTaskEnv` → centralizes handling of `task / reward_fn / max_steps / step_count`;
- For Agent: `ConversationAgentBase` → centralizes handling of conversation history `_messages` and trajectories `_trajectory`.

The result is: **the top-level logic can use a single unified loop to handle all scenarios**, without caring whether it’s single-turn/multi-turn/tools/games.

------

## 2. Env-side design principles: BaseEnv / BaseTaskEnv / concrete Envs

### 2.1 BaseEnv: the minimal, stable common interface

`BaseEnv` exists for a simple reason: **give all environments a unified “wiring standard.”**
 It does very little, but it’s crucial:

- Abstract methods:

  ```python
  reset() -> (observation, info)
  step(action) -> (observation, reward, done, info)
  ```

- Common utilities:

  - `step_async`: wraps synchronous `step` in a thread pool for concurrency;
  - `close`: resource cleanup;
  - `from_dict`: create envs from config dicts.

Why keep it so thin?

- So the orchestrator/trainer only depends on this single interface;
- So “special” environments (multi-agent, third-party gym wrappers, etc.) have room:
   they can inherit `BaseEnv` directly without being constrained by `task / reward_fn / max_steps`.

### 2.2 BaseTaskEnv: absorb 80% of the common logic

In practice, most task-style environments share several patterns:

1. They have a `task` dict as initial configuration: question text, ground truth, metadata;
2. They use a `reward_fn(task, action)` to compute rewards;
3. An episode allows only a limited number of steps (`max_steps`), to prevent infinite loops;
4. Every step needs to handle template logic: `step_count` + `done` + `info`.

If every Env implements `reset` / `step` from scratch, you’ll see a lot of copy–paste.
 `BaseTaskEnv` uses the **Template Method pattern** to unify this:

```python
class BaseTaskEnv(BaseEnv):
  def __init__(..., task=None, reward_fn=None, max_steps=1, ...):
    self.task = task or {}
    self.reward_fn = reward_fn
    self.max_steps = max_steps
    self.step_count = 0

  def reset(self):
    self.step_count = 0
    obs = self._initial_observation()
    return obs, {}

  def step(self, action):
    self.step_count += 1
    result = self._step_impl(action)  # EnvStepResult
    done = result.done or (self.step_count >= self.max_steps)
    return result.observation, result.reward, done, result.info

  def _initial_observation(self): ...
  def _step_impl(self, action): ...
```

**Benefits:**

- Env subclasses only need to worry about “what’s truly different”:
  - How to produce the initial observation;
  - How each action updates state & computes reward & decides done.
- If you later want to add cross-cutting functionality, like:
  - Timeouts;
  - Unified per-step metrics (step_count, truncated flags, etc.);
     you only modify `BaseTaskEnv` once, and all task-style envs get it.

### 2.3 Single-turn vs multi-turn: `max_steps + done`

Instead of having two completely different interfaces.

- Single-turn envs (e.g. `TaskEnvironment`):
  - Set `max_steps=1` in `__init__`;
  - Directly use `done=True` in `_step_impl`.
- Multi-turn envs (e.g. `ToolEnvironment` / `GameEnvironment`):
  - Use `max_steps>1`;
  - Only set `done=True` in `_step_impl` when certain conditions hold (finish / game over);
  - `BaseTaskEnv` enforces `max_steps` as the final safety net.

This means **the Trainer never needs to know “single-turn vs multi-turn”**—it just runs a `while not done` loop.

### 2.4 Three concrete Env designs

#### 2.4.1 TaskEnvironment (single-turn Q&A)

Goal: **“Ask a question, the model answers once, I give a reward, episode ends.”**

- Initial observation = `task` dict (`{"question": "...", "answer": "...", ...}`)
- At each step:
  - Env receives the action (model’s answer string), passes it to `reward_fn`;
  - Emits `reward`, `done=True`, and `observation={}`.
- Because it inherits `BaseTaskEnv(max_steps=1)`, even if the Agent misbehaves, it can’t go beyond one step.

This is the classic “LLM Q&A grading” RL environment.

#### 2.4.2 ToolEnvironment (multi-step tool usage)

Goal: **“The model can call tools over multiple steps, then produce a final answer that gets scored by a reward function.”**

Key design points:

1. **Action format is complex**: it’s not just a string but:
   - Intermediate steps: a list of tool calls (function name + arguments + id);
   - Final step: either a plain string or a `finish` tool.
2. **Env decides when the episode ends**:
   - If `action` is a string → end immediately;
   - If `action` is a list containing tools' info → wrap info and continue;
   - If we exceed `max_steps` → forced termination (via `BaseTaskEnv`).
3. **Reward is only given at the end**:
   - Intermediate steps only execute tools, reward=0.0, `done=False`;
   - Final step extracts the “LLM final answer” and calls `reward_fn(task, answer)`.
4. **Tool execution logic is abstracted out**:
   - `ToolManager` + `ToolCall` handle mapping from name → actual tool implementation;
   - The Env just organizes the input format and passes it to `ToolManager`;
   - `ToolManager` handles parallel execution, error handling, etc.

As a result:

- The Env manages the lifecycle of the whole tool-calling episode;
- The Agent only needs to produce standardized tool-call JSON; it doesn’t need to know how tools actually run.

#### 2.4.3 GameEnvironment (games / simulations)

Goal: **“Use an LLM as a game agent that interacts multiple times with a stateful game.”**

Design principle: **decouple game logic from RL as much as possible**, so we have:

- A pure game class `GuessNumberGame`:
  - `reset(task) -> state`
  - `step(action, state) -> (next_state, reward, done, info)`
  - `get_observation(state) -> obs_for_agent`
- Env `GameEnvironment` acts as glue:
  - `_initial_observation`:
    - `self._state = game.reset(self.task)`
    - `return game.get_observation(self._state)`
  - `_step_impl(action)`:
    - Calls `game.step` to update state and get reward/done;
    - Uses `game.get_observation` to generate the observation for the Agent.

Benefits:

- Game logic can be unit-tested independently (without any RL framework);
- To switch games, you mostly change the `Game` class; the Env template is largely reusable;
- From the RL system’s perspective, all games look the same: they just implement `reset / step`.

------

## 3. Agent-side design principles: LLMBaseAgent / ConversationAgentBase

### 3.1 LLMBaseAgent: constraining what an Agent must be able to do

`LLMBaseAgent` is the core abstraction:

- It must expose:
  - `chat_completions`: the messages to send to the LLM on the next call;
  - `trajectory`: the current episode’s trajectory so far.
- It must implement:
  - `update_from_env`: update internal state based on env feedback;
  - `update_from_model`: turn an LLM response into an Action;
  - `reset`.

This guarantees: **the execution engine can drive any Agent just by depending on `LLMBaseAgent`.**

### 3.2 ConversationAgentBase: factoring out shared logic of chat-style Agents

In practice, most Agents share these behaviors:

1. Maintain a `messages` list like `[{"role": "system", ...}, {"role": "user", ...}, ...]`;
2. Whenever the env sends an observation, append some user/tool-type messages;
3. Whenever the LLM responds, append an assistant message;
4. Record a `Step` (obs, response, action, reward, etc.).

If each Agent implemented this independently, it’d be error-prone and inconsistent.
 `ConversationAgentBase` centralizes this boilerplate:

- Internal state:

  ```python
  self._messages
  self._trajectory
  self._obs_cache
  ```

- Hooks it offers:

  ```python
  def _init_messages(self):
    # Default: just system prompt
    self._messages = [{"role": "system", "content": self.system_prompt}]
  
  def _observation_to_messages(self, observation):
    # Default: handle {"question": ...} and str
  ```

- Common `update_from_env`:

  ```python
  def update_from_env(self, observation, reward, done, info, **kwargs):
    step = self.get_current_state()
    if step:
      step.observation = observation
      step.reward = reward
      step.done = done
      step.info = info or {}
  
    self._obs_cache = observation
    if observation is not None:
      self._observation_to_messages(observation)
  ```

- In your subclasses, you only need to implement:

  ```python
  def update_from_model(self, response, **kwargs) -> Action:
    # 1) Append response as an assistant message
    # 2) Create a new Step, fill in chat_completions / action / observation / model_response
    # 3) Append the Step to trajectory
    # 4) Return an Action
  ```

### 3.3 Three concrete agents designs

#### 3.3.1 ModelAgent: the simplest “Q&A Agent”

Characteristics:

- Inherits `ConversationAgentBase`;
- Does not override `_init_messages` or `_observation_to_messages` (uses defaults);
- `update_from_model`:
  - Treats the response directly as the action to send to the `TaskEnvironment`;
  - Performs no structural parsing.

Paired with `TaskEnvironment`, this gives you the simplest “single-turn RL Q&A” chain:

- obs: `{"question": "..."}` → becomes a user message;
- LLM produces one answer → `ModelAgent` records a Step and returns the action string;
- Env uses `reward_fn` to evaluate the string answer.

#### 3.3.2 ToolAgent: an Agent that can call tools

Differences mainly in two places:

1. `_init_messages`: the system prompt concatenates `tools_prompt` (tool schemas);

2. `_observation_to_messages`:

   - If `observation` has `tool_outputs`, each output is turned into:

     ```python
     {
       "role": "tool",
       "tool_call_id": call_id,
       "content": "Tool returned result: " + output,
     }
     ```

   - This way, the next LLM call can “see” tool results.

3. `update_from_model` is more complex:

   - It doesn’t just treat the response as a plain string; instead:

     - It uses a parser to extract a list of tool calls (name + arguments) from the response;
     - If parsing yields nothing, it falls back to a `"finish"` function call.

   - The final return value is:

     ```python
     Action(action=[{
       "id": ...,
       "type": "function",
       "function": {"name": ..., "arguments": ...},
     }, ...])
     ```

Combined with `ToolEnvironment`’s `_step_impl` logic, this enables:

- Intermediate steps: ToolAgent → ToolEnvironment executes tools → observation=tool_outputs → ToolAgent continues;
- Final step: ToolAgent sends a finish / direct answer → ToolEnvironment uses `reward_fn` to grade.

#### 3.3.3 GameAgent: a game-oriented Agent (e.g., number guessing)

Difference from tools:

- In the tool scenario, intermediate obs is mostly `tool_outputs`;
- In the game scenario, intermediate obs is a “summary of game state”, like `last_guess`, board positions, score, etc.

GameAgent is actually very simple:

- `_observation_to_messages`: generates user prompts based on `{"last_guess": ...}`;
- `update_from_model`: treats the response (a numeric string) as the action, records it in a Step, and returns it.

This shows that **the same base classes can easily support very different types of tasks (Q&A / tools / games).**

------

## 4. Step / Trajectory: why record data this way?

In `agent_types.py`:

- `Step` contains:
  - `chat_completions`: the full dialogue context at that time;
  - `thought`: chain-of-thought, if you want to store it later;
  - `action`: the Agent’s action for that env step;
  - `observation`: the env’s feedback;
  - `model_response`: the raw output from the LLM;
  - `reward / done / mc_return / info`.
- `Trajectory`:
  - `task`: the episode’s task description (optional);
  - `steps`: the chronological list of Steps;
  - `reward`: total reward (or final score);
  - `to_dict()`: easy serialization/logging.

Reasons for this design:

1. **Training**: for RL / offline RL / preference learning, you need full state–action–reward sequences;
2. **Analysis & Debugging**: you can replay a trajectory to inspect every prompt/response/tool call/observation/reward;
3. **Unified format**: whether it’s a Task, Tool, or Game environment, everything ends up as the same `Trajectory`, so training code can be fully generic.

------

## 5. Three typical pipelines (logical flow)

Here are textual “flow diagrams” for the three main combinations.

### 5.1 Single-turn Q&A: TaskEnvironment + ModelAgent

1. `env.reset()` → `obs={"question": "...", "answer": "..."}`;
2. `agent.reset()` → `_messages=[{"role": "system", ...}]`;
3. `agent.update_from_env(obs, 0, False, {})`:
   - No `Step` yet (current step is None, so nothing to update);
   - `_obs_cache=obs`;
   - `_observation_to_messages` turns the question into a user message.
4. Call LLM: `response = llm(messages)`.
5. `action = agent.update_from_model(response)`:
   - Append an assistant message;
   - Create a Step recording chat/action/obs/model_response.
6. `obs2, reward, done, info = env.step(action.action)`:
   - `TaskEnvironment` calls `reward_fn` to get reward, sets `done=True`.
7. `agent.update_from_env(obs2, reward, done, info)`:
   - Update the Step’s `reward/done/info`;
   - `_obs_cache` is updated but effectively no longer needed.

Finished. The Trajectory has exactly one Step that includes the full dialogue, action, and reward.

### 5.2 Multi-step tools: ToolEnvironment + ToolAgent

Assume there are two tool calls and then a finish.

1. `env.reset()` → `obs=task`;
2. `agent.reset()` → `_messages=[system + tools_prompt]`;
3. First round:
   - `agent.update_from_env(obs, 0, False, info)`:
     - `_observation_to_messages(task)` → user message;
   - LLM sees the task + tool schema and outputs a response containing tool calls;
   - `agent.update_from_model(response)`:
     - `tool_parser.parse(response)` → `tool_calls`;
     - Records a Step where `action` is a list of tool-call dicts;
   - `env.step(action.action)`:
     - ToolEnvironment executes tools, returns `obs={"tool_outputs": {...}}`, reward=0, done=False;
4. Second round:
   - `agent.update_from_env(obs, 0, False, info)`:
     - `_observation_to_messages` converts `tool_outputs` to `role="tool"` messages;
   - LLM sees tool results, decides to call more tools or finish;
   - …repeat for several rounds…
5. Final round:
   - `agent.update_from_model` parses a finish call or nothing (fall back to finish);
   - `env.step(action.action)`:
     - Detects string / finish tool, calls `reward_fn(task, answer)`;
     - Returns reward > 0 and `done=True`.

The Trajectory will contain multiple Steps. Each Step’s `action` might be:

- In the middle: a list of tool calls;
- At the end: a wrapped finish call / direct answer.

### 5.3 Game: GameEnvironment + GameAgent

1. `env.reset()`:
   - `GameEnvironment._initial_observation`:
     - `game.reset()` → initial state;
     - `get_observation(state)` → `{"last_guess": None}`;
2. `agent.reset()`;
3. Loop:
   1. `agent.update_from_env(obs, reward, done, info)`:
      - `_observation_to_messages`:
        - `last_guess=None` → “Let’s start guessing; please give your first guess”;
        - Next round `last_guess=30` → “Last guess was 30; please give the next one”.
   2. Call LLM: it outputs a string `"42"`;
   3. `action = agent.update_from_model("42")`:
      - Records a Step (action is the string "42");
   4. `obs, reward, done, info = env.step(action.action)`:
      - `GameEnvironment._step_impl` calls `game.step("42", state)`:
        - Computes new state / reward / done / hint;
        - Produces a new observation for the Agent.
   5. Exit when `done=True` (correct guess or step limit exceeded).

The Trajectory is again a sequence of Steps; the main difference is that the env logic is a game.

------

## 6. Advantages of this design

1. **Strong decoupling**

   - The Env knows nothing about the LLM; it only cares about Actions;
   - The Agent knows nothing about how rewards are computed; it only sees observations and LLM responses;
   - Step / Trajectory act as glue for training and analysis.

2. **Unified pattern**

   - “One loop to rule them all”:

     ```python
     obs, info = env.reset()
     agent.reset()
     while not done:
       agent.update_from_env(...)
       response = llm(agent.chat_completions)
       action = agent.update_from_model(response)
       obs, reward, done, info = env.step(action.action)
     ```

   - Whether it’s a Task, Tool, or Game env, the Trainer doesn’t need any if-else branches.

3. **Easy to extend**

   - New Env: inherit `BaseTaskEnv`, implement `_initial_observation` + `_step_impl`;
   - New Agent: inherit `ConversationAgentBase`, implement `_observation_to_messages` + `update_from_model`.
