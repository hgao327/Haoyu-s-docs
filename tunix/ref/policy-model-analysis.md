# Reference Model Analysis in Tunix GRPO

## Mathematical Foundation

**π** is the mathematical symbol used to represent the **policy** function.

In GRPO/PPO code, these symbols appear:
- **π_new**: Current updated policy (distribution obtained from forward pass in learner phase)
- **π_old**: Old policy used for sampling trajectories in rollout phase (logπ_old saved during sampling)
- **π_ref**: Reference policy (reference model), a baseline model fixed before training for KL constraint

### Intuitive Understanding in LLMs

- Model = Policy π
- Prompt/conversation history = State s
- Each generated token = Choose an action a
- π provides probability distribution for all tokens (after softmax), from which we sample/greedy to get next token

## Two-Phase Process

### Rollout Phase

**Purpose**: Generate training data without parameter updates

**Process**:
1. **π_old generates responses** for given prompts (token by token)
2. **Record logπ_old** for each generated token
3. **π_ref computes logπ_ref** (optional, if β > 0 for KL regularization)

**Key**: No gradients, no updates, data collection only

### Learner Phase

**Purpose**: Update policy using collected rollout data

**Process**:
1. **π_new processes same (prompt, response)** pairs from rollout
2. **Compute loss** using logπ_new, logπ_old, logπ_ref
3. **Backpropagate and update π_new**

**Summary**:
- **Rollout phase**: π_old → Generate dialogue → Record logπ_old; π_ref → Compute logπ_ref for same dialogue (for KL)
- **Learner phase**: π_new → Compute logπ_new for same dialogue; Use logπ_new, logπ_old, logπ_ref to compute loss (GRPO/PPO)

## Tunix Code Analysis

### 1. Generation Phase: Computing `ref_per_token_logps`

Location: `_generate_and_compute_advantage(...)` in `tunix/rl/grpo/grpo_learner.py`

```python
if self.grpo_config.beta != 0.0:
  ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
      prompt_tokens=prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_value,
      eos_id=eos_value,
  )
else:
  ref_per_token_logps = None
```

- **Trigger condition**: `grpo_config.beta != 0.0`
  - Only when KL coefficient β is non-zero will it request reference model logprob computation
- After computation, `ref_per_token_logps` is packed into `TrainExample`:

```python
return TrainExample(
    ...
    ref_per_token_logps=ref_per_token_logps,
    ...
)
```

### 2. Training Phase: Using `ref_per_token_logps` in Loss

Location: `grpo_loss_fn(...)`

```python
per_token_logps = common.get_per_token_logps(model, ...)  # = logπ_new

aux = {"kl": 0.0}
if beta != 0.0:
  kl = common.compute_kl_divergence(
      per_token_logps,                    # logπ_new
      train_example.ref_per_token_logps   # logπ_ref (from previous step)
  )
  per_token_loss = per_token_loss + beta * kl
  
  # Log metric (normalized after masking)
  aux["kl"] = (kl * completion_mask).sum() / loss_denominator.mean()
```

- **Single purpose**: When β>0, use `ref_per_token_logps` with current policy's `per_token_logps` (= logπ_new) to compute KL divergence, adding `beta * KL` to `per_token_loss`

## Three Types of Forward Passes in GRPO

### 1. Reference Scoring (Given completion, compute logπ_ref)

#### A. Entry Point 1: `RLCluster` Scheduling + Device Management

```python
def get_ref_per_token_logps(...):
  with self.cluster_config.role_to_mesh[Role.REFERENCE]:         # Place on REFERENCE mesh
    self._maybe_load_model_from_cpu(                              # If offload enabled: load to device
        self.inference_worker.get_model("reference"), Role.REFERENCE
    )
    ref_per_token_logps = self.inference_worker.get_ref_per_token_logps(  # ⭐ Key: actual scoring entry
        prompt_tokens, completion_tokens, pad_id, eos_id
    )
    self._maybe_offload_model_to_cpu(
        self.inference_worker.get_model("reference"), Role.REFERENCE
    )
    return ref_per_token_logps
```

- RLCluster **doesn't handle forward details**, only manages mesh selection, load/offload, then delegates to `InferenceWorker`

#### B. Entry Point 2: `InferenceWorker` Delegates to `common`

```python
def get_ref_per_token_logps(self, prompt_tokens, completion_tokens, pad_id, eos_id):
  ref_model = self._models.get("reference")
  return common.compute_per_token_logps(              # ⭐ Key: actual logprob computation
      ref_model,
      prompt_tokens=prompt_tokens,
      completion_tokens=completion_tokens,
      pad_id=pad_id,
      eos_id=eos_id,
  )
```

#### C. Core Computation: `common.compute_per_token_logps`

```python
@nnx.jit(static_argnames=('pad_id', 'eos_id', 'stop_gradient'))
def compute_per_token_logps(model, prompt_tokens, completion_tokens, pad_id, eos_id, stop_gradient=True):
  # 1) Concatenate prompt and completion, build mask/position/attention
  prompt_completion_ids, positions, attn_mask = process_ids(
      prompt_tokens, completion_tokens, pad_id, eos_id
  )

  # 2) Single forward pass, extract completion logits → log_softmax → gather
  per_token_logps = get_per_token_logps(
      model,
      input_tokens=prompt_completion_ids,
      positions=positions,
      attn_mask=attn_mask,
      logits_to_keep=completion_tokens.shape[1],   # Only keep response segment logits
  )

  # 3) Reference only scores, no updates
  if stop_gradient:
    per_token_logps = jax.lax.stop_gradient(per_token_logps)
  return per_token_logps
```

Supporting functions (both in `common.py`):

```python
def process_ids(prompt_tokens, completion_tokens, pad_id, eos_id):
  prompt_completion_ids = jnp.concat([prompt_tokens, completion_tokens], axis=1)
  prompt_mask = prompt_tokens != pad_id
  completion_mask = make_completion_mask(completion_tokens, eos_tok=eos_id)  # Valid until first EOS
  prompt_completion_mask = jnp.concatenate([prompt_mask, completion_mask], axis=-1)
  positions = build_positions_from_mask(prompt_completion_mask)
  attn_mask = make_causal_attn_mask(prompt_completion_mask)                  # Lower triangular + padding
  return prompt_completion_ids, positions, attn_mask

def get_per_token_logps(model, input_tokens, positions, attn_mask, logits_to_keep):
  logits, _ = model(input_tokens, positions=positions, attention_mask=attention_mask, cache=None)
  logits = logits[:, -logits_to_keep - 1 : -1, :]         # Extract response segment logits (left-aligned)
  input_tokens = input_tokens[:, -logits_to_keep:]        # Response segment actual token ids
  return selective_log_softmax(logits, input_tokens)      # log_softmax → take_along_axis gather
```

> **Summary**: Reference scoring = Forward pass on (prompt + existing completion) → Extract response logits → Convert to per-token logπ_ref. No sampling, no parameter updates (`stop_gradient=True`)

### 2. Old Policy Scoring (Given completion, compute logπ_old)

File: `RLCluster`

```python
def get_old_per_token_logps(self, prompt_tokens, completion_tokens):
  with self.cluster_config.role_to_mesh[Role.ROLLOUT]:
    model = self.rollout.model()
    self._maybe_load_model_from_cpu(model, Role.ROLLOUT)
    if self.cluster_config.offload_to_cpu:
      self.rollout.update_params(nnx.state(model))     # Sync latest weights to rollout engine
    per_token_logps = self.rollout.get_per_token_logps( # ⭐ Key: rollout side scoring entry
        prompt_tokens, completion_tokens
    )
    ...
    return per_token_logps
```

- Similar to reference scoring, computes per-token logprob for **given** completion, but using **π_old** (rollout policy)
- Result used for PPO/GRPO **ratio denominator**: `r = exp(logπ_new - logπ_old)`

### 3. New Policy Forward Pass in Loss Computation

File: `grpo_loss_fn`

```python
per_token_logps = common.get_per_token_logps(model=π_new, ...)   # This is logπ_new
old_per_token_logps = train_example.old_per_token_logps or stop_grad(per_token_logps)
seq_importance_ratio = per_token_logps - old_per_token_logps     # = logπ_new - logπ_old
coef_1 = jnp.exp(seq_importance_ratio)                           # r
# ... clipping + advantage assembly into per_token_loss ...

if beta != 0.0:
  kl = common.compute_kl_divergence(
      per_token_logps,                        # logπ_new
      train_example.ref_per_token_logps       # logπ_ref (from reference scoring)
  )
  per_token_loss = per_token_loss + beta * kl # Add KL to loss
```

- The `common.get_per_token_logps` here uses **same implementation** as reference, but with `model` being **π_new** and **participating in backprop** (no stop_gradient)
- KL uses `compute_kl_divergence(new, ref)`, actually utilizing the **reference scoring** results

## Implementation Unification

Both Rollout and Reference use the same underlying function:

- **Rollout**:
  ```python
  self.rollout.get_per_token_logps(...)
  # Internally calls:
  return common.compute_per_token_logps(self.model(), ...)
  ```

- **Reference**:
  ```python
  self.inference_worker.get_ref_per_token_logps(...)
  # Internally calls:
  return common.compute_per_token_logps(ref_model, ...)
  ```

**Key insight**: Whether rollout model or reference model, both ultimately call the same public function **`common.compute_per_token_logps`**, just with different `model` inputs:
- Rollout uses current training policy model (continuously updated)
- Reference uses fixed reference model (typically initial policy, not updated)

**The computation method is identical, only the target model differs.**