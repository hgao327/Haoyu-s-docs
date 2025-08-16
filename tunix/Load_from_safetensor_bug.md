

# Preventing HBM OOM When Loading safetensors Models on JAX + TPU

When using **JAX with TPUs** (e.g., v5e/v5p) to load large models stored in the `safetensors` format (such as Llama3-8B), using `safetensors.load_file` directly can cause an **Out of Memory (OOM)** error on **TPU0's HBM**, even if no computation is performed afterward.


## Problem

In a TPU runtime environment, the following code can immediately crash:

```python
import safetensors.numpy as st

tensor_dict = {}
for f in files:
    tensor_dict |= st.load_file(f)  # OOM happens here
```

Error message:

```
RuntimeError: Resource exhausted: Out of memory on device TPU:0
```

Even without calling `device_put`, `jnp.array`, or using `jit/pjit`, JAX's internal **aggressive staging** mechanism will preemptively move these tensors to TPU0, exhausting HBM memory.

---

## Root Cause: JAX Automatic Staging to TPU0

On TPU runtimes, JAX enables:

* **Lazy Execution**: Operations are traced and delayed
* **Graph Compilation**: Compiled into efficient XLA graphs
* **Aggressive Staging**: Large `np.ndarray`s on the host may be proactively transferred to TPU0 HBM

When you write:

```python
tensor_dict = safetensors.load_file(f)  # returns dict[str, np.ndarray]
```

JAX assumes these arrays might be used in computations and **eagerly transfers them to TPU0** — even without further code — leading to an out-of-memory crash.

---

## Problematic Code (Triggers OOM)

```python
import safetensors.numpy as st

tensor_dict = {}
for f in files:
    tensor_dict |= st.load_file(f)  # ❌ Loads entire file and triggers staging to TPU0

# Even without device_put or any computation, OOM occurs at this step
```

---

## Correct Solution: Lazy Load + Immediate Sharding

Use `safe_open(...).get_tensor(key)` to **load tensors one by one**, and use `jax.device_put` to immediately shard them to the appropriate device.

```python
from safetensors import safe_open
import jax
import jax.numpy as jnp

with safe_open(f, framework="numpy") as sf:
    for k in sf.keys():
        v = sf.get_tensor(k)  # Lazily load only one tensor

        # Optional: permute or reshape
        if transform is not None:
            permute, reshape = transform
            if permute:
                v = v.transpose(permute)
            if reshape:
                v = v.reshape(reshape)

        arr = jnp.array(v)  # Explicitly control when data enters device

        # Immediately place on correct shard (avoid TPU0 accumulation)
        if shard is not None:
            subdict[final_key] = jax.device_put(arr, shard[final_key])
        else:
            subdict[final_key] = jax.device_put(arr, jax.devices()[0])
```

