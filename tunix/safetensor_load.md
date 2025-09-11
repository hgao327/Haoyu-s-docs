## safetensors → NNX Weight Mapping Bug Analysis

### Problem Description

The weight mapping from safetensors to NNX incorrectly reorders the axes for Q/K/V matrices. The model computation expects `einsum('BTD, NDH -> BTNH')`, requiring weights in **(N,D,H)** shape, but after loading they become **(N,H,D)** (or equivalent misorderings). This axis semantic mismatch causes numerical divergence starting from `q_einsum`, resulting in different first-step logits compared to the normal environment, ultimately producing garbled output.

### Problem Verification

The issue can be confirmed by comparing these computations:

```python
q_impl = attn.q_einsum(x_norm)
q_ref = jnp.einsum('BTD, NDH -> BTNH', x_norm, attn.q_einsum.w.value)
# In the abnormal environment: ||q_impl - q_ref|| is significantly non-zero
# Using NHD ordering actually produces closer results, confirming the axis misorder
```

### Root Cause

PyTorch exports weights in `(out_features, in_features)` format. For Q projection, `q_proj.weight` has shape `(N*H, D)`, which we first `permute(1,0)` to get `(D, N*H)`.

The **incorrect approach** is directly reshaping to `(N, D, H)`. Under C-order (last dimension contiguous in memory), this splits the contiguous `N*H` axis and puts `N` first, scattering data that should belong to the same head. The numerical semantics become equivalent to **(N,H,D)**—correct shape and norm, but wrong axis semantics, causing `einsum('BTD,NDH')` to produce incorrect results from the first step. K/V have the same issue: expecting `(K,D,H)` but easily mis-split into `(K,H,D)`.

### The Nature of Reshape

`reshape` doesn't change the underlying data order; it only re-partitions continuous memory according to new dimensions. Under default C-order, the last dimension is the fastest-changing axis in memory. Therefore, directly reshaping from `(D, N*H)` to `(N, D, H)` cuts the last dimension's continuous blocks into `N × H`, then places `N` first, changing how elements are allocated to each axis and scattering blocks that should stay within the same head.

The correct approach is to first `reshape (D,N,H)` (keeping D first to preserve head block continuity), then `transpose` to `(N,D,H)`; or use `einops.rearrange(W, 'd (n h) -> n d h', n=N, h=H)` to explicitly express "split then rearrange".

### Concrete Example: What Really Happens in Memory

Let's break down exactly what `reshape` does. **`reshape` itself cannot "move order around"**—it only changes how we partition the data. But because we change the target shape's dimension order, it appears as if an axis was moved to the front.

**1. What's actually in memory?** Suppose we have `(D, N*H)` with `D=2, N=2, H=2`, data is:

```
[[0, 1, 2, 3],
 [4, 5, 6, 7]]
```

Memory layout (C-order, row-major) is just 1D:

```
0,1,2,3, 4,5,6,7
```

**2. Correct reshape** Target is `(D, N, H) = (2, 2, 2)`

- First split by D=2, each block length = N*H = 4
- Each block split by N=2, each sub-block length H=2

Result:

```
[
  [[0,1],[2,3]],
  [[4,5],[6,7]]
]
```

This is `(D,N,H)`. Then `transpose(1,0,2)` to get `(N,D,H)`:

```
[
  [[0,1],[4,5]],
  [[2,3],[6,7]]
]
```

**3. Incorrect reshape** Direct `(D,N*H) -> (N,D,H) = (2,2,2)`, what does the framework do?

- Target shape = (N=2, D=2, H=2), **first dimension N=2 means split into 2 blocks, each length 4**
- First block: `0,1,2,3` → reshape to (D=2,H=2) → `[[0,1],[2,3]]`
- Second block: `4,5,6,7` → reshape to (2,2) → `[[4,5],[6,7]]`

Combined:

```
[
  [[0,1],[2,3]],
  [[4,5],[6,7]]
]
```

The shape looks like `(N,D,H)`, but notice:

- Correct: head0 is `[0,1],[4,5]`, head1 is `[2,3],[6,7]`
- Incorrect: head0 is `[0,1],[2,3]`, head1 is `[4,5],[6,7]`

**Data that should belong to the same head got scattered and mis-grouped.**

**Summary:** `reshape` never "secretly reorders"—we gave it the wrong target shape, making it interpret the data with incorrect partitioning.

### Fix Solution

**Mapping stage**: Isolate `D` first, then split `N/H` or `K/H`

- Q: `(N*H,D) → (D,N*H) → (D,N,H)`, finally transpose to `(N,D,H)`
- K/V: `(K*H,D) → (D,K*H) → (D,K,H)`, then transpose to `(K,D,H)`

**Preprocessing fallback**: Uniformly correct possible misorderings `(D,N,H)` / `(N,H,D)` / `(K,H,D)` to target `(N,D,H)` / `(K,D,H)` (or fused `(3,N,D,H)`), with strict assertions; fail explicitly on error to prevent silent misalignment.

After the fix, `q_impl` matches the reference `einsum('BTD,NDH')` with near-zero difference, first-step logits become identical across environments, and garbled output disappears.

### Key Takeaway

Always trust `einsum` as the source of truth. When you see `BTD, NDH -> BTNH`, ensure weights land as **(N,D,H)**; similarly align K/V to **(K,D,H)**. As long as you reshape in "D first, then N/H" order before transposing, you won't hit this pitfall again.