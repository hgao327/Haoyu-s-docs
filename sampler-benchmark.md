# Tunix vs vLLM Sampler Performance Benchmark Report

## Overview

This document presents a comprehensive benchmark design and results for Large Language Model (LLM) inference performance. The primary objective is to compare the Vanilla Sampler integrated within the Tunix framework against the vLLM Sampler on Google's Tensor Processing Unit (TPU) hardware. The evaluation encompasses popular LLM models to thoroughly assess performance bottlenecks and optimization benefits under different configurations.

## Test Environment

- **Hardware Platform**: TPU v5e-4
- **Test Models**: Llama-3.1-8B, Qwen2-7B
- **Dataset**: GSM8K (Grade School Math 8K)
- **Test Samples**: 1,319 mathematical reasoning problems

## Dataset and Input Specification

### GSM8K Dataset

- **Dataset**: Grade School Math 8K (elementary school math word problems)
- **Test Samples**: 1,319 mathematical word problems
- **Original Format**: Each sample contains `question` and `answer` fields

### Input Data Processing

#### Original Data Example

```json
{
    "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}
```

#### vLLM Input Processing

```python
# 1. Basic formatting
raw_prompts = [f"Question: {example['question']}\nAnswer:" for example in test_dataset]

# 2. Chat template application
prompts = tokenizer.apply_chat_template([{"role": "user", "content": p}], ...)

# 3. Process all 1,319 prompts at once
outputs = llm.generate(prompts, sampling_params)
```

#### Tunix Input Processing

```python
# 1. Same basic formatting
raw_prompts = [f"Question: {example['question']}\nAnswer:" for example in test_dataset]

# 2. Same chat template application
prompts = templatize(raw_prompts, tokenizer)

# 3. Batch processing
for i in range(0, len(prompts), batch_size):
    batch_prompts = prompts[i:i+batch_size]
    outputs = tunix_sampler(batch_prompts, ...)
```

#### Final Input Format Example

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

### Input Data Consistency

- **Same Data Source**: Both frameworks use identical GSM8K test set
- **Same Preprocessing**: Unified prompt format and chat template
- **Same Tokenizer**: Using the same model tokenizer ensures consistent tokenization

## Methodology

### vLLM Testing Strategy

In this performance benchmark, we selected the industry-leading vLLM inference framework to precisely evaluate model performance on the GSM8K dataset for high-throughput scenarios. We utilize the `vllm.generate` method, passing all test prompts at once.

**Strategy Rationale**:

- **PagedAttention**: Efficient attention mechanism memory management
- **Continuous Batching**: Token-level dynamic scheduling
- **Global Optimization**: Fully leverage vLLM's internal scheduler for resource optimization
- **Hardware Utilization Maximization**: Eliminate GPU idle time, improve overall throughput

### Tunix Testing Strategy

For the Tunix framework, we adopted a more fine-grained control strategy:

1. **Model Loading**: Use `tunix.models.llama.params.create_model_from_safe_tensors` API to load from local checkpoints
2. **Sharding Strategy**: Apply TP/FSDP sharding optimization
3. **Sampler Initialization**: Configure core sampler through `tunix.generate.sampler.Sampler` API
4. **Batch Testing**: Call sampler with fixed batch_size in loops to test performance under different parallel loads

## Test Configuration

### vLLM Configuration Parameters

```bash
python vllm_test.py \
  --model=meta-llama/Llama-3.1-8B \
  --tensor_parallel_size=4 \
  --task=generate \
  --max_model_len=1024 \
  --hide-outputs
```

| Parameter Category      | Parameter              | Value                   | Description                   |
| ----------------------- | ---------------------- | ----------------------- | ----------------------------- |
| **Model Parameters**    | model                  | meta-llama/Llama-3.1-8B | Test model                    |
|                         | max_model_len          | 1024                    | Maximum model sequence length |
| **Sampling Parameters** | max_tokens             | 256                     | Maximum generation length     |
|                         | temperature            | 0.0                     | Deterministic sampling        |
|                         | top_p                  | 1.0                     | Default setting               |
|                         | top_k                  | -1                      | Disable Top-K                 |
| **Data Parameters**     | num_samples            | -1                      | Use full GSM8K test set       |
| **Engine Parameters**   | tensor_parallel_size   | 4                       | Tensor parallelism            |
|                         | gpu_memory_utilization | 0.9                     | GPU memory utilization        |

### Tunix Configuration Parameters

| Parameter Category        | Parameter   | Test Values | Description                       |
| ------------------------- | ----------- | ----------- | --------------------------------- |
| **Generation Parameters** | max_tokens  | 64, 256     | Maximum generation length         |
|                           | cache_size  | 512         | KV cache size                     |
| **Batch Parameters**      | batch_size  | 1, 10, 50   | Batch processing size             |
| **Sampling Parameters**   | temperature | 0.0         | Deterministic sampling (inferred) |
|                           | echo        | False       | Return generated content only     |

## Benchmark Results

### Llama-3.1-8B Performance Comparison

#### vLLM Performance

```
[Overall Summary]
  Total Time:                 107.71 s
  Number of Prompts:          1,319

[Latency Metrics]
  Avg. Latency per Request:   81.66 ms/request

[Token Stats]
  Total Input Tokens:         84,015
  Total Generated Tokens:     128,661
  Grand Total Tokens:         212,676

[Throughput Metrics]
  Request Throughput:         12.25 requests/sec
  Input Token Throughput:     780.00 tokens/sec
  Output Token Throughput:    1,194.50 tokens/sec
  Total Token Throughput:     1,974.50 tokens/sec
```

#### Tunix Performance Matrix

| Configuration      | Total Time(s) | Request Throughput(req/s) | Token Throughput(tokens/s) | Avg Latency(ms) |
| ------------------ | ------------- | ------------------------- | -------------------------- | --------------- |
| **max_tokens=64**  |               |                           |                            |                 |
| batch_size=1       | 579.79        | 2.27                      | 374.57                     | 439.57          |
| batch_size=10      | 350.59        | 3.76                      | 619.44                     | 265.80          |
| batch_size=50      | 267.53        | 4.93                      | 811.77                     | 202.83          |
| **max_tokens=256** |               |                           |                            |                 |
| batch_size=1       | 1,224.72      | 1.08                      | 278.07                     | 928.52          |
| batch_size=10      | 473.41        | 2.79                      | 720.68                     | 358.92          |
| batch_size=50      | 323.17        | 4.08                      | 1,055.94                   | 245.01          |

### Qwen2-7B Performance Comparison

#### vLLM Performance

```
[Overall Summary]
  Total Time:                 117.66 s
  Number of Prompts:          1,319

[Latency Metrics]
  Avg. Latency per Request:   89.20 ms/request

[Token Stats]
  Total Input Tokens:         85,776
  Total Generated Tokens:     222,609
  Grand Total Tokens:         308,385

[Throughput Metrics]
  Request Throughput:         11.21 requests/sec
  Input Token Throughput:     729.01 tokens/sec
  Output Token Throughput:    1,891.96 tokens/sec
  Total Token Throughput:     2,620.98 tokens/sec
```

#### Tunix Performance

*Qwen2-7B Tunix test results pending*

## Performance Comparison Summary

### Llama-3.1-8B Complete Comparison Table

| Framework | Configuration            | Total Time(s) | Request Throughput(req/s) | Token Throughput(tokens/s) | Avg Latency(ms) | Input Tokens | Generated Tokens | Total Tokens |
| --------- | ------------------------ | ------------- | ------------------------- | -------------------------- | --------------- | ------------ | ---------------- | ------------ |
| **vLLM**  | Default config           | 107.71        | 12.25                     | 1,974.50                   | 81.66           | 84,015       | 128,661          | 212,676      |
| **Tunix** | max_tokens=64, batch=1   | 579.79        | 2.27                      | 374.57                     | 439.57          | 130,180      | 86,993           | 217,173      |
| **Tunix** | max_tokens=64, batch=10  | 350.59        | 3.76                      | 619.44                     | 265.80          | 130,180      | 86,991           | 217,171      |
| **Tunix** | max_tokens=64, batch=50  | 267.53        | 4.93                      | 811.77                     | 202.83          | 130,180      | 86,992           | 217,172      |
| **Tunix** | max_tokens=256, batch=1  | 1,224.72      | 1.08                      | 278.07                     | 928.52          | 130,180      | 210,378          | 340,558      |
| **Tunix** | max_tokens=256, batch=10 | 473.41        | 2.79                      | 720.68                     | 358.92          | 130,180      | 210,998          | 341,178      |
| **Tunix** | max_tokens=256, batch=50 | 323.17        | 4.08                      | 1,055.94                   | 245.01          | 130,180      | 211,070          | 341,250      |

### Qwen2-7B Complete Comparison Table

| Framework | Configuration  | Total Time(s) | Request Throughput(req/s) | Token Throughput(tokens/s) | Avg Latency(ms) | Input Tokens | Generated Tokens | Total Tokens |
| --------- | -------------- | ------------- | ------------------------- | -------------------------- | --------------- | ------------ | ---------------- | ------------ |
| **vLLM**  | Default config | 117.66        | 11.21                     | 2,620.98                   | 89.20           | 85,776       | 222,609          | 308,385      |
| **Tunix** | Pending test   | -             | -                         | -                          | -               | -            | -                | -            |

## Appendix

### Test Commands

#### vLLM Test Command

```bash
python vllm_test.py \
  --model=meta-llama/Llama-3.1-8B \
  --tensor_parallel_size=4 \
  --task=generate \
  --max_model_len=1024 \
  --hide-outputs
```

#### Tunix Test Configuration

```python
class Config:
    MODEL_CP_PATH = "meta-llama/Llama-3.1-8B"
    max_tokens = [64, 256]
    cache_size = 512
    batch_size = [1, 10, 50]
    echo = False
    num_samples = -1
    hide_outputs = True
```

### Complete Dataset Information

The test uses GSM8K (Grade School Math 8K) dataset:

- **Training Set**: 7,473 samples
- **Test Set**: 1,319 samples (used in this test)
- **Task Type**: Elementary school math word problem reasoning
- **Evaluation Metric**: Numerical answer accuracy
