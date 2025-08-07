#!/bin/bash
set -e

# --- Script Description ---
# This script runs the vLLM performance benchmark with a Llama-3.1-8B model.
# ---

echo "Starting vLLM benchmark for Llama-3.1-8B model..."

# Run the vLLM benchmark script with the specified parameters
python vllm_benchmark.py \
  --model=meta-llama/Llama-3.1-8B \
  --tensor_parallel_size=4 \
  --task=generate \
  --max_model_len=1024 \
  --hide-outputs

echo "vLLM tpu_commons sampler benchmark completed."
