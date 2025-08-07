#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting Tunix Sampler Benchmark Setup..."

# --- 1. Environment Setup and Dependencies ---
echo "Uninstalling existing JAX installations..."
pip uninstall -y $(pip list --format=freeze | grep -i '^jax' | cut -d= -f1) || true # Allow failure if JAX isn't installed

echo "Installing jax[tpu]..."
pip install jax[tpu]

echo "Installing other Python dependencies..."
pip install -q huggingface_hub datasets transformers

# --- 2. Hugging Face Login ---
echo "Logging into Hugging Face. Please follow the prompts."
huggingface-cli login

# --- 3. Run the Python Benchmark Script ---
echo "Executing the Python benchmark script..."

# Define your configuration parameters here or pass them as arguments to the script
# Example:
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
MAX_TOKENS=64
CACHE_SIZE=512
BATCH_SIZE=100
NUM_SAMPLES=-1 # -1 for full dataset
HIDE_OUTPUTS="False" # "True" or "False"

python vanilla_benchmark.py \
    --model_id "${MODEL_ID}" \
    --max_tokens "${MAX_TOKENS}" \
    --cache_size "${CACHE_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --num_samples "${NUM_SAMPLES}" \
    --hide_outputs "${HIDE_OUTPUTS}"

echo "Tunix Vanilla Sampler Benchmark Finished."