import argparse  # Import argparse for command-line arguments
from pprint import pprint
import time
from typing import List
from datasets import load_dataset
from flax import nnx
from huggingface_hub import snapshot_download
import jax
from transformers import AutoTokenizer
from tunix.generate import sampler as sampler_lib
from tunix.models.llama3 import model, params


# ==================== Configuration Class ====================
class Config:
  """Configuration for the Tunix Sampler Benchmark."""

  model_id: str
  max_tokens: int
  cache_size: int
  echo: bool = True
  num_samples: int = -1  # Use entire dataset
  batch_size: int = 100
  hide_outputs: bool = False
  model_cp_path: str = ""  # This will be set after download


# ==================== Utility Functions ====================
def templatize(prompts: List[str], tokenizer) -> List[str]:
  """Apply tokenizer's chat template to prompts."""
  return [
      tokenizer.apply_chat_template(
          [{"role": "user", "content": p}],
          tokenize=False,
          add_generation_prompt=True,
      )
      for p in prompts
  ]


# ==================== Main Benchmark Logic ====================
def main(config: Config):
  """Main function: Run Tunix sampler performance test."""

  print("Starting Tunix Sampler initialization...")

  # Download model from Hugging Face
  print(f"Downloading model from Hugging Face: {config.model_id}...")
  ignore_patterns = [
      "*.pth",  # Ignore PyTorch .pth weight files
  ]
  config.model_cp_path = snapshot_download(
      repo_id=config.model_id, ignore_patterns=ignore_patterns
  )
  print(f"Model successfully downloaded to: {config.model_cp_path}")

  # Load tokenizer
  print("Loading tokenizer...")
  tokenizer = AutoTokenizer.from_pretrained(config.model_cp_path)
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  print("Tokenizer loaded successfully")

  # Load Tunix model
  print("Loading Tunix model...")
  model_config = model.ModelConfig.llama3_8b()
  MESH = [(1, 4), ("fsdp", "tp")]
  mesh = jax.make_mesh(*MESH)
  with mesh:
    llama3 = params.create_model_from_safe_tensors(
        config.model_cp_path, model_config, mesh
    )
    nnx.display(llama3)  # Optional: display model info
  print("Tunix model loaded successfully")

  # Initialize sampler
  print("Initializing sampler...")
  tunix_sampler = sampler_lib.Sampler(
      llama3,
      tokenizer,
      sampler_lib.CacheConfig(
          cache_size=config.cache_size,
          num_layers=32,
          num_kv_heads=8,
          head_dim=128,
      ),
  )
  print("Sampler initialized successfully")

  # Load GSM8K dataset
  print("Loading the GSM8K dataset...")
  dataset = load_dataset("gsm8k", "main")

  full_dataset_size = len(dataset["test"])
  if config.num_samples == -1:
    samples_to_use = full_dataset_size
    print(
        "Default setting detected. Using the full test set with"
        f" {samples_to_use} samples."
    )
  else:
    samples_to_use = min(config.num_samples, full_dataset_size)
    print(f"Using the specified {samples_to_use} samples for testing.")

  test_dataset = dataset["test"].select(range(samples_to_use))

  # Prepare prompts
  raw_prompts = [
      f"Question: {example['question']}\nAnswer:" for example in test_dataset
  ]
  ground_truths = [example["answer"] for example in test_dataset]

  # Apply chat template
  prompts = templatize(raw_prompts, tokenizer)

  print("Dataset prepared.")
  print(
      f"Batch configuration: batch_size={config.batch_size},"
      f" total_samples={len(prompts)}"
  )
  print("-" * 50)

  # Warm-up run
  if len(prompts) > 2:
    print("Starting warm-up run...")
    warmup_prompts = prompts[: min(2, config.batch_size)]
    _ = tunix_sampler(
        warmup_prompts,
        total_generation_steps=config.max_tokens,
        echo=config.echo,
    )
    print("Warm-up complete.")
  else:
    print("Too few samples, skipping warm-up.")
  print("-" * 50)

  # Performance benchmark with batching
  print("Starting performance benchmark...")
  start_time = time.perf_counter()

  # Process in batches
  all_outputs = []
  total_batches = (len(prompts) + config.batch_size - 1) // config.batch_size

  print(
      f"Processing {len(prompts)} prompts in {total_batches} batches of size"
      f" {config.batch_size}"
  )

  for i in range(0, len(prompts), config.batch_size):
    batch_idx = i // config.batch_size + 1
    batch_prompts = prompts[i : i + config.batch_size]

    if batch_idx % 10 == 0 or batch_idx == total_batches:
      print(
          "  Processing batch"
          f" {batch_idx}/{total_batches} ({len(batch_prompts)} prompts)..."
      )

    batch_outputs = tunix_sampler(
        batch_prompts,
        total_generation_steps=config.max_tokens,
        echo=config.echo,
    )
    all_outputs.extend(batch_outputs.text)

  end_time = time.perf_counter()

  # Create a mock outputs object for compatibility
  class MockOutputs:

    def __init__(self, texts):
      self.text = texts

  outputs = MockOutputs(all_outputs)

  # Calculate metrics
  total_time = end_time - start_time
  num_prompts = len(prompts)

  total_input_tokens = 0
  total_generated_tokens = 0

  for i, (prompt, output_text) in enumerate(zip(prompts, outputs.text)):
    input_tokens = len(tokenizer.encode(prompt))
    total_input_tokens += input_tokens

    if config.echo:
      total_output_tokens = len(tokenizer.encode(output_text))
      generated_tokens = max(0, total_output_tokens - input_tokens)
    else:
      generated_tokens = len(tokenizer.encode(output_text))

    total_generated_tokens += generated_tokens

  total_tokens = total_input_tokens + total_generated_tokens

  # Print performance results
  print("\n" + "=" * 25 + " Performance Evaluation Results " + "=" * 25)
  print("\n[Overall Summary]")
  print(f"  Total Time:                 {total_time:.2f} s")
  print(f"  Number of Prompts:          {num_prompts}")
  print(f"  Batch Size:                 {config.batch_size}")
  print(f"  Total Batches:              {total_batches}")

  print("\n[Latency Metrics]")
  avg_latency_per_request = total_time / num_prompts * 1000
  print(
      f"  Avg. Latency per Request:   {avg_latency_per_request:.2f} ms/request"
  )
  avg_latency_per_batch = total_time / total_batches * 1000
  print(f"  Avg. Latency per Batch:     {avg_latency_per_batch:.2f} ms/batch")

  print("\n[Token Stats]")
  print(f"  Total Input Tokens:         {total_input_tokens}")
  print(f"  Total Generated Tokens:     {total_generated_tokens}")
  print(f"  Grand Total Tokens:         {total_tokens}")

  print("\n[Throughput Metrics]")
  print(
      "  Request Throughput:         "
      f"{num_prompts / total_time:.2f} requests/sec"
  )
  print(
      "  Batch Throughput:           "
      f"{total_batches / total_time:.2f} batches/sec"
  )
  print(
      "  Input Token Throughput:     "
      f"{total_input_tokens / total_time:.2f} tokens/sec"
  )
  print(
      "  Output Token Throughput:    "
      f"{total_generated_tokens / total_time:.2f} tokens/sec"
  )
  print(
      "  Total Token Throughput:     "
      f"{total_tokens / total_time:.2f} tokens/sec"
  )

  print("\n" + "=" * 73 + "\n")

  if not config.hide_outputs:
    print("-" * 20 + " Generated Outputs Comparison " + "-" * 20)
    max_outputs_to_show = min(5, len(outputs.text))
    for i in range(max_outputs_to_show):
      prompt = raw_prompts[i]
      generated_text = outputs.text[i]
      ground_truth = ground_truths[i]

      print(f"Sample {i+1}:")
      print(f"Prompt: {prompt!r}")
      print(f"Generated: {generated_text!r}")
      print(f"Ground Truth: {ground_truth!r}")
      print("-" * 50)

    if len(outputs.text) > max_outputs_to_show:
      num_outputs = len(outputs.text)
      print(f"... (showing {max_outputs_to_show} of {num_outputs} outputs)")
  else:
    print(
        "Individual outputs were hidden as requested by the '--hide-outputs'"
        " flag."
    )


# ==================== Argument Parsing and Execution ====================
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Tunix Sampler Performance Test")
  parser.add_argument(
      "--model_id",
      type=str,
      default="meta-llama/Llama-3.1-8B-Instruct",
      help="Hugging Face model ID to download and test.",
  )
  parser.add_argument(
      "--max_tokens",
      type=int,
      default=64,
      help="Maximum tokens to generate per sample.",
  )
  parser.add_argument(
      "--cache_size",
      type=int,
      default=512,
      help="Cache size for the Tunix sampler (must be > max_tokens).",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=100,
      help="Batch size for processing prompts.",
  )
  parser.add_argument(
      "--num_samples",
      type=int,
      default=-1,
      help="Number of samples to use from the dataset (-1 for full dataset).",
  )
  parser.add_argument(
      "--hide_outputs",
      type=lambda x: (str(x).lower() == "true"),
      default=False,
      help="Hide individual generated outputs.",
  )

  args = parser.parse_args()

  # Create Config object from parsed arguments
  app_config = Config()
  app_config.model_id = args.model_id
  app_config.max_tokens = args.max_tokens
  app_config.cache_size = args.cache_size
  app_config.batch_size = args.batch_size
  app_config.num_samples = args.num_samples
  app_config.hide_outputs = args.hide_outputs

  print("Starting Tunix Performance Test")
  print(
      f"Configuration: max_tokens={app_config.max_tokens}, "
      f"cache_size={app_config.cache_size}, "
      f"batch_size={app_config.batch_size}"
  )
  print("=" * 50)
  main(app_config)
  print("\nTest completed successfully!")
