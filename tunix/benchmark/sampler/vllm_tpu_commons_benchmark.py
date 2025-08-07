import os
import time

# Import the datasets library
from datasets import load_dataset
import vllm.envs as envs
from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser

from tpu_commons.core import disagg_utils


def create_parser():
    """Creates a flexible argument parser."""
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="meta-llama/Llama-3.2-1B-Instruct", max_tokens=256)

    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens",
                                type=int,
                                help="Maximum number of tokens to generate.")
    sampling_group.add_argument("--temperature",
                                type=float,
                                help="Controls randomness in generation.")
    sampling_group.add_argument("--top-p",
                                type=float,
                                help="Nucleus sampling probability.")
    sampling_group.add_argument("--top-k",
                                type=int,
                                help="Top-k sampling.")
    
    # Add dataset-related parameters
    dataset_group = parser.add_argument_group("Dataset parameters")
    dataset_group.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of samples from the GSM8K dataset to use for testing "
             "(default: -1, which means use the entire test set)")
    
    # Add option to hide output
    parser.add_argument(
        "--hide-outputs",
        action="store_true",
        help="If set, do not print individual prompt/generated text outputs.")

    return parser


def main(args: dict):
    """Main function to run the LLM generation and performance benchmark."""
    # Pop arguments not used by LLM directly
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")
    num_samples = args.pop("num_samples")
    hide_outputs = args.pop("hide_outputs")

    # Create an LLM instance
    llm = LLM(**args)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature if temperature is not None else 0.0,
        top_p=top_p if top_p is not None else 1.0,
        top_k=top_k if top_k is not None else -1,
        stop=["Question:"]
    )

    # --- Load and prepare the GSM8K dataset ---
    print("Loading the GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    
    full_dataset_size = len(dataset['test'])
    if num_samples == -1:
        samples_to_use = full_dataset_size
        print(f"Default setting detected. Using the full test set with {samples_to_use} samples.")
    else:
        samples_to_use = min(num_samples, full_dataset_size)
        print(f"Using the specified {samples_to_use} samples for testing.")

    test_dataset = dataset['test'].select(range(samples_to_use))
    
    prompts = [f"Question: {example['question']}\nAnswer:" for example in test_dataset]
    ground_truths = [example['answer'] for example in test_dataset]
    print("Dataset prepared.")
    print("-" * 50)

    # --- Warm-up Run ---
    if len(prompts) > 2:
        print("Starting warm-up run...")
        _ = llm.generate(prompts[:2], sampling_params)
        print("Warm-up complete.")
    else:
        print("Too few samples, skipping warm-up.")
    print("-" * 50)

    # --- Performance Benchmark ---
    print("Starting performance benchmark...")
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.perf_counter()

    # --- Calculate Metrics ---
    total_time = end_time - start_time
    num_prompts = len(prompts)
    
    # Token stats
    total_input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    total_generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    total_tokens = total_input_tokens + total_generated_tokens

    # --- Print Performance Results ---
    print("\n" + "=" * 25 + " Performance Evaluation Results " + "=" * 25)
    
    # Overall Summary
    print("\n[Overall Summary]")
    print(f"  Total Time:          {total_time:.2f} s")
    print(f"  Number of Prompts:   {num_prompts}")

    # Latency Metrics
    print("\n[Latency Metrics]")
    print(f"  Avg. Latency per Request:  {total_time / num_prompts * 1000:.2f} ms/request")

    # Token Stats
    print("\n[Token Stats]")
    print(f"  Total Input Tokens:        {total_input_tokens}")
    print(f"  Total Generated Tokens:    {total_generated_tokens}")
    print(f"  Grand Total Tokens:        {total_tokens}")

    # Throughput Metrics
    print("\n[Throughput Metrics]")
    print(f"  Request Throughput:        {num_prompts / total_time:.2f} requests/sec")
    print(f"  Input Token Throughput:    {total_input_tokens / total_time:.2f} tokens/sec")
    print(f"  Output Token Throughput:   {total_generated_tokens / total_time:.2f} tokens/sec")
    print(f"  Total Token Throughput:    {total_tokens / total_time:.2f} tokens/sec")
    
    print("\n" + "=" * 73 + "\n")

    # --- Print Individual Outputs ---
    if not hide_outputs:
        print("-" * 20 + " Generated Outputs Comparison " + "-" * 20)
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            ground_truth = ground_truths[i]
            
            print(f"Prompt: {prompt!r}")
            print(f"Generated: {generated_text!r}")
            print(f"Ground Truth: {ground_truth!r}")
            print("-" * 50)
    else:
        print("Individual outputs were hidden as requested by the '--hide-outputs' flag.")


if __name__ == "__main__":
    os.environ['SKIP_JAX_PRECOMPILE'] = '1'
    parser = create_parser()
    args: dict = vars(parser.parse_args())

    if not disagg_utils.is_disagg_enabled():
        main(args)
    else:
        from unittest.mock import patch
        from tpu_commons.core.core_tpu import EngineCore as TPUEngineCore
        from tpu_commons.core.core_tpu import EngineCoreProc as TPUEngineCoreProc
        with patch('vllm.v1.engine.core.EngineCore', TPUEngineCore):
            with patch('vllm.v1.engine.core.EngineCoreProc', TPUEngineCoreProc):
                main(args)
