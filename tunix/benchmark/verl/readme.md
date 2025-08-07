# VERL Benchmark Setup Guide


## Step 1: Configure Docker Root Directory

Since the benchmark will run in a Docker container, you need to first configure a directory on the host machine for sharing with the container. This is typically used to store code, datasets, and models.

In your `run_setup_verl.sh` script, find the following content:

```bash
# Create 'verl' container
echo "--- Creating 'verl' container ---"
sudo docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN \
  -v "$(pwd)":/workspace/verl_space \
  --name verl verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1-deepep sleep infinity
```

Modify `$(pwd)` to the **absolute path** of your desired host machine root directory.

For example, if you want to use `/scratch/haoyu` as the root directory, modify it to:

```bash
-v /scratch/haoyu:/workspace/verl
```

Otherwise, the default root path will be used (not recommended).

## Step 2: Run Installation Script

Execute the `setup_verl.sh` script to automatically complete environment initialization. This script will handle Docker creation and configuration.

```bash
bash setup_verl.sh
```

## Step 3: Enter Container

After the `setup_verl.sh` script runs successfully, it will create a container named `verl`. Enter the container terminal with the following command:

```bash
docker exec -it verl bash
```

Once inside the container, navigate to the `/workspace/verl` directory, where your project code is located.

## Step 4: Download Dataset

The benchmark requires the GSM8K dataset. Inside the container, run the following commands to download the dataset:

```bash
# Navigate to data preprocessing directory
cd examples/data_preprocess

# Download GSM8K dataset
python3 gsm8k.py --local_dir ~/data/gsm8k
```

## Step 5: Login to Hugging Face and WandB

To download models (such as Llama) and record experimental results, you need to log in to Hugging Face and Weights & Biases (WandB).

```bash
# Login to Hugging Face
huggingface-cli login

# Login to WandB
wandb login
```

## Step 6: Run Benchmark Script

Once all configurations and datasets are ready, you can run the benchmark script:

```bash
# Change script path accordingly
bash run_llama3_1B_grpo.sh
```

This script will use your configured environment to execute GRPO training on the Llama 3 1B model.

## Troubleshooting: Handling Training Crashes

If training crashes occur, the crashed tasks may become zombie processes that continue to occupy GPU resources. Run the following commands to restart and clean up resources:

```bash
# Exit container
exit

# Restart Docker service
sudo systemctl restart docker

# Restart verl container
docker restart verl

# Re-enter container
docker exec -it verl bash
```
