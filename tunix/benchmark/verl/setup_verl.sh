#!/bin/bash

set -e

echo "--- Starting Docker configuration ---"

DOCKER_CONFIG_PATH="/etc/docker/daemon.json"
CONFIG_TO_ADD='{"runtimes": {"nvidia": {"path": "nvidia-container-runtime", "runtimeArgs": []}}}'
TMP_FILE="/tmp/daemon.json.tmp"

# Check and install jq utility
echo "--- Checking for jq installation..."
if ! command -v jq &> /dev/null; then
    echo "jq not found, installing..."
    sudo apt-get update
    sudo apt-get install -y jq
fi

# Check if config file exists or is empty, and initialize it
echo "--- Checking if $DOCKER_CONFIG_PATH exists or is empty..."
if [ ! -f "$DOCKER_CONFIG_PATH" ] || [ ! -s "$DOCKER_CONFIG_PATH" ]; then
    echo "Config file not found or empty, creating with '{}'..."
    echo "{}" | sudo tee "$DOCKER_CONFIG_PATH" > /dev/null
fi

# Merge Nvidia configurations
echo "--- Merging configuration..."
sudo sh -c "jq '. + $CONFIG_TO_ADD' $DOCKER_CONFIG_PATH > $TMP_FILE"

# Move temporary file to correct location
echo "--- Moving config to final location..."
sudo mv "$TMP_FILE" "$DOCKER_CONFIG_PATH"

echo "Configuration updated."

# Restart Docker service
echo "--- Restarting Docker service..."
sudo systemctl daemon-reload
sudo systemctl restart docker
echo "Docker service restarted."

# Docker container operations
echo "--- Starting Docker operations ---"

# Launch my_vllm_app container
echo "--- Starting my_vllm_app container on port 8000 ---"
sudo docker run --rm --privileged --net=host --name my_vllm_app -p 8000:8000 verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1-deepep || true
echo "my_vllm_app container started."

# Clean up old 'verl' container
echo "--- Cleaning up old 'verl' container ---"
docker stop verl &>/dev/null || true
docker rm verl &>/dev/null || true
echo "Old 'verl' container removed (if it existed)."

# Create 'verl' container
echo "--- Creating 'verl' container ---"
sudo docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN \
  -v "$(pwd)":/workspace/verl_space \
  --name verl verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1-deepep sleep infinity
echo "'verl' container created."

# Start 'verl' container
echo "--- Starting 'verl' container ---"
docker start verl
echo "'verl' container started."

# Enter 'verl' container for initialization
echo "--- Entering 'verl' container for setup ---"
docker exec -it verl bash -c "
  echo '--- Running setup commands inside container ---';
  cd verl_space;
  git clone https://github.com/volcengine/verl /workspace/verl;
  cd verl;
  pip3 install --no-deps -e .;
  echo '--- Setup finished. Type exit to leave container. ---';
  bash
"
echo "Exited 'verl' container session."
echo "--- Script execution completed ---"