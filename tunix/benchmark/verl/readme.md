# VERL Benchmark Setup Guide

这份指南将指导你如何设置 VERL 基准测试环境，包括 Docker 配置、数据集下载、账户登录和脚本运行。

### **步骤 1: 配置 Docker 根目录**

由于基准测试将在 Docker 容器中进行，你需要首先配置一个宿主机上的目录，用于与容器共享。这通常用于存储代码、数据集和模型。

在你的 `run_setup_verl.sh` 脚本中，找到脚本中内容 `# Create 'verl' container
echo "--- Creating 'verl' container ---"
sudo docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN \
  -v "$(pwd)":/workspace/verl_space \
  --name verl verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1-deepep sleep infinity` ，并修改 `$(pwd)` 为你想要设置的宿主机根目录的**绝对路径**。

例如，如果你希望将 `/scratch/haoyu` 作为根目录，则修改为：

```bash
-v /scratch/haoyu:/workspace/verl
```
否则默认使用根路径（不推荐）

### **步骤 2: 运行安装脚本**

执行 `setup_verl.sh` 脚本来自动完成环境的初始化。这个脚本将处理 Docker 的创建和配置。

```bash
bash setup_verl.sh
```

### **步骤 3: 进入容器**

`setup_verl.sh` 脚本运行成功后，会创建一个名为 `verl` 的容器。通过以下命令进入容器的终端：

```bash
docker exec -it verl bash
```

进入容器后，进入 `/workspace/verl` 目录下，这里是你的项目代码所在的位置。

### **步骤 4: 下载数据集**

基准测试需要使用 GSM8K 数据集。在容器内部，运行以下命令来下载数据集：

```bash
# 进入数据预处理目录
cd examples/data_preprocess

# 下载 GSM8K 数据集
python3 gsm8k.py --local_dir ~/data/gsm8k
```

### **步骤 5: 登录 Hugging Face 和 WandB**

为了下载模型（如 Llama）和记录实验结果，你需要登录 Hugging Face 和 Weights & Biases (WandB)。

```bash
# 登录 Hugging Face
huggingface-cli login

# 登录 WandB
wandb login
```

### **步骤 6: 运行基准测试脚本**

所有配置和数据集都准备好后，你就可以运行基准测试脚本了：

```bash
# 假设你在项目根目录
bash run_llama3_1B_grpo.sh
```

这个脚本将使用你配置好的环境来执行 Llama 3 1B 模型的 GRPO 训练。




如果训练中出现训练崩溃，崩溃的任务可能变为僵尸进程，持续占用gpu资源，运行以下命令restart清理资源：
exit 退出container
sudo systemctl restart docker
docker restart verl
docker exec -it verl bash #重新进入
