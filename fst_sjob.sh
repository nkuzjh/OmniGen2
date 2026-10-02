#!/bin/bash
#SBATCH --job-name=omnigen2_aligned
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem=80G
#SBATCH --partition=gbunchQ2
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=fst_slogs/%j.out  # 将日志统一收纳到 logs 文件夹
#SBATCH --error=fst_slogs/%j.err

# 1. 创建日志目录，防止因找不到目录而无法输出日志
source ~/.bashrc
mkdir -p fst_slogs

# 2. 远程集群 Conda 环境初始化 (直接使用 conda activate 会在非交互 shell 中报错)
# 这里使用通用 hook，或者替换为你自己的路径：source ~/miniconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
# conda activate UniLIP

# 检查CUDA和GPU状态
echo "=== CUDA和GPU检查 ==="
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}'); print(f'BF16 Supported: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}')"

# 检查GPU数量
echo "=== GPU数量检查 ==="
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"

# 3. 动态生成 Master Port，防止在共享节点上发生端口冲突
MASTER_PORT=$((10000 + $RANDOM % 20000))

# 4. 启动训练
# 移除了 CUDA_VISIBLE_DEVICES=0，SLURM 会自动分配并隔离 GPU，强行指定 0 可能会找不到卡
NUM_PROCESSES=2 nohup bash scripts/run_csgo_seen10.sh train --experiment csgo_seen10_exp32gen_aligned --seed 42 --micro-batch-size 64 --gradient-accumulation-steps 1 --resume-from-checkpoint latest >omnigen2_aligned.nohup.out1 2>&1 &
