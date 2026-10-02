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

# 必须从 OmniGen2 项目根目录提交
cd "${SLURM_SUBMIT_DIR:?}"

export OMNIGEN2_PYTHON="$PWD/.venv/bin/python"
export PYTHONUNBUFFERED=1

# 使用与原训练一致的 HF 缓存目录
source <("$OMNIGEN2_PYTHON" scripts/download_csgo_seen10_assets.py --print-env)

# 前台运行，不使用 nohup、& 或额外日志重定向
export NUM_PROCESSES=2
exec bash scripts/run_csgo_seen10.sh train \
  --experiment csgo_seen10_exp32gen_aligned \
  --seed 42 \
  --micro-batch-size 64 \
  --gradient-accumulation-steps 1 \
  --resume-from-checkpoint latest
