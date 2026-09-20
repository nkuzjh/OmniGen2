# OmniGen2 on CSGO Benchmark v2 Seen-10

本接入只负责 generation：`seen_discrete_test` 与 `seen_continuous`。训练使用
`seen_train`，完整 `seen_validation` diffusion loss 选择 `best`；两种推理共用同一个
冻结的 LoRA + numeric-pose adapter checkpoint。数据始终从发布的 manifest/split、
radar 和 calibration 读取，不扫描图片目录重新划分。

## 本机独立环境

本机是 Blackwell GPU，实际 smoke 使用系统已有的 PyTorch 2.12/CUDA 13，并把项目
依赖装在项目自己的 `.venv`；没有向 UniLIP conda 环境安装模型依赖：

```bash
cd /home/jiahao/task/OmniGen2
/home/jiahao/miniconda3/bin/python -m venv --system-site-packages .venv
.venv/bin/python -m pip install \
  'transformers==4.51.3' 'diffusers==0.35.2' 'peft==0.17.1' \
  timm omegaconf python-dotenv tensorboard torchdiffeq pytest
```

官方模型权重可让 Hugging Face 在首次运行时下载，或指定已完整下载的本地目录：

```bash
export OMNIGEN2_MODEL_PATH=/absolute/path/to/OmniGen2
export OMNIGEN2_VAE_MODEL_PATH=/absolute/path/to/FLUX.1-dev
export OMNIGEN2_TEXT_ENCODER_MODEL_PATH=/absolute/path/to/Qwen2.5-VL-3B-Instruct
export OMNIGEN2_PYTHON=/home/jiahao/task/OmniGen2/.venv/bin/python
```

本服务器代理偶尔会令 Hugging Face Xet token refresh 报 SSL EOF。统一脚本默认导出
`HF_HUB_DISABLE_XET=1`，改用支持断点续传的普通 Hub HTTP 下载；如确认 Xet 可用，可在
命令前显式设置 `HF_HUB_DISABLE_XET=0`。直接运行 Python 入口时建议先执行：

```bash
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=30
```

训练配置中的 VAE、text encoder 与基础 transformer 分别使用
`black-forest-labs/FLUX.1-dev`、`Qwen/Qwen2.5-VL-3B-Instruct` 和
`OmniGen2/OmniGen2`，需要它们已在 Hugging Face cache 中或当前账户可下载。
若下载在模型加载前失败，输出目录中只会有复制的配置和日志；再次执行同一条 train
命令会安全重试。出现 checkpoint、metrics 或模型资产后仍保持严格的显式 resume 要求。

## 已执行的 smoke

```bash
scripts/run_csgo_seen10.sh smoke --seed 0
```

该命令运行数据/pose/checkpoint 回归测试，并执行真实数据 batch、tiny OmniGen2
forward/backward、checkpoint 严格回读和一张标准 JPEG，再直接调用共享评测器的
`smoke discrete --limit 1`。tiny 模型仅验证接线，不是正式 benchmark 结果。产物为：

```text
outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_0/smoke/smoke_report.json
outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_0/smoke/tiny_checkpoint.pt
outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_0/smoke/discrete/gen_imgs/cs_agency/file_num68_frame_421.jpg
```

## 正式训练、转换、推理与评测

便捷入口（默认单进程；多卡可在 train 前设置 `NUM_PROCESSES`）：

```bash
scripts/run_csgo_seen10.sh train   --seed 0
scripts/run_csgo_seen10.sh convert --seed 0
scripts/run_csgo_seen10.sh infer   --seed 0 --task all
scripts/run_csgo_seen10.sh eval    --seed 0 --task all
```

恢复原生 Accelerate checkpoint：

```bash
scripts/run_csgo_seen10.sh train --seed 0 --resume-from-checkpoint latest
```

等价的直接命令如下：

```bash
cd /home/jiahao/task/OmniGen2

.venv/bin/python -m accelerate.commands.launch \
  --num_machines 1 --num_processes 1 --mixed_precision bf16 \
  train_seen10.py \
  --config options/csgo_seen10_lora.yml \
  --seed 0 \
  --output-root outputs/csgo_benchmark_v2_seen10/OmniGen2

.venv/bin/python convert_ckpt_to_hf_format.py \
  --config_path options/csgo_seen10_lora.yml \
  --model_path outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_0/train/best \
  --save_path outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_0/train/inference_adapter_best

.venv/bin/python infer_seen10.py \
  --task all --seed 0 \
  --data-root /home/jiahao/task/UniLIP/data/csgo_benchmark_v2 \
  --output-root outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_0 \
  --model-path "${OMNIGEN2_MODEL_PATH:-OmniGen2/OmniGen2}" \
  --adapter-path outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_0/train/inference_adapter_best

/home/jiahao/miniconda3/envs/UniLIP/bin/python \
  /home/jiahao/task/csgo_benchmark_v2_eval_general/run_eval.py discrete \
  --pred-root outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_0/discrete \
  --data-root /home/jiahao/task/UniLIP/data/csgo_benchmark_v2 \
  --output outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_0/evaluation/discrete

/home/jiahao/miniconda3/envs/UniLIP/bin/python \
  /home/jiahao/task/csgo_benchmark_v2_eval_general/run_eval.py continuous \
  --pred-root outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_0/continuous \
  --data-root /home/jiahao/task/UniLIP/data/csgo_benchmark_v2 \
  --output outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_0/evaluation/continuous
```

配置默认 4,000 optimizer steps，step 800/1600/2400/3200/4000 各验证和保存一次。
`late` 指向最后一次，`best` 指向最低 validation loss，`latest` 保留给原生恢复；loss
曲线为 `train/loss_curve.png`。正式结果路径为：

```text
outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_<seed>/train/checkpoint-<step>/
outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_<seed>/train/{late,best,latest}
outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_<seed>/discrete/gen_imgs/<map>/<frame>.jpg
outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_<seed>/continuous/gen_imgs/<map>/<frame>.jpg
outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_<seed>/evaluation/{discrete,continuous}/summary_equal_map.json
```

当前 `RUN_FULL=0`，因此没有启动正式 50,000 样本训练和 32,800 张推理，也没有可填
Table 1 的正式指标。
