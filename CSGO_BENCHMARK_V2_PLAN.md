# OmniGen2 → CSGO Benchmark v2 Seen-10 接入方案

## 范围与边界

- 模型类型是 `GENERATION`，只接入 discrete generation 和 continuous generation，不新增 localization 头。
- 训练仅读 `seen_train`，验证仅读 `seen_validation`，同一个冻结 checkpoint 分别推理 `seen_discrete_test` 和 `seen_continuous`。
- 只修改 OmniGen2 项目；`DATA_ROOT` 和现有共享评测器保持不变。

## 数据与条件接入

1. 新增 manifest-driven Seen-10 dataset：固定按发布的地图顺序读取
   `minimal_dataset_report.json`、`benchmark_manifest.json`、`splits/seen/*`、
   `images/<map>/<file_frame>.jpg`、`radars/<map>/*` 和发布的 Z calibration；不扫描图片树重建 split。
2. 输入 radar 继续使用 OmniGen2 原生 reference-image condition；目标 FPV 仅在训练/验证时打开，推理数据集不读取目标帧。
3. 5DoF 按 `[x/1024, y/1024, (z-z_min)/(z_max-z_min), pitch/(2π), yaw/(2π)]`
   归一化。新增可训练 pose MLP，把数值向量投影成额外的 text-feature condition token；不依赖自然语言坐标。

## 训练、checkpoint 与推理

- 扩展原生 `train.py` 使其选择 Seen-10 dataset、向 diffusion transformer 传入 pose token，并使用完整 `seen_validation` diffusion loss 选择 best checkpoint。
- 保留 Accelerate/FSDP、optimizer、transport loss 和 LoRA 路径；LoRA 训练时额外解冻 pose MLP。
- 配置的总 step 可被 5 整除，eval/save interval 均为 `max_train_steps / 5`；每次保存更新原生恢复别名 `latest` 和协议要求的 `late`，验证最优更新 `best`。训练日志同时写 JSONL，结束时生成 loss 曲线。
- 扩展 checkpoint 转换，使 LoRA 权重和 pose MLP 权重一起进入可推理目录。
- 推理严格按 split/clip/frame 顺序处理，每个 sample 只使用 radar+pose，保存 448×448 RGB JPEG 并写推理 provenance。

## 推理加速方案（不含 compile）

保留旧入口 `scripts/run_csgo_seen10.sh infer --seed 0 --task all`，不要求用户增加参数即可
使用加速路径。推理分两级优化：

1. **第一优先级：batch denoising + 小批量 VAE decode。** 默认最大生成 batch 为 16，遇到
   CUDA OOM 自动按 16、8、4、2、1 减半并重试；VAE decode microbatch 默认为 1，以限制解码
   峰值显存。可使用 `INFERENCE_BATCH_SIZE`、`INFERENCE_DECODE_BATCH_SIZE` 环境变量，或
   `--batch-size`、`--vae-decode-batch-size` 参数覆盖。batch 16 还没有正式 checkpoint 上的
   实测结果，应该根据 OOM 回退和实测吞吐选择实际 batch。
2. **第二优先级：缓存并复用重复输入计算。** 十张 radar map 的 VAE posterior 参数每次运行
   只编码一次；每个样本使用独立 reference-posterior seed 采样，diffusion 初始 noise 也按
   task/sample ID 使用独立 seed。缓存 negative-prompt embedding 和 RoPE 频率，LoRA 加载后
   默认 fuse 一次。共享 radar PIL 在缓存生命周期内保持只读。

以上优化不改变模型权重、prompt、采样步数、guidance 或输出尺寸；目标是保持原有生成质量。
没有启用 `torch.compile` 或近似缓存。验收时在同一 checkpoint、seed 和样本上比较 batch 1
与运行实际采用的 batch，记录图像吞吐和显存，并使用共享评测器比较 discrete/continuous
指标；正式输出仍通过原 eval 命令检查。脚本入口、参数和验证流程见
[`CSGO_SEEN10.md`](CSGO_SEEN10.md#推理加速设置)。
正式 pending manifest 将请求的生成/decode batch 视为 provenance，断点续推需保持这两个参数；
运行内的自动 OOM 减半不改变该约束。

## 拟修改/新增文件

- `omnigen2/dataset/csgo_seen10_dataset.py`：数据合同、pose 归一化、split/clip 顺序。
- `omnigen2/models/transformers/transformer_omnigen2.py`：pose numeric-token adapter。
- `omnigen2/pipelines/omnigen2/pipeline_omnigen2.py`：推理时传递 pose condition。
- `train.py`：Seen-10 dataset、pose、validation、best/late/latest、loss 曲线。
- `convert_ckpt_to_hf_format.py`：保存 LoRA 外的 pose adapter。
- `options/csgo_seen10_lora.yml`：Seen-10 单种子训练配置。
- `train_seen10.py`、`infer_seen10.py`：轻量 train/infer wrapper。
- `scripts/run_csgo_seen10.sh`：`smoke|train|convert|infer|eval|all`，支持 `--seed` 和默认
  batch16 的推理；可用 `INFERENCE_BATCH_SIZE`、`INFERENCE_DECODE_BATCH_SIZE` 或对应 CLI
  参数覆盖。
- `tests/test_csgo_seen10_dataset.py`、`tests/test_pose_conditioning.py`、
  `tests/test_csgo_training_helpers.py`、`tests/test_infer_seen10_batching.py`、
  `tests/test_pipeline_inference_caches.py`：数据合同、condition token、checkpoint 回读、批推理、
  OOM 降级、缓存与输出 identity 的快速回归。
- `CSGO_SEEN10.md`：环境、直接命令、checkpoint/结果路径。

## 验收

1. 静态/单元测试：split 数量与顺序、路径映射、pose 值、continuous clip identity、pose adapter forward/backward 和权重回读。
2. 最小 smoke：真实数据 batch；一次 forward/backward；checkpoint save/load；一张 448×448 RGB JPEG；共享评测器 `smoke discrete --limit 1`。
3. `RUN_FULL=0`，因此不启动 50,000 样本完整训练及 32,800 张正式推理；保留无需改代码即可在后续启动 seed 0/1/2 的命令。
