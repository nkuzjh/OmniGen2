# OmniGen2 接入 CSGO Benchmark v2 Seen-10

本文统一维护**已经实现**的运行行为、环境权重、命令、输出和有证据的状态记录。命令从 `/home/jiahao/task/OmniGen2` 执行。任务仅为 **radar/map + 当前 5DoF pose → FPV**，不增加定位任务，不包含 CrossMap-4。

文档分工以及环境/资产/路径初始化方式参考 ControlAR，但不复制其 AR 模型、compiled 推理或运行结果：

- 本文：legacy/aligned 的运行说明、实际命令和带日期的状态快照。
- [CSGO_SEEN10_PLAN.md](CSGO_SEEN10_PLAN.md)：设计依据、模块职能、配方来源、实施边界与验收标准。
- [CSGO_ALIGNED_VALIDATION.md](CSGO_ALIGNED_VALIDATION.md)：2026-09-27 aligned 实施时的详细验收记录，不是持续更新的训练日志。
- 原 [CSGO_SEEN10_ALIGNED.md](CSGO_SEEN10_ALIGNED.md) 保留旧链接入口，不再重复维护命令。
- [CSGO_BENCHMARK_V2_PLAN.md](CSGO_BENCHMARK_V2_PLAN.md) 与 `csgo_benchmark_v2_start.md` 保留首次接入历史，不覆盖当前配置和实际代码。

## 1. 实验范围与比较口径

主参考为 UniLIP generation-only `exp32_gen`；joint generation+localization `exp32` 只作次要对照。aligned 对齐数据、可用条件信息、224/448 尺寸、generation 曝光预算和评测口径；OmniGen2 官方 LoRA/优化配方与原生 flow 采样作为模型差异披露，不声称参数量、FLOPs 或优化行为完全相同。

| 实验 | 入口/配置 | 输出根目录（相对项目根目录） | 结果选点 |
| --- | --- | --- | --- |
| legacy 首次接入 | 不传 `--experiment`；`options/csgo_seen10_lora.yml` | `outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_0` | wrapper 推理默认 `best` |
| exp32_gen aligned | `--experiment csgo_seen10_exp32gen_aligned`；同名 `options/*.yml` | `outputs/csgo_seen10_exp32gen_aligned/OmniGen2/seed_42` | 主结果 `late`，`best` 补充 |

| 项目 | legacy 默认单卡 | aligned |
| --- | --- | --- |
| radar / FPV 尺寸 | 448 / 448 | 224 / 448 |
| pose 条件 | pose 文本 + 数值 pose MLP/token | 只用 pose 文本，不创建数值模块 |
| 可训练模块 | attention LoRA + pose MLP | 仅 attention LoRA |
| LoRA r / alpha / dropout | 8 / 8 / 0 | 8 / 8 / 0 |
| LoRA LR / pose LR | 8e-7 / 1e-4 | 8e-7 / 无 |
| AdamW betas / weight decay | (0.9, 0.95) / 0.01 | 同左 |
| warmup / 后续 scheduler | 100 updates / constant | 官方 500 updates / constant |
| effective batch | 1 | 128；仅约束实际 world × micro × accumulation |
| 配置 updates / 曝光量 | 4,000 / 4,000 | 19,500 / 2,496,000 |
| 完整验证与保存 | 800/1600/2400/3200/4000 | 4000/8000/12000/16000/19500 |
| 文本 / reference dropout | 1e-4 / 0 | 1e-4 / 丢文本后以0.5概率丢reference |
| TF32 | 开启 | 关闭 |

表内 legacy 预算是**配置默认值，不是已完成训练记录**；实际状态见第8节。已训练过的 CSGO checkpoint 不得作为 aligned 初始化权重；不新增 aux_loc_loss、perception_loss 等任务。

## 2. 数据、条件与输出边界

数据根目录：`/home/jiahao/task/UniLIP/data/csgo_benchmark_v2`。使用发布的 manifest、selection、split、radar 映射与 frozen calibration，不扫描图片目录重新构造 split。

固定地图：`cs_agency`、`cs_italy`、`de_ancient`、`de_anubis`、`de_dust2`、`de_inferno`、`de_mirage`、`de_nuke`、`de_overpass`、`de_train`。

| split | 数量 | target FPV |
| --- | ---: | --- |
| seen_train | 50,000（每地图5,000） | 训练可读取 |
| seen_validation | 5,000（每地图500） | 验证可读取 |
| seen_discrete_test | 20,000（每地图2,000） | 推理不读取 |
| seen_continuous | 12,800；200 clips × 64 frames | 推理不读取 |

模型只使用当前 radar/map、地图名、当前 `[x,y,z,pitch,yaw]` 与固定任务文本。文本包含归一化 pose，顺序固定、数值用 `.8g` 格式，不是新增数值 adapter：

```text
x / 1024
y / 1024
(z - frozen_z_min_map) / (frozen_z_max_map - frozen_z_min_map)
pitch / (2*pi)  # 协议中的angle_v
yaw / (2*pi)    # 协议中的angle_h
```

逐地图 z 范围使用发布的 exact calibration，不重新统计测试集。aligned 文本上限888 tokens并检查截断；当前全数据审计最大136 tokens。

图像只做 RGB、确定性 PIL bicubic resize 和原生归一化：aligned radar224、target448，legacy 两者448；无随机 crop、flip、颜色扰动或擦除。aligned 保留官方 CFG 正则：丢文本概率1e-4，**只在丢文本时**以0.5概率丢radar，总radar丢弃概率5e-5。验证和推理不做条件 dropout。

两类推理均构造 `CSGOSeen10Dataset(load_target=False)`，并检查返回值不含 target。元数据可保留目标路径，但不打开为模型输入。不使用相邻/历史/未来真实 FPV，也不使用前一生成帧；连续集逐帧独立，保留 manifest 的 clip/frame identity 与顺序。

每个 condition 只生成一张 **448×448 RGB JPEG**，相对路径 `gen_imgs/<map>/<file_frame>.jpg`，采用 Pillow 默认 JPEG 编码，与已核对的 UniLIP saver 一致。不做 best-of-N，不用测试指标挑选 checkpoint 或调参。

## 3. aligned 模块、训练预算与恢复

### 3.1 模型与可训练状态

Qwen 负责文本编码，不是生成 FPV 的 Transformer。radar 走 FLUX VAE reference-image 路径，**不经过 Qwen 视觉塔**。文本、radar latent 和带噪 target latent 在 OmniGen2 生成 Transformer 汇合，按原生 linear velocity flow 目标训练，推理由 Euler 更新 latent 后经 VAE 解码。具体连接器前缀和职能见方案文档第2节。

LoRA suffix 为 `to_q`、`to_k`、`to_v`、`to_out.0`，覆盖 `layers.0..31.attn` 及 `context_refiner`、`ref_image_refiner`、`noise_refiner` 各2个block。共152个目标线性层、304个A/B张量，r8/alpha8/dropout0，**5,107,200**个可训练参数，约占生成Transformer的0.1286%，占含Qwen/VAE的训练模型总参数0.07151%。

其余全部冻结：Qwen、VAE、所有基座权重、MLP、文本/图像投影、输出模块和 image-index embedding。optimizer 仅收录 LoRA。完整逐参数审计为每个训练run的 `train/parameter_audit.json`；没有额外的 pose 参数组。

唯一LR组8e-7；AdamW betas=(0.9,0.95)、weight decay=0.01、eps=1e-8、clip norm=1。500次optimizer updates的warmup从1e-18开始，随后constant；不按batch放大LR。BF16、gradient checkpointing开启，EMA/TF32关闭，保留原生 lognorm timestep/dynamic time shift。

aligned 单独固定数值执行策略：PyTorch deterministic algorithms（遇到不支持的非确定性算子报错）、cuDNN deterministic/benchmark=False、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、原生Triton RMSNorm前后向已有的4-warps配置。保留原生SDPA，不改变loss，不开启 `torch.compile`，旧入口不采用此设置。策略写入checkpoint合同；已测同软硬件/同batch拓扑的恢复逐位一致，不承诺跨硬件或并行拓扑一致。

### 3.2 预算与多卡

```text
实际 world_size × micro_batch_per_device × gradient_accumulation = 128
默认 1 × 1 × 128；已测 1 × 2 × 64；4 × 4 × 8 等组合也合法
每 epoch：49,920 源样本 / 128 = 390 updates，尾部80条不构成不足128的更新
390 × 50 = 19,500 optimizer updates
19,500 × 128 = 2,496,000 次generation曝光 = 49.92个完整数据epoch
```

只要求三个因子为正整数且乘积128，不分别固定micro或accumulation。`max_optimizer_steps=19500`为终止条件；scheduler、日志global step和checkpoint均按optimizer update，不按microstep。CFG分支及token展开不重复计数。aligned单机多卡使用Accelerate/DDP；legacy原有多卡FSDP路径不变。

### 3.3 保存、验证与恢复

正式实验仅在 **4000、8000、12000、16000、19500** 保存，每次使用完整5,000条validation计算原生generation loss。验证随机seed固定4242，且保留训练RNG；分布式验证不重复padding样本。

- `best`：五次validation loss最低者，同loss保留更早者。
- `late`：只在正式step19500完成时指向该checkpoint，是主比较结果。
- `latest`：当前run最近的完整恢复点，不是任意最新日志step。

checkpoint在梯度累计边界保存LoRA、AdamW、scheduler、step、Python/NumPy/CPU/CUDA RNG、sampler epoch/cursor；BF16无AMP scaler，metadata明确记录 `scaler=null`。冻结基座以官方snapshot引用复用。原子落盘并写入 `COMPLETE` 后才更新相对symlink，不额外复制三份checkpoint。

恢复只接受当前run内最新的完整checkpoint，并校验配置、源码、基础模型、数据和参数审计身份；不能混用smoke/正式run，也不能原地退回更早step。改变合法batch拓扑可恢复optimizer/sampler，但会警告并重新播种各rank随机流，不再是逐位续跑。要逐位恢复应保持原执行组合、代码及依赖环境。

legacy使用原生Accelerate checkpoint；其`late`在每次保存时更新，与aligned“仅最终19500”的语义不同。legacy的`latest`用于显式恢复，wrapper默认转换/推理`best`。两者的checkpoint格式和别名不能混用。

## 4. 新服务器初始化、官方权重与路径

### 4.1 准备流程

参考 ControlAR 的“环境 / 资产 / 路径检查”分工，新增独立脚本；不复制其模型、训练配方或编译推理。面向 **Linux + NVIDIA GPU**，不要求新服务器用户名、项目安装位置或 Conda 环境名相同。Git 不包含 Python 环境、模型权重、Benchmark 数据、共享评测器或历史输出；须分别准备，**不要直接复制旧服务器的 `.venv`**。

从新服务器 OmniGen2 checkout 根目录执行。以下均为用户手动命令，本次实现没有安装环境或下载大权重：

```bash
# 1. 只安装独立训练环境；默认项目 .venv，不影响 UniLIP/评测器。
bash scripts/setup_csgo_seen10.sh --env-only

# 2. 官方基础权重。仅 aligned：约23.56 GB；需要legacy整套推理则用 --profile all。
# FLUX.1-dev需先在HF接受访问条款，并使用有权限的账户登录。
.venv/bin/hf auth login
.venv/bin/python scripts/download_csgo_seen10_assets.py --profile aligned

# 3. 显式绑定此次下载的固定revision，避免依赖HF可变/缺失的refs/main。
# 此命令只打印并设置三个模型路径，不下载；在后续训练/推理使用的同一shell执行。
source <(.venv/bin/python scripts/download_csgo_seen10_assets.py --print-env)

# 4. 数据和共享评测器由用户另行部署；示例替换为目标机器真实位置。
export CSGO_DATA_ROOT="/srv/datasets/csgo_benchmark_v2"
export SHARED_EVAL_DIR="/srv/projects/csgo_benchmark_v2_eval_general"
# 遵循共享评测器自己的README准备环境/指标资产；不要在OmniGen2里复制指标实现。
bash "$SHARED_EVAL_DIR/setup_env.sh"

# 5. 只读检查：不安装、不下载、不初始化CUDA、不创建run目录。
bash scripts/setup_csgo_seen10.sh --check --profile aligned
bash scripts/run_csgo_seen10.sh train --print-paths
bash scripts/run_csgo_seen10.sh train \
  --experiment csgo_seen10_exp32gen_aligned --print-paths
bash scripts/run_csgo_seen10.sh eval \
  --experiment csgo_seen10_exp32gen_aligned --print-paths

# 6. 在目标GPU节点显式验证CUDA；仅8×8 BF16矩阵，不载模型、不启动训练。
bash scripts/setup_csgo_seen10.sh --check-cuda
```

若只查看安装方式，运行 `bash scripts/setup_csgo_seen10.sh --help`。不加参数的 setup 是**安装环境 + 下载/检查 all 资产**，不是只读操作；`--env-only` 跳过资产，`--check --env-only` 仅检查环境。对既有完整环境只读检查、不执行 pip，不升级或降级；既有目录非环境或依赖不完整时保留原样并报错，显式选择另一目录：

```bash
export OMNIGEN2_PYTHON="$PWD/.venv-csgo-seen10/bin/python"
OMNIGEN2_BOOTSTRAP_PYTHON=/path/to/python3.11 \
  bash scripts/setup_csgo_seen10.sh --env-only
# 自定义环境时，下文所有 .venv/bin/python 改用 "$OMNIGEN2_PYTHON"。
```

fresh 安装选 Python3.11/3.12；找不到合适解释器时尝试 Conda 创建隔离 Python3.11。可用 `OMNIGEN2_BOOTSTRAP_PYTHON` / `OMNIGEN2_CONDA_EXE` 显式指定。新环境使用 PyTorch2.7.1 + torchvision0.22.1 的 cu128 配对，依据 [PyTorch官方版本表](https://pytorch.org/get-started/previous-versions/)，与 ControlAR 的新机环境后端一致；这不改变 OmniGen2 官方 LoRA/优化配方。其他直接依赖固定在 `requirements-csgo-seen10.txt`，原始 `requirements.txt` 保留不改。不是所有传递依赖的带哈希锁文件；新安装会记录实际安装版本。FlashAttention、xformers、bitsandbytes、torchaudio不作为必需依赖。

`OMNIGEN2_TORCH_BACKEND=cpu` 仅供 CPU 环境检查；实际训练/推理需要兼容驱动、BF16 GPU 与原生 Triton。GPU内核/显存是否适合仍需目标机器的 `--check-cuda` 和第7节独立 smoke；CPU导入成功不等于新机器的真实模型验收。**尚未在另一台服务器完成 fresh 安装及 GPU smoke**，不承诺跨 GPU/库版本逐位一致。

### 4.2 官方基础资产与离线缓存

aligned 从官方原始基础权重开始，固定清单 `scripts/csgo_seen10_assets.json` 使用与此前验收一致的revision；文件大小、LFS SHA256 / Git blob SHA1 来自官方 HF 固定提交的文件元数据。

| 用途 | 官方仓库 | 固定revision |
| --- | --- | --- |
| 生成Transformer / Euler scheduler | `OmniGen2/OmniGen2` | `df5dca8a981d74e6c3af214c145f5c735fe72367` |
| Qwen文本模型 / tokenizer / processor | `Qwen/Qwen2.5-VL-3B-Instruct` | `66285546d2b821cf421d4f5eb2576359d3770cd3` |
| VAE | `black-forest-labs/FLUX.1-dev` 的 `vae/` | `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21` |

| 资产profile | 文件数 / 清单大小 | 覆盖范围 |
| --- | --- | --- |
| `aligned` | 19 / 23.56 GB | 三份官方组件；aligned训练和推理，也覆盖legacy训练需要的基础组件 |
| `legacy` 或 `all`（默认） | 38 / 38.93 GB | 以上加OmniGen2 bundled mllm、processor、VAE等，供legacy完整pipeline推理 |

只下载FLUX的VAE，不下载其完整生成模型。容量不含Python环境、数据、输出及下载临时余量。下载调用 [HF官方缓存接口](https://huggingface.co/docs/huggingface_hub/guides/download)，按不可变commit逐文件获取、复用已有有效缓存；支持Hub未完成文件的续传。已有文件大小/hash不匹配时拒绝覆盖，要求排查或换独立cache；不会改写 `refs/main`、CSGO checkpoint或正在运行任务的权重。

```bash
# 仅查看文件清单和容量，不联网、不读大权重、不写文件：
python3 scripts/download_csgo_seen10_assets.py --dry-run --profile aligned
# 独立执行离线完整字节/hash检查；缺失或损坏返回非0。
python3 scripts/download_csgo_seen10_assets.py --check --profile aligned
# 缓存可迁移到大容量磁盘；所有后续命令使用相同环境变量。
export HF_HUB_CACHE="/srv/cache/huggingface/hub"
.venv/bin/python scripts/download_csgo_seen10_assets.py --profile aligned
source <(python3 scripts/download_csgo_seen10_assets.py --print-env)
```

缓存选择：`--cache-dir` > `HF_HUB_CACHE` > `HUGGINGFACE_HUB_CACHE` > `HF_HOME/hub` > `XDG_CACHE_HOME/huggingface/hub`（默认用户 `~/.cache/huggingface/hub`）。`--print-env`输出shell安全引用的三个snapshot路径，本身不是完整性证明；先用 `--check` 核对。`OMNIGEN2_MODEL_PATH`、`OMNIGEN2_VAE_MODEL_PATH`、`OMNIGEN2_TEXT_ENCODER_MODEL_PATH`继续支持手动覆盖；本地路径建议绝对路径。

离线转运HF缓存时保留 `models--.../snapshots/<revision>` 与 `blobs/` 及相对符号链接，不能只复制指向缺失blob的snapshot目录。aligned只接受官方repo ID或能核实repo/revision身份的HF snapshot，不能任意重命名为普通模型目录。下载固定revision不保证 `refs/main` 存在，因此新服务器应执行上述 `--print-env` 绑定，不能仅下载后依赖默认repo ID。不会自动将当前机器的旧CSGO模型当成基础权重。

三个模型变量都传给训练及aligned推理；legacy推理只从 `OMNIGEN2_MODEL_PATH` 加载完整pipeline，不另用外置Qwen/FLUX。**aligned资产齐全不等于legacy整套推理已就绪**。`--check`核对的是固定官方清单，不验证用户任意自定义模型目录。

下载器和wrapper默认 `HF_HUB_DISABLE_XET=1`、download timeout120、etag timeout30；支持显式HF环境变量、`HF_ENDPOINT`与已授权HF凭证。FLUX访问被拒绝时应先完成授权，不绕过门禁、不在脚本或文档中存储token。项目内历史 `pretrained_models/OmniGen2-ms` 尚有7个 `.aria2` sidecar，不能当作完整已验收模型。

### 4.3 服务器路径与旧命令兼容

新wrapper参数：`--data-root PATH`、`--eval-root PATH`、`--eval-python PATH`（别名 `--unilip-python`）、`--print-paths`。两种profile均支持；未传新参数时保留当前机器默认。相对机器路径锚定项目根目录，不跟随启动命令的cwd；Python路径不会穿透 `bin/python` symlink，保留venv身份。

| 用途 | 选择规则（从高到低） |
| --- | --- |
| 训练/推理Python | `OMNIGEN2_PYTHON` > checkout内 `.venv/bin/python` |
| 数据 | CLI `--data-root` > `CSGO_DATA_ROOT` > `CSGO_BENCHMARK_V2_DATA` > `DATA_ROOT` > 配置自定义路径 > 已存在的旧默认 > checkout同级 `UniLIP/data/csgo_benchmark_v2` |
| 共享评测器 | CLI `--eval-root` > `SHARED_EVAL_DIR` > `CSGO_EVAL_ROOT` > 已存在旧默认 > checkout同级 `csgo_benchmark_v2_eval_general` |
| 评测Python | CLI `--eval-python/--unilip-python` > **所选共享评测器** `.venv/bin/python` > `EVAL_PYTHON` > `UNILIP_PYTHON` > 旧UniLIP Python |

旧数据、评测器、评测Python默认分别为 `/home/jiahao/task/UniLIP/data/csgo_benchmark_v2`、`/home/jiahao/task/csgo_benchmark_v2_eval_general`、`/home/jiahao/miniconda3/envs/UniLIP/bin/python`。显式设置不存在路径时报告错误，**不静默换到其他数据/评测器**。自动使用共享评测器自己的venv是本次新增规则；若需强制复现历史评测环境，用 `--eval-python` 指明，并核对共享评测器的指标配置/版本。

`--print-paths`只需标准库解释器，报告来源、exists、ready，允许在训练环境、数据或评测环境尚未安装时运行；返回0仅表示检查命令执行成功，**不是所有路径ready，也不是数据协议/指标资产检查通过**。wrapper对配置仅支持现有的简单`data.data_root`标量；自定义YAML若使用anchor、插值或flow等复杂表示，会明确拒绝，需用`--data-root`/环境变量指定（直接训练入口仍由OmegaConf读取配置）。训练/推理不要求评测Python存在；正式eval以及legacy含评测的smoke仍严格检查共享评测器。aligned `--dry-run`继续只打印命令。

```bash
# 路径覆盖示例；依旧是检查，不启动正式实验。
bash scripts/run_csgo_seen10.sh train \
  --experiment csgo_seen10_exp32gen_aligned \
  --data-root "/srv/datasets/csgo_benchmark_v2" \
  --eval-root "/srv/projects/csgo_benchmark_v2_eval_general" \
  --print-paths
bash scripts/run_csgo_seen10.sh train \
  --experiment csgo_seen10_exp32gen_aligned \
  --num-processes 4 --micro-batch-size 4 --gradient-accumulation-steps 8 \
  --dry-run
```

数据须完整迁移发布的manifest、selection、split、calibration、radars、images，保持相对布局及文件内容，不重新扫描拆split。路径就绪后执行第7节数据审计，不能用目录存在代替协议检查。

训练进程数仍为 `NUM_PROCESSES`（aligned另有 `--num-processes`）；只校验 world×micro×accumulation=128。正式seed42和checkout内独立aligned root不变。推理默认batch16、decode1，原有batch/缓存/fuse/OOM开关不变，不启用compile。

本次提供**新服务器从官方基座初始化、重新开独立实验**的能力，不自动迁移旧checkpoint合同。aligned恢复仍严格检查data/base绝对路径、代码和配置身份；拷贝checkpoint换盘后可能被拒绝，不能手改metadata/指纹绕过。预测目录也有身份校验，不续写其他机器/旧checkpoint的不完整输出。跨机器迁移精确resume需另行审计批准，跨GPU/库版本不承诺逐位一致。

### 4.4 本机历史环境与本次验收边界

2026-09-27此前GPU验收环境为Python3.13.11；项目 `.venv` 中Transformers4.51.3、Diffusers0.35.2、PEFT0.17.1；继承Miniconda的PyTorch2.12.0、Accelerate1.11.0、OmegaConf2.3.0，CUDA13、RTX PRO6000 Blackwell96GB。这是**混合site-packages历史环境**，不是fresh安装配方。新的只读检查会警告系统包继承及Python版本差异，但不替换该环境。原先 `venv --system-site-packages` 的本机准备方式不再推荐用于新服务器。

本次新机能力验证采用隔离mock/路径与命令回归、当前环境只读检查和已有资产hash检查；未安装新环境、未下载权重、未修改共享评测器、未执行新GPU smoke或正式训练/推理。具体测试数与当前缺失项见第8节新增记录；此前真实两步训练/恢复证据仍以历史 [验收报告](CSGO_ALIGNED_VALIDATION.md)为准。

## 5. 手动训练、恢复、推理和评测

以下为用户手动命令，文档整理不会执行它们。`all`会串行运行正式训练/转换/推理/评测，不是只读检查。

### 5.1 legacy：保留原命令

```bash
cd /home/jiahao/task/OmniGen2
bash scripts/run_csgo_seen10.sh train --seed 0
# 有完整checkpoint后恢复：
bash scripts/run_csgo_seen10.sh train --seed 0 --resume-from-checkpoint latest
bash scripts/run_csgo_seen10.sh convert --seed 0
bash scripts/run_csgo_seen10.sh infer --seed 0 --task all
bash scripts/run_csgo_seen10.sh eval --seed 0 --task all
```

转换固定读取 `seed_0/train/best`，写入 `seed_0/train/inference_adapter_best`；推理同时加载LoRA与numeric-pose adapter。两任务可分开使用 `--task discrete` / `--task continuous`。legacy默认steps28、bf16，可由 `NUM_INFERENCE_STEPS`、`INFERENCE_DTYPE`覆盖；这些环境变量不是aligned配方的覆盖入口。

### 5.2 aligned：显式选择独立实验

```bash
# 单卡已通过真实模型smoke的组合；默认不传micro/accum时为1×128。
NUM_PROCESSES=1 bash scripts/run_csgo_seen10.sh train \
  --experiment csgo_seen10_exp32gen_aligned --seed 42 \
  --micro-batch-size 2 --gradient-accumulation-steps 64

# 同一run、原执行配置恢复（有完整checkpoint后才使用）：
NUM_PROCESSES=1 bash scripts/run_csgo_seen10.sh train \
  --experiment csgo_seen10_exp32gen_aligned --seed 42 \
  --micro-batch-size 2 --gradient-accumulation-steps 64 \
  --resume-from-checkpoint latest

# 多卡首次启动的互斥示例；需实际有4张可见GPU，未做真实多GPU验收：
NUM_PROCESSES=4 bash scripts/run_csgo_seen10.sh train \
  --experiment csgo_seen10_exp32gen_aligned --seed 42 \
  --micro-batch-size 4 --gradient-accumulation-steps 8
```

可在aligned命令后加 `--dry-run`只打印实际命令、不写run目录。多个首次启动示例不能重复写入同一个run；有运行任务时不要再启动一个写同目录的进程。

```bash
# 主结果：训练结束的同一个late用于discrete和continuous。
bash scripts/run_csgo_seen10.sh convert \
  --experiment csgo_seen10_exp32gen_aligned --seed 42 --checkpoint late
bash scripts/run_csgo_seen10.sh infer \
  --experiment csgo_seen10_exp32gen_aligned --seed 42 --inference-seed 42 \
  --checkpoint late --task all --batch-size 16 --vae-decode-batch-size 1
bash scripts/run_csgo_seen10.sh eval \
  --experiment csgo_seen10_exp32gen_aligned --seed 42 --checkpoint late --task all

# 补充结果：独立best预测/评测目录。
bash scripts/run_csgo_seen10.sh convert \
  --experiment csgo_seen10_exp32gen_aligned --seed 42 --checkpoint best
bash scripts/run_csgo_seen10.sh infer \
  --experiment csgo_seen10_exp32gen_aligned --seed 42 --inference-seed 42 \
  --checkpoint best --task all --batch-size 16 --vae-decode-batch-size 1
bash scripts/run_csgo_seen10.sh eval \
  --experiment csgo_seen10_exp32gen_aligned --seed 42 --checkpoint best --task all
```

分任务时用 `--task discrete`或`--task continuous`，保持checkpoint和inference seed相同。`--seed`选择训练run，`--inference-seed`默认42；当前输出路径**不含inference seed子目录**，不能更改seed后复用同一目录。

### 推理加速设置

不启用 `torch.compile`。aligned固定原生FlowMatch Euler 28个timesteps、text CFG4/image CFG1、原生dynamic time shift、BF16及官方VAE；当前CFG每步两次Transformer前向，即56次/batch，Qwen/VAE另计。不用测试指标选择这些参数。

1. 第一优先级：batch denoising，默认请求batch16，CUDA OOM自动减半至最低1；VAE decode microbatch默认1。
2. 第二优先级：缓存每地图radar PIL、VAE posterior参数、negative-prompt embedding和RoPE；按每样本reference seed重新采样posterior，LoRA加载后一次性fuse。不做近似特征复用。

diffusion noise与reference posterior的随机流均按seed/task/sample identity派生，OOM重试重建generator。不同batch和浮点归约不保证图像逐位一致；pending manifest锁定请求的batch/decode batch等参数，恢复须用原启动参数。`--no-oom-fallback`、`--no-fuse-lora`可关闭相应路径，改变配置需新输出。

```bash
# 保留legacy环境变量用法：
INFERENCE_BATCH_SIZE=8 INFERENCE_DECODE_BATCH_SIZE=1 \
  bash scripts/run_csgo_seen10.sh infer --seed 0 --task all
# aligned请用CLI显式覆盖decode设置：
bash scripts/run_csgo_seen10.sh infer \
  --experiment csgo_seen10_exp32gen_aligned --seed 42 --checkpoint late \
  --task all --batch-size 4 --vae-decode-batch-size 1
```

当前仅batch2少量真实aligned推理有验收记录；未测正式checkpoint的batch16吞吐，不承诺ControlAR的约9小时速度适用于OmniGen2。

## 6. 输出、共享评测与隔离

aligned目录结构如下，阶段尚未执行时对应文件不会存在：

```text
outputs/csgo_seen10_exp32gen_aligned/OmniGen2/seed_42/
├── train/
│   ├── runtime-config-*.json
│   ├── parameter_audit*.json
│   ├── logs/train_metrics.jsonl
│   ├── checkpoint-{4000,8000,12000,16000,19500}/
│   │   ├── transformer_lora/{adapter_model.safetensors,adapter_config.json}
│   │   ├── optimizer.pt / scheduler.pt / rng-rank*.pt
│   │   └── aligned_state.json / COMPLETE
│   └── best / late / latest              # 相对symlink
├── adapters/{best,late}/
├── predictions/{best,late}/{discrete,continuous}/
│   ├── gen_imgs/<map>/<file_frame>.jpg
│   └── inference_manifest.json
└── evaluation/{best,late}/{discrete,continuous}/
```

legacy对应 `outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_<seed>/`，其 `train/`下为原生checkpoint、`best/late/latest`、`inference_adapter_best`及训练完成后的`loss_curve.png`；预测直接放在run下的`discrete/continuous`，评测在`evaluation/{discrete,continuous}`。

推理先核验checkpoint/protocol/prompt脚本身份和已有JPEG完整性；配置不符、哈希变化或未经记录的旧图片都拒绝混入。不把其他checkpoint输出复制到新run，也不续写历史不完整目录。相同身份的中断推理可用同命令恢复；完成manifest再次运行只校验并复用已有图。转换目标非空时拒绝覆盖。

唯一正式评测器是 `/home/jiahao/task/csgo_benchmark_v2_eval_general/run_eval.py`，采用其 `benchmark_v2.yaml`：

| 任务 | 指标 | 聚合与时序协议 |
| --- | --- | --- |
| discrete | PSNR↑、SSIM↑、LPIPS↓、Boundary_F1↑、FID↓ | 逐地图后equal-map macro |
| continuous | PSNR↑、SSIM↑、LPIPS↓、TWE↓、TDE↓、FVD↓ | equal-map macro；clip16、stride16、FVD224 |

共享配置固定frame-difference threshold2、min_track_len4、每地图20clips×64frames；Boundary_F1 edge quantile0.85、tolerance2pixels。正式评测要求完整覆盖，缺图/多图拒绝，已有正式输出不覆盖。OmniGen2的eval wrapper直接调用共享评测器；**没有ControlAR的独立评测前checkpoint-index preflight**，不要将两项目能力混写。

## 7. 检查与隔离 smoke

只打印命令、不启动实验的aligned检查：

```bash
bash scripts/run_csgo_seen10.sh train \
  --experiment csgo_seen10_exp32gen_aligned --dry-run
```

以下命令需用户安排资源后执行，本次文档整理不运行：

```bash
# aligned smoke action只运行pytest，不自动加载4B模型短训。
bash scripts/run_csgo_seen10.sh smoke --experiment csgo_seen10_exp32gen_aligned
# 全量metadata/文本长度、各地图图像变换和target禁读抽查；需本地Qwen tokenizer，输出须不存在。
.venv/bin/python scripts/audit_csgo_aligned.py \
  --output outputs/aligned_smoke/manual_audit/data_audit.json
```

回归测试含CPU/Gloo双进程测试，不等于多GPU实训；当前模块导入还会查询CUDA属性，不能保证在完全无CUDA环境运行整个测试集。

真实模型短训仍使用有效batch128；必须设置独立、尚未占用且包含路径组件`aligned_smoke`的root。下面完成1步，再新进程恢复到第2步；smoke每步保存验证，正式仍只在五个节点保存：

```bash
NUM_PROCESSES=1 bash scripts/run_csgo_seen10.sh train \
  --experiment csgo_seen10_exp32gen_aligned --seed 42 \
  --output-root outputs/aligned_smoke/manual_check/OmniGen2 \
  --micro-batch-size 2 --gradient-accumulation-steps 64 \
  --smoke --stop-after-updates 1 --max-validation-batches 2
NUM_PROCESSES=1 bash scripts/run_csgo_seen10.sh train \
  --experiment csgo_seen10_exp32gen_aligned --seed 42 \
  --output-root outputs/aligned_smoke/manual_check/OmniGen2 \
  --micro-batch-size 2 --gradient-accumulation-steps 64 \
  --smoke --stop-after-updates 2 --max-validation-batches 2 --resume-from-checkpoint latest
bash scripts/run_csgo_seen10.sh convert \
  --experiment csgo_seen10_exp32gen_aligned --seed 42 \
  --output-root outputs/aligned_smoke/manual_check/OmniGen2 --checkpoint latest --smoke
bash scripts/run_csgo_seen10.sh infer \
  --experiment csgo_seen10_exp32gen_aligned --seed 42 \
  --output-root outputs/aligned_smoke/manual_check/OmniGen2 --checkpoint latest \
  --task all --batch-size 2 --max-samples 2
bash scripts/run_csgo_seen10.sh eval \
  --experiment csgo_seen10_exp32gen_aligned --seed 42 \
  --output-root outputs/aligned_smoke/manual_check/OmniGen2 --checkpoint latest \
  --task all --smoke --max-samples 2
```

重复smoke要换新目录；不要复用本机已验收的artifact目录。连续少于64帧时wrapper使用共享evaluator的frame-only smoke，只验证paired metrics，不宣称TWE/TDE/FVD通过。恢复加载成功也不自动等于逐位恢复；严格对照需从同一个checkpoint分叉，方法和报告见验收文档。

legacy `bash scripts/run_csgo_seen10.sh smoke --seed 0`另有历史语义：运行回归、真实数据+tiny模型forward/backward/严格回读/一张JPEG，再调用共享discrete smoke。已有`seed_0/smoke`时不覆盖。它不是aligned真实4B模型验收。

## 8. 已有结果、速度与执行状态

以下为 **2026-09-27 04:15 HKT（2026-09-26 20:15 UTC）文档整理时的只读快照**，不是持续监控，也不代表本次启动了任务：

| 项目 | 已核对状态与证据 |
| --- | --- |
| 当前OmniGen2进程 | 未发现训练/推理/评测进程；tmux `omnigen2`只有空闲bash终端 |
| legacy seed0训练 | `outputs/csgo_benchmark_v2_seen10/OmniGen2/seed_0/train/logs/20260921-145015.log`在Prepare dataset阶段报OmegaConf scalar错误；无checkpoint和训练metrics，不填完成step |
| legacy正式预测/指标 | 未发现完整预测或正式evaluation结果，不能填benchmark表格 |
| aligned seed42正式run | `outputs/csgo_seen10_exp32gen_aligned/OmniGen2`尚不存在，无正式训练/推理/评测结果 |
| aligned实现验收 | `outputs/aligned_smoke/20260927_acceptance/`有独立短训、恢复、预测与smoke评测；不属于正式run |

共享GPU上现有OpenPI/ControlAR任务不属于OmniGen2，本次未停止或改动。旧日志记录历史失败，不代表当前代码仍存在同一错误；此前修复及短训证据应分别阅读。

aligned验收摘要（详见 [CSGO_ALIGNED_VALIDATION.md](CSGO_ALIGNED_VALIDATION.md)）：

- 69项测试、8项子测试通过；四split计数、10地图、200×64连续协议、224/448及文本长度检查通过。target禁读为代码检查与抽样文件打开拦截，不是读取全部测试FPV。
- 单卡micro2×累计64，完整有效batch128，完成2个optimizer updates；验证仅2条。152个LoRA目标、5,107,200个trainable与optimizer成员严格一致。
- `deterministic_continuous`与`deterministic_resumed`的checkpoint2中304个LoRA张量、AdamW、scheduler、RNG及进度metadata逐位相同，见`deterministic_resume_comparison.json`。早期原生自动选核未通过的对照保留，不将其与最终通过结果混淆。
- 最终同一checkpoint离散2张、连续2张，均448×448 RGB；重复推理只校验并复用已有输出。共享discrete smoke验证4个paired/boundary指标，continuous只验证单帧PSNR/SSIM/LPIPS，结果明确 `smoke_only=true`。
- 未实测多GPU真实训练、完整5,000条validation、完整连续时序指标、FID/FVD或正式全量生成。不把CPU DDP、两步训练或单帧评测写成这些项目已通过。

两步确定性短训合计203秒，约100秒/update；机械外推19,500步约23天，另加完整验证，但共卡负载、预热及batch会改变吞吐，**不是正式ETA或长期稳定性结论**。更大micro、多GPU及batch16推理速度均未实测。完整adapter-only checkpoint约59MiB，五个约295MiB；官方base通过snapshot复用，图片建议额外预留数GB。

历史`RUN_FULL=0`/`RUN_FORMAL=0`表示对应接入轮次的授权边界，不是脚本永久禁止正式执行。正式实验由用户手动启动；新状态应依据实际日志/metadata更新本节，不覆写历史验收报告。

### 8.1 新服务器初始化增补验收：2026-09-27 05:12 HKT

此次仅实现环境/资产/路径能力并验证，不是另一次正式实验：

| 检查 | 实际结果 |
| --- | --- |
| `.venv/bin/python -m pytest -q tests` | 最终93 passed、14 subtests passed，13.26秒；20项已有Torch弃用/可选FlashAttention警告，无失败 |
| `bash scripts/setup_csgo_seen10.sh --check --env-only` | 退出0；当前混合环境CPU导入与Qwen类检查通过，CUDA未初始化；报告继承base、Python3.13及可选cv2/wandb缺失，不安装补齐 |
| `python scripts/download_csgo_seen10_assets.py --check --profile aligned` | 退出1且正确报告未齐备：19项中18项size/hash通过；缺Qwen `generation_config.json`（216字节），未自动下载 |
| legacy/aligned `--print-paths` | 数据、训练Python、共享评测器与其venv解析正确；无模型加载/输出目录写入 |
| aligned 4卡×micro4×accum8 `--dry-run` | 正确生成DDP启动命令，有效batch128；未启动GPU作业 |
| 兼容/隔离 | 两份CSGO配置、原requirements及严格训练源码指纹覆盖文件与本次修改前一致；mock验证缺失/损坏资产拒绝、现有环境保护、含空格/非cwd/显式坏路径和配置路径优先级；`bash -n`、`git diff --check`通过 |

Qwen缺失项属于本次固定完整清单中的生成配置元数据，不代表18项已通过hash的文件损坏，也不推翻此前GPU smoke记录。用户可按第4节下载命令手动补齐，再运行`--check`。未对legacy/all全部大权重另做完整hash验收，不把aligned结果泛化为legacy资产就绪。

未实际执行新环境pip/Conda安装、远端服务器GPU检查、真实模型短训/推理或全量评测；没有停止已有ControlAR/OpenPI任务、改写checkpoint、覆盖预测或创建正式aligned run。目标机器仍须完成第4节安装/CUDA检查及第7节独立smoke，之后由用户决定正式启动。
