# OmniGen2 Seen-10：设计决策与 aligned 实现

本文记录设计依据、配方来源、模块职能、实现边界与验收标准；实际运行命令、环境、输出和带日期状态统一维护在 [CSGO_SEEN10.md](CSGO_SEEN10.md)。详细的2026-09-27实施验收保存在 [CSGO_ALIGNED_VALIDATION.md](CSGO_ALIGNED_VALIDATION.md)，旧 [CSGO_SEEN10_ALIGNED.md](CSGO_SEEN10_ALIGNED.md)仅保留跳转。

参考ControlAR的**文档分工和证据记录方式**；后续新增环境/资产/路径初始化也参考其分工，但不复制Canny/DINO/VQ/GPT结构、数值pose注入、LoRA r32方案或compiled batch16。以下aligned已经实现并完成限定范围smoke；正式训练状态以主文档的日期化快照为准。

## 1. 决策来源与比较边界

最终采用的方案名为 `csgo_seen10_exp32gen_aligned`。它是generation-only基线，主对照UniLIP `exp32_gen`；joint `exp32`是次要对照。数据、当前样本条件、图像尺寸、有效batch、曝光预算和评测规则对齐，不把跨架构参数量或采样FLOPs机械做成一致。

| 设置 | 来源与最终决定 |
| --- | --- |
| LoRA r8 / alpha8 / dropout0、attention-only | 官方 `options/ft_lora.yml` 的rank/dropout与 `train.py` 的alpha/target构造；不是沿用exp32_gen的r32 |
| LR8e-7、AdamW(.9,.95)、weight decay.01、500-step warmup后constant | OmniGen2官方微调配置与原生scheduler；作为模型差异披露，不按batch放大 |
| Qwen/VAE/主干基座冻结，只有attention LoRA训练 | 官方LoRA路径；不另外给Qwen或连接器加入exp32_gen式LoRA |
| radar224 / FPV448 | 用户最终修订；对齐exp32_gen，覆盖前一版“保留官方尺寸预算”的方案 |
| 只用文本pose | 用户批准去掉aligned新增数值pose MLP；旧legacy保留双路条件 |
| 无随机图像增强，保留原生归一化与CFG条件dropout | 官方预处理原则与生成标签一致性；不对radar/pose或FPV执行不一致的几何增强 |
| effective batch128、19500 updates、2496000曝光 | exp32_gen样本预算；实际world × micro × accumulation只校验乘积 |
| 4000/8000/12000/16000/19500完整验证和保存 | 用户最终修订，覆盖早期3900间隔方案 |
| PyTorch/cuDNN/cuBLAS确定性设置、原生RMSNorm固定4-warps | 实施中为精确resume加入的工程设置；不是官方CSGO训练推荐，不改变loss/模型数学 |
| 原生flow训练与Euler28步/CFG4/1推理 | OmniGen2生成机制；不替换成AR loss或未经审核的20-NFE配置 |

“官方配方”表示本地官方配置提供的微调起点，不表示已在CSGO上验证最优或长期稳定。这里只测试接线、预算、恢复和产物合同；不依据少量测试图的指标选择超参数。

### 1.1 与官方微调、exp32_gen的横向比较

| 项目 | OmniGen2官方微调 | UniLIP exp32_gen | 已实现OmniGen2 aligned |
| --- | --- | --- | --- |
| 任务 | 通用生成/编辑 | CSGO generation-only | CSGO generation-only |
| 输入图 / 输出图 | reference像素预算按图序为1024²/1024²/768²/512²，输出最多1024²，最长边2048；不是固定方图 | radar224 / FPV448 | radar224 / FPV448 |
| 条件理解语言模块 | 冻结Qwen | 活跃LLM LoRA r32/alpha64/dropout.05 | 冻结Qwen；地图名+pose文本+固定指令 |
| 主要生成网络 | attention LoRA r8/alpha8/dropout0 | 生成expert attention+MLP LoRA r32/alpha64/dropout.05 | 官方attention LoRA，152个目标层 |
| connector / 小型生成模块 | LoRA以外基座冻结 | generation connector LoRA r16；部分projector/query全量 | 原生文本/图像投影冻结，不新增数值adapter |
| 图像tokenizer | 冻结FLUX VAE | 冻结VAE | 冻结FLUX VAE |
| 训练目标 | 原生flow matching | 原生flow matching | 原生linear velocity flow |
| LR / AdamW betas / weight decay | 8e-7 / (.9,.95) / .01 | 1e-4 / (.9,.999) / 0 | 同官方 |
| scheduler | warmup500，从1e-18到8e-7，后constant | warmup ratio.003，cosine至1e-5 | 同官方，按optimizer update推进 |
| 有效batch | 配置global16；micro2，实际仍取决于启动world | 128 | 128，默认1×1×128，支持其他合法组合 |
| updates / 曝光 | 配置4000，曝光依实际启动 | 19500 / 2496000 | 19500 / 2496000 |
| 验证与选择 | 官方配置validation500、checkpoint1000，无此benchmark选点规则 | 原训练eval_strategy=no，用final | 五次完整5000验证；late主结果、best补充 |
| 正式结果状态 | 不是本项目CSGO结果 | 已有历史final模型 | 本地状态和证据见主文档第8节 |

官方证据：[ft_lora.yml](options/ft_lora.yml)、[train.py](train.py)的LoRA构造、[StepLRScheduler](omnigen2/optim/scheduler/step_lr.py)。UniLIP参考证据为 `/home/jiahao/task/UniLIP/csgo_configs/exp32_gen.yaml`、`record.md`、实际训练代码和 `outputs/csgo_1b/exp32_gen` 的trainer/adapter元数据；不能只凭配置注释推断完成updates。OmniGen2当前合同以 [aligned配置](options/csgo_seen10_exp32gen_aligned.yml)、[实际训练入口](omnigen2/aligned_training.py) 和run内metadata为准。

## 2. 模块职能与条件流

OmniGen2的Qwen与生成Transformer是两个不同职责的模块：前者把条件文本编码为特征，后者结合reference latent和噪声latent预测生成更新。不能因为二者都是Transformer就把统一主干重复当成LLM和DiT注入两套LoRA。

```text
地图名 + 归一化pose文本 + 固定指令
  → 冻结Qwen → caption_embedder → context_refiner ┐
                                                │
radar224 → 冻结VAE encoder → ref_image_patch_embedder
  → ref_image_refiner ────────────────────────────┼→ layers.0..31
                                                │   → norm_out（含输出投影）
FPV448 → 冻结VAE encoder → 原生flow加噪             │   → velocity预测
  → x_embedder → noise_refiner ──────────────────┘

推理：FPV分支从随机latent开始，不读取目标FPV；Euler更新latent → 冻结VAE decoder → RGB448
```

图中caption/patch embedding是投影层，三个refiner是含attention/MLP的原生生成模块，不是一个独立的“Qwen视觉塔→LLM视觉连接器”。radar不走Qwen视觉塔。aligned保留这些原生投影，但冻结它们，只在refiner的attention注入官方LoRA。

以下前缀相对生成Transformer实例；Qwen和VAE为独立实例，不属于同一参数名前缀空间：

| UniLIP参考职能 | OmniGen2真实模块/前缀 | aligned状态 | 可训练参数 |
| --- | --- | --- | ---: |
| 条件视觉encoder | radar使用独立 `vae.encoder.*` / `vae.quant_conv.*` | 全冻结；Qwen视觉塔不在此路径 | 0 |
| 活跃条件LLM | 独立Qwen文本模型 | 冻结，保留文本特征计算；这是官方配方差异 | 0 |
| 条件进入生成网络的投影 | `time_caption_embed.caption_embedder.*`、`ref_image_patch_embedder.*` | 冻结，不强行改成r16 connector LoRA | 0 |
| 条件文本refinement | `context_refiner.{0,1}.attn.*` | attention LoRA r8/alpha8/dropout0；其余冻结 | 268,800 |
| 条件图像refinement | `ref_image_refiner.{0,1}.attn.*` | 同上 | 268,800 |
| 生成噪声refinement | `noise_refiner.{0,1}.attn.*` | 同上 | 268,800 |
| 主要生成expert | `layers.{0..31}.attn.*` | 同上；不训练MLP基座 | 4,300,800 |
| 小型生成投影/嵌入 | `x_embedder.*`、`time_caption_embed.timestep_embedder.*`、`image_index_embedding`、`norm_out.*` | 冻结，不新增query/adapter | 0 |
| 图像tokenizer/decoder | 独立VAE的其余参数 | 全冻结 | 0 |
| 新数值pose模块 | `pose_adapter` | 不创建，无额外pose token | 0 |

### 2.1 精确LoRA目标与审计

对以下每个block只注入一次，A随机初始化、B零初始化、alpha/r=1：

```text
layers.{0..31}.attn.{to_q,to_k,to_v,to_out.0}
context_refiner.{0,1}.attn.{to_q,to_k,to_v,to_out.0}
ref_image_refiner.{0,1}.attn.{to_q,to_k,to_v,to_out.0}
noise_refiner.{0,1}.attn.{to_q,to_k,to_v,to_out.0}
```

q/out为2520→2520，k/v为2520→840；每block的LoRA参数为
`8 × [(2520+2520)+(2520+840)+(2520+840)+(2520+2520)] = 134400`。
38个block合计5,107,200参数、152个线性层、304个A/B张量。所有LoRA属于唯一8e-7参数组，weight decay0.01；没有full-train小模块。

实际审计：生成Transformer总参数3,972,268,600，冻结3,967,161,400；Qwen冻结3,085,938,688，VAE冻结83,819,683。trainable占Transformer约0.1286%，占上述全部模型约0.07151%。分母包括冻结组件，不能只按LoRA或当前激活token统计。逐参数 `requires_grad`、shape、numel和optimizer成员检查由 `_parameter_audit`执行并落盘，不以日志中的单一总数代替。

## 3. 预算、恢复与实验隔离的设计

预算是128个**真实源样本**/update ×19500；尾部不足128的数据不形成一次update，不把CFG分支、reference或latent token额外计数。sampler每epoch打乱50000条，使用49920条（390updates），50轮达成2496000曝光/49.92等价完整epoch。实际world × micro × accumulation只校验乘积；源码中任何固定micro/accum相等检查都不应作为额外实验约束。

累计loss按固定448 latent元素数做全局归一化，并考虑DDP梯度平均；scheduler只按optimizer update推进。原生flow训练不改成跨模型统一loss。多卡验证按rank分配不重复的样本，以全局numerator/denominator聚合，避免padding重复或平均各rank均值的偏差。

只有4000/8000/12000/16000/19500是正式完整验证和保存节点；best在这五个点内部选择，late是final。不同模型loss不能直接横比。UniLIP历史结果使用final，因此外部模型best只作补充，不称为严格相同选点规则。

checkpoint保存发生在累计边界，记录完整可训练状态、optimizer/scheduler/RNG、sampler cursor及base/data/code身份。冻结基座通过官方snapshot重建，不从旧CSGO微调模型初始化。恢复允许合法batch拓扑改变，但那不保证同随机轨迹；要求逐位续跑时保持执行拓扑和环境。

早期同checkpoint恢复在原生自动选核下，RNG和scheduler相同但参数更新不逐位相同。aligned固定数值执行策略后，两步真实GPU对照通过。该策略不替换原生算子数学，不要求其他架构或旧legacy采用相同后端，也不承诺跨机器逐位恢复。完整失败/通过证据均留在验收报告，不删除失败记录以只展示成功。

新旧root、checkpoint、adapter转换、prediction、manifest、evaluation及日志分离。正式root/seed固定为已批准实验，smoke只在含 `aligned_smoke` 的独立root；不停止已有任务，不续写其他实验未完成的推理目录。输出完整性与同身份断点续跑的区别、当前wrapper真实参数见主文档。

## 4. 已实施文件分工

| 文件 | 职责 |
| --- | --- |
| `options/csgo_seen10_lora.yml` | legacy配方保留，不改变含义 |
| `options/csgo_seen10_exp32gen_aligned.yml` | 新独立配方、尺寸、预算和milestones |
| `scripts/run_csgo_seen10.sh` / `scripts/run_csgo_aligned.py` | 显式experiment路由；legacy旧命令不变；aligned train/convert/infer/eval/smoke/dry-run |
| `train_seen10.py` / `train.py` | 配置构建、旧训练入口及aligned分发；启动早期设置aligned cuBLAS环境 |
| `omnigen2/aligned_training.py` | 独立预算/累计/DDP、官方LoRA、逐参数审计、完整验证、checkpoint与精确恢复 |
| `omnigen2/dataset/csgo_seen10_dataset.py` | 同协议读取、224/448可选尺寸、文本pose、target隔离及禁止静默截断；legacy默认保留 |
| `convert_ckpt_to_hf_format.py` | aligned adapter-only转换与身份检查，不导出数值pose；legacy sidecar保留 |
| `infer_seen10.py` | 两profile隔离、官方base验证、原生采样、逐样本seed、cache/fuse/OOM回退、manifest与图像完整性 |
| `omnigen2/pipelines/lora_pipeline.py` | 兼容Diffusers旧字典与0.35.2 tuple返回值，保持LoRA加载/融合接口 |
| `scripts/audit_csgo_aligned.py` | 全量split/token长度与按地图图像/target隔离抽查报告 |
| `scripts/compare_aligned_checkpoints.py` | 比较两个受信smoke checkpoint的LoRA、optimizer、scheduler、RNG、进度与validation metadata |
| `tests/test_aligned_{training,ddp,launcher,conversion}.py`及相关dataset/inference/loader测试 | 新旧边界与关键数值语义回归 |
| `scripts/setup_csgo_seen10.sh` / `scripts/check_csgo_environment.py` / `requirements-csgo-seen10.txt` | 新服务器隔离环境安装、已有环境只读保护、CPU导入与显式CUDA小检查 |
| `scripts/download_csgo_seen10_assets.py` / `scripts/csgo_seen10_assets.json` | 固定官方revision、按profile获取组件、离线size/hash检查、HF snapshot环境变量输出 |
| `csgo_runtime_paths.py`及`tests/test_csgo_{runtime_paths,environment,assets}.py` | 服务器路径/共享评测环境选择、无副作用检查与隔离回归 |

未修改UniLIP、ControlAR或共享指标实现。portable路径解析和自动环境引导现已实现，具体接口见主文档第4节；ControlAR的compiled采样和checkpoint-index preflight仍未引入。

## 5. 验收标准、已测范围与后续记录规则

1. 配置/旧命令兼容，正式与smoke root明确隔离；只能显式选择新experiment。
2. 全部split/地图/样本数、selection/calibration身份符合协议；reference224、FPV448、无不一致随机增强；推理target不得被打开为输入。
3. 152个LoRA目标完整覆盖，304个张量/5,107,200参数；Qwen/VAE/基座冻结，optimizer成员逐参数对账。
4. 合法world/micro/accum组合只按有效batch128校验；每update真实样本128，scheduler/global step只在optimizer update后推进；拒绝尾部短batch更新。
5. 五个完整验证节点及best/late规则正确；在独立smoke中从同一checkpoint分叉验证状态和后续更新一致，不能只验证文件能加载。
6. 同一checkpoint的离散/连续少量生成；448 RGB JPEG、clip/frame identity、逐样本随机流和已有输出完整性；精确说明batch/数值后端的一致性范围。
7. 只调用共享evaluator，明确frame-only与完整时序、distribution指标的差别；不把smoke指标报告为正式benchmark结果。

2026-09-27已执行的验收包括69项测试、8项子测试、单卡micro2×累计64的两步真实训练/逐位恢复、两任务各2张推理及共享evaluator frame smoke。完整原始产物在 `outputs/aligned_smoke/20260927_acceptance/`，详见 [验收报告](CSGO_ALIGNED_VALIDATION.md)。两进程CPU/Gloo检查不等于多GPU实训；完整5000验证、时序/FID/FVD、长期收敛、batch16吞吐均未通过这次小测得到验证。

上一次文档整理仅合并职责、核对现状与修正说明；后续新服务器初始化变更见下节，不倒写为历史GPU验收。新状态在主文档维护，历史报告保留当时范围；正式实验仍由用户手动启动。

## 6. 新服务器初始化增补（2026-09-27）

日常操作简化为主文档第4节的两组命令：环境准备、权重下载并设置当前终端模型路径。FLUX首次网页授权和服务器登录作为下载前提单独简述，已完成时跳过，不增加日常检查命令。脚本内部的必要校验不变，下面保留设计与验收记录，不作为额外的逐项操作清单。

- 实验语义不变：两份CSGO YAML、训练/模型/数据核心、19500步和有效batch128、LoRA/LR、224/448、milestones与指标实现不改。仅入口解析服务器路径，保留旧命令。
- 环境与模型资产分开准备；Linux fresh采用Python3.11/3.12与PyTorch2.7.1/torchvision0.22.1 cu128，直接依赖独立固定。已有可用环境只读保留；不同环境不宣称逐位等价，新机实际GPU运行尚待验收。
- 官方资产按不可变revision及字节hash审计；aligned不下载Omni bundled语言模型或FLUX完整生成模型，legacy/all补齐原生pipeline。通过显式三个snapshot路径固定加载，既不移动旧权重，也不改共享HF refs/main。
- 数据按CLI/env/config/兼容默认解析；评测优先所选共享评测器自己的venv，显式CLI仍最高优先级，不把训练venv用于指标兜底。只读print-paths可在缺依赖机器运行，正式执行按动作检查必需路径。
- 不自动重写checkpoint/provenance。严格源码指纹覆盖的核心文件保持不变；数据/base路径变化仍可能使旧resume被拒绝。目标是新机从官方权重开新实验，不是静默解除旧实验恢复合同。
- 验收包括路径优先级/含空格/无评测环境训练/非cwd、旧命令、多卡命令规划、mock安装幂等/现有目录保护、离线缺失/损坏资产拒绝及本机只读检查。真实fresh安装、目标GPU smoke、正式实验均由用户手动执行；这次不下载大权重、不抢占现有任务。
