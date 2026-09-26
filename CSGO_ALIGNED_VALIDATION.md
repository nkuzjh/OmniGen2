# aligned 实施验收记录

记录日期：2026-09-27（Asia/Hong_Kong）。对应实施轮次仅进行实现和隔离smoke，未启动正式训练或全量推理。
本文件保留当时实施与验收范围，不作为持续运行状态。当前运行说明与手动命令见
[CSGO_SEEN10.md](CSGO_SEEN10.md)，设计依据与验收标准见
[CSGO_SEEN10_PLAN.md](CSGO_SEEN10_PLAN.md)。

## 隔离边界

- 新profile为`csgo_seen10_exp32gen_aligned`，只有显式`--experiment`才进入新入口。
- 旧配置及旧实验目录未覆盖；未停止其他训练/推理任务。
- 全部本次运行产物位于`outputs/aligned_smoke/20260927_acceptance/`，不是正式结果。
- 使用本地缓存的官方基础模型；仅补齐少量processor/scheduler元数据，未下载大权重。
- 共享评测器及其指标实现未修改。短训练产物不得用于报告模型性能。

## 数据与参数实测

完整协议报告：`outputs/aligned_smoke/20260927_acceptance/data_audit.json`。

| 项目 | 实测 |
|---|---|
| train / validation / discrete / continuous | 50000 / 5000 / 20000 / 12800 |
| 地图与连续clip | 固定Seen-10；200 clips × 64 frames |
| radar / FPV | 224×224 / 448×448 |
| 最长文本token数（同上四split） | 136 / 135 / 136 / 135，均小于888 |
| 推理target隔离 | `load_target=False`；抽样文件打开拦截检查target读取0次 |
| 可训练 | 152个attention目标层，304个LoRA张量，共5,107,200参数 |
| 冻结生成Transformer基座 | 3,967,161,400参数 |
| 冻结Qwen / VAE | 3,085,938,688 / 83,819,683参数 |
| trainable占比 | Transformer约0.1286%；含Qwen/VAE约0.07151% |
| optimizer审计 | 参数集合与全部trainable LoRA集合完全一致；无frozen参数 |

全部87,800条记录核对split身份及文本长度；图像变换和target禁读是各split/地图抽样检查，
不是打开全部目标图。完整逐参数名称、shape、numel和optimizer成员存于每个train目录的
`parameter_audit.json`。协议哈希：

```text
benchmark_manifest.json
4debad27e0d481d31587a537325d6781247934551a7248885e686ed94046ba47
calibration/z_calibration.json
67436a888e0f79520bf1ab156009f7384b3a1ade7638b44513f31b2a9b6c14ee
```

基础snapshot：OmniGen2 `df5dca8a981d74e6c3af214c145f5c735fe72367`；
Qwen `66285546d2b821cf421d4f5eb2576359d3770cd3`；
FLUX `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`。

## 自动测试与兼容检查

运行命令：

```bash
OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python -m pytest -q tests
bash -n scripts/run_csgo_seen10.sh
git diff --check
```

最终回归：**69 passed，8 subtests passed**，15.50秒；20条已有依赖弃用/可选flash-attn警告。
原始输出：`outputs/aligned_smoke/20260927_acceptance/tests_final.log`。
回归覆盖配置、batch乘积、milestone与别名、sampler/worker/恢复、数据隔离、LoRA覆盖、
adapter转换、输出hash和旧入口。两个真实CPU/Gloo进程验证2×4×16=128的累计梯度、
AdamW更新，以及最后一个microbatch不使用reference分支的情况；这不是多GPU模型训练实测。
CPU tiny-model恢复有逐位检查；真实模型恢复记录见下一节。

当前环境Diffusers 0.35.2的LoRA私有接口返回`(state_dict, metadata)`，已兼容旧版字典返回；
真实safetensors加载、融合、卸载测试通过。离线加载显式指定权重文件名。

## 真实GPU训练与恢复

设备：单张RTX PRO 6000 Blackwell 96GB，与既有其他任务共用；micro=2、accumulation=64，
每个update真实使用128条源样本。验证仅取2条，不能称为完整5,000条验证。

首次保留原生自动选核的同checkpoint对照（`frozen_continuous` vs `frozen_resumed`）：
起点checkpoint-1文件相同，结束时RNG、scheduler、sampler和步数相同，但152个LoRA-B张量
不逐位相同，最大绝对差约2.56e-9。失败证据保留在`frozen_resume_comparison.json`，
没有将该轮标记为精确恢复通过。

最终仅为aligned固定PyTorch/cuDNN/cuBLAS确定性设置，以及原生Triton RMSNorm的4-warps
配置后，重新从官方基座运行两步，再复制同一个checkpoint-1到独立目录恢复step2。
未改变模型/loss/优化配方或原生SDPA接口。

对照目录为`deterministic_continuous`与`deterministic_resumed`，结果：

| 比较项 | 结果 |
|---|---|
| 304个LoRA张量 | 全部逐位相等，最大绝对差0 |
| AdamW完整optimizer状态 | 逐位相等 |
| scheduler、Python/NumPy/CPU/CUDA RNG | 全部相等 |
| step、sampler epoch/cursor、配置指纹、参数审计指纹 | 全部相等 |
| step2 train loss / grad norm | 0.39517821128271063 / 0.008229636587202549，两路相同 |
| validation loss（2条） | 0.32438122496312977，两路相同 |
| 总结 | `bitwise_equal=true` |

报告：`outputs/aligned_smoke/20260927_acceptance/deterministic_resume_comparison.json`。
smoke的best指向checkpoint-1（同loss保留更早者），latest指向checkpoint-2；没有late，
因为late只能代表正式19,500步结束。这次逐位恢复通过仅限已测相同软硬件和batch拓扑；
不等价于跨机器/多GPU/改变micro或accumulation后仍逐位一致。

## 推理与共享评测

最终重新完成`deterministic_continuous/OmniGen2/seed_42/`下checkpoint-2的转换，
使用同一冻结adapter运行两类推理（早期`frozen_resumed`的独立smoke证据也保留）：

- 离散2张、连续2张，batch=2、28步、bf16、seed42、text CFG4/image CFG1。
- 四张均为448×448 RGB JPEG；连续frame identity保持原始顺序，互不构成时序条件。
- 重复运行相同推理命令，两个任务都输出“all 2 verified outputs already exist”，
  校验图像hash后复用，无重新生成/覆盖。
- 共享评测器discrete smoke运行PSNR、SSIM、LPIPS、Boundary_F1（2张）；
  continuous frame-only smoke运行PSNR、SSIM、LPIPS（1帧）。
- 最终报告分别为`eval_discrete_final_smoke.json`、`eval_continuous_final_smoke.json`，均声明
  `smoke_only=true`、`formal=false`、`official_output_written=false`。

**未实测**：多GPU真实模型训练、完整5,000条validation、完整64帧时序评测、FID/FVD，
以及19,500步训练和全量推理。少量smoke不能证明训练收敛或生成质量。

## 资源预估

固定数值执行设置后的单卡micro2两步短训用时203秒，约100秒/update，
仅按此短样本外推19,500步约23天，另加完整验证；
共卡负载、预热与正式batch都会影响该估计，不能当作稳定吞吐基准。
多卡速度和更大microbatch未实测，不推断线性加速。
adapter-only完整checkpoint约59MiB，五个约295MiB；官方基础权重通过snapshot引用复用。
JPEG存储随内容变化，正式32,800张建议预留数GB以上空间；本次smoke单独保留，不混入正式结果。
