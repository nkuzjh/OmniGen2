"""Exact, adapter-only training path for csgo_seen10_exp32gen_aligned.

The legacy trainer deliberately does not share its optimizer, sampler or
checkpoint semantics with this profile.
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
import numbers
import os
import random
import shutil
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer, Qwen2_5_VLModel

from omnigen2.dataset.csgo_seen10_dataset import CSGOSeen10Collator, CSGOSeen10Dataset, SEEN_MAPS
from omnigen2.models.transformers.repo import OmniGen2RotaryPosEmbed
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.transport import create_transport


PROFILE = "csgo_seen10_exp32gen_aligned"
MILESTONES = (4000, 8000, 12000, 16000, 19500)
LORA_SUFFIXES = ("to_q", "to_k", "to_v", "to_out.0")
GLOBAL_BATCH = 128
TOTAL_UPDATES = 19500
EXECUTION_POLICY = {
    "cublas_workspace_config": ":4096:8",
    "torch_deterministic_algorithms": "error",
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "tf32": False,
    "triton_rmsnorm_num_warps": 4,
    "triton_rmsnorm_autotune_configs": 1,
    "sdpa": "native",
}
OFFICIAL_BASES = {
    "pretrained_model_path": "OmniGen2/OmniGen2",
    "pretrained_vae_model_name_or_path": "black-forest-labs/FLUX.1-dev",
    "pretrained_text_encoder_model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",
}


def _configure_execution_policy():
    """Select deterministic native kernels for this profile before tensor work."""
    from omnigen2.ops.triton import layer_norm

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = EXECUTION_POLICY["cublas_workspace_config"]
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for name in ("_layer_norm_fwd_1pass_kernel", "_layer_norm_bwd_kernel"):
        kernel = getattr(layer_norm, name)
        selected = [config for config in kernel.configs if config.num_warps == 4]
        if len(selected) != 1:
            raise RuntimeError(f"Expected one native num_warps=4 Triton config for {name}")
        kernel.configs = selected
        kernel.cache.clear()
    return dict(EXECUTION_POLICY)


def _official_source(value: str, model_id: str) -> bool:
    if value == model_id:
        return True
    path = Path(value).expanduser().resolve()
    parts = path.parts
    cache_name = "models--" + model_id.replace("/", "--")
    return path.is_dir() and cache_name in parts and "snapshots" in parts and not any(
        part.startswith("checkpoint-") for part in parts
    )


def validate_config(args, world_size: int) -> tuple[int, int, int]:
    """Return microbatch, accumulation and records per rank per update."""
    if args.get("experiment") != PROFILE or args.data.dataset_type != "csgo_seen10":
        raise ValueError("Aligned trainer requires the aligned Seen-10 profile")
    train = args.train
    for field in ("batch_size", "gradient_accumulation_steps", "global_batch_size",
                  "max_optimizer_steps", "max_train_steps"):
        value = train[field]
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"train.{field} must be an integer")
    micro, accumulation = int(train.batch_size), int(train.gradient_accumulation_steps)
    if min(micro, accumulation, world_size) <= 0:
        raise ValueError("Batch factors and world size must be positive")
    if micro * accumulation * world_size != GLOBAL_BATCH:
        raise ValueError(
            f"Effective batch must be exactly 128 source records; got "
            f"{micro} * {accumulation} * {world_size}"
        )
    if int(train.get("global_batch_size", GLOBAL_BATCH)) != GLOBAL_BATCH:
        raise ValueError("train.global_batch_size must be 128")
    if int(train.max_optimizer_steps) != TOTAL_UPDATES or int(train.max_train_steps) != TOTAL_UPDATES:
        raise ValueError("Aligned optimizer budget must be 19500")
    if tuple(int(x) for x in train.checkpoint_steps) != MILESTONES:
        raise ValueError("Aligned checkpoint steps differ from the approved milestones")
    checks = {
        "learning_rate": 8e-7, "adam_beta1": .9, "adam_beta2": .95,
        "adam_weight_decay": .01, "adam_epsilon": 1e-8,
        "lora_rank": 8, "lora_alpha": 8, "lora_dropout": 0, "ema_decay": 0,
        "warmup_t": 500, "warmup_lr_init": 1e-18, "max_grad_norm": 1,
    }
    for key, expected in checks.items():
        if not math.isclose(float(train[key]), expected, rel_tol=1e-8, abs_tol=1e-20):
            raise ValueError(f"train.{key} must be {expected}")
    if train.get("mixed_precision") != "bf16" or not train.get("gradient_checkpointing"):
        raise ValueError("Aligned training requires bf16 and gradient checkpointing")
    if train.get("allow_tf32") or train.get("scale_lr") or train.get("use_8bit_adam"):
        raise ValueError("TF32, LR scaling and 8-bit Adam are disabled for aligned training")
    if not train.get("lora_ft"):
        raise ValueError("Aligned training requires LoRA")
    if train.get("lr_scheduler") != "timm_constant_with_warmup" or not train.get("warmup_prefix") or train.get("t_in_epochs"):
        raise ValueError("Aligned scheduler must use update-based timm constant warmup")
    for key, model_id in OFFICIAL_BASES.items():
        if not _official_source(str(args.model[key]), model_id):
            raise ValueError(f"Aligned training requires official {model_id} base")
    if int(args.data.reference_image_size) != 224 or int(args.data.target_image_size) != 448:
        raise ValueError("Aligned reference/target image sizes must be 224/448")
    if int(args.data.maximum_text_tokens) != 888:
        raise ValueError("Aligned maximum text tokens must be 888")
    if args.data.get("train_split") != "seen_train" or args.data.get("validation_split") != "seen_validation":
        raise ValueError("Aligned split protocol requires seen_train/seen_validation")
    if not args.data.get("use_chat_template"):
        raise ValueError("Aligned text conditioning requires the official chat template")
    if (args.transport.get("snr_type") != "lognorm" or not args.transport.get("do_shift") or
            not args.transport.get("dynamic_time_shift") or args.transport.get("time_shift_version", "v1") != "v1"):
        raise ValueError("Aligned transport settings differ from official profile")
    if not math.isclose(float(args.data.prompt_dropout_prob), 1e-4) or not math.isclose(float(args.data.ref_img_dropout_prob), .5):
        raise ValueError("Aligned prompt/reference dropout must be 1e-4/0.5")
    if args.model.get("pose_conditioning", False) or args.model.get("arch_opt", {}).get("pose_conditioning", False):
        raise ValueError("Aligned profile may not enable a numeric pose adapter")
    if args.val.get("max_validation_batches") is not None and not args.get("smoke", False):
        raise ValueError("Formal aligned validation must be complete")
    if args.get("smoke", False):
        stop = args.get("stop_after_updates")
        if stop is None or not 0 < int(stop) <= TOTAL_UPDATES:
            raise ValueError("Smoke stop-after-updates must be in 1..19500")
        if "aligned_smoke" not in Path(args.output_dir).parts:
            raise ValueError("Smoke output must be isolated below aligned_smoke")
    elif args.get("stop_after_updates") is not None:
        raise ValueError("stop-after-updates requires smoke mode")
    return micro, accumulation, micro * accumulation


def epoch_batch_indices(length: int, seed: int, epoch: int, update: int, rank: int, world_size: int) -> list[int]:
    """A full global batch, shuffled without replacement within an epoch."""
    full_updates = length // GLOBAL_BATCH
    if full_updates < 1 or not 0 <= update < full_updates:
        raise ValueError("Dataset has no full 128-record update at this cursor")
    permutation = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(length)
    selected = permutation[update * GLOBAL_BATCH:(update + 1) * GLOBAL_BATCH]
    local = GLOBAL_BATCH // world_size
    return selected[rank * local:(rank + 1) * local].tolist()


class _SeededDataset(Dataset):
    """Make augmentation/dropout independent of worker prefetch and resume."""

    def __init__(self, source: Dataset, seed: int):
        self.source, self.seed = source, int(seed)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, key: tuple[int, int]):
        epoch, index = key
        sample_seed = int(np.random.SeedSequence([self.seed, epoch, index]).generate_state(1)[0])
        python_state, numpy_state = random.getstate(), np.random.get_state()
        with torch.random.fork_rng(devices=[]):
            try:
                random.seed(sample_seed)
                np.random.seed(sample_seed)
                torch.random.default_generator.manual_seed(sample_seed)
                return self.source[index]
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)


class _EpochMicrobatches(Sampler[list[tuple[int, int]]]):
    def __init__(self, length, seed, epoch, start_update, rank, world_size, micro):
        self.length, self.seed, self.epoch = length, seed, epoch
        self.start_update, self.rank, self.world_size, self.micro = start_update, rank, world_size, micro

    def __iter__(self):
        permutation = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch])).permutation(self.length)
        local = GLOBAL_BATCH // self.world_size
        for update in range(self.start_update, self.length // GLOBAL_BATCH):
            full = permutation[update * GLOBAL_BATCH:(update + 1) * GLOBAL_BATCH]
            rank_indices = full[self.rank * local:(self.rank + 1) * local]
            for start in range(0, local, self.micro):
                yield [(self.epoch, int(i)) for i in rank_indices[start:start + self.micro]]

    def __len__(self):
        return (self.length // GLOBAL_BATCH - self.start_update) * (GLOBAL_BATCH // self.world_size // self.micro)


class WarmupConstant:
    def __init__(self, optimizer, next_update=1):
        self.optimizer = optimizer
        self.next_update = int(next_update)
        self._set_lr()

    @staticmethod
    def rate(update):
        if update <= 500:
            return 1e-18 + (8e-7 - 1e-18) * (update - 1) / 500
        return 8e-7

    def _set_lr(self):
        for group in self.optimizer.param_groups:
            group["lr"] = self.rate(self.next_update)

    def step(self):
        self.next_update += 1
        self._set_lr()

    def state_dict(self):
        return {"next_update": self.next_update}

    def load_state_dict(self, state):
        self.next_update = int(state["next_update"])
        self._set_lr()


def _parameter_audit(model, text_encoder, vae, optimizer):
    model_named = dict(model.named_parameters())
    trainable = {name: p for name, p in model_named.items() if p.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Only LoRA parameters may be trainable")
    optimizer_params = [p for group in optimizer.param_groups for p in group["params"]]
    if len(optimizer_params) != len(trainable) or {id(p) for p in optimizer_params} != {id(p) for p in trainable.values()}:
        raise RuntimeError("Optimizer parameter identities differ from trainable LoRA identities")
    if any(p.requires_grad for p in text_encoder.parameters()) or any(p.requires_grad for p in vae.parameters()):
        raise RuntimeError("Text encoder and VAE must be frozen")
    expected = {f"{prefix}.{block}.attn.{suffix}" for prefix, count in
                (("noise_refiner", 2), ("ref_image_refiner", 2), ("context_refiner", 2), ("layers", 32))
                for block in range(count) for suffix in LORA_SUFFIXES}
    actual = {name for name, _ in model.named_modules() if name in expected}
    if actual != expected:
        raise RuntimeError(f"Expected all 152 attention targets; missing {sorted(expected - actual)[:8]}")
    covered = {name.rsplit(".lora_", 1)[0] for name in trainable}
    if covered != expected:
        raise RuntimeError(f"LoRA parameter targets differ: missing={sorted(expected-covered)[:8]}, extra={sorted(covered-expected)[:8]}")
    return {
        "profile": PROFILE,
        "lora_targets": sorted(expected),
        "trainable": [{"name": name, "shape": list(p.shape), "numel": p.numel(), "runtime_id": id(p)} for name, p in trainable.items()],
        "frozen_transformer": [{"name": name, "shape": list(p.shape), "numel": p.numel()} for name, p in model_named.items() if not p.requires_grad],
        "frozen_text_encoder": [{"name": name, "shape": list(p.shape), "numel": p.numel()} for name, p in text_encoder.named_parameters()],
        "frozen_vae": [{"name": name, "shape": list(p.shape), "numel": p.numel()} for name, p in vae.named_parameters()],
        "optimizer_trainable_names": [next(name for name, item in trainable.items() if item is p) for p in optimizer_params],
    }


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _contract_identity(args):
    from huggingface_hub import snapshot_download

    sources = {}
    for key, model_id in OFFICIAL_BASES.items():
        supplied = str(args.model[key])
        snapshot = Path(supplied).resolve() if supplied != model_id else Path(
            snapshot_download(model_id, local_files_only=True)
        ).resolve()
        if snapshot.parent.name != "snapshots" or snapshot.parent.parent.name != "models--" + model_id.replace("/", "--"):
            raise ValueError(f"Cannot identify official cached revision for {model_id}")
        sources[key] = {"repo_id": model_id, "revision": snapshot.name,
                        "snapshot_path": str(snapshot)}
    data_root = Path(args.data.data_root).resolve()
    files = [data_root / "benchmark_manifest.json", data_root / "minimal_dataset_report.json",
             data_root / "calibration" / "z_calibration.json"]
    for map_name in SEEN_MAPS:
        for split_name in ("train.json", "validation.json"):
            files.append(data_root / "splits" / "seen" / map_name / split_name)
    contract_files = {}
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Aligned contract file missing: {path}")
        contract_files[str(path.relative_to(data_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "profile": PROFILE, "seed": int(args.seed),
        "base_sources": sources, "data_root": str(data_root),
        "contract_files": contract_files,
        "reference": 224, "target": 448, "text_tokens": 888,
        "validation_seed": int(args.val.get("seed", 4242)),
        "code_fingerprint": _code_fingerprint(),
        "execution_policy": dict(EXECUTION_POLICY),
    }


def _config_fingerprint(args):
    return hashlib.sha256(json.dumps(_contract_identity(args), sort_keys=True).encode()).hexdigest()


def _code_fingerprint():
    project = Path(__file__).resolve().parents[1]
    sources = (
        "omnigen2/aligned_training.py",
        "train.py",
        "omnigen2/dataset/csgo_seen10_dataset.py",
        "omnigen2/transport/transport.py",
        "omnigen2/models/transformers/transformer_omnigen2.py",
        "omnigen2/models/attention_processor.py",
        "omnigen2/ops/triton/layer_norm.py",
    )
    hashes = {name: hashlib.sha256((project / name).read_bytes()).hexdigest()
              for name in sources}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)


def _save_checkpoint(accelerator, args, model, optimizer, scheduler, step, epoch, cursor, val_loss, audit_hash, topology_changed):
    from train import _atomic_update_checkpoint_link

    output = Path(args.output_dir)
    final = output / f"checkpoint-{step}"
    staging_name = [f".checkpoint-{step}.incomplete-{uuid.uuid4().hex}" if accelerator.is_main_process else None]
    if accelerator.num_processes > 1:
        torch.distributed.broadcast_object_list(staging_name, src=0)
    staging = output / staging_name[0]
    if accelerator.is_main_process:
        if final.exists():
            raise FileExistsError(f"Checkpoint already exists: {final}")
        staging.mkdir()
    accelerator.wait_for_everyone()
    torch.save(_rng_state(), staging / f"rng-rank{accelerator.process_index}.pt")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        adapter_dir = staging / "transformer_lora"
        adapter_dir.mkdir()
        state = get_peft_model_state_dict(unwrapped, adapter_name="default")
        save_file({key: value.detach().cpu().contiguous() for key, value in state.items()}, adapter_dir / "adapter_model.safetensors")
        unwrapped.peft_config["default"].save_pretrained(adapter_dir)
        torch.save(optimizer.state_dict(), staging / "optimizer.pt")
        torch.save(scheduler.state_dict(), staging / "scheduler.pt")
        _write_json(staging / "aligned_state.json", {
            "profile": PROFILE, "global_step": step, "sampler_epoch": epoch,
            "sampler_update_cursor": cursor, "global_batch_size": GLOBAL_BATCH,
            "world_size": accelerator.num_processes,
            "micro_batch_size": int(args.train.batch_size),
            "gradient_accumulation_steps": int(args.train.gradient_accumulation_steps),
            "config_fingerprint": _config_fingerprint(args), "parameter_audit_sha256": audit_hash,
            "validation_loss": val_loss, "smoke": bool(args.get("smoke", False)),
            "topology_changed_resume": topology_changed, "scaler": None,
            "validation_seed": int(args.val.get("seed", 4242)),
            "code_fingerprint": _code_fingerprint(),
            "execution_policy": dict(EXECUTION_POLICY),
            "contract_identity": _contract_identity(args),
        })
        (staging / "COMPLETE").write_text("aligned adapter checkpoint\n", encoding="utf-8")
        os.replace(staging, final)
        _atomic_update_checkpoint_link(str(output), "latest", final.name)
        if step == TOTAL_UPDATES and not args.get("smoke", False):
            _atomic_update_checkpoint_link(str(output), "late", final.name)
    accelerator.wait_for_everyone()


def _resolve_resume(args):
    from train import _checkpoint_directories

    requested = args.get("resume_from_checkpoint")
    if not requested:
        return None
    output = Path(args.output_dir).resolve()
    completed = []
    for candidate_step, candidate_name in _checkpoint_directories(str(output)):
        candidate = output / candidate_name
        if (candidate / "COMPLETE").is_file():
            completed.append((candidate_step, candidate))
    if not completed:
        raise FileNotFoundError("Requested aligned checkpoint does not exist")
    if requested == "latest":
        step, path = max(completed)
    else:
        path = Path(requested)
        if not path.is_absolute():
            path = output / path
        path = path.resolve()
        if path.parent != output:
            raise ValueError("Aligned resume checkpoint must be in the current run root")
        matches = [(item_step, item_path) for item_step, item_path in completed if item_path == path]
        if not matches:
            raise FileNotFoundError(f"Requested aligned checkpoint is not complete: {path}")
        step = matches[0][0]
        if step < max(candidate_step for candidate_step, _ in completed):
            raise ValueError("Resume from an older checkpoint requires a new isolated run root")
    if not (path / "COMPLETE").is_file():
        raise ValueError("Aligned checkpoint is incomplete")
    state = json.loads((path / "aligned_state.json").read_text(encoding="utf-8"))
    if state["profile"] != PROFILE or state["global_step"] != step:
        raise ValueError("Checkpoint is not an aligned checkpoint")
    if state["config_fingerprint"] != _config_fingerprint(args):
        raise ValueError("Aligned checkpoint base/data fingerprint mismatch")
    if state["smoke"] != bool(args.get("smoke", False)):
        raise ValueError("Smoke and formal checkpoints cannot cross-resume")
    return path, state


def _recover_history(output: Path, resume_step: int, accelerator):
    """Make JSONL and best pointer agree with complete checkpoints on resume."""
    from train import _atomic_update_checkpoint_link, _checkpoint_directories

    metrics = output / "logs" / "train_metrics.jsonl"
    best_loss, best_name = float("inf"), None
    for checkpoint_step, name in _checkpoint_directories(str(output)):
        if checkpoint_step > resume_step:
            continue
        checkpoint = output / name
        if not (checkpoint / "COMPLETE").is_file():
            continue
        state = json.loads((checkpoint / "aligned_state.json").read_text(encoding="utf-8"))
        value = state.get("validation_loss")
        if value is not None and float(value) < best_loss:
            best_loss, best_name = float(value), name
    if accelerator.is_main_process:
        if metrics.exists():
            valid = []
            for line in metrics.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    break  # A crash may leave one partial trailing line.
                if int(record["step"]) <= resume_step:
                    valid.append(line)
            repaired = "\n".join(valid) + ("\n" if valid else "")
            temporary = metrics.with_name(f".{metrics.name}.repair-{uuid.uuid4().hex}")
            temporary.write_text(repaired, encoding="utf-8")
            os.replace(temporary, metrics)
        if best_name is not None:
            _atomic_update_checkpoint_link(str(output), "best", best_name)
    accelerator.wait_for_everyone()
    return best_loss


def _validate(accelerator, model, dataset, collator, text_encoder, vae, freqs_cis, transport, seed, max_batches=None):
    from train import _prepare_diffusion_batch

    unwrapped = accelerator.unwrap_model(model)
    was_training = unwrapped.training
    unwrapped.eval()
    total = torch.zeros(2, device=accelerator.device, dtype=torch.float64)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), _rng_state()
    try:
        with torch.no_grad():
            indices = range(accelerator.process_index, len(dataset), accelerator.num_processes)
            for batch_no, index in enumerate(indices):
                if max_batches is not None and batch_no >= int(max_batches):
                    break
                sample_seed = int(np.random.SeedSequence([int(seed), int(index)]).generate_state(1)[0])
                random.seed(sample_seed)
                np.random.seed(sample_seed)
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)
                prepared = _prepare_diffusion_batch(collator([dataset[index]]), text_encoder, vae,
                                                    torch.bfloat16, accelerator.device, freqs_cis, seen10=False)
                autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if
                            torch.device(accelerator.device).type == "cuda" else contextlib.nullcontext())
                with autocast:
                    losses = transport.training_losses(unwrapped, prepared["output_latents"], prepared["model_kwargs"],
                        process_index=0, num_processes=1, reduction="sum")
                total[0] += losses["loss"].detach().double().sum()
                total[1] += prepared["token_counts"].double().sum()
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        _set_rng_state(torch_state)
        unwrapped.train(was_training)
    total = accelerator.reduce(total, reduction="sum")
    if total[1].item() <= 0:
        raise ValueError("Validation has no samples")
    value = (total[0] / total[1]).item()
    if not math.isfinite(value):
        raise FloatingPointError(f"Nonfinite aligned validation loss: {value}")
    return value


def main(args):
    from train import _atomic_update_checkpoint_link, _prepare_diffusion_batch

    execution_policy = _configure_execution_policy()
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[
        DistributedDataParallelKwargs(find_unused_parameters=True)
    ])
    micro, accumulation, _ = validate_config(args, accelerator.num_processes)
    output = Path(args.output_dir)
    resume = _resolve_resume(args)
    preflight_error = [None]
    if accelerator.is_main_process:
        try:
            if not resume and output.exists() and any(output.iterdir()):
                raise FileExistsError(f"Aligned output already exists: {output}")
            output.mkdir(parents=True, exist_ok=True)
            (output / "logs").mkdir(exist_ok=True)
            if not resume:
                shutil.copyfile(args.config_file, output / Path(args.config_file).name)
            else:
                _atomic_update_checkpoint_link(str(output), "latest", resume[0].name)
        except (OSError, ValueError) as error:
            preflight_error[0] = str(error)
    if accelerator.num_processes > 1:
        torch.distributed.broadcast_object_list(preflight_error, src=0)
    if preflight_error[0] is not None:
        raise RuntimeError(preflight_error[0])
    accelerator.wait_for_everyone()
    set_seed(int(args.seed), device_specific=True)
    contract = _contract_identity(args)
    if accelerator.is_main_process:
        launch_record = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "resume_from_checkpoint": str(resume[0]) if resume else None,
            "world_size": accelerator.num_processes,
            "micro_batch_size": micro,
            "gradient_accumulation_steps": accumulation,
            "effective_global_batch_size": GLOBAL_BATCH,
            "contract_identity": contract,
            "code_fingerprint": _code_fingerprint(),
            "validation_seed": int(args.val.get("seed", 4242)),
            "execution_policy": execution_policy,
            "resolved_config": OmegaConf.to_container(args, resolve=True),
        }
        _write_json(output / f"runtime-config-{uuid.uuid4().hex}.json", launch_record)
    accelerator.wait_for_everyone()

    model = OmniGen2Transformer2DModel.from_pretrained(contract["base_sources"]["pretrained_model_path"]["snapshot_path"],
                                                       subfolder="transformer", pose_conditioning=False)
    if getattr(model, "pose_adapter", None) is not None or getattr(model, "pose_conditioner", None) is not None:
        raise ValueError("Official aligned base must not have a pose adapter")
    if getattr(model, "peft_config", None):
        raise ValueError("Aligned training must start from the official base without preloaded adapters")
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=8, lora_alpha=8, lora_dropout=0,
                                 init_lora_weights="gaussian", target_modules=list(LORA_SUFFIXES)))
    model.enable_gradient_checkpointing()
    model.train()
    text_source = contract["base_sources"]["pretrained_text_encoder_model_name_or_path"]["snapshot_path"]
    tokenizer = AutoTokenizer.from_pretrained(text_source)
    tokenizer.padding_side = "right"
    text_encoder = Qwen2_5_VLModel.from_pretrained(text_source,
                                                   torch_dtype=torch.bfloat16).to(accelerator.device)
    vae = AutoencoderKL.from_pretrained(contract["base_sources"]["pretrained_vae_model_name_or_path"]["snapshot_path"],
                                       subfolder=args.model.get("vae_subfolder", "vae"))
    vae = vae.to(accelerator.device, dtype=torch.bfloat16)
    text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-18, betas=(.9, .95), weight_decay=.01, eps=1e-8)
    scheduler = WarmupConstant(optimizer)
    audit = _parameter_audit(model, text_encoder, vae, optimizer)
    adapter_numel = sum(item["numel"] for item in audit["trainable"])
    if adapter_numel != 5_107_200:
        raise RuntimeError(f"Aligned LoRA parameter count must be 5,107,200, got {adapter_numel}")
    if accelerator.is_main_process:
        print(f"Aligned trainable LoRA: {adapter_numel:,} / {sum(p.numel() for p in model.parameters()):,} "
              f"transformer parameters; {len(audit['lora_targets'])} attention targets", flush=True)
    audit_json = json.dumps(audit, sort_keys=True).encode()
    stable_audit = dict(audit)
    stable_audit["trainable"] = [{key: value for key, value in item.items() if key != "runtime_id"}
                                 for item in audit["trainable"]]
    audit_hash = hashlib.sha256(json.dumps(stable_audit, sort_keys=True).encode()).hexdigest()
    if accelerator.is_main_process:
        first_audit = output / "parameter_audit.json"
        if not first_audit.exists():
            first_audit.write_bytes(audit_json)
        (output / f"parameter_audit-{uuid.uuid4().hex}.json").write_bytes(audit_json)
    accelerator.wait_for_everyone()

    data_common = dict(data_root=args.data.data_root, tokenizer=tokenizer,
                       use_chat_template=args.data.get("use_chat_template", True),
                       max_input_pixels=args.data.get("max_input_pixels", 224 * 224),
                       max_output_pixels=args.data.get("max_output_pixels", 448 * 448),
                       max_side_length=args.data.get("max_side_length", 448),
                       reference_image_size=224, target_image_size=448, load_target=True)
    train_dataset = CSGOSeen10Dataset(split=args.data.get("train_split", "seen_train"),
                                     prompt_dropout_prob=1e-4, ref_img_dropout_prob=.5, **data_common)
    val_dataset = CSGOSeen10Dataset(split=args.data.get("validation_split", "seen_validation"),
                                   prompt_dropout_prob=0, ref_img_dropout_prob=0, **data_common)
    if len(train_dataset) != 50000:
        raise ValueError(f"Expected 50000 native train samples, got {len(train_dataset)}")
    if len(val_dataset) != 5000:
        raise ValueError(f"Expected 5000 native validation samples, got {len(val_dataset)}")
    updates_per_epoch = len(train_dataset) // GLOBAL_BATCH
    if updates_per_epoch <= 0:
        raise ValueError("Seen train split has fewer than 128 records")
    collator = CSGOSeen10Collator(tokenizer=tokenizer, max_token_len=888, check_truncation=True)
    transport = create_transport("Linear", "velocity", None, None, None,
        snr_type=args.transport.snr_type, do_shift=args.transport.do_shift,
        seq_len=448 * 448 // 16 // 16, dynamic_time_shift=args.transport.get("dynamic_time_shift", False),
        time_shift_version=args.transport.get("time_shift_version", "v1"))
    freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(model.config.axes_dim_rope,
                                                      model.config.axes_lens, theta=10000)
    model = accelerator.prepare(model)
    step = 0
    topology_changed = False
    if resume:
        path, state = resume
        current_topology_changed = (
            state["world_size"] != accelerator.num_processes or state["micro_batch_size"] != micro or
            state["gradient_accumulation_steps"] != accumulation
        )
        topology_changed = bool(state.get("topology_changed_resume", False)) or current_topology_changed
        if current_topology_changed and accelerator.is_main_process:
            warnings.warn("Changed batch topology: optimizer/sampler resume is exact, per-rank stochastic draws are reseeded", stacklevel=2)
        if state["parameter_audit_sha256"] != audit_hash:
            raise ValueError("Trainable/frozen parameter audit differs from checkpoint")
        adapter = load_file(path / "transformer_lora" / "adapter_model.safetensors", device="cpu")
        result = set_peft_model_state_dict(accelerator.unwrap_model(model), adapter, adapter_name="default")
        missing_lora = [key for key in result.missing_keys if "lora_" in key]
        if result.unexpected_keys or missing_lora:
            raise ValueError(f"Adapter resume mismatch: {result}")
        optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False))
        scheduler.load_state_dict(torch.load(path / "scheduler.pt", map_location="cpu", weights_only=False))
        step = int(state["global_step"])
        if state["sampler_epoch"] != step // updates_per_epoch or state["sampler_update_cursor"] != step % updates_per_epoch:
            raise ValueError("Checkpoint sampler cursor does not match optimizer step")
        if current_topology_changed:
            set_seed(int(args.seed) + step, device_specific=True)
        else:
            _set_rng_state(torch.load(path / f"rng-rank{accelerator.process_index}.pt", map_location="cpu", weights_only=False))

    limit = int(args.stop_after_updates) if args.get("smoke", False) else TOTAL_UPDATES
    if step >= limit:
        raise ValueError("Resume checkpoint is already at or beyond requested stop")
    if scheduler.next_update != step + 1:
        raise ValueError("Scheduler update count differs from checkpoint step")
    seeded_dataset = _SeededDataset(train_dataset, int(args.seed))
    workers = int(args.train.get("dataloader_num_workers", 0))
    metrics = output / "logs" / "train_metrics.jsonl"
    best = _recover_history(output, step, accelerator) if resume else float("inf")
    starting_step, started_at = step, time.monotonic()

    while step < limit:
        epoch, start = divmod(step, updates_per_epoch)
        sampler = _EpochMicrobatches(len(train_dataset), int(args.seed), epoch, start,
                                     accelerator.process_index, accelerator.num_processes, micro)
        loader = DataLoader(seeded_dataset, batch_sampler=sampler, num_workers=workers,
                            collate_fn=collator, pin_memory=True,
                            generator=torch.Generator().manual_seed(int(args.seed) + epoch))
        batches = iter(loader)
        for _ in range(start, updates_per_epoch):
            if step >= limit:
                break
            optimizer.zero_grad(set_to_none=True)
            local_numerator = torch.zeros((), device=accelerator.device, dtype=torch.float64)
            local_tokens = torch.zeros((), device=accelerator.device, dtype=torch.float64)
            for micro_index in range(accumulation):
                batch = next(batches)
                prepared = _prepare_diffusion_batch(batch, text_encoder, vae, torch.bfloat16,
                                                    accelerator.device, freqs_cis, seen10=False)
                context = accelerator.no_sync(model) if micro_index < accumulation - 1 else contextlib.nullcontext()
                with context:
                    losses = transport.training_losses(model, prepared["output_latents"], prepared["model_kwargs"],
                        process_index=accelerator.process_index, num_processes=accelerator.num_processes, reduction="sum")
                    numerator = losses["loss"].sum()
                    tokens = prepared["token_counts"].sum()
                    expected_tokens = GLOBAL_BATCH * (16 * 56 * 56)
                    accelerator.backward(numerator * accelerator.num_processes / expected_tokens)
                local_numerator += numerator.detach().double()
                local_tokens += tokens.detach().double()
            totals = accelerator.reduce(torch.stack((local_numerator, local_tokens)), reduction="sum")
            if not bool(torch.isfinite(totals).all()):
                raise FloatingPointError(f"Nonfinite aligned training loss at update {step + 1}")
            if int(totals[1].item()) != GLOBAL_BATCH * (16 * 56 * 56):
                raise ValueError("An update did not contain exactly 128 target latents at 448x448")
            grad_norm = accelerator.clip_grad_norm_(trainable, float(args.train.max_grad_norm))
            if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
                raise FloatingPointError(f"Nonfinite aligned gradient norm at update {step + 1}")
            grad_norm_value = float(grad_norm)
            used_lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            step += 1
            if accelerator.is_main_process and (step == starting_step + 1 or step % 50 == 0 or step in MILESTONES or step == limit):
                elapsed = time.monotonic() - started_at
                rate = (step - starting_step) / max(elapsed, 1e-9)
                eta = (limit - step) / max(rate, 1e-9)
                print(f"aligned update {step}/{limit} loss={float(totals[0] / totals[1]):.8f} "
                      f"grad_norm={grad_norm_value:.6g} lr={used_lr:.3e} "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
            val_loss = None
            milestone = step in MILESTONES
            smoke_save = bool(args.get("smoke", False))
            if milestone or smoke_save:
                val_loss = _validate(accelerator, model, val_dataset, collator, text_encoder, vae,
                                     freqs_cis, transport, int(args.val.get("seed", 4242)),
                                     max_batches=args.val.get("max_validation_batches") if args.get("smoke", False) else None)
                _save_checkpoint(accelerator, args, model, optimizer, scheduler, step,
                                 step // updates_per_epoch, step % updates_per_epoch,
                                 val_loss, audit_hash, topology_changed)
                if accelerator.is_main_process and val_loss < best:
                    _atomic_update_checkpoint_link(str(output), "best", f"checkpoint-{step}")
                    best = val_loss
            if accelerator.is_main_process:
                with metrics.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": step, "loss": float(totals[0] / totals[1]),
                                             "grad_norm": grad_norm_value, "lr": used_lr,
                                             "val_loss": val_loss}, allow_nan=False) + "\n")
            if step >= limit:
                break
    accelerator.wait_for_everyone()
    accelerator.end_training()
