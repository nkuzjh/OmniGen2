import dotenv

dotenv.load_dotenv(override=True)

import time
from copy import deepcopy
import argparse
import logging
import math
import json
import os
import re
import shutil
import inspect
import uuid
from functools import partial
from pathlib import Path
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import numpy as np

import matplotlib.pyplot as plt

import torch

import torch.nn.functional as F
import torch.utils.checkpoint

from torchvision.transforms.functional import crop, to_pil_image, to_tensor

from einops import repeat, rearrange

import accelerate
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import transformers
from transformers import AutoTokenizer
from transformers import Qwen2_5_VLModel as TextEncoder

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from peft import LoraConfig

from omnigen2.training_utils import EMAModel
from omnigen2.utils.logging_utils import TqdmToLogger
from omnigen2.transport import create_transport
from omnigen2.dataset.omnigen2_train_dataset import OmniGen2TrainDataset, OmniGen2Collator
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.models.transformers.repo import OmniGen2RotaryPosEmbed


logger = get_logger(__name__)

    
def parse_args(root_path) -> OmegaConf:
    parser = argparse.ArgumentParser(description="OmniGen2 training script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (YAML format)",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=None,
        help="Global batch size.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Data path.",
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)

    output_dir = os.path.join(root_path, 'experiments', conf.name)
    conf.root_dir = root_path
    conf.output_dir = output_dir
    conf.config_file = args.config

    # Override config with command line arguments
    if args.global_batch_size is not None:
        conf.train.global_batch_size = args.global_batch_size
    
    if args.data_path is not None:
        if conf.data.get("dataset_type") == "csgo_seen10":
            conf.data.data_root = args.data_path
        else:
            conf.data.data_path = args.data_path
    return conf

def setup_logging(args: OmegaConf, accelerator: Accelerator) -> None:
    """
    Set up logging configuration for training.
    
    Args:
        accelerator: Accelerator instance
        args: Configuration object
        logging_dir: Directory for log files
    """

    logging_dir = Path(args.output_dir, "logs")
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        shutil.copy(args.config_file, args.output_dir)
        
        # Create logging directory and file handler
        os.makedirs(logging_dir, exist_ok=True)
        log_file = Path(logging_dir, f'{time.strftime("%Y%m%d-%H%M%S")}.log')

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.logger.addHandler(file_handler)

    # Configure basic logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # Set verbosity for different processes
    log_level = logging.INFO if accelerator.is_local_main_process else logging.ERROR
    transformers.utils.logging.set_verbosity(log_level)
    diffusers.utils.logging.set_verbosity(log_level)


def log_model_info(name: str, model: torch.nn.Module):
    """Logs parameter counts for a given model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"--- {name} ---")
    logger.info(model)
    logger.info(f"Total parameters (M): {total_params / 1e6:.2f}")
    logger.info(f"Trainable parameters (M): {trainable_params / 1e6:.2f}")


def log_time_distribution(transport, device, args):
    """Samples time steps from transport and plots their distribution."""
    with torch.no_grad():
        dummy_tensor = torch.randn((64, 16, int(math.sqrt(args.data.max_output_pixels) / 8), int(math.sqrt(args.data.max_output_pixels) / 8)), device=device)
        ts = torch.cat([transport.sample(dummy_tensor, AcceleratorState().process_index, AcceleratorState().num_processes)[0] for _ in range(1000)], dim=0)
    
    ts_np = ts.cpu().numpy()
    percentile_70 = np.percentile(ts_np, 70)
    
    plt.figure(figsize=(10, 6))
    plt.hist(ts_np, bins=50, edgecolor='black', alpha=0.7, label="Time Step Distribution")
    plt.axvline(percentile_70, color='red', linestyle='dashed', linewidth=2, label=f'70th Percentile = {percentile_70:.2f}')
    plt.title('Distribution of Sampled Time Steps (t)')
    plt.xlabel('Time Step (t)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = Path(args.output_dir) / 't_distribution.png'
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Time step distribution plot saved to {save_path}")


_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def _checkpoint_directories(output_dir):
    """Return real checkpoint directories as (step, basename), excluding links."""
    checkpoints = []
    if not os.path.isdir(output_dir):
        return checkpoints
    with os.scandir(output_dir) as entries:
        for entry in entries:
            match = _CHECKPOINT_PATTERN.fullmatch(entry.name)
            if match is None or entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                continue
            checkpoints.append((int(match.group(1)), entry.name))
    return sorted(checkpoints)


def _checkpoint_link_target(output_dir, link_name):
    """Return a valid real checkpoint target for a checkpoint pointer symlink."""
    link_path = os.path.join(output_dir, link_name)
    if not os.path.islink(link_path):
        return None
    target_path = os.path.realpath(link_path)
    target_name = os.path.basename(target_path)
    match = _CHECKPOINT_PATTERN.fullmatch(target_name)
    if (
        match is None
        or os.path.dirname(target_path) != os.path.realpath(output_dir)
        or not os.path.isdir(target_path)
        or os.path.islink(target_path)
    ):
        return None
    return target_name


def _atomic_update_checkpoint_link(output_dir, link_name, checkpoint_name):
    """Atomically point a relative symlink at a saved checkpoint directory."""
    if link_name not in {"latest", "late", "best"}:
        raise ValueError(f"Unsupported checkpoint link name: {link_name}")
    checkpoint_path = os.path.join(output_dir, checkpoint_name)
    match = _CHECKPOINT_PATTERN.fullmatch(checkpoint_name)
    if match is None or not os.path.isdir(checkpoint_path) or os.path.islink(checkpoint_path):
        raise FileNotFoundError(f"Not a saved checkpoint directory: {checkpoint_path}")

    link_path = os.path.join(output_dir, link_name)
    if os.path.lexists(link_path) and not os.path.islink(link_path):
        raise FileExistsError(f"Refusing to replace non-symlink checkpoint pointer: {link_path}")

    temporary_link = os.path.join(
        output_dir, f".{link_name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        os.symlink(checkpoint_name, temporary_link)
        os.replace(temporary_link, link_path)
    finally:
        if os.path.lexists(temporary_link):
            os.unlink(temporary_link)


def _prune_old_checkpoints(output_dir, total_limit):
    """Apply the configured limit without deleting latest/best or symlink entries."""
    if total_limit is None:
        return
    total_limit = int(total_limit)
    if total_limit <= 0:
        raise ValueError("logger.checkpoints_total_limit must be positive or null")

    checkpoints = _checkpoint_directories(output_dir)
    protected = {
        target
        for target in (
            _checkpoint_link_target(output_dir, "latest"),
            _checkpoint_link_target(output_dir, "late"),
            _checkpoint_link_target(output_dir, "best"),
        )
        if target is not None
    }
    while len(checkpoints) >= total_limit:
        removable = next((item for item in checkpoints if item[1] not in protected), None)
        if removable is None:
            logger.info("Keeping protected latest/best checkpoint despite checkpoint limit")
            break
        _, name = removable
        shutil.rmtree(os.path.join(output_dir, name))
        checkpoints.remove(removable)


def _resolve_resume_checkpoint(output_dir, requested):
    """Resolve latest or an explicit checkpoint path and parse its step safely."""
    if requested == "latest":
        checkpoint_name = _checkpoint_link_target(output_dir, "latest")
        if checkpoint_name is None:
            checkpoints = _checkpoint_directories(output_dir)
            checkpoint_name = checkpoints[-1][1] if checkpoints else None
        if checkpoint_name is None:
            return None, None
        checkpoint_path = os.path.join(output_dir, checkpoint_name)
    else:
        requested_path = Path(os.path.expanduser(str(requested)))
        if requested_path.is_absolute():
            checkpoint_path = requested_path
        elif os.path.lexists(os.path.join(output_dir, str(requested))):
            checkpoint_path = Path(output_dir) / str(requested)
        else:
            checkpoint_path = requested_path
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
        checkpoint_path = checkpoint_path.resolve()

    checkpoint_name = os.path.basename(os.path.realpath(checkpoint_path))
    match = _CHECKPOINT_PATTERN.fullmatch(checkpoint_name)
    if match is None:
        raise ValueError(
            "Resume checkpoint directory must be named checkpoint-<step>, got "
            f"{checkpoint_name!r}"
        )
    return os.path.realpath(checkpoint_path), int(match.group(1))


def _pose_dimensions(args):
    arch_opt = args.model.get("arch_opt", {})
    input_dim = int(args.model.get("pose_input_dim", arch_opt.get("pose_input_dim", 5)))
    hidden_dim = int(args.model.get("pose_hidden_dim", arch_opt.get("pose_hidden_dim", 512)))
    if input_dim != 5:
        raise ValueError(f"Seen-10 pose input must have 5 normalized values, got {input_dim}")
    if hidden_dim <= 0:
        raise ValueError("model.pose_hidden_dim must be positive")
    return input_dim, hidden_dim


def _load_transformer(args, seen10):
    if not seen10:
        return OmniGen2Transformer2DModel.from_pretrained(
            args.model.pretrained_model_path, subfolder="transformer"
        )

    input_dim, hidden_dim = _pose_dimensions(args)
    init_parameters = inspect.signature(OmniGen2Transformer2DModel.__init__).parameters
    pose_kwargs = {}
    if "pose_conditioning" in init_parameters:
        pose_kwargs["pose_conditioning"] = True
    elif "pose_conditioning" not in init_parameters:
        raise TypeError("The transformer constructor does not support pose_conditioning")

    input_key = next(
        (key for key in ("pose_input_dim", "pose_conditioning_input_dim") if key in init_parameters),
        None,
    )
    if input_key is not None:
        pose_kwargs[input_key] = input_dim
    elif getattr(OmniGen2Transformer2DModel, "pose_input_dim", 5) != input_dim:
        raise TypeError("The transformer pose input dimension does not match the Seen-10 schema")

    hidden_key = next(
        (
            key
            for key in ("pose_hidden_dim", "pose_conditioning_hidden_dim")
            if key in init_parameters
        ),
        None,
    )
    if hidden_key is None:
        raise TypeError("The transformer constructor does not expose a pose hidden dimension")
    pose_kwargs[hidden_key] = hidden_dim

    model = OmniGen2Transformer2DModel.from_pretrained(
        args.model.pretrained_model_path,
        subfolder="transformer",
        **pose_kwargs,
    )
    pose_module = _get_pose_module(model)
    if pose_module is None:
        enable_pose = getattr(model, "enable_pose_conditioning", None)
        if enable_pose is None:
            raise TypeError("The loaded transformer has no trainable pose conditioner")
        enable_pose(hidden_dim)
    return model


def _get_pose_module(model):
    pose_module = getattr(model, "pose_conditioner", None)
    if pose_module is None:
        pose_module = getattr(model, "pose_adapter", None)
    return pose_module


def _make_trainable_parameter_groups(model, args, seen10):
    pose_module = _get_pose_module(model)
    pose_parameters = list(pose_module.parameters()) if pose_module is not None else []
    if seen10 and not pose_parameters:
        raise ValueError("Seen-10 training requires a trainable pose conditioner")
    if seen10:
        if not args.train.get("lora_ft", False):
            raise ValueError("Seen-10 configuration must enable train.lora_ft")
        for parameter in pose_parameters:
            parameter.requires_grad_(True)

    pose_ids = {id(parameter) for parameter in pose_parameters}
    lora_parameters = []
    trainable_parameters = []
    seen_parameter_ids = set()
    unexpected = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen_parameter_ids:
            continue
        seen_parameter_ids.add(id(parameter))
        trainable_parameters.append(parameter)
        if id(parameter) in pose_ids:
            continue
        lora_parameters.append(parameter)
        if seen10 and "lora_" not in name:
            unexpected.append(name)

    if seen10 and unexpected:
        raise RuntimeError(
            "Seen-10 LoRA training found non-LoRA trainable parameters outside the pose MLP: "
            + ", ".join(unexpected[:8])
        )
    if not trainable_parameters:
        raise ValueError("No trainable transformer parameters were found")

    if not seen10:
        return trainable_parameters, [{"params": trainable_parameters, "lr": args.train.learning_rate}]

    if not lora_parameters:
        raise ValueError("Seen-10 LoRA training found no trainable LoRA parameters")
    pose_learning_rate = args.train.get("pose_learning_rate", None)
    if pose_learning_rate is None or float(pose_learning_rate) <= 0:
        raise ValueError("Seen-10 configuration requires a positive train.pose_learning_rate")
    parameter_groups = [
        {"params": lora_parameters, "lr": float(args.train.learning_rate)},
        {"params": pose_parameters, "lr": float(pose_learning_rate)},
    ]
    if len({id(parameter) for group in parameter_groups for parameter in group["params"]}) != len(
        [parameter for group in parameter_groups for parameter in group["params"]]
    ):
        raise RuntimeError("Duplicate transformer parameter found in optimizer groups")
    return trainable_parameters, parameter_groups


def _call_with_supported_kwargs(constructor, kwargs):
    parameters = inspect.signature(constructor).parameters
    accepts_arbitrary = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    filtered = kwargs if accepts_arbitrary else {
        key: value for key, value in kwargs.items() if key in parameters
    }
    return constructor(**filtered)


def _to_python_config_value(value):
    """Convert OmegaConf nodes while leaving ordinary Python values intact."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_object(value)
    return value


def _make_csgo_seen10_dataset(args, tokenizer, split, load_target):
    from omnigen2.dataset.csgo_seen10_dataset import CSGOSeen10Dataset

    train_split = args.data.get("train_split", "seen_train")
    dataset_kwargs = {
        "data_root": args.data.data_root,
        "split": split,
        "tokenizer": tokenizer,
        "use_chat_template": args.data.get("use_chat_template", True),
        "max_input_pixels": _to_python_config_value(
            args.data.get("max_input_pixels", 1024 * 1024)
        ),
        "max_output_pixels": args.data.get("max_output_pixels", 1024 * 1024),
        "max_side_length": args.data.get("max_side_length", 2048),
        "load_target": load_target,
        "image_size": args.data.get("image_size", 448),
        "prompt_dropout_prob": (
            args.data.get("prompt_dropout_prob", 0.0) if split == train_split else 0.0
        ),
        "ref_img_dropout_prob": (
            args.data.get("ref_img_dropout_prob", 0.0) if split == train_split else 0.0
        ),
    }
    return _call_with_supported_kwargs(CSGOSeen10Dataset, dataset_kwargs)


def _encode_vae_image(vae, image, weight_dtype, device):
    if image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(
            "Each OmniGen2 VAE input must have shape [3,H,W] or [1,3,H,W], "
            f"got {tuple(image.shape)}"
        )
    image = image.to(device=device, dtype=vae.dtype)
    latent = vae.encode(image).latent_dist.sample()
    if vae.config.shift_factor is not None:
        latent = latent - vae.config.shift_factor
    if vae.config.scaling_factor is not None:
        latent = latent * vae.config.scaling_factor
    return latent.to(dtype=weight_dtype)


def _prepare_diffusion_batch(
    batch,
    text_encoder,
    vae,
    weight_dtype,
    device,
    freqs_cis,
    seen10,
):
    """Encode one collated batch for either the train or validation forward."""
    input_images = batch["input_images"]
    output_image = batch["output_image"]
    text_input_ids = batch["text_ids"].to(device=device)
    text_mask = batch["text_mask"].to(device=device)

    with torch.no_grad():
        text_feats = text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_mask,
            output_hidden_states=False,
        ).last_hidden_state

        input_latents = []
        for references in input_images:
            if references is not None and len(references) > 0:
                input_latents.append(
                    [
                        _encode_vae_image(vae, image, weight_dtype, device).squeeze(0)
                        for image in references
                    ]
                )
            else:
                input_latents.append(None)

        output_latents = [
            _encode_vae_image(vae, image, weight_dtype, device).squeeze(0)
            for image in output_image
        ]

    model_kwargs = dict(
        text_hidden_states=text_feats,
        text_attention_mask=text_mask,
        ref_image_hidden_states=input_latents,
        freqs_cis=freqs_cis,
    )
    if seen10:
        pose_values = batch.get("pose_values")
        if pose_values is None:
            raise KeyError("CSGOSeen10Collator must return pose_values")
        if not torch.is_tensor(pose_values):
            pose_values = torch.as_tensor(pose_values)
        model_kwargs["pose_values"] = pose_values.to(device=device, dtype=torch.float32)

    token_counts = torch.tensor(
        [latent.numel() for latent in output_latents],
        device=device,
        dtype=torch.long,
    )
    return {
        "input_images": input_images,
        "output_image": output_image,
        "text_input_ids": text_input_ids,
        "output_latents": output_latents,
        "model_kwargs": model_kwargs,
        "token_counts": token_counts,
    }


def _diffusion_training_losses(transport, model, prepared_batch, accelerator):
    return transport.training_losses(
        model,
        prepared_batch["output_latents"],
        prepared_batch["model_kwargs"],
        process_index=AcceleratorState().process_index,
        num_processes=AcceleratorState().num_processes,
        reduction="sum",
    )


def _evaluation_rng(accelerator, seed):
    device = torch.device(accelerator.device)
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    return torch.random.fork_rng(devices=devices), seed + accelerator.process_index


def _evaluate_seen_validation(
    accelerator,
    model,
    validation_dataloader,
    text_encoder,
    vae,
    weight_dtype,
    freqs_cis,
    transport,
    max_validation_batches=None,
    seed=0,
):
    """Compute validation diffusion loss, globally weighted by latent elements."""
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=accelerator.device, dtype=torch.long)
    rng_context, rank_seed = _evaluation_rng(accelerator, int(seed))

    try:
        with rng_context:
            torch.random.default_generator.manual_seed(rank_seed)
            if torch.device(accelerator.device).type == "cuda":
                with torch.cuda.device(accelerator.device):
                    torch.cuda.manual_seed(rank_seed)
            with torch.no_grad():
                for batch_index, batch in enumerate(validation_dataloader):
                    if max_validation_batches is not None and batch_index >= int(max_validation_batches):
                        break
                    prepared = _prepare_diffusion_batch(
                        batch,
                        text_encoder,
                        vae,
                        weight_dtype,
                        accelerator.device,
                        freqs_cis,
                        seen10=True,
                    )
                    loss_dict = _diffusion_training_losses(transport, model, prepared, accelerator)
                    local_losses = loss_dict["loss"].detach().to(dtype=torch.float64)
                    gathered_losses, gathered_tokens = accelerator.gather_for_metrics(
                        (local_losses, prepared["token_counts"])
                    )
                    total_loss += gathered_losses.sum()
                    total_tokens += gathered_tokens.sum()
    finally:
        model.train(was_training)

    accelerator.wait_for_everyone()
    if total_tokens.item() == 0:
        raise ValueError("seen_validation produced no samples; cannot select a checkpoint")
    return (total_loss / total_tokens).item()


def _load_metric_history(metrics_path):
    records = []
    if not os.path.isfile(metrics_path):
        return records
    with open(metrics_path, "r", encoding="utf-8") as metrics_file:
        for line in metrics_file:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Ignoring malformed train_metrics.jsonl line")
    return records


def _write_loss_curve(metrics_path, output_path):
    records = _load_metric_history(metrics_path)
    train_points = [
        (int(item["step"]), float(item["loss"]))
        for item in records
        if item.get("step") is not None and item.get("loss") is not None
    ]
    validation_points = [
        (int(item["step"]), float(item["val_loss"]))
        for item in records
        if item.get("step") is not None and item.get("val_loss") is not None
    ]
    if not train_points:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(
        [point[0] for point in train_points],
        [point[1] for point in train_points],
        label="train loss",
        linewidth=1.2,
    )
    if validation_points:
        plt.plot(
            [point[0] for point in validation_points],
            [point[1] for point in validation_points],
            marker="o",
            label="seen_validation loss",
        )
    plt.xlabel("optimizer step")
    plt.ylabel("diffusion loss")
    plt.title("OmniGen2 Seen-10 training loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _save_accelerate_checkpoint(accelerator, args, global_step, seen10):
    if accelerator.is_main_process:
        _prune_old_checkpoints(
            args.output_dir,
            args.logger.get("checkpoints_total_limit", None),
        )
    accelerator.wait_for_everyone()

    checkpoint_name = f"checkpoint-{global_step}"
    save_path = os.path.join(args.output_dir, checkpoint_name)
    if os.path.lexists(save_path):
        raise FileExistsError(
            f"Refusing to overwrite existing checkpoint: {save_path}"
        )
    accelerator.save_state(save_path)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process and seen10:
        _atomic_update_checkpoint_link(args.output_dir, "latest", checkpoint_name)
        # The benchmark contract names the final-checkpoint pointer ``late``.
        # Keep ``latest`` as the native resume alias and expose both atomically.
        _atomic_update_checkpoint_link(args.output_dir, "late", checkpoint_name)
    accelerator.wait_for_everyone()
    logger.info(f"Saved state to {save_path}")
    
    
def main(args):
    seen10 = args.data.get("dataset_type", None) == "csgo_seen10"
    if seen10:
        if "max_train_steps" not in args.train:
            raise ValueError("Seen-10 training requires train.max_train_steps")
        max_train_steps = int(args.train.max_train_steps)
        if max_train_steps < 5 or max_train_steps % 5:
            raise ValueError("Seen-10 train.max_train_steps must be >= 5 and divisible by 5")
        expected_interval = max_train_steps // 5
        if int(args.val.get("validation_steps", -1)) != expected_interval:
            raise ValueError(
                "Seen-10 val.validation_steps must equal train.max_train_steps / 5 "
                f"({expected_interval})"
            )
        if int(args.logger.checkpointing_steps) != expected_interval:
            raise ValueError(
                "Seen-10 logger.checkpointing_steps must equal train.max_train_steps / 5 "
                f"({expected_interval})"
            )
        max_validation_batches = args.val.get("max_validation_batches", None)
        if max_validation_batches is not None and int(max_validation_batches) <= 0:
            raise ValueError("val.max_validation_batches must be positive or null")
        metrics_path = os.path.join(args.output_dir, "logs", "train_metrics.jsonl")
        if not args.resume_from_checkpoint and (
            _checkpoint_directories(args.output_dir)
            or os.path.isfile(metrics_path)
            or os.path.lexists(os.path.join(args.output_dir, "latest"))
        ):
            raise FileExistsError(
                "Seen-10 output already contains a checkpoint or training metrics; "
                "resume explicitly instead of overwriting it"
            )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=Path(args.output_dir, 'logs'))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.train.gradient_accumulation_steps,
        mixed_precision=args.train.mixed_precision,
        log_with=_to_python_config_value(args.logger.log_with),
        project_config=accelerator_project_config,
    )

    setup_logging(args, accelerator)
    
    # Reproducibility
    if args.seed is not None:
        set_seed(args.seed, device_specific=args.get('device_specific_seed', False))

    # Set performance flags
    if args.train.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.train.get('benchmark_cudnn', False):
        torch.backends.cudnn.benchmark = True

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    ema_decay = args.train.get('ema_decay', 0)

    model = _load_transformer(args, seen10)
    model.train()

    # model = OmniGen2Transformer2DModel(**args.model.arch_opt)
    # model.train()

    freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
        model.config.axes_dim_rope,
        model.config.axes_lens,
        theta=10000,
    )

    # if args.model.get("pretrained_model_path", None) is not None:
    #     logger.info(f"Loading model parameters from: {args.model.pretrained_model_path}")
    #     state_dict = torch.load(args.model.pretrained_model_path, map_location="cpu")
    #     missing, unexpect = model.load_state_dict(state_dict, strict=False)
    #     logger.info(
    #         f"missed parameters: {missing}",
    #     )
    #     logger.info(f"unexpected parameters: {unexpect}")

    if ema_decay != 0:
        model_ema = deepcopy(model)
        model_ema._requires_grad = False

    text_tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained_text_encoder_model_name_or_path)
    text_tokenizer.padding_side = "right"

    if accelerator.is_main_process:
        text_tokenizer.save_pretrained(os.path.join(args.output_dir, 'tokenizer'))

    text_encoder = TextEncoder.from_pretrained(
        args.model.pretrained_text_encoder_model_name_or_path,
        torch_dtype=weight_dtype,
    )
    if args.model.get('resize_token_embeddings', False):
        text_encoder.resize_token_embeddings(len(text_tokenizer))

    if accelerator.is_main_process:
        text_encoder.save_pretrained(os.path.join(args.output_dir, 'text_encoder'))

    log_model_info("text_encoder", text_encoder)

    vae = AutoencoderKL.from_pretrained(
        args.model.pretrained_vae_model_name_or_path,
        subfolder=args.model.get("vae_subfolder", "vae"),
    )
    
    logger.info(vae)
    logger.info("***** Move vae, text_encoder to device and cast to weight_dtype *****")
    # Move vae, unet, text_encoder and controlnet_ema to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    args.train.lora_ft = args.train.get('lora_ft', False)
    if args.train.lora_ft:
        model.requires_grad_(False)

        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

        # now we will add new LoRA weights the transformer layers
        lora_config = LoraConfig(
            r=args.train.lora_rank,
            lora_alpha=args.train.lora_rank,
            lora_dropout=args.train.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        model.add_adapter(lora_config)

    if args.train.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if args.train.scale_lr:
        args.train.learning_rate = (
            args.train.learning_rate * args.train.gradient_accumulation_steps * args.train.batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    log_model_info("transformer", model)

    # Optimizer creation
    trainable_params, optimizer_param_groups = _make_trainable_parameter_groups(
        model, args, seen10
    )
    
    optimizer = optimizer_class(
        optimizer_param_groups,
        lr=args.train.learning_rate,
        betas=(args.train.adam_beta1, args.train.adam_beta2),
        weight_decay=args.train.adam_weight_decay,
        eps=args.train.adam_epsilon,
    )

    logger.info("***** Prepare dataset *****")

    with accelerator.main_process_first():
        if seen10:
            from omnigen2.dataset.csgo_seen10_dataset import CSGOSeen10Collator

            train_dataset = _make_csgo_seen10_dataset(
                args,
                text_tokenizer,
                split=args.data.get("train_split", "seen_train"),
                load_target=True,
            )
            validation_dataset = _make_csgo_seen10_dataset(
                args,
                text_tokenizer,
                split=args.data.get("validation_split", "seen_validation"),
                load_target=True,
            )
            collate_fn = CSGOSeen10Collator(
                tokenizer=text_tokenizer,
                max_token_len=args.data.maximum_text_tokens,
            )
        else:
            train_dataset = OmniGen2TrainDataset(
                args.data.data_path,
                tokenizer=text_tokenizer,
                use_chat_template=args.data.use_chat_template,
                prompt_dropout_prob=args.data.get('prompt_dropout_prob', 0.0),
                ref_img_dropout_prob=args.data.get('ref_img_dropout_prob', 0.0),
                max_input_pixels=_to_python_config_value(
                    args.data.get('max_input_pixels', 1024 * 1024)
                ),
                max_output_pixels=args.data.get('max_output_pixels', 1024 * 1024),
                max_side_length=args.data.get('max_side_length', 2048),
            )
            validation_dataset = None
            collate_fn = OmniGen2Collator(
                tokenizer=text_tokenizer,
                max_token_len=args.data.maximum_text_tokens,
            )

    # default: 1000 steps, linear noise schedule
    transport = create_transport(
        "Linear",
        "velocity",
        None,
        None,
        None,
        snr_type=args.transport.snr_type,
        do_shift=args.transport.do_shift,
        seq_len=args.data.max_output_pixels // 16 // 16,
        dynamic_time_shift=args.transport.get("dynamic_time_shift", False),
        time_shift_version=args.transport.get("time_shift_version", "v1"),
    )  # default: velocity;

    # Log time distribution for analysis
    if accelerator.is_main_process:
        log_time_distribution(transport, accelerator.device, args)

    logger.info(f"Number of training samples: {len(train_dataset)}")
    if seen10:
        logger.info(f"Number of seen_validation samples: {len(validation_dataset)}")
        if len(train_dataset) == 0 or len(validation_dataset) == 0:
            raise ValueError("Seen-10 train and validation datasets must both be non-empty")

    if args.seed is not None and args.get("workder_specific_seed", False):
        from omnigen2.utils.reproducibility import worker_init_fn

        worker_init_fn = partial(
            worker_init_fn,
            num_processes=AcceleratorState().num_processes,
            num_workers=args.train.dataloader_num_workers,
            process_index=AcceleratorState().process_index,
            seed=args.seed,
            same_seed_per_epoch=args.get("same_seed_per_epoch", False),
        )
    else:
        worker_init_fn = None

    logger.info("***** Prepare dataLoader *****")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train.batch_size,
        num_workers=args.train.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        collate_fn=collate_fn,
    )
    validation_dataloader = None
    if seen10:
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            shuffle=False,
            batch_size=args.train.batch_size,
            num_workers=args.train.dataloader_num_workers,
            worker_init_fn=worker_init_fn,
            drop_last=False,
            collate_fn=collate_fn,
        )

    logger.info(f"{args.train.batch_size=} {args.train.gradient_accumulation_steps=} {accelerator.num_processes=} {args.train.global_batch_size=}")
    assert (
        args.train.batch_size
        * args.train.gradient_accumulation_steps
        * accelerator.num_processes
        == args.train.global_batch_size
    ), (
        f"{args.train.batch_size=} * {args.train.gradient_accumulation_steps=} * {accelerator.num_processes=} should be equal to {args.train.global_batch_size=}"
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.train.gradient_accumulation_steps)
    if 'max_train_steps' not in args.train:
        args.train.max_train_steps = args.train.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.train.lr_scheduler == 'timm_cosine':
        from omnigen2.optim.scheduler.cosine_lr import CosineLRScheduler

        lr_scheduler = CosineLRScheduler(optimizer=optimizer,
                                         t_initial=args.train.t_initial,
                                         lr_min=args.train.lr_min,
                                         cycle_decay=args.train.cycle_decay,
                                         warmup_t=args.train.warmup_t,
                                         warmup_lr_init=args.train.warmup_lr_init,
                                         warmup_prefix=args.train.warmup_prefix,
                                         t_in_epochs=args.train.t_in_epochs)
    elif args.train.lr_scheduler == 'timm_constant_with_warmup':
        from omnigen2.optim.scheduler.step_lr import StepLRScheduler

        lr_scheduler = StepLRScheduler(
            optimizer=optimizer,
            decay_t=1,
            decay_rate=1,
            warmup_t=args.train.warmup_t,
            warmup_lr_init=args.train.warmup_lr_init,
            warmup_prefix=args.train.warmup_prefix,
            t_in_epochs=args.train.t_in_epochs,
        )
    else:
        lr_scheduler = get_scheduler(
            args.train.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.train.lr_warmup_steps,
            num_training_steps=args.train.max_train_steps,
            num_cycles=args.train.lr_num_cycles,
            power=args.train.lr_power,
        )

    logger.info("***** Prepare everything with our accelerator *****")

    if args.train.ema_decay != 0 and seen10:
        model, model_ema, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
            model, model_ema, optimizer, train_dataloader, validation_dataloader, lr_scheduler
        )
        model_ema = EMAModel(model_ema.parameters(), decay=ema_decay, model_cls=type(unwrap_model(model)), model_config=model_ema.config)
    elif args.train.ema_decay != 0:
        model, model_ema, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, model_ema, optimizer, train_dataloader, lr_scheduler
        )
        model_ema = EMAModel(model_ema.parameters(), decay=ema_decay, model_cls=type(unwrap_model(model)), model_config=model_ema.config)
    elif seen10:
        model, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, validation_dataloader, lr_scheduler
        )
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.train.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.train.max_train_steps = args.train.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.train.num_train_epochs = math.ceil(args.train.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("OmniGen2", init_kwargs={"wandb": {"name": args.name}})

    # Train!
    total_batch_size = args.train.batch_size * accelerator.num_processes * args.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.train.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.train.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.train.max_train_steps}")
    global_step = 0
    first_epoch = 0
    resume_batches_to_skip = 0

    metrics_path = os.path.join(args.output_dir, "logs", "train_metrics.jsonl")
    metrics_history = _load_metric_history(metrics_path) if seen10 else []
    validation_history = [
        float(item["val_loss"])
        for item in metrics_history
        if item.get("val_loss") is not None
    ]
    best_validation_loss = min(validation_history) if validation_history else float("inf")
        
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        resume_path, resume_step = _resolve_resume_checkpoint(
            args.output_dir, args.resume_from_checkpoint
        )
        if resume_path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {resume_path}")
            accelerator.load_state(resume_path)
            global_step = resume_step
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            resume_batches_to_skip = (
                global_step % num_update_steps_per_epoch
            ) * args.train.gradient_accumulation_steps
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.train.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
        file=TqdmToLogger(logger, level=logging.INFO)
    )

    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                logger.info(f"***** Wandb log dir: {tracker.run.dir} *****")
    
    for epoch in range(first_epoch, args.train.num_train_epochs):
        if 'max_train_steps' in args.train and global_step >= args.train.max_train_steps:
            break
        epoch_dataloader = train_dataloader
        if epoch == first_epoch and resume_batches_to_skip > 0:
            epoch_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_batches_to_skip
            )
        resume_batches_to_skip = 0
        for step, batch in enumerate(epoch_dataloader):
            # Number of bins, for loss recording
            n_loss_bins = 10
            # Create bins for t
            loss_bins = torch.linspace(0.0, 1.0, n_loss_bins + 1, device=accelerator.device)
            # Initialize occurrence and sum tensors
            bin_occurrence = torch.zeros(n_loss_bins, device=accelerator.device)
            bin_sum_loss = torch.zeros(n_loss_bins, device=accelerator.device)

            prepared_batch = _prepare_diffusion_batch(
                batch,
                text_encoder,
                vae,
                weight_dtype,
                accelerator.device,
                freqs_cis,
                seen10=seen10,
            )
            input_images = prepared_batch['input_images']
            output_image = prepared_batch['output_image']
            text_input_ids = prepared_batch['text_input_ids']
            output_latents = prepared_batch['output_latents']

            with accelerator.accumulate(model):
                local_num_tokens_in_batch = prepared_batch["token_counts"].sum()
                
                num_tokens_in_batch = accelerator.gather(local_num_tokens_in_batch).sum().item()
                loss_dict = _diffusion_training_losses(
                    transport, model, prepared_batch, accelerator
                )
                loss = loss_dict["loss"].sum()
                loss = (loss * accelerator.gradient_state.num_steps * accelerator.num_processes) / num_tokens_in_batch
                total_loss = loss

                accelerator.backward(total_loss)

                bin_indices = torch.bucketize(
                    loss_dict["t"].to(device=loss_bins.device), loss_bins, right=True
                ) - 1
                detached_loss = loss_dict["loss"].detach()
                local_num_tokens = prepared_batch["token_counts"]

                # Iterate through each bin index to update occurrence and sum
                for i in range(n_loss_bins):
                    mask = bin_indices == i  # Mask for elements in the i-th bin
                    bin_occurrence[i] = bin_occurrence[i] + local_num_tokens[mask].sum()  # Count occurrences in the i-th bin
                    bin_sum_loss[i] = bin_sum_loss[i] + detached_loss[mask].sum()  # Sum loss values in the i-th bin

                avg_loss = accelerator.gather(loss.detach().repeat(args.train.batch_size)).mean() / accelerator.gradient_state.num_steps
                avg_total_loss = accelerator.gather(total_loss.detach().repeat(args.train.batch_size)).mean() / accelerator.gradient_state.num_steps

                bin_occurrence = accelerator.gather(rearrange(bin_occurrence, "b -> 1 b")).sum(dim=0)
                bin_sum_loss = accelerator.gather(rearrange(bin_sum_loss, "b -> 1 b")).sum(dim=0)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.train.max_grad_norm)

                optimizer.step()
                if 'timm' in args.train.lr_scheduler:
                    lr_scheduler.step(global_step)
                else:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.train.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                logs = {"loss": avg_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                logs["total_loss"] = avg_total_loss.detach().item()

                for i in range(n_loss_bins):
                    if bin_occurrence[i] > 0:
                        bin_avg_loss = (bin_sum_loss[i] / bin_occurrence[i]).item()
                        logs[f"loss-bin{i+1}-{n_loss_bins}"] = bin_avg_loss

                if ema_decay != 0:
                    model_ema.step(model.parameters())
                    
                global_step += 1
                validation_loss = None

                if global_step % args.logger.checkpointing_steps == 0:
                    _save_accelerate_checkpoint(
                        accelerator, args, global_step, seen10=seen10
                    )
                    if seen10:
                        validation_loss = _evaluate_seen_validation(
                            accelerator,
                            model,
                            validation_dataloader,
                            text_encoder,
                            vae,
                            weight_dtype,
                            freqs_cis,
                            transport,
                            max_validation_batches=args.val.get(
                                "max_validation_batches", None
                            ),
                            seed=args.get("seed", 0),
                        )
                        if not math.isfinite(validation_loss):
                            raise FloatingPointError(
                                f"seen_validation loss is not finite at step {global_step}: "
                                f"{validation_loss}"
                            )
                        if accelerator.is_main_process:
                            if validation_loss < best_validation_loss:
                                checkpoint_name = f"checkpoint-{global_step}"
                                _atomic_update_checkpoint_link(
                                    args.output_dir, "best", checkpoint_name
                                )
                                best_validation_loss = validation_loss
                                logger.info(
                                    "New best seen_validation loss %.8f at step %d",
                                    validation_loss,
                                    global_step,
                                )
                        accelerator.wait_for_everyone()

                if seen10:
                    metric_record = {
                        "step": int(global_step),
                        "loss": float(logs["loss"]),
                        "lr": float(logs["lr"]),
                        "val_loss": (
                            float(validation_loss) if validation_loss is not None else None
                        ),
                    }
                    if accelerator.is_main_process:
                        with open(metrics_path, "a", encoding="utf-8") as metrics_file:
                            metrics_file.write(
                                json.dumps(metric_record, allow_nan=False) + "\n"
                            )
                            metrics_file.flush()

                if accelerator.is_main_process:
                    if 'train_visualization_steps' in args.val and (global_step - 1) % args.val.train_visualization_steps == 0:
                        num_samples = min(args.val.get('num_train_visualization_samples', 3), args.train.batch_size)
                        with torch.no_grad():
                            for i in range(num_samples):
                                model_pred = loss_dict['pred'][i]
                                pred_image = loss_dict['xt'][i] + (1 - loss_dict['t'][i]) * model_pred

                                if vae.config.scaling_factor is not None:
                                    pred_image = pred_image / vae.config.scaling_factor
                                if vae.config.shift_factor is not None:
                                    pred_image = pred_image + vae.config.shift_factor
                                pred_image = vae.decode(
                                    pred_image.unsqueeze(0).to(dtype=weight_dtype),
                                    return_dict=False,
                                )[0]
                                pred_image = pred_image.clamp(-1, 1)

                                vis_images = [output_image[i]] + [pred_image]
                                if input_images[i] is not None:
                                    vis_images = input_images[i] + vis_images

                                # Concatenate input images of different sizes horizontally
                                max_height = max(img.shape[-2] for img in vis_images)
                                total_width = sum(img.shape[-1] for img in vis_images)
                                canvas = torch.zeros((3, max_height, total_width), device=vis_images[0].device)
                                
                                current_x = 0
                                for img in vis_images:
                                    h, w = img.shape[-2:]
                                    # Place image at the top of canvas
                                    canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
                                    current_x += w
                                
                                to_pil_image(canvas).save(os.path.join(args.output_dir, f"input_visualization_{global_step}_{i}_t{loss_dict['t'][i]}.png"))
                                
                                input_ids = text_input_ids[i].detach().cpu()
                                instruction = text_tokenizer.decode(input_ids, skip_special_tokens=False)

                                with open(os.path.join(args.output_dir, f"instruction_{global_step}_{i}.txt"), "w", encoding='utf-8') as f:
                                    f.write(f"token len: {len(input_ids)}\ntext: {instruction}")

                progress_bar.set_postfix(**logs)
                progress_bar.update(1)

                accelerator.log(logs, step=global_step)

            if 'max_train_steps' in args.train and global_step >= args.train.max_train_steps:
                break

    checkpoints = _checkpoint_directories(args.output_dir)
    if global_step > 0 and (
        not checkpoints or checkpoints[-1][0] < global_step
    ):
        _save_accelerate_checkpoint(
            accelerator, args, global_step, seen10=seen10
        )

    if seen10:
        accelerator.wait_for_everyone()
        checkpoints = _checkpoint_directories(args.output_dir)
        if checkpoints and accelerator.is_main_process:
            _atomic_update_checkpoint_link(
                args.output_dir, "latest", checkpoints[-1][1]
            )
            _atomic_update_checkpoint_link(
                args.output_dir, "late", checkpoints[-1][1]
            )
        if accelerator.is_main_process:
            _write_loss_curve(
                metrics_path,
                os.path.join(args.output_dir, "loss_curve.png"),
            )
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    root_path = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args(root_path)
    main(args)
