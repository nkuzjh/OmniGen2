#!/usr/bin/env python3
"""Thin, seed-aware launcher for OmniGen2 CSGO Benchmark v2 training."""

from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path

from omegaconf import OmegaConf

from train import main as train_main


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "options" / "csgo_seen10_lora.yml"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "csgo_benchmark_v2_seen10" / "OmniGen2"
)

_BOOTSTRAP_FILES = {
    "tokenizer": frozenset(
        {
            "added_tokens.json",
            "chat_template.jinja",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        }
    ),
    "text_encoder": frozenset(
        {
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        }
    ),
}
_SHARDED_BOOTSTRAP_WEIGHT_PATTERNS = (
    re.compile(r"model-\d{5}-of-\d{5}\.safetensors\Z"),
    re.compile(r"pytorch_model-\d{5}-of-\d{5}\.bin\Z"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OmniGen2 on the manifest-driven CSGO v2 Seen-10 split."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--pretrained-model-path", default=None)
    parser.add_argument("--pretrained-vae-model-path", default=None)
    parser.add_argument("--pretrained-text-encoder-model-path", default=None)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help=(
            "Checkpoint path/name, or 'latest'. Existing training progress is rejected "
            "without this flag; failed bootstrap attempts with recognized tokenizer/text "
            "encoder assets may retry."
        ),
    )
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=None)
    return parser.parse_args()


def _is_retryable_bootstrap_output(output_dir: Path, config_path: Path) -> bool:
    """Allow retry after failure before any model/training state was written.

    setup_logging and model initialization can leave a copied config, logs, and
    the recognized tokenizer/text-encoder files saved before training starts.
    These derived bootstrap artifacts can coexist with a retry. Checkpoints,
    metrics, visualizations, symlinks, and unknown files or directories remain
    protected by the normal no-overwrite rule.
    """
    if output_dir.is_symlink() or not output_dir.is_dir():
        return False

    def is_allowed_pretrained_artifact(path: Path) -> bool:
        allowed_files = _BOOTSTRAP_FILES[path.name]
        if path.is_symlink() or not path.is_dir():
            return False
        for artifact in path.iterdir():
            if artifact.is_symlink() or not artifact.is_file():
                return False
            if artifact.name in allowed_files:
                continue
            if path.name == "text_encoder" and any(
                pattern.fullmatch(artifact.name)
                for pattern in _SHARDED_BOOTSTRAP_WEIGHT_PATTERNS
            ):
                continue
            return False
        return True

    for entry in output_dir.iterdir():
        if entry.name == config_path.name:
            if (
                entry.is_symlink()
                or not entry.is_file()
                or entry.read_bytes() != config_path.read_bytes()
            ):
                return False
            continue
        if entry.name == "logs":
            if entry.is_symlink() or not entry.is_dir():
                return False
            for log_entry in entry.iterdir():
                if (
                    log_entry.is_symlink()
                    or not log_entry.is_file()
                    or log_entry.suffix != ".log"
                ):
                    return False
            continue
        if entry.name in _BOOTSTRAP_FILES:
            if not is_allowed_pretrained_artifact(entry):
                return False
            continue
        return False
    return True


def build_config(cli: argparse.Namespace):
    config_path = Path(cli.config).expanduser().resolve()
    conf = OmegaConf.load(config_path)
    if conf.data.get("dataset_type") != "csgo_seen10":
        raise ValueError("Seen-10 launcher requires data.dataset_type=csgo_seen10")

    output_dir = (
        Path(cli.output_root).expanduser().resolve()
        / f"seed_{cli.seed}"
        / "train"
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not cli.resume_from_checkpoint:
        if _is_retryable_bootstrap_output(output_dir, config_path):
            warnings.warn(
                f"Retrying {output_dir}: it contains only bootstrap config/log files "
                "and recognized tokenizer/text_encoder assets, with no checkpoint "
                "or training progress.",
                stacklevel=2,
            )
        else:
            raise FileExistsError(
                f"Refusing to overwrite non-empty training output: {output_dir}. "
                "Pass --resume-from-checkpoint latest (or an explicit checkpoint) to resume."
            )

    conf.seed = cli.seed
    conf.root_dir = str(PROJECT_ROOT)
    conf.output_dir = str(output_dir)
    conf.config_file = str(config_path)
    conf.resume_from_checkpoint = cli.resume_from_checkpoint

    model_overrides = {
        "pretrained_model_path": getattr(cli, "pretrained_model_path", None),
        "pretrained_vae_model_name_or_path": getattr(
            cli, "pretrained_vae_model_path", None
        ),
        "pretrained_text_encoder_model_name_or_path": getattr(
            cli, "pretrained_text_encoder_model_path", None
        ),
    }
    for key, value in model_overrides.items():
        if value is not None:
            conf.model[key] = value

    overrides = {
        "max_train_steps": cli.max_train_steps,
        "global_batch_size": cli.global_batch_size,
        "batch_size": cli.batch_size,
        "gradient_accumulation_steps": cli.gradient_accumulation_steps,
        "dataloader_num_workers": cli.dataloader_num_workers,
    }
    for key, value in overrides.items():
        if value is not None:
            conf.train[key] = value

    if cli.max_validation_batches is not None:
        conf.val.max_validation_batches = cli.max_validation_batches

    max_steps = int(conf.train.max_train_steps)
    if max_steps < 5 or max_steps % 5:
        raise ValueError("train.max_train_steps must be >= 5 and divisible by 5")
    interval = max_steps // 5
    conf.val.validation_steps = interval
    conf.val.train_visualization_steps = interval
    conf.logger.checkpointing_steps = interval

    expected_global = (
        int(conf.train.batch_size)
        * int(conf.train.gradient_accumulation_steps)
        * int(os.environ.get("WORLD_SIZE", "1"))
    )
    if cli.global_batch_size is None:
        conf.train.global_batch_size = expected_global
    return conf


if __name__ == "__main__":
    train_main(build_config(parse_args()))
