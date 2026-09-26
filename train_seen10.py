#!/usr/bin/env python3
"""Thin, seed-aware launcher for OmniGen2 CSGO Benchmark v2 training."""

from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path

from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "options" / "csgo_seen10_lora.yml"
ALIGNED_CONFIG = PROJECT_ROOT / "options" / "csgo_seen10_exp32gen_aligned.yml"
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
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-after-updates", type=int, default=None)
    parser.add_argument("--output-root", default=None)
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
    parser.add_argument("--micro-batch-size", type=int, default=None)
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
    experiment = getattr(cli, "experiment", None)
    config_value = cli.config or (str(ALIGNED_CONFIG) if experiment == "csgo_seen10_exp32gen_aligned" else str(DEFAULT_CONFIG))
    config_path = Path(config_value).expanduser().resolve()
    conf = OmegaConf.load(config_path)
    if conf.data.get("dataset_type") != "csgo_seen10":
        raise ValueError("Seen-10 launcher requires data.dataset_type=csgo_seen10")

    smoke = bool(getattr(cli, "smoke", False))
    stop_after_updates = getattr(cli, "stop_after_updates", None)
    micro_batch_size = getattr(cli, "micro_batch_size", None)
    aligned = experiment == "csgo_seen10_exp32gen_aligned" or conf.get("experiment") == "csgo_seen10_exp32gen_aligned"
    seed = cli.seed if cli.seed is not None else (42 if aligned else 0)
    if experiment is not None and experiment != conf.get("experiment", experiment):
        raise ValueError("CLI experiment and configuration experiment disagree")
    if aligned:
        if experiment != "csgo_seen10_exp32gen_aligned":
            raise ValueError("Aligned training requires explicit --experiment csgo_seen10_exp32gen_aligned")
        if cli.max_train_steps is not None:
            raise ValueError("Aligned optimizer budget is fixed; use --stop-after-updates for smoke runs")
        if stop_after_updates is not None and not smoke:
            raise ValueError("--stop-after-updates is only available with --smoke")
        if smoke and stop_after_updates is None:
            raise ValueError("--smoke requires --stop-after-updates")
        if smoke and (cli.output_root is None or "aligned_smoke" not in Path(cli.output_root).parts):
            raise ValueError("Smoke output root must include an aligned_smoke directory")
        if not smoke and cli.output_root is not None and "aligned_smoke" in Path(cli.output_root).parts:
            raise ValueError("Formal aligned output cannot use an aligned_smoke directory")
        if cli.batch_size is not None and micro_batch_size is not None and cli.batch_size != micro_batch_size:
            raise ValueError("--batch-size and --micro-batch-size disagree")

    output_root = cli.output_root or str(
        PROJECT_ROOT / "outputs" / "csgo_seen10_exp32gen_aligned" / "OmniGen2"
        if aligned else DEFAULT_OUTPUT_ROOT
    )
    if aligned and not smoke and Path(output_root).expanduser().resolve() != (
        PROJECT_ROOT / "outputs" / "csgo_seen10_exp32gen_aligned" / "OmniGen2"
    ).resolve():
        raise ValueError("Formal aligned output root must be outputs/csgo_seen10_exp32gen_aligned/OmniGen2")
    if aligned and not smoke and seed != 42:
        raise ValueError("Formal aligned seed must be 42")
    output_dir = (
        Path(output_root).expanduser().resolve()
        / f"seed_{seed}"
        / "train"
    )
    if not aligned and output_dir.exists() and any(output_dir.iterdir()) and not cli.resume_from_checkpoint:
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

    conf.seed = seed
    conf.root_dir = str(PROJECT_ROOT)
    conf.output_dir = str(output_dir)
    conf.config_file = str(config_path)
    conf.resume_from_checkpoint = cli.resume_from_checkpoint
    if aligned:
        conf.experiment = "csgo_seen10_exp32gen_aligned"
        conf.smoke = smoke
        conf.stop_after_updates = stop_after_updates

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
        "batch_size": micro_batch_size if micro_batch_size is not None else cli.batch_size,
        "gradient_accumulation_steps": cli.gradient_accumulation_steps,
        "dataloader_num_workers": cli.dataloader_num_workers,
    }
    for key, value in overrides.items():
        if value is not None:
            conf.train[key] = value

    if cli.max_validation_batches is not None:
        conf.val.max_validation_batches = cli.max_validation_batches

    if aligned:
        if not smoke and conf.val.get("max_validation_batches") is not None:
            raise ValueError("Formal aligned validation must use all seen_validation samples")
        if int(conf.train.max_optimizer_steps) != 19500 or int(conf.train.max_train_steps) != 19500:
            raise ValueError("Aligned optimizer budget must be 19500")
        if list(conf.train.checkpoint_steps) != [4000, 8000, 12000, 16000, 19500]:
            raise ValueError("Aligned checkpoint_steps must be [4000, 8000, 12000, 16000, 19500]")
        return conf

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


def _prepare_cli_environment(cli: argparse.Namespace) -> None:
    if cli.experiment == "csgo_seen10_exp32gen_aligned":
        # Set this before importing train/torch so cuBLAS sees it on first use.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


if __name__ == "__main__":
    cli = parse_args()
    _prepare_cli_environment(cli)
    from train import main as train_main

    train_main(build_config(cli))
