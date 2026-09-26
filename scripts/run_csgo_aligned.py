#!/usr/bin/env python3
"""Explicit, isolated aligned profile; legacy shell invocation remains unchanged."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


PROJECT = Path(__file__).resolve().parents[1]
EXPERIMENT = "csgo_seen10_exp32gen_aligned"
DEFAULT_ROOT = PROJECT / "outputs" / EXPERIMENT / "OmniGen2"
EVALUATOR = Path("/home/jiahao/task/csgo_benchmark_v2_eval_general")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("train", "convert", "infer", "eval", "smoke", "all"))
    p.add_argument("--experiment", required=True, choices=(EXPERIMENT,))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--inference-seed", type=int, default=42)
    p.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--config", type=Path, default=PROJECT / "options" / f"{EXPERIMENT}.yml")
    p.add_argument("--num-processes", type=int, default=int(os.environ.get("NUM_PROCESSES", "1")))
    p.add_argument("--micro-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=128)
    p.add_argument("--dataloader-num-workers", type=int, default=None)
    p.add_argument("--resume-from-checkpoint")
    p.add_argument("--checkpoint", choices=("best", "late", "latest"), default="late")
    p.add_argument("--task", choices=("discrete", "continuous", "all"), default="all")
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("INFERENCE_BATCH_SIZE", "16")))
    p.add_argument("--vae-decode-batch-size", type=int, default=1)
    p.add_argument("--no-oom-fallback", action="store_true")
    p.add_argument("--no-fuse-lora", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--stop-after-updates", type=int)
    p.add_argument("--max-validation-batches", type=int)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--dry-run", action="store_true", help="Print commands only; no directory or model writes.")
    args = p.parse_args(argv)
    if args.seed < 0 or args.inference_seed < 0:
        p.error("seeds must be non-negative")
    if args.action in {"train", "all"}:
        batch = (args.num_processes, args.micro_batch_size, args.gradient_accumulation_steps)
        if min(batch) < 1 or batch[0] * batch[1] * batch[2] != 128:
            p.error("world_size * micro_batch_size * gradient_accumulation_steps must equal 128")
    for name in ("batch_size", "vae_decode_batch_size", "max_samples", "stop_after_updates", "max_validation_batches"):
        value = getattr(args, name)
        if value is not None and value < 1:
            p.error(f"{name} must be positive")
    if (args.stop_after_updates is not None or args.max_validation_batches is not None) and not args.smoke:
        p.error("short training/validation limits require --smoke and an isolated output root")
    if args.smoke or args.max_samples is not None:
        root = args.output_root.expanduser().resolve()
        if root == DEFAULT_ROOT or DEFAULT_ROOT in root.parents:
            p.error("smoke/limited runs require --output-root outside the formal aligned root")
    return args


def commands(args):
    python = os.environ.get("OMNIGEN2_PYTHON", str(PROJECT / ".venv/bin/python"))
    eval_python = os.environ.get("UNILIP_PYTHON", "/home/jiahao/miniconda3/envs/UniLIP/bin/python")
    root = args.output_root.expanduser().resolve() / f"seed_{args.seed}"
    adapter = root / "adapters" / args.checkpoint
    predictions = root / "predictions" / args.checkpoint
    data_root = "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2"
    model = os.environ.get("OMNIGEN2_MODEL_PATH", "OmniGen2/OmniGen2")
    vae = os.environ.get("OMNIGEN2_VAE_MODEL_PATH", "black-forest-labs/FLUX.1-dev")
    text = os.environ.get("OMNIGEN2_TEXT_ENCODER_MODEL_PATH", "Qwen/Qwen2.5-VL-3B-Instruct")
    selected = ("train", "convert", "infer", "eval") if args.action == "all" else (args.action,)
    result = []
    for action in selected:
        if action == "train":
            command = [python, "-m", "accelerate.commands.launch", "--num_machines", "1",
                       "--num_processes", str(args.num_processes), "--mixed_precision", "bf16"]
            if args.num_processes > 1:
                command += ["--multi_gpu"]
            command += ["train_seen10.py", "--experiment", EXPERIMENT, "--config", str(args.config),
                        "--seed", str(args.seed), "--output-root", str(args.output_root),
                        "--micro-batch-size", str(args.micro_batch_size),
                        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
                        "--pretrained-model-path", model, "--pretrained-vae-model-path", vae,
                        "--pretrained-text-encoder-model-path", text]
            for option, value in (("--resume-from-checkpoint", args.resume_from_checkpoint),
                                  ("--dataloader-num-workers", args.dataloader_num_workers),
                                  ("--stop-after-updates", args.stop_after_updates),
                                  ("--max-validation-batches", args.max_validation_batches)):
                if value is not None:
                    command += [option, str(value)]
            if args.smoke:
                command += ["--smoke"]
            result.append(command)
        elif action == "convert":
            command = [python, "convert_ckpt_to_hf_format.py", "--config_path", str(args.config),
                       "--model_path", str(root / "train" / args.checkpoint), "--save_path", str(adapter)]
            if args.smoke:
                command += ["--allow-smoke"]
            result.append(command)
        elif action == "infer":
            command = [python, "infer_seen10.py", "--experiment", EXPERIMENT,
                       "--seed", str(args.inference_seed), "--task", args.task, "--data-root", data_root,
                       "--output-root", str(predictions), "--model-path", model,
                       "--vae-model-path", vae, "--text-encoder-model-path", text,
                       "--adapter-path", str(adapter), "--num-inference-steps", "28", "--dtype", "bf16",
                       "--batch-size", str(args.batch_size), "--vae-decode-batch-size", str(args.vae_decode_batch_size)]
            if args.max_samples is not None:
                command += ["--max-samples", str(args.max_samples)]
            if args.no_oom_fallback:
                command += ["--no-oom-fallback"]
            if args.no_fuse_lora:
                command += ["--no-fuse-lora"]
            result.append(command)
        elif action == "eval":
            for task in ("discrete", "continuous") if args.task == "all" else (args.task,):
                command = [eval_python, str(EVALUATOR / "run_eval.py")]
                if args.smoke:
                    command += ["smoke", task, "--limit", str(args.max_samples or 1)]
                    if task == "continuous" and (args.max_samples or 1) < 64:
                        command += ["--frame-only"]
                else:
                    command += [task]
                command += ["--pred-root", str(predictions / task), "--data-root", data_root]
                if not args.smoke:
                    command += ["--output", str(root / "evaluation" / args.checkpoint / task)]
                result.append(command)
        else:
            result.append([python, "-m", "pytest", "-q", "tests"])
    return result


def main(argv=None):
    args = parse_args(argv)
    planned = commands(args)
    if args.dry_run:
        print(json.dumps({"experiment": EXPERIMENT, "commands": planned}, indent=2))
        return 0
    for command in planned:
        print("+ " + shlex.join(command), flush=True)
        subprocess.run(command, cwd=PROJECT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
