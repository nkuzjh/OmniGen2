#!/usr/bin/env python3
"""Read-only environment check; intentionally never imports omnigen2."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import os
import sys
from pathlib import Path


REQUIRED_DISTRIBUTIONS = (
    "torch", "torchvision", "numpy", "Pillow", "einops", "timm",
    "accelerate", "transformers", "diffusers", "peft", "huggingface-hub",
    "tokenizers", "safetensors", "omegaconf", "torchdiffeq",
    "python-dotenv", "matplotlib", "scipy", "tqdm",
)
REQUIRED_MODULES = (
    "torch", "torchvision", "numpy", "PIL", "einops", "timm",
    "accelerate", "transformers", "diffusers", "peft", "huggingface_hub",
    "tokenizers", "safetensors", "omegaconf", "torchdiffeq",
    "dotenv", "matplotlib", "scipy", "tqdm",
)
OPTIONAL_MODULES = ("cv2", "wandb")


def environment_identity(expected: Path) -> None:
    actual = Path(sys.prefix).resolve()
    if actual != expected.resolve():
        raise SystemExit(f"Python belongs to {actual}, expected {expected}; preserve this path and choose its own bin/python")
    cfg = expected / "pyvenv.cfg"
    if cfg.is_file():
        if sys.prefix == sys.base_prefix:
            raise SystemExit(f"{expected} is not an active virtual environment")
        config = cfg.read_text(encoding="utf-8").lower()
        if "include-system-site-packages = true" in config:
            print("WARNING: this venv inherits packages from its base Python; existing environment is read-only", file=sys.stderr)
        elif "include-system-site-packages = false" not in config:
            raise SystemExit(f"{cfg} has unknown package isolation; preserved unchanged")
    elif not (expected / "conda-meta").is_dir():
        raise SystemExit(f"{expected} is neither a venv nor a Conda environment; preserved unchanged")


def check_imports(*, strict_extras: bool = False) -> None:
    # Optional integrations such as peft/DeepSpeed may probe CUDA at import.
    # Hide devices in this short-lived process so the CPU check stays CPU-only.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    missing = []
    for name in REQUIRED_DISTRIBUTIONS:
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        raise SystemExit("Missing core distributions: " + ", ".join(missing))
    failures = []
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit("Core import failures:\n  " + "\n  ".join(failures))
    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel
        _ = (AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel)
    except Exception as exc:
        raise SystemExit(f"Required Transformers Qwen2.5-VL classes unavailable: {type(exc).__name__}: {exc}") from exc
    optional_failures = []
    for module in OPTIONAL_MODULES:
        try:
            importlib.import_module(module)
        except Exception as exc:
            optional_failures.append(f"{module}: {type(exc).__name__}: {exc}")
    if optional_failures:
        message = "Optional import failures:\n  " + "\n  ".join(optional_failures)
        if strict_extras:
            raise SystemExit(message)
        print("WARNING: " + message, file=sys.stderr)
    import torch

    if torch.cuda.is_initialized():
        raise SystemExit("CUDA initialized during CPU-only import check")
    print(f"Environment: {sys.prefix} (Python {sys.version.split()[0]})")
    print(f"torch {metadata.version('torch')}; torchvision {metadata.version('torchvision')}; CPU imports passed; CUDA uninitialized")
    if not (3, 11) <= sys.version_info[:2] <= (3, 12):
        print("WARNING: existing Python is outside the fresh-install target range 3.11-3.12", file=sys.stderr)


def check_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable to PyTorch in this environment")
    device = torch.device("cuda:0")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("GPU does not support BF16 required for training")
    matrix = torch.eye(8, device=device, dtype=torch.bfloat16)
    result = matrix @ matrix
    torch.cuda.synchronize(device)
    if not torch.allclose(result, matrix):
        raise SystemExit("CUDA BF16 matrix operation failed")
    capability = torch.cuda.get_device_capability(device)
    print(f"CUDA matrix check passed: {torch.cuda.get_device_name(device)}, sm_{capability[0]}{capability[1]}, runtime {torch.version.cuda}")


def check_fresh_pins(requirements: Path) -> None:
    expected = {"torch": "2.7.1", "torchvision": "0.22.1"}
    for raw in requirements.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            expected[name] = version
    mismatches = []
    for name, version in expected.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = "missing"
        # Official CUDA wheels carry a local suffix such as +cu128.
        if actual.split("+", 1)[0] != version:
            mismatches.append(f"{name}: expected {version}, found {actual}")
    if mismatches:
        raise SystemExit("Fresh environment pins were not retained:\n  " + "\n  ".join(mismatches))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-prefix", type=Path, required=True)
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--cuda-only", action="store_true", help="internal second phase after CPU import check")
    parser.add_argument("--fresh-pins", type=Path, help="verify all fresh-install pins after pip")
    args = parser.parse_args()
    environment_identity(args.expected_prefix)
    if args.cuda_only:
        check_cuda()
    elif not args.identity_only:
        check_imports(strict_extras=bool(args.fresh_pins))
        if args.fresh_pins:
            check_fresh_pins(args.fresh_pins)


if __name__ == "__main__":
    main()
