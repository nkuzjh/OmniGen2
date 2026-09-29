#!/usr/bin/env python3
"""Read-only environment check; intentionally never imports omnigen2."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import os
import re
import sys
from pathlib import Path


REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements-csgo-seen10.txt"
# The requirements file is authoritative. Every declared direct dependency must
# have an import probe; adding a package cannot silently bypass this check.
DISTRIBUTION_MODULES = {
    "torch": "torch", "torchvision": "torchvision", "numpy": "numpy",
    "pillow": "PIL", "einops": "einops", "timm": "timm",
    "accelerate": "accelerate", "transformers": "transformers",
    "diffusers": "diffusers", "peft": "peft", "huggingface-hub": "huggingface_hub",
    "datasets": "datasets", "tokenizers": "tokenizers", "safetensors": "safetensors",
    "omegaconf": "omegaconf", "pyyaml": "yaml", "packaging": "packaging",
    "opencv-python-headless": "cv2", "scipy": "scipy", "torchdiffeq": "torchdiffeq",
    "wandb": "wandb", "tensorboard": "tensorboard", "matplotlib": "matplotlib",
    "tqdm": "tqdm", "python-dotenv": "dotenv", "ninja": "ninja",
    "wheel": "wheel", "pytest": "pytest",
}
REQUIRED_SYMBOLS = {
    "datasets": ("load_dataset", "concatenate_datasets"),
    "transformers": ("AutoTokenizer", "AutoProcessor", "Qwen2_5_VLModel", "Qwen2_5_VLForConditionalGeneration"),
    "torch.utils.tensorboard": ("SummaryWriter",),
    "accelerate": ("Accelerator", "init_empty_weights"),
    "diffusers": ("AutoencoderKL", "FlowMatchEulerDiscreteScheduler"),
    "peft": ("LoraConfig", "get_peft_model_state_dict", "set_peft_model_state_dict"),
    "huggingface_hub": ("hf_hub_download", "snapshot_download"),
}


def requirement_pins(path: Path = REQUIREMENTS) -> dict[str, str]:
    pins = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if not match:
            raise ValueError(f"Expected an exact dependency pin in {path}: {line}")
        name, version = match.groups()
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        if canonical in pins:
            raise ValueError(f"Duplicate dependency: {name}")
        pins[canonical] = version
    return pins


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
            print("WARNING: this venv inherits base packages; repairs must stay inside the selected venv and preserve core versions", file=sys.stderr)
        elif "include-system-site-packages = false" not in config:
            raise SystemExit(f"{cfg} has unknown package isolation; preserved unchanged")
    elif not (expected / "conda-meta").is_dir():
        raise SystemExit(f"{expected} is neither a venv nor a Conda environment; preserved unchanged")


def check_imports(*, strict_extras: bool = False, requirements: Path = REQUIREMENTS) -> None:
    # Optional integrations such as peft/DeepSpeed may probe CUDA at import.
    # Hide devices in this short-lived process so the CPU check stays CPU-only.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    pins = requirement_pins(requirements)
    unmapped = set(pins) - DISTRIBUTION_MODULES.keys()
    if unmapped:
        raise SystemExit("Dependencies missing import probes: " + ", ".join(sorted(unmapped)))
    missing = []
    for name in pins:
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    if missing:
        raise SystemExit("Missing declared distributions: " + ", ".join(missing) +
                         ". Run bash scripts/setup_csgo_seen10.sh --env-only to install all missing dependencies.")
    failures = []
    for module in dict.fromkeys(DISTRIBUTION_MODULES[name] for name in pins):
        try:
            importlib.import_module(module)
        except Exception as exc:
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit("Declared dependency import failures:\n  " + "\n  ".join(failures))
    for module, symbols in REQUIRED_SYMBOLS.items():
        try:
            imported = importlib.import_module(module)
            for symbol in symbols:
                getattr(imported, symbol)
        except Exception as exc:
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
    if failures:
        raise SystemExit("Required runtime interfaces unavailable:\n  " + "\n  ".join(failures))
    import torch

    # CUDA wheels supply the matching Triton version through Torch dependencies.
    # Do not import OmniGen2's native kernels here: their decorators probe a GPU.
    if torch.version.cuda is not None and sys.platform == "linux":
        try:
            importlib.import_module("triton")
        except ImportError as exc:
            raise SystemExit("CUDA PyTorch requires its matching Triton dependency; rerun environment setup") from exc
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
    expected = requirement_pins(requirements)
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
        check_imports(requirements=args.fresh_pins or REQUIREMENTS)
        if args.fresh_pins:
            check_fresh_pins(args.fresh_pins)


if __name__ == "__main__":
    main()
