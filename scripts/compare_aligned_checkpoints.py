#!/usr/bin/env python3
"""Compare two trusted local smoke checkpoints without loading base models."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def differences(left, right, prefix=""):
    if isinstance(left, torch.Tensor):
        return [] if isinstance(right, torch.Tensor) and torch.equal(left, right) else [prefix]
    if isinstance(left, np.ndarray):
        return [] if isinstance(right, np.ndarray) and np.array_equal(left, right) else [prefix]
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            return [prefix + ".keys"]
        return [item for key in left for item in differences(left[key], right[key], f"{prefix}.{key}")]
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            return [prefix + ".length"]
        return [item for index, (x, y) in enumerate(zip(left, right))
                for item in differences(x, y, f"{prefix}.{index}")]
    return [] if left == right else [prefix]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"left": str(args.left.resolve()), "right": str(args.right.resolve()), "smoke_only": True}
    states = [json.loads((root / "aligned_state.json").read_text()) for root in (args.left, args.right)]
    assert all(state["smoke"] for state in states), "This comparison is for smoke artifacts only"
    keys = ("global_step", "sampler_epoch", "sampler_update_cursor", "config_fingerprint",
            "parameter_audit_sha256", "validation_loss", "world_size", "micro_batch_size", "gradient_accumulation_steps")
    report["metadata_differences"] = differences({k: states[0][k] for k in keys}, {k: states[1][k] for k in keys})
    adapters = [load_file(str(root / "transformer_lora/adapter_model.safetensors")) for root in (args.left, args.right)]
    report["adapter_tensor_count"] = len(adapters[0])
    report["adapter_differences"] = differences(*adapters)
    report["max_adapter_absolute_difference"] = max(
        (adapters[0][key] - adapters[1][key]).abs().max().item() for key in adapters[0]
    )
    for filename in ("optimizer.pt", "scheduler.pt", "rng-rank0.pt"):
        payload = [torch.load(root / filename, map_location="cpu", weights_only=False) for root in (args.left, args.right)]
        report[filename + "_differences"] = differences(*payload)
    report["bitwise_equal"] = not any(value for key, value in report.items() if key.endswith("_differences"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["bitwise_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
