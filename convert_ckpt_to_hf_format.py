import argparse
import hashlib
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file as load_safetensors_file

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline


POSE_ADAPTER_CONFIG_NAME = "pose_adapter_config.json"
POSE_ADAPTER_WEIGHTS_NAME = "pose_adapter.bin"
ALIGNED_EXPERIMENT = "csgo_seen10_exp32gen_aligned"
ALIGNED_ADAPTER_CONFIG_NAME = "aligned_adapter_config.json"
ALIGNED_TARGET_MODULES = ("to_k", "to_q", "to_v", "to_out.0")
ALIGNED_LORA_MODULE_COUNT = 152
ALIGNED_LORA_TENSOR_COUNT = 304
ALIGNED_LORA_PARAMETER_COUNT = 5_107_200


def _expected_aligned_lora_shapes():
    expected = {}
    for stack, depth in (("noise_refiner", 2), ("ref_image_refiner", 2),
                         ("context_refiner", 2), ("layers", 32)):
        for index in range(depth):
            for target in ALIGNED_TARGET_MODULES:
                module = f"{stack}.{index}.attn.{target}"
                output_dim = 840 if target in ("to_k", "to_v") else 2520
                expected[f"{module}.lora_A.weight"] = (8, 2520)
                expected[f"{module}.lora_B.weight"] = (output_dim, 8)
    return expected


def _convert_aligned_checkpoint(conf, model_path, save_path, config_path, *, allow_smoke=False):
    """Export the aligned attention LoRA without constructing a pose-enabled model."""
    if conf.get("experiment") != ALIGNED_EXPERIMENT:
        raise ValueError("Aligned conversion requires the aligned experiment config")
    data = conf.data
    train = conf.train
    expected = {
        "reference_image_size": 224,
        "target_image_size": 448,
        "lora_rank": 8,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
    }
    observed = {
        "reference_image_size": data.get("reference_image_size"),
        "target_image_size": data.get("target_image_size"),
        "lora_rank": train.get("lora_rank"),
        "lora_alpha": train.get("lora_alpha"),
        "lora_dropout": train.get("lora_dropout"),
    }
    for key, value in expected.items():
        if observed[key] != value:
            raise ValueError(f"Aligned config {key} must be {value!r}, got {observed[key]!r}")
    if not train.get("lora_ft", False):
        raise ValueError("Aligned conversion requires LoRA fine-tuning")
    if bool(conf.model.arch_opt.get("pose_conditioning", False)) or bool(conf.model.get("pose_conditioning", False)):
        raise ValueError("Aligned conversion requires pose conditioning disabled")

    checkpoint = Path(model_path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Aligned conversion requires a checkpoint directory: {checkpoint}")
    if not (checkpoint / "COMPLETE").is_file():
        raise FileNotFoundError(f"Aligned checkpoint is not complete: {checkpoint / 'COMPLETE'}")
    for forbidden in (POSE_ADAPTER_CONFIG_NAME, POSE_ADAPTER_WEIGHTS_NAME):
        if (checkpoint / forbidden).exists():
            raise ValueError(f"Aligned checkpoint contains a pose adapter: {checkpoint / forbidden}")
    metadata_path = checkpoint / "aligned_state.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Aligned checkpoint metadata missing: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        checkpoint_metadata = json.load(metadata_file)
    if checkpoint_metadata.get("experiment", checkpoint_metadata.get("profile")) != ALIGNED_EXPERIMENT:
        raise ValueError("Checkpoint profile does not match the aligned experiment")
    if checkpoint_metadata.get("smoke") and not allow_smoke:
        raise ValueError("Refusing to convert a smoke checkpoint for formal inference; pass --allow-smoke")
    if checkpoint_metadata.get("global_step") is None:
        raise ValueError("Aligned checkpoint metadata is missing global_step")
    contract = checkpoint_metadata.get("contract_identity")
    if not isinstance(contract, dict) or contract.get("profile") != ALIGNED_EXPERIMENT:
        raise ValueError("Aligned checkpoint is missing its training contract identity")
    contract_fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if checkpoint_metadata.get("config_fingerprint") != contract_fingerprint:
        raise ValueError("Aligned checkpoint contract fingerprint is inconsistent")
    official_sources = {
        "pretrained_model_path": "OmniGen2/OmniGen2",
        "pretrained_vae_model_name_or_path": "black-forest-labs/FLUX.1-dev",
        "pretrained_text_encoder_model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",
    }
    sources = contract.get("base_sources")
    if not isinstance(sources, dict):
        raise ValueError("Aligned checkpoint is missing base source identities")
    for key, repo_id in official_sources.items():
        source = sources.get(key)
        if not isinstance(source, dict) or source.get("repo_id") != repo_id:
            raise ValueError(f"Aligned checkpoint base {key} is not the official {repo_id}")
        if not isinstance(source.get("revision"), str) or len(source["revision"]) != 40:
            raise ValueError(f"Aligned checkpoint base {key} has no pinned revision")

    adapter_dir = checkpoint / "transformer_lora"
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Aligned checkpoint LoRA directory missing: {adapter_dir}")
    for forbidden in (POSE_ADAPTER_CONFIG_NAME, POSE_ADAPTER_WEIGHTS_NAME):
        if (adapter_dir / forbidden).exists():
            raise ValueError(f"Aligned LoRA contains a pose adapter: {adapter_dir / forbidden}")
    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_weights_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_config_path.is_file() or not adapter_weights_path.is_file():
        raise FileNotFoundError(
            f"Aligned checkpoint requires adapter_config.json and adapter_model.safetensors in {adapter_dir}"
        )
    with adapter_config_path.open("r", encoding="utf-8") as adapter_config_file:
        adapter_config = json.load(adapter_config_file)
    if adapter_config.get("r") != 8 or adapter_config.get("lora_alpha") != 8:
        raise ValueError("Aligned LoRA adapter must use rank=alpha=8")
    if adapter_config.get("lora_dropout") != 0.0:
        raise ValueError("Aligned LoRA adapter must use dropout=0")
    if set(adapter_config.get("target_modules", ())) != set(ALIGNED_TARGET_MODULES):
        raise ValueError("Aligned LoRA adapter must target attention projections only")
    state_dict = load_safetensors_file(str(adapter_weights_path), device="cpu")
    expected_shapes = _expected_aligned_lora_shapes()
    if len(state_dict) != ALIGNED_LORA_TENSOR_COUNT:
        raise ValueError(
            f"Aligned LoRA needs {ALIGNED_LORA_TENSOR_COUNT} tensors; found {len(state_dict)}"
        )
    modules: dict[str, set[str]] = {}
    parameter_count = 0
    for name, tensor in state_dict.items():
        if name not in expected_shapes:
            raise ValueError(f"Aligned LoRA tensor is not an official attention target: {name}")
        if tuple(tensor.shape) != expected_shapes[name]:
            raise ValueError(
                f"Aligned LoRA tensor {name} has shape {tuple(tensor.shape)}, "
                f"expected {expected_shapes[name]}"
            )
        if "pose" in name.lower():
            raise ValueError(f"Aligned LoRA contains pose weights: {name}")
        if not (name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight")):
            raise ValueError(f"Aligned LoRA contains an unexpected tensor: {name}")
        if tensor.ndim != 2:
            raise ValueError(f"Aligned LoRA tensor {name} must be a matrix")
        kind = "A" if name.endswith(".lora_A.weight") else "B"
        module = name.removesuffix(f".lora_{kind}.weight")
        modules.setdefault(module, set()).add(kind)
        parameter_count += tensor.numel()
        rank = tensor.shape[0] if kind == "A" else tensor.shape[1]
        if rank != 8:
            raise ValueError(f"Aligned LoRA tensor {name} has rank {rank}, expected 8")
        if not any(f".{module}.lora_" in name for module in ALIGNED_TARGET_MODULES):
            raise ValueError(f"Aligned LoRA tensor targets a non-attention module: {name}")
    if len(modules) != ALIGNED_LORA_MODULE_COUNT or any(parts != {"A", "B"} for parts in modules.values()):
        raise ValueError(
            f"Aligned LoRA requires {ALIGNED_LORA_MODULE_COUNT} complete A/B attention modules; "
            f"found {len(modules)}"
        )
    if set(state_dict) != set(expected_shapes):
        missing = sorted(set(expected_shapes) - set(state_dict))
        raise ValueError(f"Aligned LoRA is missing official attention tensors: {missing[:3]}")
    if parameter_count != ALIGNED_LORA_PARAMETER_COUNT:
        raise ValueError(
            f"Aligned LoRA requires {ALIGNED_LORA_PARAMETER_COUNT} parameters; found {parameter_count}"
        )

    OmniGen2Pipeline.save_lora_weights(
        save_directory=save_path,
        transformer_lora_layers=state_dict,
    )
    output_config = {
        "experiment": ALIGNED_EXPERIMENT,
        **expected,
        "pose_conditioning": "text_only",
        "target_modules": list(ALIGNED_TARGET_MODULES),
        "global_step": checkpoint_metadata.get("global_step"),
        "smoke": bool(checkpoint_metadata.get("smoke", False)),
        "config_fingerprint": checkpoint_metadata.get("config_fingerprint"),
        "parameter_audit_sha256": checkpoint_metadata.get("parameter_audit_sha256"),
        "contract_identity": contract,
        "source_config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "source_checkpoint": str(checkpoint),
        "source_config": str(Path(config_path).expanduser().resolve()),
    }
    with open(os.path.join(save_path, ALIGNED_ADAPTER_CONFIG_NAME), "w", encoding="utf-8") as config_file:
        json.dump(output_config, config_file, indent=2, sort_keys=True)
        config_file.write("\n")


def load_training_state_dict(model_path):
    """Load either an Accelerate checkpoint directory or a state-dict file."""
    if os.path.isdir(model_path):
        candidates = (
            "model.safetensors",
            "pytorch_model_fsdp.bin",
            "pytorch_model.bin",
            "model.bin",
        )
        resolved_path = next(
            (
                os.path.join(model_path, filename)
                for filename in candidates
                if os.path.isfile(os.path.join(model_path, filename))
            ),
            None,
        )
        if resolved_path is None:
            raise FileNotFoundError(
                f"No supported model state was found in checkpoint directory {model_path}; "
                f"expected one of {candidates}"
            )
    else:
        resolved_path = model_path
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"Model checkpoint does not exist: {resolved_path}")

    if resolved_path.endswith(".safetensors"):
        return load_safetensors_file(resolved_path, device="cpu")
    return torch.load(resolved_path, mmap=True, map_location="cpu", weights_only=True)


def save_pose_adapter(transformer, save_directory):
    """Save the numeric-pose adapter beside a converted Transformer/LoRA."""
    pose_adapter = getattr(transformer, "pose_adapter", None)
    if pose_adapter is None:
        return None

    os.makedirs(save_directory, exist_ok=True)
    config = {
        "input_dim": pose_adapter.input_dim,
        "pose_hidden_dim": pose_adapter.hidden_dim,
        "output_dim": pose_adapter.output_dim,
    }
    state_dict = {
        name: value.detach().to(device="cpu")
        for name, value in pose_adapter.state_dict().items()
    }

    config_path = os.path.join(save_directory, POSE_ADAPTER_CONFIG_NAME)
    weights_path = os.path.join(save_directory, POSE_ADAPTER_WEIGHTS_NAME)
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")
    torch.save(state_dict, weights_path)
    return weights_path


def load_pose_adapter(transformer, load_directory):
    """Enable and load a pose adapter sidecar into a base Transformer.

    Returns ``None`` when the directory contains no pose sidecar and the
    Transformer has pose conditioning disabled. A pose-enabled Transformer
    requires the sidecar so callers cannot silently run with random weights.
    """
    config_path = os.path.join(load_directory, POSE_ADAPTER_CONFIG_NAME)
    weights_path = os.path.join(load_directory, POSE_ADAPTER_WEIGHTS_NAME)
    has_config = os.path.isfile(config_path)
    has_weights = os.path.isfile(weights_path)
    pose_adapter = getattr(transformer, "pose_adapter", None)

    if not has_config and not has_weights:
        if pose_adapter is not None:
            raise FileNotFoundError(
                f"pose conditioning is enabled, but no adapter sidecar exists in {load_directory}"
            )
        return None
    if has_config != has_weights:
        raise FileNotFoundError(
            f"incomplete pose adapter sidecar in {load_directory}; expected both "
            f"{POSE_ADAPTER_CONFIG_NAME} and {POSE_ADAPTER_WEIGHTS_NAME}"
        )

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if config.get("input_dim") != 5:
        raise ValueError(f"unsupported pose adapter input_dim: {config.get('input_dim')}")
    if config.get("output_dim") != transformer.config.text_feat_dim:
        raise ValueError(
            "pose adapter output width does not match the Transformer's text_feat_dim: "
            f"{config.get('output_dim')} != {transformer.config.text_feat_dim}"
        )

    hidden_dim = config.get("pose_hidden_dim", config.get("hidden_dim"))
    if hidden_dim is None:
        raise ValueError("pose adapter config is missing pose_hidden_dim")
    pose_adapter = transformer.enable_pose_conditioning(hidden_dim=hidden_dim)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    pose_adapter.load_state_dict(state_dict, strict=True)
    return pose_adapter


def main(args):
    import dotenv

    dotenv.load_dotenv(override=True)

    config_path = args.config_path
    model_path = args.model_path
    save_path = args.save_path
    if os.path.exists(save_path) and (
        not os.path.isdir(save_path) or any(os.scandir(save_path))
    ):
        raise FileExistsError(f"Refusing to overwrite non-empty conversion output: {save_path}")

    conf = OmegaConf.load(config_path)
    if conf.get("experiment") == ALIGNED_EXPERIMENT:
        _convert_aligned_checkpoint(
            conf, model_path, save_path, config_path,
            allow_smoke=getattr(args, "allow_smoke", False),
        )
        return
    arch_opt = conf.model.arch_opt

    arch_opt = OmegaConf.to_object(arch_opt)
    # Convert lists to tuples in conf.model.arch_opt
    for key, value in arch_opt.items():
        if isinstance(value, list):
            arch_opt[key] = tuple(value)

    with init_empty_weights():
        transformer = OmniGen2Transformer2DModel(**arch_opt)

        if conf.train.get('lora_ft', False):
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]

            # now we will add new LoRA weights the transformer layers
            lora_config = LoraConfig(
                r=conf.train.lora_rank,
                lora_alpha=conf.train.lora_rank,
                lora_dropout=conf.train.lora_dropout,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            transformer.add_adapter(lora_config)

    state_dict = load_training_state_dict(model_path)
    missing, unexpect = transformer.load_state_dict(
        state_dict, assign=True, strict=False
    )
    print(f"missed parameters: {missing}")
    print(f"unexpected parameters: {unexpect}")
    if missing or unexpect:
        raise RuntimeError(
            "Refusing to export an incomplete checkpoint: "
            f"missing={missing[:20]}, unexpected={unexpect[:20]}"
        )

    if conf.train.get('lora_ft', False):
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        OmniGen2Pipeline.save_lora_weights(
            save_directory=save_path,
            transformer_lora_layers=transformer_lora_layers,
        )
        save_pose_adapter(transformer, save_path)
    else:
        transformer.save_pretrained(save_path)
        save_pose_adapter(transformer, save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--allow-smoke", action="store_true", help="Allow conversion of an aligned smoke checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
