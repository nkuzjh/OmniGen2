import argparse
import json
import os

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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
