#!/usr/bin/env python3
"""Run a CPU-sized, real-data Seen-10 plumbing smoke test.

This deliberately does not claim to test the released 29 GiB weights.  It uses
the production dataset adapter and production OmniGen2 Transformer class with a
tiny configuration, then exercises forward/backward, checkpoint reload and the
strict JPEG identity expected by the shared evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from omnigen2.dataset.csgo_seen10_dataset import CSGOSeen10Collator, CSGOSeen10Dataset
from omnigen2.models.transformers import block_lumina2, transformer_omnigen2
from omnigen2.models.transformers.components import swiglu as torch_swiglu
from omnigen2.models.transformers.repo import OmniGen2RotaryPosEmbed
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/home/jiahao/task/UniLIP/data/csgo_benchmark_v2")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "csgo_benchmark_v2_seen10" / "OmniGen2"


class _SmokeTokenizer:
    """Small deterministic tokenizer sufficient for the real dataset collator."""

    def __call__(
        self,
        texts,
        *,
        padding="longest",
        max_length=16,
        truncation=True,
        return_tensors="pt",
    ):
        del padding, truncation, return_tensors
        token_rows = []
        for text in texts:
            words = text.split()[:max_length]
            token_rows.append([1 + sum(word.encode("utf-8")) % 251 for word in words] or [1])
        width = max(len(row) for row in token_rows)
        ids = torch.zeros(len(token_rows), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(token_rows):
            ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[index, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def _disable_gpu_only_fast_paths() -> None:
    # The released implementation selects Triton kernels whenever installed;
    # this smoke intentionally runs on CPU and tests the same PyTorch modules.
    block_lumina2.RMSNorm = torch.nn.RMSNorm
    transformer_omnigen2.RMSNorm = torch.nn.RMSNorm
    block_lumina2.swiglu = torch_swiglu

    def _no_flash_attention():
        raise ImportError("CPU Seen-10 smoke uses the PyTorch attention processor")

    transformer_omnigen2.OmniGen2AttnProcessorFlash2Varlen = _no_flash_attention


def _make_tiny_transformer() -> OmniGen2Transformer2DModel:
    _disable_gpu_only_fast_paths()
    model = OmniGen2Transformer2DModel(
        patch_size=2,
        in_channels=2,
        hidden_size=32,
        num_layers=1,
        num_refiner_layers=0,
        num_attention_heads=4,
        num_kv_heads=2,
        multiple_of=8,
        norm_eps=1e-5,
        axes_dim_rope=(2, 2, 4),
        axes_lens=(64, 64, 64),
        text_feat_dim=16,
        timestep_scale=1.0,
        pose_conditioning=True,
        pose_input_dim=5,
        pose_hidden_dim=8,
    )
    # OmniGen2 intentionally zero-initializes modulation/output projections.
    # Make the tiny smoke non-degenerate so pose gradients and a model-derived
    # image can be asserted after just one optimizer step.
    with torch.no_grad():
        for block in model.layers:
            block.norm1.linear.weight.normal_(mean=0.0, std=0.02)
            block.norm1.linear.bias.normal_(mean=0.0, std=0.02)
        model.norm_out.linear_1.weight.normal_(mean=0.0, std=0.02)
        model.norm_out.linear_1.bias.normal_(mean=0.0, std=0.02)
        model.norm_out.linear_2.weight.normal_(mean=0.0, std=0.02)
    return model


def _image_latent(image: torch.Tensor) -> torch.Tensor:
    if tuple(image.shape) != (3, 448, 448):
        raise ValueError(f"Expected a 3x448x448 processed image, got {tuple(image.shape)}")
    return F.interpolate(
        image[:2].unsqueeze(0), size=(8, 8), mode="bilinear", align_corners=False
    )


def _text_features(text_ids: torch.Tensor) -> torch.Tensor:
    scales = torch.arange(1, 17, dtype=torch.float32).view(1, 1, 16)
    return torch.sin(text_ids.to(dtype=torch.float32).unsqueeze(-1) * scales / 97.0)


def _forward(model, sample, tokenizer, freqs_cis):
    tokens = tokenizer([sample["instruction"]], max_length=16)
    radar_latent = _image_latent(sample["input_images"][0])
    # The initial noise is deterministic and independent from the unavailable
    # test target. Radar and numeric pose are the only sample conditions.
    hidden_states = torch.zeros_like(radar_latent)
    return model(
        hidden_states=hidden_states,
        timestep=torch.full((1,), 0.5),
        text_hidden_states=_text_features(tokens["input_ids"]),
        freqs_cis=freqs_cis,
        text_attention_mask=tokens["attention_mask"],
        ref_image_hidden_states=[[radar_latent.squeeze(0)]],
        pose_values=sample["pose_values"].reshape(1, 5),
    )


def _atomic_jpeg(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        image.convert("RGB").save(temporary, format="JPEG")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    torch.manual_seed(args.seed)
    data_root = Path(args.data_root).expanduser().resolve()
    seed_root = Path(args.output_root).expanduser().resolve() / f"seed_{args.seed}"
    smoke_root = seed_root / "smoke"
    if smoke_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite smoke output: {smoke_root}. Move it aside before rerunning."
        )

    seed_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".smoke-staging-", dir=seed_root))
    try:
        tokenizer = _SmokeTokenizer()
        train_dataset = CSGOSeen10Dataset(
            data_root,
            "seen_train",
            tokenizer=tokenizer,
            use_chat_template=False,
            load_target=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=CSGOSeen10Collator(tokenizer, max_token_len=16),
        )
        train_batch = next(iter(train_loader))
        if tuple(train_batch["pose_values"].shape) != (1, 5):
            raise RuntimeError("Dataset/collator smoke did not produce one normalized 5DoF row")

        model = _make_tiny_transformer()
        freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
            model.config.axes_dim_rope, model.config.axes_lens, theta=10000
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        train_sample = train_dataset[0]
        target_latent = _image_latent(train_sample["output_image"])
        prediction = _forward(model, train_sample, tokenizer, freqs_cis)
        loss = F.mse_loss(prediction, target_latent)
        loss.backward()
        pose_gradient = model.pose_adapter.network[0].weight.grad
        if pose_gradient is None or not torch.isfinite(pose_gradient).all() or pose_gradient.abs().sum() == 0:
            raise RuntimeError("Numeric 5DoF adapter did not receive a finite, non-zero gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        checkpoint_path = staging_root / "tiny_checkpoint.pt"
        checkpoint_tmp = staging_root / ".tiny_checkpoint.pt.tmp"
        torch.save(
            {
                "model": model.state_dict(),
                "seed": args.seed,
                "train_sample_id": train_sample["sample_id"],
                "smoke_only": True,
            },
            checkpoint_tmp,
        )
        os.replace(checkpoint_tmp, checkpoint_path)

        reloaded = _make_tiny_transformer()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        reloaded.load_state_dict(checkpoint["model"], strict=True)
        reloaded.eval()

        # load_target=False is the production inference safety boundary: the
        # target path is metadata only and no target/neighbor frame is opened.
        discrete_dataset = CSGOSeen10Dataset(
            data_root,
            "seen_discrete_test",
            tokenizer=tokenizer,
            use_chat_template=False,
            load_target=False,
        )
        discrete_sample = discrete_dataset[0]
        if discrete_sample["output_image"] is not None or discrete_sample["output_image_path"] is not None:
            raise RuntimeError("Inference dataset unexpectedly loaded a target image")
        with torch.no_grad():
            generated_latent = _forward(reloaded, discrete_sample, tokenizer, freqs_cis)[0]
        third_channel = generated_latent.mean(dim=0, keepdim=True)
        generated_rgb = torch.sigmoid(torch.cat((generated_latent, third_channel), dim=0))
        generated_rgb = F.interpolate(
            generated_rgb.unsqueeze(0),
            size=(448, 448),
            mode="bilinear",
            align_corners=False,
        )[0]
        generated_uint8 = (
            generated_rgb.mul(255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        )
        image = Image.fromarray(generated_uint8)
        prediction_path = (
            staging_root
            / "discrete"
            / "gen_imgs"
            / discrete_sample["map_name"]
            / f"{discrete_sample['file_frame']}.jpg"
        )
        _atomic_jpeg(image, prediction_path)
        with Image.open(prediction_path) as check_image:
            if check_image.format != "JPEG" or check_image.mode != "RGB" or check_image.size != (448, 448):
                raise RuntimeError(
                    "Smoke output is not an exact 448x448 RGB JPEG: "
                    f"{check_image.format}, {check_image.mode}, {check_image.size}"
                )

        report = {
            "smoke_only": True,
            "formal_result": False,
            "seed": args.seed,
            "dataset_batch": {
                "train_sample_id": train_sample["sample_id"],
                "radar_shape": list(train_sample["input_images"][0].shape),
                "target_shape": list(train_sample["output_image"].shape),
                "pose_shape": list(train_batch["pose_values"].shape),
            },
            "forward_backward": {
                "loss": float(loss.detach()),
                "pose_gradient_l1": float(pose_gradient.detach().abs().sum()),
            },
            "checkpoint": {
                "path": str(smoke_root / checkpoint_path.relative_to(staging_root)),
                "sha256": _sha256(checkpoint_path),
                "reloaded_strict": True,
            },
            "prediction": {
                "sample_id": discrete_sample["sample_id"],
                "path": str(smoke_root / prediction_path.relative_to(staging_root)),
                "sha256": _sha256(prediction_path),
                "size": [448, 448],
                "mode": "RGB",
                "format": "JPEG",
                "target_loaded": False,
            },
        }
        report_path = staging_root / "smoke_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging_root, smoke_root)
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
