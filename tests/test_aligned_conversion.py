import json
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from convert_ckpt_to_hf_format import _convert_aligned_checkpoint, _expected_aligned_lora_shapes


class TestAlignedConversion(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.checkpoint = self.root / "checkpoint-100"
        adapter_dir = self.checkpoint / "transformer_lora"
        adapter_dir.mkdir(parents=True)
        (self.checkpoint / "COMPLETE").write_text("", encoding="utf-8")
        self.contract = {
            "profile": "csgo_seen10_exp32gen_aligned",
            "base_sources": {
                key: {"repo_id": repo, "revision": "a" * 40, "snapshot_path": "/cached/snapshot"}
                for key, repo in {
                    "pretrained_model_path": "OmniGen2/OmniGen2",
                    "pretrained_vae_model_name_or_path": "black-forest-labs/FLUX.1-dev",
                    "pretrained_text_encoder_model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",
                }.items()
            },
        }
        self.fingerprint = hashlib.sha256(json.dumps(self.contract, sort_keys=True).encode()).hexdigest()
        (self.checkpoint / "aligned_state.json").write_text(
            json.dumps({"profile": "csgo_seen10_exp32gen_aligned", "global_step": 100,
                        "smoke": False, "contract_identity": self.contract,
                        "config_fingerprint": self.fingerprint}),
            encoding="utf-8",
        )
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({
                "r": 8, "lora_alpha": 8, "lora_dropout": 0.0,
                "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
            }),
            encoding="utf-8",
        )
        tensors = {
            name: torch.zeros(shape)
            for name, shape in _expected_aligned_lora_shapes().items()
        }
        save_file(tensors, adapter_dir / "adapter_model.safetensors")
        self.conf = OmegaConf.create({
            "experiment": "csgo_seen10_exp32gen_aligned",
            "data": {"reference_image_size": 224, "target_image_size": 448},
            "train": {"lora_ft": True, "lora_rank": 8, "lora_alpha": 8, "lora_dropout": 0.0},
            "model": {"arch_opt": {"pose_conditioning": False}, "pose_conditioning": False},
        })
        (self.root / "config.yml").write_text("experiment: csgo_seen10_exp32gen_aligned\n", encoding="utf-8")

    def test_exports_adapter_only_with_aligned_metadata(self):
        destination = self.root / "converted"
        destination.mkdir()
        with patch("convert_ckpt_to_hf_format.OmniGen2Pipeline.save_lora_weights") as save:
            _convert_aligned_checkpoint(self.conf, self.checkpoint, destination, self.root / "config.yml")
        save.assert_called_once()
        metadata = json.loads((destination / "aligned_adapter_config.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["reference_image_size"], 224)
        self.assertEqual(metadata["pose_conditioning"], "text_only")
        self.assertFalse((destination / "pose_adapter.bin").exists())

    def test_rejects_smoke_without_explicit_permission(self):
        (self.checkpoint / "aligned_state.json").write_text(
            json.dumps({"profile": "csgo_seen10_exp32gen_aligned", "global_step": 100,
                        "smoke": True, "contract_identity": self.contract,
                        "config_fingerprint": self.fingerprint}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "smoke checkpoint"):
            _convert_aligned_checkpoint(self.conf, self.checkpoint, self.root / "converted", self.root / "config.yml")

    def test_rejects_pose_sidecar(self):
        (self.checkpoint / "pose_adapter.bin").write_bytes(b"pose")
        with self.assertRaisesRegex(ValueError, "pose adapter"):
            _convert_aligned_checkpoint(self.conf, self.checkpoint, self.root / "converted", self.root / "config.yml")

    def test_rejects_wrong_module_even_with_correct_count_and_numel(self):
        adapter_path = self.checkpoint / "transformer_lora" / "adapter_model.safetensors"
        from safetensors.torch import load_file
        tensors = load_file(adapter_path)
        tensor = tensors.pop("layers.0.attn.to_q.lora_A.weight")
        tensors["transformer_blocks.0.attn.to_q.lora_A.weight"] = tensor
        save_file(tensors, adapter_path)
        with self.assertRaisesRegex(ValueError, "not an official attention target"):
            _convert_aligned_checkpoint(self.conf, self.checkpoint, self.root / "converted", self.root / "config.yml")


if __name__ == "__main__":
    unittest.main()
