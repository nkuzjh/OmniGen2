import argparse
import os
import tempfile
import unittest
import warnings
from pathlib import Path

import torch
from safetensors.torch import save_file

from convert_ckpt_to_hf_format import load_training_state_dict
from train import (
    _atomic_update_checkpoint_link,
    _checkpoint_link_target,
    _encode_vae_image,
)
from train_seen10 import build_config


class _LatentDistribution:
    def __init__(self, tensor):
        self.tensor = tensor

    def sample(self):
        return self.tensor


class _FakeVAE:
    dtype = torch.float32

    class config:
        shift_factor = None
        scaling_factor = None

    def __init__(self):
        self.seen_shape = None

    def encode(self, image):
        self.seen_shape = tuple(image.shape)
        if image.ndim != 4:
            raise ValueError("expected NCHW")
        return type("Encoded", (), {"latent_dist": _LatentDistribution(image[:, :2])})()


class Seen10TrainingHelperTests(unittest.TestCase):
    def test_vae_helper_adds_single_image_batch_dimension(self):
        vae = _FakeVAE()
        result = _encode_vae_image(
            vae,
            torch.zeros(3, 448, 448),
            torch.float32,
            torch.device("cpu"),
        )
        self.assertEqual(vae.seen_shape, (1, 3, 448, 448))
        self.assertEqual(tuple(result.shape), (1, 2, 448, 448))
        with self.assertRaisesRegex(ValueError, "VAE input"):
            _encode_vae_image(
                vae,
                torch.zeros(2, 3, 448, 448),
                torch.float32,
                torch.device("cpu"),
            )

    def test_late_and_best_checkpoint_links_are_relative_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoint-2").mkdir()
            (root / "checkpoint-4").mkdir()
            _atomic_update_checkpoint_link(directory, "late", "checkpoint-2")
            _atomic_update_checkpoint_link(directory, "best", "checkpoint-2")
            _atomic_update_checkpoint_link(directory, "late", "checkpoint-4")
            self.assertEqual(os.readlink(root / "late"), "checkpoint-4")
            self.assertEqual(_checkpoint_link_target(directory, "late"), "checkpoint-4")
            self.assertEqual(_checkpoint_link_target(directory, "best"), "checkpoint-2")

    def test_accelerate_safetensors_checkpoint_directory_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = torch.arange(6, dtype=torch.float32).reshape(2, 3)
            save_file({"weight": expected}, str(Path(directory) / "model.safetensors"))
            actual = load_training_state_dict(directory)
            self.assertTrue(torch.equal(actual["weight"], expected))

    def test_launcher_forces_exactly_five_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(__file__).parents[1] / "options" / "csgo_seen10_lora.yml"
            namespace = argparse.Namespace(
                config=str(config_path),
                seed=3,
                output_root=directory,
                resume_from_checkpoint=None,
                pretrained_model_path="/models/omnigen2",
                pretrained_vae_model_path="/models/flux",
                pretrained_text_encoder_model_path="/models/qwen",
                max_train_steps=10,
                max_validation_batches=1,
                global_batch_size=None,
                batch_size=1,
                gradient_accumulation_steps=1,
                dataloader_num_workers=0,
            )
            config = build_config(namespace)
            self.assertEqual(config.logger.checkpointing_steps, 2)
            self.assertEqual(config.val.validation_steps, 2)
            self.assertEqual(config.output_dir, str(Path(directory).resolve() / "seed_3" / "train"))
            self.assertEqual(config.model.pretrained_model_path, "/models/omnigen2")
            self.assertEqual(config.model.pretrained_vae_model_name_or_path, "/models/flux")
            self.assertEqual(
                config.model.pretrained_text_encoder_model_name_or_path, "/models/qwen"
            )

    def test_launcher_retries_bootstrap_only_output_but_protects_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(__file__).parents[1] / "options" / "csgo_seen10_lora.yml"
            train_root = Path(directory) / "seed_0" / "train"
            logs = train_root / "logs"
            logs.mkdir(parents=True)
            (train_root / config_path.name).write_bytes(config_path.read_bytes())
            (logs / "failed-download.log").write_text("network failure\n", encoding="utf-8")
            namespace = argparse.Namespace(
                config=str(config_path),
                seed=0,
                output_root=directory,
                resume_from_checkpoint=None,
                pretrained_model_path=None,
                pretrained_vae_model_path=None,
                pretrained_text_encoder_model_path=None,
                max_train_steps=10,
                max_validation_batches=1,
                global_batch_size=None,
                batch_size=1,
                gradient_accumulation_steps=1,
                dataloader_num_workers=0,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                build_config(namespace)
            self.assertTrue(any("bootstrap config/log files" in str(item.message) for item in caught))

            (train_root / "t_distribution.png").write_bytes(b"progress")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                build_config(namespace)


if __name__ == "__main__":
    unittest.main()
