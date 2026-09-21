import argparse
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

from convert_ckpt_to_hf_format import load_training_state_dict
from train import (
    _atomic_update_checkpoint_link,
    _checkpoint_link_target,
    _encode_vae_image,
    _make_csgo_seen10_dataset,
    _to_python_config_value,
)
from train_seen10 import _is_retryable_bootstrap_output, build_config
from omnigen2.dataset import csgo_seen10_dataset


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
    def test_config_conversion_accepts_omegaconf_and_python_values(self):
        self.assertEqual(_to_python_config_value(200704), 200704)
        self.assertEqual(_to_python_config_value([200704, 409600]), [200704, 409600])
        self.assertEqual(
            _to_python_config_value({"limits": [200704, 409600]}),
            {"limits": [200704, 409600]},
        )
        self.assertEqual(
            _to_python_config_value(OmegaConf.create({"limits": [200704, 409600]})),
            {"limits": [200704, 409600]},
        )
        self.assertEqual(
            _to_python_config_value(OmegaConf.create([200704, 409600])),
            [200704, 409600],
        )

    def test_seen10_dataset_accepts_scalar_max_input_pixels_from_omegaconf(self):
        args = SimpleNamespace(
            data=OmegaConf.create(
                {
                    "data_root": "/unused",
                    "train_split": "seen_train",
                    "max_input_pixels": 200704,
                }
            )
        )
        expected_dataset = object()

        with patch.object(
            csgo_seen10_dataset,
            "CSGOSeen10Dataset",
            return_value=expected_dataset,
        ) as constructor:
            result = _make_csgo_seen10_dataset(
                args,
                tokenizer=None,
                split="seen_train",
                load_target=True,
            )

        self.assertIs(result, expected_dataset)
        self.assertEqual(constructor.call_args.kwargs["max_input_pixels"], 200704)

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

            tokenizer_dir = train_root / "tokenizer"
            tokenizer_dir.mkdir()
            for name in (
                "added_tokens.json",
                "merges.txt",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ):
                (tokenizer_dir / name).write_text("{}", encoding="utf-8")

            text_encoder_dir = train_root / "text_encoder"
            text_encoder_dir.mkdir()
            for name in (
                "config.json",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json",
            ):
                (text_encoder_dir / name).write_text("bootstrap", encoding="utf-8")

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
            self.assertTrue(
                any(
                    "recognized tokenizer/text_encoder assets" in str(item.message)
                    for item in caught
                )
            )

            (train_root / "t_distribution.png").write_bytes(b"progress")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                build_config(namespace)

    def test_retryable_bootstrap_output_rejects_unknown_entries_symlinks_and_progress(self):
        config_path = Path(__file__).parents[1] / "options" / "csgo_seen10_lora.yml"

        def create_retryable_output(root):
            root.mkdir(parents=True)
            (root / config_path.name).write_bytes(config_path.read_bytes())
            logs = root / "logs"
            logs.mkdir()
            (logs / "failed-bootstrap.log").write_text("failed\n", encoding="utf-8")
            tokenizer_dir = root / "tokenizer"
            tokenizer_dir.mkdir()
            for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
                (tokenizer_dir / name).write_text("bootstrap", encoding="utf-8")
            text_encoder_dir = root / "text_encoder"
            text_encoder_dir.mkdir()
            for name in ("config.json", "model.safetensors", "model.safetensors.index.json"):
                (text_encoder_dir / name).write_text("bootstrap", encoding="utf-8")
            return root

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_output = create_retryable_output(root / "valid")
            self.assertTrue(_is_retryable_bootstrap_output(valid_output, config_path))

            invalid_outputs = []

            unknown_file = create_retryable_output(root / "unknown_file")
            (unknown_file / "tokenizer" / "custom.bin").write_bytes(b"unknown")
            invalid_outputs.append(unknown_file)

            unknown_directory = create_retryable_output(root / "unknown_directory")
            (unknown_directory / "text_encoder" / "optimizer").mkdir()
            invalid_outputs.append(unknown_directory)

            top_level_directory = create_retryable_output(root / "top_level_directory")
            (top_level_directory / "unexpected").mkdir()
            invalid_outputs.append(top_level_directory)

            symlink_output = create_retryable_output(root / "symlink")
            (symlink_output / "tokenizer" / "linked-tokenizer.json").symlink_to(
                symlink_output / "tokenizer" / "tokenizer.json"
            )
            invalid_outputs.append(symlink_output)

            root_symlink_target = create_retryable_output(root / "root_symlink_target")
            root_symlink = root / "root_symlink"
            root_symlink.symlink_to(root_symlink_target, target_is_directory=True)
            invalid_outputs.append(root_symlink)

            checkpoint_output = create_retryable_output(root / "checkpoint")
            (checkpoint_output / "checkpoint-10").mkdir()
            invalid_outputs.append(checkpoint_output)

            metrics_output = create_retryable_output(root / "metrics")
            (metrics_output / "train_metrics.jsonl").write_text("{}\n", encoding="utf-8")
            invalid_outputs.append(metrics_output)

            visualization_output = create_retryable_output(root / "visualization")
            (visualization_output / "t_distribution.png").write_bytes(b"progress")
            invalid_outputs.append(visualization_output)

            for output in invalid_outputs:
                with self.subTest(output=output.name):
                    self.assertFalse(_is_retryable_bootstrap_output(output, config_path))


if __name__ == "__main__":
    unittest.main()
