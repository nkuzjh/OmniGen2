"""Portable path selection and wrapper inspection tests; no model imports."""

from __future__ import annotations

import json
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import csgo_runtime_paths as paths


class RuntimePathTests(unittest.TestCase):
    def test_config_reader_rejects_unsupported_yaml_and_skips_low_priority_config(self):
        unsupported = (
            "data: {data_root: /tmp/data}\n",
            "data: &shared\n  data_root: /tmp/data\n",
            "data:\n  nested:\n    data_root: /tmp/data\n",
            "data:\n  data_root: ${oc.env:CSGO_DATA_ROOT}\n",
            "data:\n  data_root: >\n    /tmp/data\n",
            "data:\n  data_root: !!str /tmp/data\n",
        )
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "custom.yml"
            for source in unsupported:
                config.write_text(source)
                with self.subTest(source=source):
                    with self.assertRaisesRegex(ValueError, "pass --data-root"):
                        paths.config_data_root(config, env={})
                    self.assertIsNone(paths.config_data_root(config, explicit="cli data", env={}))
                    self.assertIsNone(paths.config_data_root(config,
                                                              env={"CSGO_DATA_ROOT": "env data"}))
            config.write_text('data:\n  data_root: "data with spaces" # comment\n')
            self.assertEqual(paths.config_data_root(config, env={}), "data with spaces")

    def test_training_config_overrides_only_data_root(self):
        from train_seen10 import build_config

        with tempfile.TemporaryDirectory() as temp:
            custom_config = Path(temp) / "custom.yml"
            original = (ROOT / "options/csgo_seen10_lora.yml").read_text()
            custom_config.write_text(original.replace(
                "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2", "custom data"
            ))
            namespace = argparse.Namespace(
                config=str(custom_config), experiment=None, seed=7, smoke=False, stop_after_updates=None,
                output_root=temp, data_root=None, resume_from_checkpoint=None,
                pretrained_model_path=None, pretrained_vae_model_path=None,
                pretrained_text_encoder_model_path=None, max_train_steps=None,
                max_validation_batches=None, global_batch_size=None, batch_size=None,
                micro_batch_size=None, gradient_accumulation_steps=None,
                dataloader_num_workers=None,
            )
            before = build_config(namespace)
            self.assertEqual(before.data.data_root, str(ROOT / "custom data"))
            namespace.data_root = "alternate data"
            after = build_config(namespace)
            self.assertEqual(after.data.data_root, str(ROOT / "alternate data"))
            self.assertEqual(after.seed, before.seed)
            self.assertEqual(after.output_dir, before.output_dir)
            self.assertEqual(after.train.max_train_steps, before.train.max_train_steps)
            self.assertEqual(after.train.global_batch_size, before.train.global_batch_size)

    def test_aligned_noncwd_config_and_output_paths_agree(self):
        with tempfile.TemporaryDirectory() as temp:
            custom_config = Path(temp) / "aligned config.yml"
            original = (ROOT / "options/csgo_seen10_exp32gen_aligned.yml").read_text()
            custom_config.write_text(original.replace(
                "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2", "custom data"
            ))
            env = os.environ.copy()
            for name in ("CSGO_DATA_ROOT", "CSGO_BENCHMARK_V2_DATA", "DATA_ROOT"):
                env.pop(name, None)
            env["OMNIGEN2_MODEL_PATH"] = "./models/local"
            command = ["bash", str(ROOT / "scripts/run_csgo_seen10.sh"), "all", "--experiment",
                       "csgo_seen10_exp32gen_aligned", "--config", str(custom_config),
                       "--output-root", "aligned_smoke/portable", "--smoke",
                       "--stop-after-updates", "1", "--dry-run"]
            result = subprocess.run(command, cwd="/", env=env, text=True, capture_output=True, check=True)
            train, convert, infer, evaluation = json.loads(result.stdout)["commands"][:4]
            output = str(ROOT / "aligned_smoke/portable")
            self.assertEqual(train[train.index("--output-root") + 1], output)
            self.assertEqual(train[train.index("--config") + 1], str(custom_config))
            self.assertEqual(train[train.index("--data-root") + 1], str(ROOT / "custom data"))
            self.assertEqual(train[train.index("--pretrained-model-path") + 1], str(ROOT / "models/local"))
            self.assertEqual(convert[convert.index("--model_path") + 1], output + "/seed_42/train/late")
            self.assertEqual(infer[infer.index("--output-root") + 1], output + "/seed_42/predictions/late")
            self.assertEqual(evaluation[evaluation.index("--pred-root") + 1],
                             output + "/seed_42/predictions/late/discrete")

    def test_priority_relative_and_spaces(self):
        with tempfile.TemporaryDirectory(prefix="csgo paths ") as temp:
            root = Path(temp) / "checkout"
            root.mkdir()
            cli = root / "cli data"
            cli.mkdir()
            env_data = root / "env data"
            env_data.mkdir()
            env = {"CSGO_DATA_ROOT": "env data", "DATA_ROOT": "lower priority"}
            self.assertEqual(paths.data_root("cli data", root=root, env=env).path, cli)
            self.assertEqual(paths.data_root(root=root, env=env).path, env_data)
            self.assertEqual(paths.data_root(root=root, env=env).source, "CSGO_DATA_ROOT")
            self.assertEqual(paths.data_root(config_value="configured data", root=root, env={}).path,
                             root / "configured data")
            self.assertEqual(paths.data_root(config_value="configured data", root=root, env=env).path,
                             env_data)
            absent = paths.data_root("missing path", root=root, env=env)
            self.assertEqual(absent.path, root / "missing path")
            self.assertFalse(absent.ready)
            self.assertEqual(paths.model_source("Org/model", root), "Org/model")
            self.assertEqual(paths.model_source("./models/local", root), str(root / "models/local"))

    def test_fallbacks_and_eval_python_priority(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "OmniGen2"
            root.mkdir()
            sibling_data = root.parent / "UniLIP/data/csgo_benchmark_v2"
            sibling_data.mkdir(parents=True)
            sibling_eval = root.parent / "csgo_benchmark_v2_eval_general"
            sibling_eval.mkdir()
            with patch.object(paths, "LEGACY_DATA_ROOT", root / "absent data"), patch.object(
                paths, "LEGACY_EVAL_ROOT", root / "absent eval"
            ):
                self.assertEqual(paths.data_root(root=root, env={}).path, sibling_data)
                self.assertEqual(paths.eval_root(root=root, env={}).path, sibling_eval)

            env_python = root / "env python"
            env_python.write_text("#!/bin/sh\n")
            env_python.chmod(0o755)
            env = {"EVAL_PYTHON": "env python", "UNILIP_PYTHON": "other python"}
            self.assertEqual(paths.eval_python(sibling_eval, root=root, env=env).path, env_python)
            shared = sibling_eval / ".venv/bin/python"
            shared.parent.mkdir(parents=True)
            shared.symlink_to(sys.executable)
            selected = paths.eval_python(sibling_eval, root=root, env=env)
            self.assertEqual(selected.path, shared)
            self.assertEqual(selected.source, "shared evaluator .venv")
            self.assertEqual(paths.eval_python(sibling_eval, "env python", root=root, env=env).path, env_python)
            self.assertEqual(paths.model_python(root=root, env={}).path, root / ".venv/bin/python")

    def test_wrapper_print_paths_from_other_cwd_without_eval_env(self):
        with tempfile.TemporaryDirectory() as temp:
            missing = "missing data"
            env = os.environ.copy()
            env["CSGO_DATA_ROOT"] = missing
            env["SHARED_EVAL_DIR"] = "missing evaluator"
            env["EVAL_PYTHON"] = "missing eval python"
            env["OMNIGEN2_PYTHON"] = "missing model python"
            command = ["bash", str(ROOT / "scripts/run_csgo_seen10.sh"), "train", "--print-paths"]
            result = subprocess.run(command, cwd=temp, env=env, text=True, capture_output=True, check=True)
            report = json.loads(result.stdout)
            self.assertEqual(report["paths"]["data_root"]["path"], str(ROOT / missing))
            self.assertEqual(report["paths"]["data_root"]["source"], "CSGO_DATA_ROOT")
            self.assertFalse(report["paths"]["eval_python"]["ready"])
            self.assertFalse(report["paths"]["model_python"]["ready"])

    def test_wrapper_uses_explicit_python_without_python3_on_path(self):
        with tempfile.TemporaryDirectory() as temp:
            dirname = shutil.which("dirname")
            self.assertIsNotNone(dirname)
            Path(temp, "dirname").symlink_to(dirname)
            env = os.environ.copy()
            env["PATH"] = temp
            env["OMNIGEN2_PYTHON"] = sys.executable
            result = subprocess.run(["/bin/bash", str(ROOT / "scripts/run_csgo_seen10.sh"),
                                     "train", "--print-paths"], cwd="/", env=env,
                                    text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(result.stdout)["paths"]["model_python"]["path"],
                             sys.executable)
            Path(temp, "python").symlink_to(sys.executable)
            env["HOME"] = temp
            env["OMNIGEN2_PYTHON"] = "~/python"
            result = subprocess.run(["/bin/bash", str(ROOT / "scripts/run_csgo_seen10.sh"),
                                     "train", "--print-paths"], cwd="/", env=env,
                                    text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(result.stdout)["paths"]["model_python"]["path"],
                             str(Path(temp) / "python"))

    def test_aligned_dry_run_multigpu_and_legacy_command(self):
        env = os.environ.copy()
        env["CSGO_DATA_ROOT"] = "data with spaces"
        env["EVAL_PYTHON"] = "absent eval python"
        command = ["bash", str(ROOT / "scripts/run_csgo_seen10.sh"), "train", "--experiment",
                   "csgo_seen10_exp32gen_aligned", "--num-processes", "2",
                   "--gradient-accumulation-steps", "64", "--dry-run"]
        result = subprocess.run(command, cwd="/", env=env, text=True, capture_output=True, check=True)
        train = json.loads(result.stdout)["commands"][0]
        self.assertIn("--multi_gpu", train)
        self.assertEqual(train[train.index("--data-root") + 1], str(ROOT / "data with spaces"))
        self.assertEqual(train[train.index("--gradient-accumulation-steps") + 1], "64")

        legacy = subprocess.run(["bash", str(ROOT / "scripts/run_csgo_seen10.sh"),
                                 "infer", "--print-paths"], cwd="/", env=env,
                                text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(legacy.stdout)["action"], "infer")


if __name__ == "__main__":
    unittest.main()
