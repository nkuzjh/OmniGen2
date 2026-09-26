import copy
import math
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

from omnigen2.aligned_training import (
    EXECUTION_POLICY,
    GLOBAL_BATCH,
    MILESTONES,
    WarmupConstant,
    _EpochMicrobatches,
    _configure_execution_policy,
    _contract_identity,
    _code_fingerprint,
    _recover_history,
    _SeededDataset,
    _resolve_resume,
    _save_checkpoint,
    _set_rng_state,
    epoch_batch_indices,
    validate_config,
)
from omnigen2.optim.scheduler.step_lr import StepLRScheduler
from train_seen10 import _prepare_cli_environment


class _FakeAccelerator:
    is_main_process = True
    process_index = 0
    num_processes = 1

    def wait_for_everyone(self):
        pass

    def unwrap_model(self, model):
        return model


def _model():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    return get_peft_model(model, LoraConfig(r=2, lora_alpha=2, lora_dropout=0,
                                           target_modules=["0", "1"]))


def _update(model, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(5, 3)
    model(x).square().sum().backward()
    optimizer.step()
    scheduler.step()


class AlignedTrainingTests(unittest.TestCase):
    def test_deterministic_policy_pins_native_norm_configs_and_clears_cache(self):
        from omnigen2.ops import triton as triton_package

        def kernel():
            return SimpleNamespace(configs=[SimpleNamespace(num_warps=n) for n in (1, 4, 8)],
                                   cache={"old shape": "timing-selected config"})

        forward, backward = kernel(), kernel()
        fake_norm = SimpleNamespace(_layer_norm_fwd_1pass_kernel=forward,
                                    _layer_norm_bwd_kernel=backward)
        old_flags = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                     torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            with patch.object(triton_package, "layer_norm", fake_norm), \
                 patch.object(torch, "use_deterministic_algorithms") as set_deterministic, \
                 patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": "bad"}):
                policy = _configure_execution_policy()
                set_deterministic.assert_called_once_with(True, warn_only=False)
                self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
                self.assertFalse(torch.backends.cudnn.benchmark)
                self.assertTrue(torch.backends.cudnn.deterministic)
                self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                self.assertFalse(torch.backends.cudnn.allow_tf32)
                self.assertEqual(policy, EXECUTION_POLICY)
                for selected in (forward, backward):
                    self.assertEqual([item.num_warps for item in selected.configs], [4])
                    self.assertEqual(selected.cache, {})
        finally:
            (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) = old_flags

    def test_legacy_launcher_does_not_set_aligned_cublas_policy(self):
        with patch.dict(os.environ, {}, clear=True):
            _prepare_cli_environment(SimpleNamespace(experiment=None))
            self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", os.environ)
            _prepare_cli_environment(SimpleNamespace(experiment="csgo_seen10_exp32gen_aligned"))
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

    def test_code_fingerprint_includes_native_norm_source_and_contract_policy(self):
        config = OmegaConf.load(Path(__file__).parents[1] / "options" / "csgo_seen10_exp32gen_aligned.yml")
        contract = _contract_identity(config)
        self.assertEqual(contract["code_fingerprint"], _code_fingerprint())
        self.assertEqual(contract["execution_policy"], EXECUTION_POLICY)
        self.assertEqual(contract["validation_seed"], 4242)

    def test_final_checkpoint_and_best_tie_recover_from_complete_state(self):
        self.assertEqual(MILESTONES, (4000, 8000, 12000, 16000, 19500))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            config = OmegaConf.create({"output_dir": directory,
                "train": {"batch_size": 1, "gradient_accumulation_steps": 128},
                "val": {"seed": 4242},
                "smoke": False, "resume_from_checkpoint": "latest"})
            model = _model()
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=8e-7)
            scheduler = WarmupConstant(optimizer)
            accelerator = _FakeAccelerator()
            with patch("omnigen2.aligned_training._config_fingerprint", return_value="fingerprint"), \
                 patch("omnigen2.aligned_training._contract_identity", return_value={"fixed": True}):
                _save_checkpoint(accelerator, config, model, optimizer, scheduler,
                                 4000, 10, 94, 0.5, "audit", False)
                _save_checkpoint(accelerator, config, model, optimizer, scheduler,
                                 19500, 50, 0, 0.5, "audit", False)
                self.assertEqual(os.readlink(root / "latest"), "checkpoint-19500")
                self.assertEqual(os.readlink(root / "late"), "checkpoint-19500")
                self.assertFalse((root / "checkpoint-20000").exists())
                (root / "latest").unlink()
                (root / "latest").symlink_to("checkpoint-4000")
                path, state = _resolve_resume(config)
                self.assertEqual(path.name, "checkpoint-19500")
                self.assertEqual(state["global_step"], 19500)
                metrics = root / "logs" / "train_metrics.jsonl"
                metrics.write_text('{"step":4000,"loss":1}\n{"step":', encoding="utf-8")
                self.assertEqual(_recover_history(root, 19500, accelerator), 0.5)
                self.assertEqual(os.readlink(root / "best"), "checkpoint-4000")
                self.assertEqual(metrics.read_text(encoding="utf-8"), '{"step":4000,"loss":1}\n')

    def test_batch_plan_never_uses_short_tail_or_duplicates_within_update(self):
        length = 389  # Three full 128-record updates plus five unused records.
        all_steps = []
        for update in range(length // GLOBAL_BATCH):
            rank_batches = [epoch_batch_indices(length, 17, 0, update, rank, 4)
                            for rank in range(4)]
            flat = [index for rank_batch in rank_batches for index in rank_batch]
            self.assertEqual(len(flat), 128)
            self.assertEqual(len(set(flat)), 128)
            all_steps.extend(flat)
        self.assertEqual(len(set(all_steps)), 384)
        self.assertEqual(epoch_batch_indices(length, 17, 0, 1, 2, 4),
                         epoch_batch_indices(length, 17, 0, 1, 2, 4))
        batches = list(_EpochMicrobatches(length, 17, 0, 1, 2, 4, 4))
        self.assertEqual(len(batches), 16)
        self.assertEqual([index for batch in batches[:8] for _, index in batch],
                         epoch_batch_indices(length, 17, 0, 1, 2, 4))

    def test_worker_sample_is_repeatable_without_changing_parent_rng(self):
        class Source:
            def __len__(self):
                return 2

            def __getitem__(self, index):
                return random.random(), np.random.rand(), torch.rand(1).item(), index

        source = _SeededDataset(Source(), 42)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        first = source[(3, 1)]
        self.assertEqual(first, source[(3, 1)])
        self.assertNotEqual(first, source[(4, 1)])
        self.assertEqual(random.getstate(), before[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], before[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))

    def test_warmup_matches_repo_official_scheduler_at_each_update_boundary(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=8e-7)
        official = StepLRScheduler(optimizer, decay_t=1, decay_rate=1,
                                   warmup_t=500, warmup_lr_init=1e-18,
                                   warmup_prefix=True, t_in_epochs=False)
        for update in (1, 2, 250, 500, 501, 502, 19500):
            self.assertTrue(math.isclose(WarmupConstant.rate(update), official._get_lr(update - 1)[0],
                                         rel_tol=1e-12, abs_tol=1e-25))

    def test_config_batch_factor_and_protocol_guards(self):
        path = Path(__file__).parents[1] / "options" / "csgo_seen10_exp32gen_aligned.yml"
        config = OmegaConf.load(path)
        config.smoke = False
        config.output_dir = "/tmp/formal/seed_42/train"
        self.assertEqual(validate_config(config, 1), (1, 128, 128))
        config.train.batch_size = 2
        config.train.gradient_accumulation_steps = 64
        self.assertEqual(validate_config(config, 1), (2, 64, 128))
        config.train.batch_size = 3
        with self.assertRaisesRegex(ValueError, "Effective batch"):
            validate_config(config, 1)
        config.train.batch_size = 2
        config.data.validation_split = "seen_discrete_test"
        with self.assertRaisesRegex(ValueError, "split protocol"):
            validate_config(config, 1)

    def test_adapter_optimizer_scheduler_and_rng_resume_match_continuous(self):
        torch.manual_seed(9)
        model = _model()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=8e-7)
        scheduler = WarmupConstant(optimizer)
        _update(model, optimizer, scheduler)
        saved_model = copy.deepcopy(model)
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        saved_scheduler = scheduler.state_dict()
        saved_rng = torch.get_rng_state().clone()
        _update(model, optimizer, scheduler)
        continuous = get_peft_model_state_dict(model, adapter_name="default")

        with tempfile.TemporaryDirectory() as directory:
            config = OmegaConf.create({"output_dir": directory,
                "train": {"batch_size": 1, "gradient_accumulation_steps": 128},
                "val": {"seed": 4242},
                "smoke": True})
            checkpoint_optimizer = _optimizer_for_state(saved_model, saved_optimizer)
            checkpoint_scheduler = WarmupConstant(checkpoint_optimizer)
            checkpoint_scheduler.load_state_dict(saved_scheduler)
            with patch("omnigen2.aligned_training._config_fingerprint", return_value="fingerprint"), \
                 patch("omnigen2.aligned_training._contract_identity", return_value={"fixed": True}):
                torch.set_rng_state(saved_rng)
                _save_checkpoint(_FakeAccelerator(), config, saved_model,
                    optimizer=checkpoint_optimizer, scheduler=checkpoint_scheduler,
                    step=1, epoch=0, cursor=1, val_loss=1.0, audit_hash="audit", topology_changed=False)
                config.resume_from_checkpoint = "latest"
                path, state = _resolve_resume(config)
                self.assertEqual(state["global_step"], 1)
                self.assertTrue((path / "COMPLETE").is_file())
                torch.manual_seed(9)
                resumed = _model()
                result = set_peft_model_state_dict(resumed,
                    load_file(path / "transformer_lora" / "adapter_model.safetensors"), adapter_name="default")
                self.assertFalse([key for key in result.missing_keys if "lora_" in key])
                resumed_optimizer = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=8e-7)
                resumed_optimizer.load_state_dict(torch.load(path / "optimizer.pt", weights_only=False))
                resumed_scheduler = WarmupConstant(resumed_optimizer)
                resumed_scheduler.load_state_dict(torch.load(path / "scheduler.pt", weights_only=False))
                _set_rng_state(torch.load(path / "rng-rank0.pt", weights_only=False))
                _update(resumed, resumed_optimizer, resumed_scheduler)
                for key, expected in continuous.items():
                    self.assertTrue(torch.equal(expected, get_peft_model_state_dict(resumed)[key]), key)
                self.assertEqual(resumed_scheduler.next_update, scheduler.next_update)


def _optimizer_for_state(model, state):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=8e-7)
    optimizer.load_state_dict(state)
    return optimizer
