"""CPU-only tests of command isolation; these never launch training."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aligned_launcher", ROOT / "scripts/run_csgo_aligned.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
PROFILE = ["--experiment", launcher.EXPERIMENT]


@pytest.mark.parametrize("world,micro,accum", [(1, 1, 128), (1, 8, 16), (2, 4, 16), (4, 4, 8), (8, 4, 4)])
def test_batch_product_only(world, micro, accum):
    args = launcher.parse_args(["train", *PROFILE, "--num-processes", str(world),
                               "--micro-batch-size", str(micro), "--gradient-accumulation-steps", str(accum)])
    command = launcher.commands(args)[0]
    assert ("--multi_gpu" in command) == (world > 1)
    assert "--use_fsdp" not in command
    assert command[command.index("--micro-batch-size") + 1] == str(micro)


def test_invalid_effective_batch_rejected():
    with pytest.raises(SystemExit):
        launcher.parse_args(["train", *PROFILE, "--micro-batch-size", "2"])


def test_train_and_inference_seed_independent():
    args = launcher.parse_args(["infer", *PROFILE, "--seed", "7", "--inference-seed", "42", "--checkpoint", "best"])
    command = launcher.commands(args)[0]
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--output-root") + 1].endswith("seed_7/predictions/best")


def test_formal_smoke_root_rejected():
    with pytest.raises(SystemExit):
        launcher.parse_args(["infer", *PROFILE, "--max-samples", "1"])


def test_smoke_evaluator_uses_actual_cli(tmp_path):
    args = launcher.parse_args(["eval", *PROFILE, "--smoke", "--output-root", str(tmp_path), "--max-samples", "2"])
    commands = launcher.commands(args)
    assert len(commands) == 2
    assert all("--output" not in command for command in commands)
    assert "--frame-only" in commands[1]
    assert "--frame-only" not in commands[0]


def test_limits_are_not_formal_training_overrides(tmp_path):
    with pytest.raises(SystemExit):
        launcher.parse_args(["train", *PROFILE, "--output-root", str(tmp_path), "--stop-after-updates", "1"])


def test_dry_run_no_output_directories(tmp_path, capsys):
    target = tmp_path / "absent"
    launcher.main(["train", *PROFILE, "--output-root", str(target), "--dry-run"])
    assert not target.exists()
    assert "train_seen10.py" in capsys.readouterr().out
