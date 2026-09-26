"""Isolated CLI tests: no packages are installed and no assets are downloaded."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "setup_csgo_seen10.sh"
FAKE_PYTHON = """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
with Path(os.environ['TEST_CALLS']).open('a') as handle:
    handle.write(json.dumps({'exe': sys.argv[0], 'args': sys.argv[1:]}) + '\\n')
args = sys.argv[1:]
if args[:2] == ['-m', 'venv']:
    target = Path(args[2])
    (target / 'bin').mkdir(parents=True)
    py = target / 'bin' / 'python'
    py.write_text(Path(__file__).read_text())
    py.chmod(0o755)
    (target / 'pyvenv.cfg').write_text('include-system-site-packages = false\\n')
elif os.environ.get('FAIL_CHECK') == '1' and '--expected-prefix' in args and '--identity-only' not in args:
    sys.exit(1)
"""


class EnvironmentSetupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="omnigen2 env test ")
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name) / "project with spaces"
        scripts = self.project / "scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(SOURCE, scripts / SOURCE.name)
        (scripts / "check_csgo_environment.py").write_text("# mocked by fake Python\n")
        (scripts / "download_csgo_seen10_assets.py").write_text("# mocked by fake Python\n")
        (self.project / "requirements-csgo-seen10.txt").write_text("numpy==2.2.6\n")
        self.calls_file = Path(self.temp.name) / "calls.jsonl"
        self.env_dir = self.project / ".venv"
        self.target_python = self.env_dir / "bin" / "python"
        self.env = dict(os.environ, TEST_CALLS=str(self.calls_file), OMNIGEN2_PYTHON=str(self.target_python))

    def make_existing(self, *, symlink: bool = False) -> None:
        self.target_python.parent.mkdir(parents=True)
        executable = self.target_python.with_name("python-real") if symlink else self.target_python
        executable.write_text(FAKE_PYTHON)
        executable.chmod(0o755)
        if symlink:
            self.target_python.symlink_to(executable.name)
        (self.env_dir / "pyvenv.cfg").write_text("include-system-site-packages = false\n")

    def invoke(self, *args: str, extra_env: dict[str, str] | None = None,
               cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        env = self.env | (extra_env or {})
        return subprocess.run(
            ["bash", str(self.project / "scripts" / SOURCE.name), *args],
            cwd=cwd or self.project,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def calls(self) -> list[dict[str, object]]:
        if not self.calls_file.exists():
            return []
        return [json.loads(line) for line in self.calls_file.read_text().splitlines()]

    def test_help_has_no_effect(self) -> None:
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("verify/download", result.stdout)
        self.assertFalse(self.env_dir.exists())
        self.assertEqual(self.calls(), [])

    def test_existing_complete_is_idempotent_without_pip(self) -> None:
        self.make_existing(symlink=True)
        for _ in range(2):
            result = self.invoke("--env-only")
            self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(call["exe"] == str(self.target_python) for call in calls))
        self.assertTrue(all("-m" not in call["args"] for call in calls))
        self.assertTrue(all("download_csgo_seen10_assets.py" not in " ".join(call["args"]) for call in calls))
        self.assertTrue(self.target_python.is_symlink())

    def test_relative_python_is_project_anchored(self) -> None:
        self.make_existing()
        result = self.invoke("--env-only", extra_env={"OMNIGEN2_PYTHON": ".venv/bin/python"},
                             cwd=Path(self.temp.name))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(call["exe"] == str(self.target_python) for call in self.calls()))

    def test_home_relative_python_expands_before_environment_check(self) -> None:
        fake_home = Path(self.temp.name) / "home with spaces"
        self.env_dir = fake_home / "another env"
        self.target_python = self.env_dir / "bin" / "python"
        self.make_existing()
        result = self.invoke("--env-only", extra_env={"HOME": str(fake_home),
                             "OMNIGEN2_PYTHON": "~/another env/bin/python"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(all(call["exe"] == str(self.target_python) for call in self.calls()))

    def test_check_assets_and_explicit_cuda_are_separate(self) -> None:
        self.make_existing()
        check = self.invoke("--check", "--profile", "aligned")
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertEqual(self.calls()[-1]["args"][-3:], ["--check", "--profile", "aligned"])
        self.calls_file.unlink()
        cpu = self.invoke("--env-only", "--check")
        self.assertEqual(cpu.returncode, 0, cpu.stderr)
        self.assertEqual(len(self.calls()), 1)
        self.assertNotIn("--cuda-only", self.calls()[0]["args"])
        self.calls_file.unlink()
        gpu = self.invoke("--check-cuda")
        self.assertEqual(gpu.returncode, 0, gpu.stderr)
        self.assertEqual(len(self.calls()), 2)
        self.assertIn("--cuda-only", self.calls()[-1]["args"])
        self.assertFalse(any("download_csgo_seen10_assets.py" in " ".join(c["args"]) for c in self.calls()))

    def test_incomplete_existing_environment_is_preserved(self) -> None:
        self.make_existing()
        sentinel = self.env_dir / "keep.txt"
        sentinel.write_text("user data")
        result = self.invoke("--env-only", extra_env={"FAIL_CHECK": "1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("preserved unchanged", result.stderr)
        self.assertIn(".venv-csgo-seen10", result.stderr)
        self.assertEqual(sentinel.read_text(), "user data")
        self.assertFalse(any("-m" in c["args"] for c in self.calls()))

    def test_existing_non_environment_directory_is_preserved(self) -> None:
        self.env_dir.mkdir()
        sentinel = self.env_dir / "keep.txt"
        sentinel.write_text("user data")
        result = self.invoke("--env-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("preserved unchanged", result.stderr)
        self.assertEqual(sentinel.read_text(), "user data")
        self.assertEqual(self.calls(), [])

    def test_fresh_mock_install_then_second_run_only_checks(self) -> None:
        bootstrap = Path(self.temp.name) / "fake bootstrap python"
        bootstrap.write_text(FAKE_PYTHON)
        bootstrap.chmod(0o755)
        first = self.invoke("--env-only", extra_env={"OMNIGEN2_BOOTSTRAP_PYTHON": str(bootstrap)})
        self.assertEqual(first.returncode, 0, first.stderr)
        calls = self.calls()
        pip_installs = [c for c in calls if c["args"][:3] == ["-m", "pip", "install"]]
        self.assertEqual(len(pip_installs), 2)
        self.assertIn("torch==2.7.1", pip_installs[0]["args"])
        self.assertIn("torchvision==0.22.1", pip_installs[0]["args"])
        self.assertIn("https://download.pytorch.org/whl/cu128", pip_installs[0]["args"])
        self.assertTrue(any(c["args"][:3] == ["-m", "pip", "check"] for c in calls))
        self.assertTrue(any(c["args"][:4] == ["-m", "pip", "freeze", "--all"] for c in calls))
        self.assertFalse(any("download_csgo_seen10_assets.py" in " ".join(c["args"]) for c in calls))
        self.calls_file.unlink()
        second = self.invoke("--env-only", extra_env={"OMNIGEN2_BOOTSTRAP_PYTHON": str(bootstrap)})
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertFalse(any("-m" in c["args"] for c in self.calls()))


if __name__ == "__main__":
    unittest.main()
