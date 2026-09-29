"""Offline repair tests; no project environment or package index is touched."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import venv
import zipfile


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "repair_csgo_environment.py"
SPEC = importlib.util.spec_from_file_location("repair_csgo_environment", SOURCE)
assert SPEC and SPEC.loader
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


class RepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="csgo repair test ")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.args = argparse.Namespace(
            requirements=self.path / "requirements.txt",
            checker=self.path / "check.py",
            expected_prefix=self.path / "environment with spaces",
            backend="cpu",
        )
        self.pins = {
            "torch": ("torch", "2.7.1"),
            "torchvision": ("torchvision", "0.22.1"),
            "datasets": ("datasets", "4.0.0"),
        }
        self.before = {"torch": "2.7.1+cpu", "torchvision": "0.22.1+cpu",
                       "torchaudio": "2.7.1+cpu", "triton": "3.3.1",
                       "nvidia-cublas-cu12": "12.8.4", "fsspec": "2025.5.0"}
        self.after = self.before | {"datasets": "4.0.0", "fsspec": "2025.3.0"}

    def test_constraints_preserve_direct_and_torch_stack_but_allow_fsspec(self) -> None:
        content = repair.constraints(self.pins, self.before)
        self.assertIn("torch==2.7.1+cpu\n", content)
        self.assertIn("torchvision==0.22.1+cpu\n", content)
        self.assertIn("triton==3.3.1\n", content)
        self.assertIn("torchaudio==2.7.1+cpu\n", content)
        self.assertIn("nvidia-cublas-cu12==12.8.4\n", content)
        self.assertNotIn("fsspec", content)

    def test_duplicate_distribution_uses_first_sys_path_hit(self) -> None:
        first = SimpleNamespace(metadata={"Name": "CSGO_Sample"}, version="1.0")
        second = SimpleNamespace(metadata={"Name": "csgo-sample"}, version="2.0")
        with mock.patch.object(repair.metadata, "distributions", return_value=[first, second]):
            self.assertEqual(repair.distributions(), {"csgo-sample": "1.0"})

    def test_dependency_closure_follows_requested_extras(self) -> None:
        metadata_by_name = {
            "csgo-parent": SimpleNamespace(requires=["csgo-child[feature]==1.0"]),
            "csgo-child": SimpleNamespace(requires=[
                "csgo-extra-only==1.0; extra == 'feature'"
            ]),
        }
        with mock.patch.object(repair.metadata, "distribution",
                               side_effect=lambda name: metadata_by_name[name]):
            issues = repair.root_issues(
                {"csgo-parent": ("csgo-parent", "1.0")},
                {"csgo-parent": "1.0", "csgo-child": "1.0"},
            )
        self.assertTrue(any("csgo-extra-only" in issue for issue in issues), issues)

    def test_missing_datasets_repairs_then_repeat_is_idempotent(self) -> None:
        with (mock.patch.object(repair, "direct_pins", return_value=self.pins),
              mock.patch.object(repair, "distributions", side_effect=[self.before, self.after, self.after]),
              mock.patch.object(repair, "root_issues", side_effect=[
                  ["missing datasets"], ["missing datasets"], [], []
              ]),
              mock.patch.object(repair, "pip_install") as pip,
              mock.patch.object(repair, "record_repair") as record,
              mock.patch.object(repair.subprocess, "run") as run):
            repair.repair(self.args)
            repair.repair(self.args)
        pip.assert_called_once()
        roots = pip.call_args.args[0]
        self.assertIn("datasets==4.0.0", roots)
        self.assertIn("torch==2.7.1+cpu", roots)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0][1], str(self.args.checker))
        record.assert_called_once()

    def test_dry_run_rejects_core_replacement_before_install(self) -> None:
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            if "--report" in command:
                report = Path(command[command.index("--report") + 1])
                report.write_text(json.dumps({"install": [
                    {"metadata": {"name": "torch", "version": "2.8.0"}}
                ]}), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)

        with mock.patch.object(repair.subprocess, "run", side_effect=fake_run):
            with self.assertRaisesRegex(RuntimeError, "replace installed protected versions"):
                repair.pip_install(["datasets==4.0.0"], self.pins, self.before)
        self.assertEqual(len(calls), 1)
        self.assertIn("--dry-run", calls[0])

    def test_failed_install_can_retry_without_changing_core(self) -> None:
        with (mock.patch.object(repair, "direct_pins", return_value=self.pins),
              mock.patch.object(repair, "distributions", side_effect=[self.before, self.before, self.after]),
              mock.patch.object(repair, "root_issues", side_effect=[
                  ["missing datasets"], ["missing datasets"],
                  ["missing datasets"], ["missing datasets"], []
              ]),
              mock.patch.object(repair, "pip_install", side_effect=[
                  subprocess.CalledProcessError(1, "pip"), None
              ]) as pip,
              mock.patch.object(repair, "record_repair") as record,
              mock.patch.object(repair.subprocess, "run")):
            with self.assertRaises(subprocess.CalledProcessError):
                repair.repair(self.args)
            record.assert_not_called()
            repair.repair(self.args)
        self.assertEqual(pip.call_count, 2)
        record.assert_called_once()

    def test_one_missing_torch_partner_with_nonmatching_version_fails(self) -> None:
        installed = {"torch": "2.6.0", "datasets": "4.0.0"}
        with (mock.patch.object(repair, "direct_pins", return_value=self.pins),
              mock.patch.object(repair, "distributions", return_value=installed),
              mock.patch.object(repair, "root_issues", return_value=["missing torchvision"]),
              mock.patch.object(repair, "pip_install") as pip):
            with self.assertRaisesRegex(RuntimeError, "preserved the installed core version"):
                repair.repair(self.args)
        pip.assert_not_called()

    def test_one_missing_torch_partner_with_other_backend_fails(self) -> None:
        installed = {"torch": "2.7.1+cu128", "datasets": "4.0.0"}
        with (mock.patch.object(repair, "direct_pins", return_value=self.pins),
              mock.patch.object(repair, "distributions", return_value=installed),
              mock.patch.object(repair, "root_issues", return_value=["missing torchvision"]),
              mock.patch.object(repair, "pip_install") as pip):
            with self.assertRaisesRegex(RuntimeError, "selected cpu wheel backend"):
                repair.repair(self.args)
        pip.assert_not_called()

    def test_one_missing_torch_partner_with_declared_version_installs_only_missing(self) -> None:
        before = {"torch": "2.7.1+cpu", "datasets": "4.0.0"}
        after = before | {"torchvision": "0.22.1+cpu"}
        with (mock.patch.object(repair, "direct_pins", return_value=self.pins),
              mock.patch.object(repair, "distributions", side_effect=[before, after, after]),
              mock.patch.object(repair, "root_issues", side_effect=[
                  ["missing torchvision"], [], []
              ]),
              mock.patch.object(repair, "pip_install") as pip,
              mock.patch.object(repair, "record_repair"),
              mock.patch.object(repair.subprocess, "run")):
            repair.repair(self.args)
        pip.assert_called_once()
        self.assertEqual(pip.call_args.args[0], ["torchvision==0.22.1"])

    def test_offline_venv_repairs_missing_transitive_and_keeps_history(self) -> None:
        target = self.args.expected_prefix
        venv.EnvBuilder(with_pip=True).create(target)
        python = target / "bin" / "python"
        wheel_temp = tempfile.TemporaryDirectory(prefix="csgo_wheels_")
        self.addCleanup(wheel_temp.cleanup)
        wheels = Path(wheel_temp.name)

        def wheel(name: str, requires: str = "") -> None:
            stem = f"{name}-1.0"
            with zipfile.ZipFile(wheels / f"{stem}-py3-none-any.whl", "w") as archive:
                archive.writestr(f"{name}/__init__.py", "")
                archive.writestr(f"{stem}.dist-info/METADATA",
                                 f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n"
                                 + (f"Requires-Dist: {requires}\n" if requires else ""))
                archive.writestr(f"{stem}.dist-info/WHEEL",
                                 "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
                archive.writestr(f"{stem}.dist-info/RECORD", "")

        wheel("csgo_child")
        wheel("csgo_sample", "csgo-child==1.0")
        self.args.requirements.write_text("csgo-sample==1.0\n", encoding="utf-8")
        self.args.checker.write_text("import sys\nassert sys.prefix != sys.base_prefix\n", encoding="utf-8")
        environment = os.environ | {"PIP_NO_INDEX": "1", "PIP_FIND_LINKS": str(wheels)}
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "--no-index",
                        "--find-links", str(wheels), "csgo-sample==1.0"],
                       check=True, capture_output=True, text=True, env=environment)
        bootstrap = (
            "import importlib.util, sys; "
            "spec=importlib.util.spec_from_file_location('repair', sys.argv[1]); "
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "module.TORCH_PINS={}; sys.argv=sys.argv[1:]; module.main()"
        )
        command = [str(python), "-c", bootstrap, str(SOURCE), "--requirements",
                   str(self.args.requirements), "--checker", str(self.args.checker),
                   "--expected-prefix", str(target), "--backend", "cpu"]
        first = subprocess.run(command, capture_output=True, text=True, env=environment)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        installed = subprocess.run([str(python), "-m", "pip", "show", "csgo-child"],
                                   capture_output=True, text=True, env=environment)
        self.assertEqual(installed.returncode, 0, installed.stderr)
        manifests = list(target.glob("csgo-repair-*-manifest.json"))
        self.assertEqual(len(manifests), 1)
        payload = json.loads(manifests[0].read_text(encoding="utf-8"))
        self.assertEqual(payload["actual_changes"]["csgo-child"]["after"], "1.0")
        self.assertTrue((target / payload["pip_freeze"]).is_file())
        second = subprocess.run(command, capture_output=True, text=True, env=environment)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(list(target.glob("csgo-repair-*-manifest.json")), manifests)
        wheel("csgo_direct")
        self.args.requirements.write_text("csgo-sample==1.0\ncsgo-direct==1.0\n", encoding="utf-8")
        third = subprocess.run(command, capture_output=True, text=True, env=environment)
        self.assertEqual(third.returncode, 0, third.stdout + third.stderr)
        self.assertEqual(len(list(target.glob("csgo-repair-*-manifest.json"))), 2)
        direct = subprocess.run([str(python), "-m", "pip", "show", "csgo-direct"],
                                capture_output=True, text=True, env=environment)
        self.assertEqual(direct.returncode, 0, direct.stderr)


if __name__ == "__main__":
    unittest.main()
