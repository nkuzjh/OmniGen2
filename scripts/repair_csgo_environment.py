#!/usr/bin/env python3
"""Repair missing CSGO dependencies in an existing environment without changing direct pins."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name
except ImportError:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name


TORCH_PINS = {"torch": "2.7.1", "torchvision": "0.22.1"}


def direct_pins(path: Path) -> dict[str, tuple[str, str]]:
    pins = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.partition("#")[0].strip()
        if not line:
            continue
        name, separator, version = line.partition("==")
        if not separator or not name.strip() or not version.strip():
            raise ValueError(f"unsupported requirement (expected name==version): {raw}")
        key = canonicalize_name(name.strip())
        if key in pins:
            raise ValueError(f"duplicate requirement: {name}")
        pins[key] = (name.strip(), version.strip())
    for name, version in TORCH_PINS.items():
        pins.setdefault(name, (name, version))
    return pins


def distributions() -> dict[str, str]:
    found = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            # importlib.metadata.distribution() also chooses the first sys.path hit.
            found.setdefault(canonicalize_name(name), dist.version)
    return found


def root_issues(pins: dict[str, tuple[str, str]], installed: dict[str, str]) -> list[str]:
    """Check only the dependency closure of this project's direct requirements."""
    issues = []
    pending = [(key, frozenset()) for key in pins]
    visited = set()
    while pending:
        key, extras = pending.pop()
        state = (key, extras)
        if state in visited:
            continue
        visited.add(state)
        if key not in installed:
            issues.append(f"missing {key}")
            continue
        try:
            dist = metadata.distribution(key)
        except metadata.PackageNotFoundError:
            issues.append(f"missing metadata for {key}")
            continue
        for raw in dist.requires or ():
            requirement = Requirement(raw)
            if requirement.marker and not any(
                requirement.marker.evaluate({"extra": extra}) for extra in ("", *sorted(extras))
            ):
                continue
            child = canonicalize_name(requirement.name)
            version = installed.get(child)
            if version is None:
                issues.append(f"{key} requires missing {requirement}")
                continue
            if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
                issues.append(f"{key} requires {requirement}, found {version}")
            pending.append((child, frozenset(requirement.extras)))
    return issues


def constraints(pins: dict[str, tuple[str, str]], installed: dict[str, str]) -> str:
    protected = set(pins) | {"triton", "torchaudio"}
    protected.update(key for key in installed if key.startswith("nvidia-"))
    return "".join(f"{key}=={installed[key]}\n" for key in sorted(protected & installed.keys()))


def pip_install(requirements: list[str], pins: dict[str, tuple[str, str]],
                before: dict[str, str], *, index: str | None = None) -> None:
    with tempfile.TemporaryDirectory(prefix="csgo-pip-plan-") as temp:
        constraint = Path(temp) / "constraints.txt"
        report = Path(temp) / "report.json"
        constraint.write_text(constraints(pins, before), encoding="utf-8")
        command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                   "--no-input", "--constraint", str(constraint)]
        if index:
            command += ["--index-url", index]
        dry_run = command + ["--dry-run", "--report", str(report), *requirements]
        print("Checking package repair plan:", ", ".join(requirements), flush=True)
        subprocess.run(dry_run, check=True)
        plan = json.loads(report.read_text(encoding="utf-8"))
        protected = set(pins) | {"triton", "torchaudio"}
        protected.update(key for key in before if key.startswith("nvidia-"))
        changes = []
        for entry in plan.get("install", []):
            name = canonicalize_name(entry["metadata"]["name"])
            version = entry["metadata"]["version"]
            if name in protected and name in before and before[name] != version:
                changes.append(f"{name}: {before[name]} -> {version}")
        if changes:
            raise RuntimeError("pip planned to replace installed protected versions: " + ", ".join(changes))
        subprocess.run(command + requirements, check=True)


def record_repair(environment: Path, before: dict[str, str], after: dict[str, str],
                  issues: list[str], backend: str) -> None:
    actual = {
        name: {"before": before.get(name), "after": after.get(name)}
        for name in sorted(before.keys() | after.keys()) if before.get(name) != after.get(name)
    }
    if not actual:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"csgo-repair-{stamp}-{os.getpid()}"
    freeze_name = stem + "-pip-freeze.txt"
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"],
                            check=True, capture_output=True, text=True).stdout
    (environment / freeze_name).write_text(freeze, encoding="utf-8")
    payload = {
        "schema_version": 1,
        "environment": str(environment),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_backend": backend,
        "previous_issues": issues,
        "actual_changes": actual,
        "distributions": after,
        "pip_freeze": freeze_name,
    }
    (environment / (stem + "-manifest.json")).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Repair record: {environment / (stem + '-manifest.json')}")


def repair(args: argparse.Namespace) -> None:
    if args.backend not in {"cu128", "cpu"}:
        raise ValueError(f"unsupported OMNIGEN2_TORCH_BACKEND={args.backend}; choose cu128 or cpu")
    pins = direct_pins(args.requirements)
    before = distributions()
    issues = root_issues(pins, before)
    if not issues:
        return
    print("Repairing project dependency closure:\n  " + "\n  ".join(issues), flush=True)
    installed = before.copy()
    missing_torch = [name for name in TORCH_PINS if name not in installed]
    if missing_torch:
        present = [name for name in TORCH_PINS if name in installed]
        if present:
            counterpart = present[0]
            if installed[counterpart].split("+", 1)[0] != TORCH_PINS[counterpart]:
                raise RuntimeError(
                    f"{missing_torch[0]} is missing, but installed {counterpart} "
                    f"{installed[counterpart]} is not the declared {TORCH_PINS[counterpart]} pair; "
                    "preserved the installed core version"
                )
            local_backend = installed[counterpart].partition("+")[2]
            if local_backend != args.backend:
                raise RuntimeError(
                    f"{missing_torch[0]} is missing, but installed {counterpart} "
                    f"{installed[counterpart]} does not identify the selected {args.backend} "
                    "wheel backend; preserved the installed core version"
                )
        index = os.environ.get("OMNIGEN2_TORCH_INDEX_URL",
                               f"https://download.pytorch.org/whl/{args.backend}")
        pip_install([f"{name}=={TORCH_PINS[name]}" for name in missing_torch],
                    pins, installed, index=index)
        installed = distributions()
    current_roots = [f"{name}=={installed.get(key, version)}"
                     for key, (name, version) in pins.items()]
    if root_issues(pins, installed):
        pip_install(current_roots, pins, installed)
    after = distributions()
    remaining = root_issues(pins, after)
    if remaining:
        raise RuntimeError("dependency closure remains incomplete after pip:\n  " + "\n  ".join(remaining))
    subprocess.run([sys.executable, str(args.checker), "--expected-prefix",
                    str(args.expected_prefix)], check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    record_repair(args.expected_prefix, before, after, issues, args.backend)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--checker", type=Path, required=True)
    parser.add_argument("--expected-prefix", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    args = parser.parse_args()
    try:
        repair(args)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"CSGO environment repair failed; environment retained for retry: {exc}") from exc


if __name__ == "__main__":
    main()
