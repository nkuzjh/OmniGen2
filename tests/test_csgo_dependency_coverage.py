"""Keep the setup manifest and CPU probes aligned with the real entrypoints."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("csgo_dependency_checker", ROOT / "scripts/check_csgo_environment.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_every_declared_distribution_has_a_cpu_import_probe():
    pins = checker.requirement_pins()
    assert set(pins) == set(checker.DISTRIBUTION_MODULES)
    assert pins["datasets"] == "4.0.0"
    assert pins["tensorboard"] == "2.21.0"
    assert pins["pyyaml"] == "6.0.3"
    assert checker.DISTRIBUTION_MODULES["pyyaml"] == "yaml"
    assert "SummaryWriter" in checker.REQUIRED_SYMBOLS["torch.utils.tensorboard"]
    assert set(checker.REQUIRED_SYMBOLS["datasets"]) == {"load_dataset", "concatenate_datasets"}
    legacy = (ROOT / "options/csgo_seen10_lora.yml").read_text()
    assert "log_with: [tensorboard]" in legacy


def test_unknown_declared_dependency_cannot_skip_checks(tmp_path):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("unprobed-dependency==1.0\n")
    with pytest.raises(SystemExit, match="missing import probes"):
        checker.check_imports(requirements=requirements)


@pytest.mark.parametrize("absent", ["datasets", "tensorboard", "pyyaml", "opencv-python-headless", "wandb"])
def test_missing_runtime_is_reported_before_any_import(monkeypatch, absent):
    def version(name):
        if name == absent:
            raise checker.metadata.PackageNotFoundError(name)
        return "1.0"

    importer = Mock(side_effect=AssertionError("No import is allowed before metadata completeness"))
    monkeypatch.setattr(checker.metadata, "version", version)
    monkeypatch.setattr(checker.importlib, "import_module", importer)
    with pytest.raises(SystemExit, match=absent):
        checker.check_imports()
    importer.assert_not_called()


def fake_import_environment(monkeypatch):
    modules = {name: SimpleNamespace() for name in checker.DISTRIBUTION_MODULES.values()}
    modules["torch"] = SimpleNamespace(version=SimpleNamespace(cuda=None),
                                       cuda=SimpleNamespace(is_initialized=lambda: False))
    for module, symbols in checker.REQUIRED_SYMBOLS.items():
        modules.setdefault(module, SimpleNamespace())
        for symbol in symbols:
            setattr(modules[module], symbol, object())
    monkeypatch.setattr(checker.metadata, "version", lambda name: "1.0")
    monkeypatch.setitem(sys.modules, "torch", modules["torch"])
    importer = Mock(side_effect=lambda name: modules[name])
    monkeypatch.setattr(checker.importlib, "import_module", importer)
    return modules, importer


def test_metadata_alone_does_not_hide_missing_runtime_interfaces(monkeypatch):
    modules, _ = fake_import_environment(monkeypatch)
    delattr(modules["datasets"], "load_dataset")
    with pytest.raises(SystemExit, match="Required runtime interfaces unavailable"):
        checker.check_imports()


def test_cpu_check_never_imports_gpu_probing_project_modules(monkeypatch):
    _, importer = fake_import_environment(monkeypatch)
    checker.check_imports()
    names = [call.args[0] for call in importer.call_args_list]
    assert "torch.utils.tensorboard" in names and "datasets" in names
    assert not any(name == "train" or name.startswith("omnigen2") for name in names)


def test_cuda_build_requires_triton_without_loading_native_kernels(monkeypatch):
    modules, importer = fake_import_environment(monkeypatch)
    modules["torch"].version.cuda = "12.8"
    modules["triton"] = SimpleNamespace()
    monkeypatch.setattr(checker.sys, "platform", "linux")
    checker.check_imports()
    importer.assert_any_call("triton")


def local_modules(name):
    path = ROOT.joinpath(*name.split("."))
    results = []
    if path.with_suffix(".py").is_file():
        results.append(path.with_suffix(".py"))
    if (path / "__init__.py").is_file():
        results.append(path / "__init__.py")
    return results


def test_entrypoint_import_graph_dependencies_are_declared():
    # Parse, never import: OmniGen2's Triton decorators query CUDA on import.
    pending = [ROOT / name for name in (
        "train_seen10.py", "train.py", "infer_seen10.py", "convert_ckpt_to_hf_format.py",
        "smoke_seen10.py", "scripts/audit_csgo_aligned.py", "scripts/download_csgo_seen10_assets.py",
    )]
    visited = set()
    external = {}
    while pending:
        source = pending.pop()
        if source in visited:
            continue
        visited.add(source)
        relative = source.relative_to(ROOT).with_suffix("")
        package = ".".join(relative.parts[:-1])
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [entry.name for entry in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = importlib.util.resolve_name("." * node.level + base, package)
                names = [base] + [base + "." + alias.name for alias in node.names if alias.name != "*"]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                files = local_modules(name)
                if files:
                    pending.extend(files)
                elif root not in sys.stdlib_module_names and not local_modules(root):
                    external.setdefault(root, []).append(f"{source.relative_to(ROOT)}:{node.lineno}")
    declared = {module.split(".")[0] for module in checker.DISTRIBUTION_MODULES.values()}
    conditional = {
        "flash_attn",  # Explicit SDPA/native fallbacks in model code.
        "bitsandbytes",  # Both CSGO configs disable 8-bit Adam.
        "torch_xla",  # Only imported for the optional TPU path.
        "torch_npu",  # Only imported when the optional NPU backend is available.
        "deepspeed",  # Guarded by is_deepspeed_zero3_enabled(); CSGO uses DDP/FSDP.
        "importlib_metadata",  # Backport only for Python < 3.8; setup uses 3.11/3.12.
        "triton",  # Supplied by matching Linux CUDA Torch wheel, checked above.
    }
    unexpected = set(external) - declared - conditional
    assert not unexpected, {name: external[name] for name in sorted(unexpected)}
    assert "datasets" in external, "The real training import chain must be traversed"
    assert ROOT / "omnigen2/aligned_training.py" in visited
