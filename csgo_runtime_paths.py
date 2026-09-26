"""Machine-local CSGO paths. This module intentionally uses only the stdlib."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
from typing import Mapping

PROJECT_ROOT = Path(__file__).resolve().parent
LEGACY_DATA_ROOT = Path("/home/jiahao/task/UniLIP/data/csgo_benchmark_v2")
LEGACY_EVAL_ROOT = Path("/home/jiahao/task/csgo_benchmark_v2_eval_general")
LEGACY_EVAL_PYTHON = Path("/home/jiahao/miniconda3/envs/UniLIP/bin/python")


@dataclass(frozen=True)
class Selection:
    path: Path
    source: str
    exists: bool
    ready: bool


def project_path(value: str | os.PathLike[str], root: Path = PROJECT_ROOT) -> Path:
    """Anchor relative paths to the checkout without resolving Python symlinks."""
    if not os.fspath(value):
        raise ValueError("Path value must not be empty")
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def model_source(value: str, root: Path = PROJECT_ROOT) -> str:
    """Anchor explicit local paths (./, ../, ~/, absolute); retain Hub IDs."""
    if value.startswith(("./", "../", "~/")) or Path(value).is_absolute():
        return str(project_path(value, root))
    return value


def config_data_root(config: str | os.PathLike[str], *, explicit: str | None = None,
                     root: Path = PROJECT_ROOT,
                     env: Mapping[str, str] | None = None) -> str | None:
    """Read only the simple data.data_root YAML scalar; no YAML dependency."""
    environment = os.environ if env is None else env
    if explicit is not None or _first_env(
        environment, ("CSGO_DATA_ROOT", "CSGO_BENCHMARK_V2_DATA", "DATA_ROOT")
    ):
        return None
    path = project_path(config, root)
    in_data = False
    child_indent = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not in_data:
            section = re.match(r"^data\s*:\s*(.*?)\s*$", line)
            if section:
                if section.group(1) and not section.group(1).startswith("#"):
                    raise ValueError(f"Unsupported YAML data section in {path}; pass --data-root")
                in_data = True
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith("\t"):
            raise ValueError(f"Unsupported YAML indentation in {path}; pass --data-root")
        indent = len(line) - len(line.lstrip(" "))
        if indent == 0:
            break
        if "\t" in line[:indent]:
            raise ValueError(f"Unsupported YAML indentation in {path}; pass --data-root")
        if child_indent is None:
            child_indent = indent
        if indent < child_indent:
            raise ValueError(f"Unsupported YAML indentation in {path}; pass --data-root")
        if indent > child_indent:
            if re.match(r"^data_root\s*:", stripped):
                raise ValueError(f"Nested data_root is unsupported in {path}; pass --data-root")
            continue
        if stripped.startswith("<<:"):
            raise ValueError(f"YAML merge is unsupported in {path}; pass --data-root")
        match = re.match(r"^data_root\s*:\s*(.*?)\s*$", stripped)
        if match:
            value = match.group(1)
            if "${" in value or value.startswith(("!", "&", "*", "|", ">", "[", "{")):
                raise ValueError(f"Unsupported YAML data_root in {path}; pass --data-root")
            if value.startswith('"'):
                selected, consumed = json.JSONDecoder().raw_decode(value)
                if not isinstance(selected, str) or (value[consumed:].strip() and
                                                     not value[consumed:].strip().startswith("#")):
                    raise ValueError(f"Invalid data.data_root in {path}")
                return selected
            if value.startswith("'"):
                end = value.rfind("'")
                if end <= 0 or (value[end + 1:].strip() and
                                not value[end + 1:].strip().startswith("#")):
                    raise ValueError(f"Invalid quoted data.data_root in {path}")
                return value[1:end].replace("''", "'")
            value = value.split(" #", 1)[0].strip()
            return None if value in ("", "null", "~") else value
    return None


def _first_env(env: Mapping[str, str], names: tuple[str, ...]) -> tuple[str, str] | None:
    return next(((env[name], name) for name in names if env.get(name)), None)


def _directory(value: str | os.PathLike[str], source: str, root: Path) -> Selection:
    path = project_path(value, root)
    return Selection(path, source, path.exists(), path.is_dir())


def _evaluator(value: str | os.PathLike[str], source: str, root: Path) -> Selection:
    selected = _directory(value, source, root)
    return Selection(selected.path, source, selected.exists,
                     selected.ready and (selected.path / "run_eval.py").is_file())


def _python(value: str | os.PathLike[str], source: str, root: Path) -> Selection:
    path = project_path(value, root)
    return Selection(path, source, path.exists(), path.is_file() and os.access(path, os.X_OK))


def data_root(explicit: str | None = None, *, config_value: str | None = None,
              root: Path = PROJECT_ROOT,
              env: Mapping[str, str] | None = None) -> Selection:
    environment = os.environ if env is None else env
    if explicit is not None:
        return _directory(explicit, "CLI --data-root", root)
    selected = _first_env(environment, ("CSGO_DATA_ROOT", "CSGO_BENCHMARK_V2_DATA", "DATA_ROOT"))
    if selected:
        return _directory(selected[0], selected[1], root)
    if config_value and project_path(config_value, root) != LEGACY_DATA_ROOT:
        return _directory(config_value, "config data.data_root", root)
    if LEGACY_DATA_ROOT.is_dir():
        return _directory(LEGACY_DATA_ROOT, "legacy default", root)
    return _directory(root.parent / "UniLIP/data/csgo_benchmark_v2", "sibling UniLIP", root)


def eval_root(explicit: str | None = None, *, root: Path = PROJECT_ROOT,
              env: Mapping[str, str] | None = None) -> Selection:
    environment = os.environ if env is None else env
    if explicit is not None:
        return _evaluator(explicit, "CLI --eval-root", root)
    selected = _first_env(environment, ("SHARED_EVAL_DIR", "CSGO_EVAL_ROOT"))
    if selected:
        return _evaluator(selected[0], selected[1], root)
    if LEGACY_EVAL_ROOT.is_dir():
        return _evaluator(LEGACY_EVAL_ROOT, "legacy default", root)
    return _evaluator(root.parent / "csgo_benchmark_v2_eval_general", "sibling evaluator", root)


def eval_python(selected_root: Path, explicit: str | None = None, *,
                root: Path = PROJECT_ROOT, env: Mapping[str, str] | None = None) -> Selection:
    environment = os.environ if env is None else env
    if explicit is not None:
        return _python(explicit, "CLI --eval-python/--unilip-python", root)
    shared = _python(selected_root / ".venv/bin/python", "shared evaluator .venv", root)
    if shared.ready:
        return shared
    selected = _first_env(environment, ("EVAL_PYTHON", "UNILIP_PYTHON"))
    if selected:
        return _python(selected[0], selected[1], root)
    return _python(LEGACY_EVAL_PYTHON, "legacy UniLIP", root)


def model_python(*, root: Path = PROJECT_ROOT,
                 env: Mapping[str, str] | None = None) -> Selection:
    environment = os.environ if env is None else env
    selected = environment.get("OMNIGEN2_PYTHON")
    if selected and "/" not in selected:
        command = shutil.which(selected)
        if command:
            return _python(command, "OMNIGEN2_PYTHON", root)
    return _python(selected or root / ".venv/bin/python",
                   "OMNIGEN2_PYTHON" if selected else "project .venv", root)


def resolve(*, data: str | None = None, data_config: str | None = None,
            evaluation: str | None = None,
            evaluation_python: str | None = None, root: Path = PROJECT_ROOT,
            env: Mapping[str, str] | None = None) -> dict[str, Selection]:
    selected_eval = eval_root(evaluation, root=root, env=env)
    return {
        "model_python": model_python(root=root, env=env),
        "data_root": data_root(data, config_value=data_config, root=root, env=env),
        "eval_root": selected_eval,
        "eval_python": eval_python(selected_eval.path, evaluation_python, root=root, env=env),
    }


def require(selection: Selection, label: str) -> Path:
    if not selection.ready:
        raise FileNotFoundError(f"{label} is missing or unusable ({selection.source}): {selection.path}")
    return selection.path


def report(paths: Mapping[str, Selection], *, action: str) -> dict[str, object]:
    return {
        "project_root": str(PROJECT_ROOT), "action": action,
        "paths": {name: {"path": str(value.path), "source": value.source,
                         "exists": value.exists, "ready": value.ready}
                  for name, value in paths.items()},
        "evaluator": str(paths["eval_root"].path / "run_eval.py"),
        "inspection_only": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("smoke", "train", "convert", "infer", "eval", "all"), default="train")
    parser.add_argument("--data-root")
    parser.add_argument("--config")
    parser.add_argument("--eval-root")
    parser.add_argument("--eval-python", "--unilip-python", dest="eval_python")
    parser.add_argument("--lines", action="store_true")
    args = parser.parse_args(argv)
    config_value = config_data_root(args.config, explicit=args.data_root) if args.config else None
    paths = resolve(data=args.data_root, data_config=config_value, evaluation=args.eval_root,
                    evaluation_python=args.eval_python)
    if args.lines:
        values = [str(paths[name].path) for name in ("model_python", "data_root", "eval_root", "eval_python")]
        if any("\n" in value or "\r" in value for value in values):
            parser.error("Paths may not contain newlines")
        print("\n".join(values))
    else:
        print(json.dumps(report(paths, action=args.action), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
