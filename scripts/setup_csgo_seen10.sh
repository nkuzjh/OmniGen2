#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REQUIREMENTS="$PROJECT_ROOT/requirements-csgo-seen10.txt"
CHECKER="$PROJECT_ROOT/scripts/check_csgo_environment.py"
ASSETS="$PROJECT_ROOT/scripts/download_csgo_seen10_assets.py"
MODE=setup
ENV_ONLY=0
PROFILE=all

usage() {
    cat <<'HELP'
Usage: bash scripts/setup_csgo_seen10.sh [--env-only] [--check | --check-cuda] [--profile aligned|legacy|all] [--help]

Default: use OmniGen2/.venv/bin/python. Create a missing isolated environment,
install pinned dependencies, then verify/download the selected official model
assets (default profile: all). Existing complete environments are only checked;
their installed packages are never installed, upgraded, or downgraded.

--env-only    Prepare/check only Python dependencies; skip all model assets.
--check       Read-only CPU import check and, unless --env-only, asset check.
--check-cuda  Explicit CPU checks plus a small CUDA BF16 matrix check; no assets.
--profile     Select aligned, legacy, or all assets (default: all).
--help        Print this help without installation, downloads, or CUDA work.

For a separate new environment, set
  OMNIGEN2_PYTHON=/path/to/OmniGen2/.venv-csgo-seen10/bin/python
Relative target paths are anchored at OmniGen2; bin/python may be a symlink.
OMNIGEN2_BOOTSTRAP_PYTHON explicitly selects a Python 3.11/3.12 base.
Otherwise python3.11, then python3.12 is preferred; Conda can create an
isolated Python 3.11 prefix when no suitable venv-capable Python is found.
OMNIGEN2_CONDA_EXE can select the Conda executable.
OMNIGEN2_TORCH_BACKEND=cu128 (default) installs torch 2.7.1 and torchvision
0.22.1 from the CUDA 12.8 wheel index. cpu is for CPU checks only: OmniGen2
Triton training requires an NVIDIA GPU. OMNIGEN2_TORCH_INDEX_URL can select a
matching PyTorch wheel mirror. Fresh CUDA/GPU operation is not certified by
the CPU import check; run --check-cuda on the destination server.
HELP
}

die() { printf 'setup_csgo_seen10: %s\n' "$*" >&2; exit 2; }

while (( $# )); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --env-only) ENV_ONLY=1; shift ;;
        --check) [[ "$MODE" == setup || "$MODE" == check ]] || die "--check conflicts with --check-cuda"; MODE=check; shift ;;
        --check-cuda) [[ "$MODE" == setup || "$MODE" == check-cuda ]] || die "--check-cuda conflicts with --check"; MODE=check-cuda; shift ;;
        --profile) (( $# >= 2 )) || die "--profile requires aligned, legacy, or all"; PROFILE="$2"; shift 2 ;;
        --profile=*) PROFILE="${1#*=}"; shift ;;
        *) die "unknown argument: $1 (use --help)" ;;
    esac
done
case "$PROFILE" in aligned|legacy|all) ;; *) die "invalid --profile=$PROFILE" ;; esac
[[ -f "$REQUIREMENTS" && -f "$CHECKER" ]] || die "missing environment files"

PYTHON="${OMNIGEN2_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
[[ "$PYTHON" != '~/'* ]] || PYTHON="${HOME:?HOME is required for ~/ paths}/${PYTHON:2}"
[[ "$PYTHON" == /* ]] || PYTHON="$PROJECT_ROOT/$PYTHON"
[[ "$(basename -- "$(dirname -- "$PYTHON")")" == bin ]] || die "OMNIGEN2_PYTHON must name an environment bin/python"
RAW_ENV_DIR="$(dirname -- "$(dirname -- "$PYTHON")")"
[[ ! -L "$RAW_ENV_DIR" ]] || die "environment directory is a symlink; preserved unchanged: $RAW_ENV_DIR"
ENV_DIR="$(realpath -m -- "$RAW_ENV_DIR")"
[[ "$ENV_DIR" != / && "$ENV_DIR" != "$PROJECT_ROOT" ]] || die "unsafe environment directory: $ENV_DIR"

check_environment() {
    [[ -x "$PYTHON" ]] || die "missing environment Python: $PYTHON"
    CUDA_VISIBLE_DEVICES= "$PYTHON" "$CHECKER" --expected-prefix "$ENV_DIR" "$@"
}

asset_action() {
    [[ -f "$ASSETS" ]] || die "missing asset downloader: $ASSETS"
    "$PYTHON" "$ASSETS" "$@" --profile "$PROFILE"
}

if [[ "$MODE" == check ]]; then
    check_environment
    if (( ! ENV_ONLY )); then asset_action --check; fi
    exit 0
fi
if [[ "$MODE" == check-cuda ]]; then
    check_environment
    "$PYTHON" "$CHECKER" --expected-prefix "$ENV_DIR" --cuda-only
    exit 0
fi

if [[ -e "$ENV_DIR" || -L "$ENV_DIR" ]]; then
    [[ -d "$ENV_DIR" && -x "$PYTHON" ]] || die "$ENV_DIR exists but is not an environment; preserved unchanged"
    check_environment --identity-only
    if ! check_environment; then
        die "existing environment is incomplete; preserved unchanged. Use OMNIGEN2_PYTHON=$PROJECT_ROOT/.venv-csgo-seen10/bin/python to create a separate environment"
    fi
else
    backend="${OMNIGEN2_TORCH_BACKEND:-cu128}"
    case "$backend" in cu128|cpu) ;; *) die "unsupported OMNIGEN2_TORCH_BACKEND=$backend; choose cu128 or cpu" ;; esac
    bootstrap="${OMNIGEN2_BOOTSTRAP_PYTHON:-}"
    python_usable() {
        "$1" -c 'import ensurepip, sys, venv; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)' >/dev/null 2>&1
    }
    if [[ -n "$bootstrap" ]]; then
        command -v "$bootstrap" >/dev/null 2>&1 || die "bootstrap Python unavailable: $bootstrap"
        python_usable "$bootstrap" || die "bootstrap Python must be 3.11/3.12 with venv and ensurepip: $bootstrap"
    else
        for candidate in python3.11 python3.12 python3; do
            if command -v "$candidate" >/dev/null 2>&1 && python_usable "$candidate"; then
                bootstrap="$candidate"
                break
            fi
        done
    fi
    if [[ -n "$bootstrap" ]]; then
        "$bootstrap" -m venv "$ENV_DIR" || die "venv creation failed; directory preserved for inspection: $ENV_DIR"
    else
        conda_exe="${OMNIGEN2_CONDA_EXE:-$(command -v conda || true)}"
        [[ -n "$conda_exe" ]] && command -v "$conda_exe" >/dev/null 2>&1 || die "Python 3.11/3.12 with venv or Conda is required"
        "$conda_exe" create --prefix "$ENV_DIR" python=3.11 pip -y || die "Conda creation failed; directory preserved for inspection: $ENV_DIR"
    fi
    check_environment --identity-only
    "$PYTHON" -m pip --version >/dev/null || die "pip missing from new environment: $ENV_DIR"
    index="${OMNIGEN2_TORCH_INDEX_URL:-https://download.pytorch.org/whl/$backend}"
    "$PYTHON" -m pip install --disable-pip-version-check --no-input --index-url "$index" 'torch==2.7.1' 'torchvision==0.22.1'
    "$PYTHON" -m pip install --disable-pip-version-check --no-input -r "$REQUIREMENTS"
    "$PYTHON" -m pip check
    check_environment --fresh-pins "$REQUIREMENTS"
    "$PYTHON" -m pip freeze --all > "$ENV_DIR/csgo-pip-freeze.txt"
    "$PYTHON" - "$ENV_DIR" "$backend" <<'PY'
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

environment = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "environment": str(environment),
    "python_executable": sys.executable,
    "python_version": sys.version.split()[0],
    "torch_backend": sys.argv[2],
    "distributions": {dist.metadata["Name"]: dist.version for dist in metadata.distributions() if dist.metadata.get("Name")},
}
(environment / "csgo-install-manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi

if (( ! ENV_ONLY )); then asset_action; fi
printf 'OmniGen2 CSGO environment ready: %s\n' "$ENV_DIR"
