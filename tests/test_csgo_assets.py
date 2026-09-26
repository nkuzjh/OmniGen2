"""Offline fixtures only: no remote requests or multi-GB weight downloads."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("csgo_assets", ROOT / "scripts/download_csgo_seen10_assets.py")
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


def tiny_repo():
    data = b"tiny official fixture"
    item = {"filename": "weights.safetensors", "size": len(data), "algorithm": "sha256",
            "digest": hashlib.sha256(data).hexdigest(), "profiles": ["aligned", "legacy"]}
    return {"repo_id": "fixture/model", "revision": "a" * 40, "env": "OMNIGEN2_MODEL_PATH", "files": [item]}, data


def test_cache_priority_and_no_creation(tmp_path):
    expected = tmp_path / "empty cache"
    assert assets.cache_root(str(expected), {"HF_HUB_CACHE": "/ignored"}) == expected
    assert assets.cache_root(None, {"HF_HOME": str(expected)}) == expected / "hub"
    assert assets.cache_root(None, {"HF_HUB_CACHE": str(expected), "HF_HOME": "/ignored"}) == expected
    assert not expected.exists()


def test_offline_check_never_creates_cache(tmp_path):
    cache = tmp_path / "does not exist"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/download_csgo_seen10_assets.py"),
                             "--check", "--profile", "aligned", "--cache-dir", str(cache), "--json"],
                            capture_output=True, text=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ready"] is False
    assert all(r["status"] == "missing" for r in payload["files"])
    assert not cache.exists()


def test_hashes_and_incomplete_files(tmp_path):
    repo, data = tiny_repo()
    path = tmp_path / "weights"
    path.with_suffix(".incomplete").write_bytes(data)
    assert assets.check_file(path, repo["files"][0]) == "missing"
    path.write_bytes(data)
    assert assets.check_file(path, repo["files"][0]) == "ok"
    path.write_bytes(b"x" * len(data))
    assert assets.check_file(path, repo["files"][0]) == "hash_mismatch"
    path.write_bytes(b"short")
    assert assets.check_file(path, repo["files"][0]) == "size_mismatch"


def test_git_blob_hash(tmp_path):
    data = b'{"config":true}\n'
    path = tmp_path / "config.json"
    path.write_bytes(data)
    asset = {"size": len(data), "algorithm": "git-blob-sha1",
             "digest": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()}
    assert assets.check_file(path, asset) == "ok"


def test_download_is_pinned_idempotent_and_never_rewrites_corrupt_files(tmp_path):
    repo, data = tiny_repo()
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        path = assets.snapshot_path(tmp_path, repo) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    rows = assets.audit_assets(tmp_path, [repo], "aligned")
    assets.download_missing(tmp_path, [repo], "aligned", rows, downloader=fetch)
    assert rows[0]["status"] == "ok"
    assert calls[0]["revision"] == "a" * 40
    assert calls[0]["cache_dir"] == str(tmp_path)
    assets.download_missing(tmp_path, [repo], "aligned", rows, downloader=fetch)
    assert len(calls) == 1
    assert not (tmp_path / "models--fixture--model/refs/main").exists()
    target = Path(rows[0]["path"])
    target.write_bytes(b"broken")
    rows = assets.audit_assets(tmp_path, [repo], "aligned")
    with pytest.raises(RuntimeError, match="not overwritten"):
        assets.download_missing(tmp_path, [repo], "aligned", rows, downloader=fetch)
    assert target.read_bytes() == b"broken"
    assert len(calls) == 1


def test_profiles_cover_training_and_native_legacy_pipeline():
    repos = assets.manifest_repositories()
    aligned = {(r["repo_id"], a["filename"]) for r, a in assets.selected_assets(repos, "aligned")}
    legacy = {(r["repo_id"], a["filename"]) for r, a in assets.selected_assets(repos, "legacy")}
    assert aligned < legacy
    assert ("OmniGen2/OmniGen2", "mllm/model-00004-of-00004.safetensors") in legacy - aligned
    assert ("OmniGen2/OmniGen2", "processor/chat_template.json") in legacy
    assert ("Qwen/Qwen2.5-VL-3B-Instruct", "model-00002-of-00002.safetensors") in aligned
    flux = [a["filename"] for r, a in assets.selected_assets(repos, "all") if r["repo_id"].startswith("black-")]
    assert set(flux) == {"vae/config.json", "vae/diffusion_pytorch_model.safetensors"}


def test_print_env_is_safely_quoted_and_does_not_create_paths(tmp_path, capsys):
    repo, _ = tiny_repo()
    cache = tmp_path / "space ' quote"
    assets.print_environment(cache, [repo])
    exports = capsys.readouterr().out
    proc = subprocess.run(["bash", "-c", exports + '\nprintf "%s" "$OMNIGEN2_MODEL_PATH"'],
                          capture_output=True, text=True, check=True)
    assert proc.stdout == str(assets.snapshot_path(cache, repo))
    assert not cache.exists()


def test_dry_run_offline(tmp_path, capsys):
    assert assets.main(["--dry-run", "--profile", "aligned", "--cache-dir", str(tmp_path / "missing")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_bytes"] > 23_000_000_000
    assert payload["ready"] is False
    assert not (tmp_path / "missing").exists()
