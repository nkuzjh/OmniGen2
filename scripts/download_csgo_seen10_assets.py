#!/usr/bin/env python3
"""Prepare pinned official HF components; --check is offline and read-only.

No Torch import, CUDA initialization, checkpoint rewrite or refs/main mutation.
Legacy training needs the same three repositories as aligned; legacy inference
additionally loads OmniGen2's bundled mllm/processor/vae components.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("csgo_seen10_assets.json")


def anchored(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT / path


def cache_root(explicit=None, environ=None) -> Path:
    env = os.environ if environ is None else environ
    if explicit:
        return anchored(explicit)
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if env.get(name):
            return anchored(env[name])
    if env.get("HF_HOME"):
        return anchored(env["HF_HOME"]) / "hub"
    return anchored(env.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface/hub"


def manifest_repositories():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload["schema_version"] != 1:
        raise ValueError("Unsupported asset manifest")
    return payload["repositories"]


def snapshot_path(root: Path, repository: dict) -> Path:
    return root / ("models--" + repository["repo_id"].replace("/", "--")) / "snapshots" / repository["revision"]


def selected_assets(repositories, profile):
    for repository in repositories:
        for asset in repository["files"]:
            if profile == "all" or profile in asset["profiles"]:
                yield repository, asset


def check_file(path: Path, asset: dict) -> str:
    if not path.is_file():
        return "missing"
    size = path.stat().st_size
    if size != asset["size"]:
        return "size_mismatch"
    algorithm = asset["algorithm"]
    if algorithm == "git-blob-sha1":
        digest = hashlib.sha1()
        digest.update(f"blob {size}\0".encode("ascii"))
    elif algorithm == "sha256":
        digest = hashlib.sha256()
    else:
        raise ValueError(f"Unknown digest algorithm: {algorithm}")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return "ok" if digest.hexdigest() == asset["digest"] else "hash_mismatch"


def audit_assets(root, repositories, profile):
    results = []
    for repository, asset in selected_assets(repositories, profile):
        path = snapshot_path(root, repository) / asset["filename"]
        results.append({"repo_id": repository["repo_id"], "filename": asset["filename"],
                        "path": str(path), "bytes": asset["size"],
                        "status": check_file(path, asset)})
    return results


def download_missing(root, repositories, profile, results, downloader=None):
    # Never repair an existing corrupt final file implicitly: the cache may be
    # shared by another experiment. A missing file may have resumable Hub .incomplete data.
    corrupt = [r for r in results if r["status"] not in ("ok", "missing")]
    if corrupt:
        raise RuntimeError("Corrupt cached assets found; use a new HF cache or investigate manually. Existing files were not overwritten.")
    missing = {(r["repo_id"], r["filename"]) for r in results if r["status"] == "missing"}
    if not missing:
        return
    if downloader is None:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download
    for repository, asset in selected_assets(repositories, profile):
        if (repository["repo_id"], asset["filename"]) not in missing:
            continue
        print(f"Downloading {repository['repo_id']}/{asset['filename']} ({asset['size']} bytes)", file=sys.stderr, flush=True)
        try:
            downloader(repo_id=repository["repo_id"], revision=repository["revision"],
                       filename=asset["filename"], cache_dir=str(root))
        except Exception as exc:
            # Do not echo credentials, request headers or tokenized URLs.
            raise RuntimeError(f"Download failed ({type(exc).__name__}) for {repository['repo_id']}/{asset['filename']}. Check network/HF_ENDPOINT and HF authentication; FLUX.1-dev requires accepted access terms.") from None
        path = snapshot_path(root, repository) / asset["filename"]
        state = check_file(path, asset)
        if state != "ok":
            raise RuntimeError(f"Downloaded file failed integrity check ({state}): {path}")
        for result in results:
            if (result["repo_id"], result["filename"]) == (repository["repo_id"], asset["filename"]):
                result["status"] = "ok"


def print_environment(root, repositories):
    # Deliberately explicit: downloading by commit does not update refs/main.
    # The caller sources these quoted paths to pin both legacy and aligned loads.
    for repository in repositories:
        print(f"export {repository['env']}={shlex.quote(str(snapshot_path(root, repository)))}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Offline full byte/hash validation; no downloads, installs or writes")
    mode.add_argument("--print-env", action="store_true", help="Print shell exports for fixed official snapshots (does not check/download)")
    mode.add_argument("--dry-run", action="store_true", help="List pinned files and total bytes without hashing/downloading")
    parser.add_argument("--profile", choices=("aligned", "legacy", "all"), default="all",
                        help="aligned components only, or legacy/all including full native inference pipeline")
    parser.add_argument("--cache-dir", help="HF Hub cache root; relative paths are anchored at checkout root")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable audit/plan")
    args = parser.parse_args(argv)
    root = cache_root(args.cache_dir)
    repositories = manifest_repositories()
    if args.print_env:
        print_environment(root, repositories)
        return 0
    if args.dry_run:
        results = [{"repo_id": repo["repo_id"], "revision": repo["revision"],
                    "filename": asset["filename"], "bytes": asset["size"], "status": "not_checked"}
                   for repo, asset in selected_assets(repositories, args.profile)]
    else:
        results = audit_assets(root, repositories, args.profile)
        if not args.check:
            download_missing(root, repositories, args.profile, results)
    ready = not args.dry_run and all(r["status"] == "ok" for r in results)
    payload = {"profile": args.profile, "cache_dir": str(root), "ready": ready,
               "total_bytes": sum(r["bytes"] for r in results), "files": results}
    if args.json or args.dry_run:
        print(json.dumps(payload, indent=2))
    else:
        for result in results:
            print(f"{result['status'].upper()}: {result['repo_id']}/{result['filename']}")
        print(f"{len(results)} files; {payload['total_bytes'] / 1e9:.2f} GB; ready={ready}")
        if ready:
            print("Pin runtime paths (also needed if refs/main is absent):")
            print_environment(root, repositories)
    return 0 if args.dry_run or ready else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
