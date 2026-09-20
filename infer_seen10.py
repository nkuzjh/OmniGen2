#!/usr/bin/env python3
"""Deterministic Seen-10 generation inference with resumable, provenance-safe output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/home/jiahao/task/UniLIP/data/csgo_benchmark_v2")
STANDARD_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "csgo_benchmark_v2_seen10" / "OmniGen2"
IMAGE_SIZE = 448
SPLITS = {
    "discrete": ("seen_discrete_test", "discrete_test.json"),
    "continuous": ("seen_continuous", "continuous_clips.json"),
}
FILE_FRAME_RE = re.compile(r"^file_num\d+_frame_\d+$")
SEED_DIRECTORY_RE = re.compile(r"^seed_\d+$")
LORA_WEIGHT_NAMES = ("pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin")
POSE_CONFIG_NAME = "pose_adapter_config.json"
POSE_WEIGHT_NAME = "pose_adapter.bin"


@dataclass(frozen=True)
class SampleIdentity:
    index: int
    sample_id: str
    map_name: str
    file_frame: str
    clip_id: str | None
    frame_index: int | None


@dataclass
class TaskPlan:
    task: str
    split: str
    dataset: Any
    task_root: Path
    identities: list[SampleIdentity]
    base_manifest: dict[str, Any]
    complete: bool = False
    existing_hashes: dict[str, str] | None = None

    @property
    def final_manifest_path(self) -> Path:
        return self.task_root / "inference_manifest.json"

    @property
    def pending_manifest_path(self) -> Path:
        return self.task_root / ".inference_manifest.pending.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CSGO Benchmark v2 Seen-10 discrete and/or continuous samples. "
            "--max-samples is smoke-only and requires a separate --output-root."
        )
    )
    parser.add_argument("--task", choices=("discrete", "continuous", "all"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root containing <task>/gen_imgs; defaults to the standard seed_<seed> result root.",
    )
    parser.add_argument("--model-path", required=True, help="Base OmniGen2 Diffusers pipeline path or Hub id.")
    parser.add_argument("--adapter-path", required=True, help="Converted LoRA directory with pose adapter sidecar.")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Smoke-test limit per task. Use a separate output root, never the formal seed_<n> root.",
    )
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    parser.add_argument("--offload", action="store_true", help="Enable Diffusers model CPU offload (CUDA required).")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("--seed must be a non-negative integer")
    if args.num_inference_steps <= 0:
        raise ValueError("--num-inference-steps must be positive")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be a positive integer")
    if args.max_samples is not None:
        if args.output_root is None:
            raise ValueError("--max-samples requires an explicit, separate --output-root for smoke outputs")
        candidate = Path(args.output_root).expanduser().resolve()
        standard_root = STANDARD_OUTPUT_ROOT.resolve()
        if candidate == standard_root:
            raise ValueError("--max-samples cannot write under the standard formal output root")
        try:
            relative = candidate.relative_to(standard_root)
        except ValueError:
            relative = None
        if relative and relative.parts and SEED_DIRECTORY_RE.fullmatch(relative.parts[0]):
            raise ValueError(
                "--max-samples cannot write under a standard formal seed_<n> output root; "
                "choose a separate smoke output directory"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read inference provenance file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Inference provenance file must contain a JSON object: {path}")
    return value


def _fsync_directory(directory: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reject_symlink_path(path: Path) -> None:
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise FileExistsError(f"Refusing to follow symlink in inference output path: {candidate}")


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> bool:
    """Atomically create JSON without replacing a pre-existing file.

    Returns True when this call created the file and False when an identical
    file was already present. A non-identical file is never overwritten.
    """
    _reject_symlink_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path(path)

    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_path, 0o644)
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            existing = _read_json_object(path)
            if existing != dict(value):
                raise FileExistsError(f"Refusing to overwrite non-identical inference manifest: {path}")
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _atomic_create_jpeg(image: Any, path: Path) -> bool:
    """Atomically create a 448x448 RGB JPEG, never replacing an existing path."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError(f"Pipeline output must be a PIL image, got {type(image).__name__}")
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Generated image must be {IMAGE_SIZE}x{IMAGE_SIZE}, got {image.size}")
    rgb_image = image.convert("RGB")
    _reject_symlink_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_path(path)
    if path.exists():
        _validate_existing_jpeg(path)
        return False

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            # Match the existing UniLIP generation path: RGB conversion followed
            # by Pillow's default JPEG encoder settings.
            rgb_image.save(temporary_file, format="JPEG")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_path, 0o644)
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            _validate_existing_jpeg(path)
            return False
        _fsync_directory(path.parent)
        return True
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _validate_existing_jpeg(path: Path) -> str:
    from PIL import Image

    _reject_symlink_path(path)
    if not path.is_file():
        raise ValueError(f"Existing generated output is not a regular JPEG file: {path}")
    try:
        with Image.open(path) as image:
            if image.format != "JPEG" or image.size != (IMAGE_SIZE, IMAGE_SIZE) or image.mode != "RGB":
                raise ValueError(
                    f"Existing output violates RGB JPEG contract at {path}: "
                    f"format={image.format!r}, size={image.size}, mode={image.mode!r}"
                )
            image.verify()
        with Image.open(path) as image:
            image.load()
            if image.format != "JPEG" or image.size != (IMAGE_SIZE, IMAGE_SIZE) or image.mode != "RGB":
                raise ValueError(f"Existing output changed while being validated: {path}")
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"Existing generated output is corrupt: {path}: {exc}") from exc
    return _sha256_file(path)


def _metadata_from(value: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = value.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata
    encoded = value.get("meta_data")
    if isinstance(encoded, str):
        try:
            decoded = json.loads(encoded)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, Mapping):
            return decoded
    return {}


def _sample_identity_from_mapping(value: Mapping[str, Any], index: int, map_order: tuple[str, ...]) -> SampleIdentity:
    metadata = _metadata_from(value)

    def field(name: str, fallback: Any = None) -> Any:
        direct = value.get(name)
        return direct if direct is not None else metadata.get(name, fallback)

    map_name = field("map_name", field("map"))
    file_frame = field("file_frame")
    if map_name not in map_order:
        raise ValueError(f"Dataset sample {index} has an unsupported map name: {map_name!r}")
    if not isinstance(file_frame, str) or FILE_FRAME_RE.fullmatch(file_frame) is None:
        raise ValueError(f"Dataset sample {index} has an invalid file_frame: {file_frame!r}")
    sample_id = field("sample_id", f"{map_name}/{file_frame}")
    expected_id = f"{map_name}/{file_frame}"
    if sample_id != expected_id:
        raise ValueError(f"Dataset sample identity mismatch at index {index}: {sample_id!r} != {expected_id!r}")

    clip_id = field("clip_id")
    frame_index = field("frame_index")
    if clip_id is not None and not isinstance(clip_id, str):
        raise ValueError(f"Dataset sample {sample_id} has a non-string clip_id")
    if frame_index is not None and (isinstance(frame_index, bool) or not isinstance(frame_index, int)):
        raise ValueError(f"Dataset sample {sample_id} has a non-integer frame_index")
    return SampleIdentity(
        index=index,
        sample_id=sample_id,
        map_name=map_name,
        file_frame=file_frame,
        clip_id=clip_id,
        frame_index=frame_index,
    )


def _sample_identity(dataset: Any, index: int, map_order: tuple[str, ...]) -> SampleIdentity:
    """Read stable metadata in dataset order without opening target images."""
    rows = getattr(dataset, "rows", None)
    if rows is not None:
        row = rows[index]
        if isinstance(row, Mapping):
            return _sample_identity_from_mapping(row, index, map_order)
    item = dataset[index]
    if not isinstance(item, Mapping):
        raise TypeError(f"Dataset item {index} must be a mapping, got {type(item).__name__}")
    return _sample_identity_from_mapping(item, index, map_order)


def _inference_inputs(dataset: Any, identity: SampleIdentity, map_order: tuple[str, ...]):
    import torch
    from PIL import Image

    item = dataset[identity.index]
    if not isinstance(item, Mapping):
        raise TypeError(f"Dataset item {identity.index} must be a mapping")
    if item.get("output_image") is not None:
        raise ValueError(
            "Inference dataset returned a target image; construct CSGOSeen10Dataset with load_target=False"
        )
    actual_identity = _sample_identity_from_mapping(item, identity.index, map_order)
    if actual_identity != identity:
        raise ValueError(f"Dataset identity changed between metadata and item access: {identity.sample_id}")

    prompt = item.get("instruction")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Dataset sample {identity.sample_id} has no instruction")

    radar_images = item.get("input_images_pil")
    if radar_images is None:
        radar_images = item.get("input_images")
    if isinstance(radar_images, Image.Image):
        radar_images = [radar_images]
    if not isinstance(radar_images, (list, tuple)) or len(radar_images) != 1:
        raise ValueError(
            f"Dataset sample {identity.sample_id} must provide exactly one radar RGB image"
        )
    radar_image = radar_images[0]
    if not isinstance(radar_image, Image.Image):
        raise TypeError(
            f"Dataset radar input must be a PIL image, got {type(radar_image).__name__}"
        )
    if radar_image.size != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Radar input must be {IMAGE_SIZE}x{IMAGE_SIZE}, got {radar_image.size}")
    if radar_image.mode != "RGB":
        radar_image = radar_image.convert("RGB")

    pose_values = torch.as_tensor(item.get("pose_values"), dtype=torch.float32)
    if pose_values.shape == (5,):
        pose_values = pose_values.unsqueeze(0)
    if pose_values.shape != (1, 5):
        raise ValueError(
            f"Dataset pose for {identity.sample_id} must have shape [5] or [1, 5], "
            f"got {tuple(pose_values.shape)}"
        )
    if not torch.isfinite(pose_values).all().item():
        raise ValueError(f"Dataset pose for {identity.sample_id} contains non-finite values")
    return prompt, [radar_image], pose_values


def _resolve_radar_paths(dataset: Any, data_root: Path, map_order: tuple[str, ...]) -> dict[str, Path]:
    paths = getattr(dataset, "radar_paths", None)
    if isinstance(paths, Mapping):
        result = {name: Path(paths[name]).expanduser().resolve() for name in map_order}
    else:
        report_path = data_root / "minimal_dataset_report.json"
        report = _read_json_object(report_path)
        entries = report.get("radars", {}).get("entries", [])
        if not isinstance(entries, list):
            raise ValueError("minimal_dataset_report.json radars.entries must be a list")
        target_by_map = {
            entry.get("map"): entry.get("target")
            for entry in entries
            if isinstance(entry, Mapping)
        }
        result = {}
        for name in map_order:
            target = target_by_map.get(name)
            if not isinstance(target, str):
                raise ValueError(f"No radar asset mapping is available for {name}")
            result[name] = (data_root / "radars" / target).resolve()

    resolved_root = data_root.resolve()
    for map_name, path in result.items():
        try:
            path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"Radar asset escapes data root for {map_name}: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Missing radar asset for {map_name}: {path}")
    return result


def _dataset_hashes(
    dataset: Any,
    data_root: Path,
    task: str,
    map_order: tuple[str, ...],
) -> tuple[dict[str, str], dict[str, str]]:
    split_name, split_filename = SPLITS[task]
    protocol_relatives = [
        "benchmark_manifest.json",
        "minimal_dataset_report.json",
        "calibration/z_calibration.json",
    ]
    protocol_relatives.extend(
        f"splits/seen/{map_name}/{split_filename}" for map_name in map_order
    )
    protocol_hashes = {
        relative: _sha256_file(data_root / relative)
        for relative in protocol_relatives
    }
    radar_paths = _resolve_radar_paths(dataset, data_root, map_order)
    radar_hashes = {
        path.relative_to(data_root.resolve()).as_posix(): _sha256_file(path)
        for path in radar_paths.values()
    }
    return protocol_hashes, radar_hashes


def _checkpoint_provenance(model_path_value: str, adapter_path_value: str) -> dict[str, Any]:
    adapter_path = Path(adapter_path_value).expanduser().resolve()
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"--adapter-path must be a local converted adapter directory: {adapter_path}")
    lora_files = [adapter_path / name for name in LORA_WEIGHT_NAMES if (adapter_path / name).is_file()]
    if not lora_files:
        raise FileNotFoundError(
            f"No standard LoRA weight file found in {adapter_path}; expected one of {LORA_WEIGHT_NAMES}"
        )
    for required_name in (POSE_CONFIG_NAME, POSE_WEIGHT_NAME):
        if not (adapter_path / required_name).is_file():
            raise FileNotFoundError(f"Required pose adapter sidecar missing: {adapter_path / required_name}")

    adapter_hashes = {
        path.name: _sha256_file(path)
        for path in [*lora_files, adapter_path / POSE_CONFIG_NAME, adapter_path / POSE_WEIGHT_NAME]
    }
    model_path = Path(model_path_value).expanduser()
    model_config_hashes: dict[str, str] = {}
    if model_path.is_dir():
        for relative in ("model_index.json", "config.json", "transformer/config.json"):
            candidate = model_path / relative
            if candidate.is_file():
                model_config_hashes[relative] = _sha256_file(candidate)

    return {
        "model_path": str(model_path.resolve()) if model_path.exists() else model_path_value,
        "model_config_sha256": model_config_hashes,
        "adapter_path": str(adapter_path),
        "adapter_asset_sha256": adapter_hashes,
    }


def _derive_sample_seed(base_seed: int, task: str, sample_id: str) -> int:
    seed_material = f"csgo_benchmark_v2\0{base_seed}\0{task}\0{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big") & ((1 << 63) - 1)


def _output_path(task_root: Path, identity: SampleIdentity) -> Path:
    return task_root / "gen_imgs" / identity.map_name / f"{identity.file_frame}.jpg"


def _output_manifest_entry(identity: SampleIdentity, relative_path: str, digest: str, seed: int) -> dict[str, Any]:
    return {
        "sample_id": identity.sample_id,
        "map_name": identity.map_name,
        "file_frame": identity.file_frame,
        "clip_id": identity.clip_id,
        "frame_index": identity.frame_index,
        "path": relative_path,
        "seed": seed,
        "sha256": digest,
    }


def _make_task_plan(
    task: str,
    dataset_class: Any,
    map_order: tuple[str, ...],
    args: argparse.Namespace,
    output_root: Path,
    checkpoint: Mapping[str, Any],
) -> TaskPlan:
    split_name, _ = SPLITS[task]
    dataset = dataset_class(
        data_root=Path(args.data_root).expanduser().resolve(),
        split=split_name,
        use_chat_template=False,
        load_target=False,
    )
    if getattr(dataset, "load_target", False) is not False:
        raise ValueError("Inference requires CSGOSeen10Dataset(load_target=False)")
    dataset_length = len(dataset)
    if dataset_length <= 0:
        raise ValueError(f"Dataset split {split_name} is empty")
    sample_count = min(dataset_length, args.max_samples) if args.max_samples is not None else dataset_length
    identities = [_sample_identity(dataset, index, map_order) for index in range(sample_count)]
    sample_ids = [identity.sample_id for identity in identities]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Dataset split {split_name} contains duplicate sample IDs")

    data_root = Path(args.data_root).expanduser().resolve()
    protocol_hashes, radar_hashes = _dataset_hashes(dataset, data_root, task, map_order)
    maps_in_order = list(dict.fromkeys(identity.map_name for identity in identities))
    sample_sequence = "\n".join(sample_ids).encode("utf-8")
    base_manifest = {
        "manifest_version": 1,
        "benchmark_id": "csgo_benchmark_v2",
        "model_name": "OmniGen2",
        "task": task,
        "split": split_name,
        "data_root": str(data_root),
        "protocol": {
            "seen_map_order": list(map_order),
            "maps_in_sample_order": maps_in_order,
            "continuous_order": "manifest clip order then frame order" if task == "continuous" else None,
            "target_images_opened": False,
            "historical_or_future_target_frames_used": False,
        },
        "protocol_asset_sha256": protocol_hashes,
        "radar_asset_sha256": radar_hashes,
        "protocol_sha256": _canonical_sha256(
            {"protocol_assets": protocol_hashes, "radar_assets": radar_hashes}
        ),
        "checkpoint": dict(checkpoint),
        "seed": args.seed,
        "sample_seed_derivation": "sha256('csgo_benchmark_v2\\0<base_seed>\\0<task>\\0<sample_id>') first 8 bytes masked to 63 bits",
        "num_inference_steps": args.num_inference_steps,
        "dtype": args.dtype,
        "offload": bool(args.offload),
        "guidance": {
            "text_guidance_scale": 4.0,
            "image_guidance_scale": 1.0,
            "cfg_range": [0.0, 1.0],
        },
        "image": {
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "radar_width": IMAGE_SIZE,
            "radar_height": IMAGE_SIZE,
            "mode": "RGB",
            "format": "JPEG",
            "encoder": "Pillow default JPEG settings",
            "align_res": False,
        },
        "sample_count": sample_count,
        "sample_ids_sha256": hashlib.sha256(sample_sequence).hexdigest(),
        "inference_script_sha256": _sha256_file(Path(__file__).resolve()),
        "smoke": args.max_samples is not None,
        "max_samples": args.max_samples,
    }
    return TaskPlan(
        task=task,
        split=split_name,
        dataset=dataset,
        task_root=output_root / task,
        identities=identities,
        base_manifest=base_manifest,
    )


def _all_jpeg_outputs(task_root: Path) -> set[Path]:
    image_root = task_root / "gen_imgs"
    if not image_root.exists():
        return set()
    if image_root.is_symlink():
        raise ValueError(f"Refusing to follow symlinked generation directory: {image_root}")
    return {path.resolve() for path in image_root.rglob("*.jpg") if path.is_file() or path.is_symlink()}


def _expected_output_paths(plan: TaskPlan) -> list[Path]:
    return [_output_path(plan.task_root, identity) for identity in plan.identities]


def _output_hash_entries(plan: TaskPlan, digests: list[str], seed: int) -> list[dict[str, Any]]:
    entries = []
    for identity, path, digest in zip(plan.identities, _expected_output_paths(plan), digests):
        entries.append(
            _output_manifest_entry(
                identity=identity,
                relative_path=path.relative_to(plan.task_root).as_posix(),
                digest=digest,
                seed=_derive_sample_seed(seed, plan.task, identity.sample_id),
            )
        )
    return entries


def _validate_complete_manifest(plan: TaskPlan, args: argparse.Namespace) -> bool:
    manifest_path = plan.final_manifest_path
    if not manifest_path.exists():
        return False
    if manifest_path.is_symlink():
        raise ValueError(f"Refusing to read a symlinked inference manifest: {manifest_path}")
    existing_manifest = _read_json_object(manifest_path)
    existing_base = {key: value for key, value in existing_manifest.items() if key != "output_hashes"}
    if existing_base != plan.base_manifest:
        raise FileExistsError(
            f"Existing {plan.task} inference manifest differs from this run; refusing to overwrite: {manifest_path}"
        )

    expected_paths = _expected_output_paths(plan)
    expected_resolved = {path.resolve() for path in expected_paths}
    unexpected = _all_jpeg_outputs(plan.task_root) - expected_resolved
    if unexpected:
        raise FileExistsError(f"Unexpected existing generation outputs under {plan.task_root}: {sorted(unexpected)[:3]}")

    digests = []
    for identity, path in zip(plan.identities, expected_paths):
        _reject_symlink_path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Completed inference manifest references a missing image; refusing implicit repair: {path}"
            )
        digests.append(_validate_existing_jpeg(path))
    expected_manifest = dict(plan.base_manifest)
    expected_manifest["output_hashes"] = _output_hash_entries(plan, digests, args.seed)
    if existing_manifest != expected_manifest:
        raise ValueError(
            f"Existing outputs or hashes disagree with {manifest_path}; refusing to overwrite or mix results"
        )

    pending_path = plan.pending_manifest_path
    if pending_path.exists():
        if pending_path.is_symlink() or _read_json_object(pending_path) != plan.base_manifest:
            raise FileExistsError(f"A conflicting partial inference marker exists: {pending_path}")
        pending_path.unlink()
        _fsync_directory(pending_path.parent)
    return True


def _prepare_partial_resume(plan: TaskPlan) -> None:
    pending_path = plan.pending_manifest_path
    if pending_path.exists():
        if pending_path.is_symlink() or _read_json_object(pending_path) != plan.base_manifest:
            raise FileExistsError(
                f"Existing partial inference marker differs from this run; refusing to mix outputs: {pending_path}"
            )
    else:
        existing = _all_jpeg_outputs(plan.task_root)
        if existing:
            raise FileExistsError(
                f"Found untracked JPEG outputs without a matching manifest; refusing to reuse them: "
                f"{sorted(existing)[:3]}"
            )
        _atomic_create_json(pending_path, plan.base_manifest)

    expected_paths = _expected_output_paths(plan)
    expected_resolved = {path.resolve() for path in expected_paths}
    unexpected = _all_jpeg_outputs(plan.task_root) - expected_resolved
    if unexpected:
        raise FileExistsError(f"Unexpected existing generation outputs under {plan.task_root}: {sorted(unexpected)[:3]}")

    plan.existing_hashes = {}
    for identity, path in zip(plan.identities, expected_paths):
        _reject_symlink_path(path)
        if path.exists():
            plan.existing_hashes[identity.sample_id] = _validate_existing_jpeg(path)


def _load_pipeline(args: argparse.Namespace):
    import torch

    from convert_ckpt_to_hf_format import load_pose_adapter
    from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline

    dtype_by_name = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    weight_dtype = dtype_by_name[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.offload and device.type != "cuda":
        raise RuntimeError("--offload requires an available CUDA device")
    if device.type == "cpu" and weight_dtype == torch.float16:
        raise ValueError("CPU inference with fp16 is unsupported; use --dtype fp32 or bf16")

    transformer = OmniGen2Transformer2DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
    )
    load_pose_adapter(transformer, args.adapter_path)
    pipeline = OmniGen2Pipeline.from_pretrained(
        args.model_path,
        transformer=transformer,
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )
    # The converted directory contains both LoRA weights and the independent
    # pose-adapter sidecar; the adapter is loaded once and shared by both tasks.
    pipeline.load_lora_weights(args.adapter_path)

    if args.offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)
    return pipeline


def _generate_one(pipeline: Any, prompt: str, radar_images: list[Any], pose_values: Any, seed: int, steps: int):
    import torch

    device = getattr(pipeline, "_execution_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipeline(
        prompt=prompt,
        input_images=radar_images,
        pose_values=pose_values,
        width=IMAGE_SIZE,
        height=IMAGE_SIZE,
        align_res=False,
        max_pixels=IMAGE_SIZE * IMAGE_SIZE,
        max_input_image_side_length=IMAGE_SIZE,
        num_inference_steps=steps,
        num_images_per_prompt=1,
        generator=generator,
        output_type="pil",
        return_dict=True,
    )
    images = getattr(result, "images", None)
    if not isinstance(images, (list, tuple)) or len(images) != 1:
        raise ValueError("OmniGen2 pipeline must return exactly one image per Seen-10 sample")
    return images[0]


def _run_task(plan: TaskPlan, pipeline: Any, args: argparse.Namespace, map_order: tuple[str, ...]) -> None:
    expected_paths = _expected_output_paths(plan)
    digests: list[str] = []
    generated = 0
    skipped = 0
    total = len(plan.identities)

    for offset, (identity, output_path) in enumerate(zip(plan.identities, expected_paths), start=1):
        _reject_symlink_path(output_path)
        if output_path.exists():
            digest = _validate_existing_jpeg(output_path)
            skipped += 1
        else:
            prompt, radar_images, pose_values = _inference_inputs(plan.dataset, identity, map_order)
            sample_seed = _derive_sample_seed(args.seed, plan.task, identity.sample_id)
            generated_image = _generate_one(
                pipeline=pipeline,
                prompt=prompt,
                radar_images=radar_images,
                pose_values=pose_values,
                seed=sample_seed,
                steps=args.num_inference_steps,
            )
            _atomic_create_jpeg(generated_image, output_path)
            digest = _validate_existing_jpeg(output_path)
            generated += 1
        digests.append(digest)

        if offset == 1 or offset % 100 == 0 or offset == total:
            print(
                f"[{plan.task}] {offset}/{total} processed "
                f"(generated={generated}, reused={skipped})",
                flush=True,
            )

    final_manifest = dict(plan.base_manifest)
    final_manifest["output_hashes"] = _output_hash_entries(plan, digests, args.seed)
    _atomic_create_json(plan.final_manifest_path, final_manifest)

    pending_path = plan.pending_manifest_path
    if pending_path.exists():
        if pending_path.is_symlink() or _read_json_object(pending_path) != plan.base_manifest:
            raise FileExistsError(f"Partial inference marker changed during generation: {pending_path}")
        pending_path.unlink()
        _fsync_directory(pending_path.parent)
    print(f"[{plan.task}] manifest: {plan.final_manifest_path}", flush=True)


def main(args: argparse.Namespace) -> None:
    _validate_args(args)
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root is not None
        else STANDARD_OUTPUT_ROOT / f"seed_{args.seed}"
    )
    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"--data-root is not a directory: {data_root}")

    from omnigen2.dataset.csgo_seen10_dataset import CSGOSeen10Dataset, SEEN_MAPS

    map_order = tuple(SEEN_MAPS)
    checkpoint = _checkpoint_provenance(args.model_path, args.adapter_path)
    tasks = ("discrete", "continuous") if args.task == "all" else (args.task,)
    plans = [
        _make_task_plan(
            task=task,
            dataset_class=CSGOSeen10Dataset,
            map_order=map_order,
            args=args,
            output_root=output_root,
            checkpoint=checkpoint,
        )
        for task in tasks
    ]

    for plan in plans:
        plan.complete = _validate_complete_manifest(plan, args)
        if not plan.complete:
            _prepare_partial_resume(plan)
        else:
            print(f"[{plan.task}] all {len(plan.identities)} verified outputs already exist; reusing manifest")

    incomplete_plans = [plan for plan in plans if not plan.complete]
    if not incomplete_plans:
        return

    pipeline = _load_pipeline(args)
    for plan in incomplete_plans:
        _run_task(plan, pipeline, args, map_order)


if __name__ == "__main__":
    main(parse_args())
