"""Manifest-driven CSGO Benchmark v2 Seen-10 dataset for OmniGen2."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


SEEN_MAPS = (
    "cs_agency",
    "cs_italy",
    "de_ancient",
    "de_anubis",
    "de_dust2",
    "de_inferno",
    "de_mirage",
    "de_nuke",
    "de_overpass",
    "de_train",
)

_BENCHMARK_ID = "csgo_benchmark_v2"
_IMAGE_SIZE = 448
_IMAGE_AREA = _IMAGE_SIZE * _IMAGE_SIZE
_IMAGE_TEMPLATE = "images/{map}/{file_frame}.jpg"
_FILE_FRAME_RE = re.compile(r"^file_num(?P<file_num>\d+)_frame_(?P<frame_id>\d+)$")
_SPLITS = {
    "seen_train": ("train", "train.json"),
    "seen_validation": ("validation", "validation.json"),
    "seen_discrete_test": ("discrete_test", "discrete_test.json"),
    "seen_continuous": ("continuous_frames", "continuous_clips.json"),
}
_COUNT_KEYS = ("train", "validation", "discrete_test", "continuous_clips", "continuous_frames")
_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")


class CSGOSeen10DatasetError(ValueError):
    """Raised when a Benchmark v2 bundle violates its published data contract."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required CSGO Benchmark v2 file not found: {path}") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CSGOSeen10DatasetError(f"Cannot read JSON file {path}: {exc}") from exc


def _require_object(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CSGOSeen10DatasetError(f"{description} must be a JSON object")
    return value


def _require_integer(value: Any, description: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CSGOSeen10DatasetError(f"{description} must be an integer >= {minimum}, got {value!r}")
    return value


def _finite_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CSGOSeen10DatasetError(f"{description} must be a finite number, got {value!r}")
    converted = float(value)
    if not math.isfinite(converted):
        raise CSGOSeen10DatasetError(f"{description} must be a finite number, got {value!r}")
    return converted


def _pixel_limit(value: Any, description: str, *, default: int) -> int:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError(f"{description} cannot be an empty sequence")
        value = value[0]
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a positive integer, got {value!r}")
    converted = int(value)
    if converted != value or converted < _IMAGE_AREA:
        raise ValueError(f"{description} must be at least {_IMAGE_AREA} to preserve 448x448 inputs")
    return converted


def _side_limit(value: Any) -> int:
    if value is None:
        return _IMAGE_SIZE
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"max_side_length must be a positive integer, got {value!r}")
    converted = int(value)
    if converted != value or converted < _IMAGE_SIZE:
        raise ValueError(f"max_side_length must be at least {_IMAGE_SIZE} to preserve 448x448 inputs")
    return converted


class CSGOSeen10Dataset(Dataset):
    """Read a published Seen-10 split and provide OmniGen2-ready samples.

    With load_target=False, intended for inference, the dataset creates the
    target path in metadata but never opens the target image. Radar and target
    images are PIL-bicubic-resized to 448x448 before the OmniGen2 processor;
    processor output is checked to remain 3x448x448.
    """

    SYSTEM_PROMPT = "You are a helpful assistant that generates high-quality images based on user instructions."
    SYSTEM_PROMPT_DROP = "You are a helpful assistant that generates images."

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        tokenizer: Any | None = None,
        use_chat_template: bool = False,
        max_input_pixels: int | Sequence[int] | None = 1024 * 1024,
        max_output_pixels: int | None = 1024 * 1024,
        max_side_length: int | None = 2048,
        load_target: bool = True,
        image_size: int = _IMAGE_SIZE,
        prompt_dropout_prob: float = 0.0,
        ref_img_dropout_prob: float = 0.0,
        *,
        image_processor: Any | None = None,
    ):
        if split not in _SPLITS:
            raise ValueError(f"Unsupported split {split!r}; expected one of {tuple(_SPLITS)}")
        if not isinstance(load_target, bool):
            raise ValueError("load_target must be a bool")
        if use_chat_template and (tokenizer is None or not hasattr(tokenizer, "apply_chat_template")):
            raise ValueError("use_chat_template=True requires a tokenizer with apply_chat_template")

        self.data_root = Path(data_root).expanduser().resolve()
        self.split = split
        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template
        self.load_target = load_target
        if image_size != _IMAGE_SIZE:
            raise ValueError(f"CSGO benchmark images must use image_size={_IMAGE_SIZE}, got {image_size!r}")
        self.image_size = image_size
        self.prompt_dropout_prob = self._probability(prompt_dropout_prob, "prompt_dropout_prob")
        self.ref_img_dropout_prob = self._probability(ref_img_dropout_prob, "ref_img_dropout_prob")
        self.max_input_pixels = _pixel_limit(
            max_input_pixels, "max_input_pixels", default=_IMAGE_AREA
        )
        self.max_output_pixels = _pixel_limit(
            max_output_pixels, "max_output_pixels", default=_IMAGE_AREA
        )
        self.max_side_length = _side_limit(max_side_length)
        self._image_processor = image_processor

        self.manifest = self._read_object("benchmark_manifest.json")
        self.report = self._read_object("minimal_dataset_report.json")
        self._validate_manifest_and_report()
        self.z_ranges = self._load_z_ranges()
        self.radar_paths = self._load_radar_paths()
        self.rows = self._load_split_rows()

    @staticmethod
    def _probability(value: Any, description: str) -> float:
        probability = _finite_number(value, description)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{description} must be between 0 and 1, got {value!r}")
        return probability

    def _read_object(self, relative_path: str) -> Mapping[str, Any]:
        value = _read_json(self.data_root / relative_path)
        return _require_object(value, relative_path)

    def _validate_manifest_and_report(self) -> None:
        manifest = self.manifest
        report = self.report
        if manifest.get("benchmark_id") != _BENCHMARK_ID:
            raise CSGOSeen10DatasetError("benchmark_manifest.json is not csgo_benchmark_v2")
        if manifest.get("schema_version") != 1:
            raise CSGOSeen10DatasetError("benchmark_manifest.json schema_version must be 1")
        benchmark = _require_object(manifest.get("benchmark"), "manifest.benchmark")
        if benchmark.get("id") != _BENCHMARK_ID or benchmark.get("strict_protocol") is not True:
            raise CSGOSeen10DatasetError("manifest benchmark id/protocol does not match strict CSGO v2")

        protocol = _require_object(manifest.get("protocol"), "manifest.protocol")
        if tuple(protocol.get("seen_maps", ())) != SEEN_MAPS:
            raise CSGOSeen10DatasetError(
                f"Seen-10 map order must be {SEEN_MAPS!r}, got {protocol.get('seen_maps')!r}"
            )
        if tuple(protocol.get("seen_splits", ())) != ("train", "validation", "discrete_test"):
            raise CSGOSeen10DatasetError("manifest.protocol.seen_splits must be train/validation/discrete_test")

        if report.get("benchmark_id") != _BENCHMARK_ID:
            raise CSGOSeen10DatasetError("minimal_dataset_report.json is not csgo_benchmark_v2")
        if report.get("schema_version") != 1:
            raise CSGOSeen10DatasetError("minimal_dataset_report.json schema_version must be 1")
        if report.get("status") != "verified":
            raise CSGOSeen10DatasetError(
                f"minimal dataset report status must be verified, got {report.get('status')!r}"
            )

        images = _require_object(report.get("images"), "report.images")
        if images.get("status") != "verified":
            raise CSGOSeen10DatasetError("report.images.status must be verified")
        if images.get("root") != "images" or images.get("target_template") != _IMAGE_TEMPLATE:
            raise CSGOSeen10DatasetError(
                f"report image layout must use images/<map>/<file_frame>.jpg; "
                f"got {images.get('target_template')!r}"
            )

        radars = _require_object(report.get("radars"), "report.radars")
        if radars.get("status") != "verified":
            raise CSGOSeen10DatasetError("report.radars.status must be verified")
        if radars.get("root") != "radars":
            raise CSGOSeen10DatasetError("report.radars.root must be radars")

        count_root = _require_object(manifest.get("counts"), "manifest.counts")
        counts = _require_object(count_root.get("seen"), "manifest.counts.seen")
        if tuple(counts.keys()) != SEEN_MAPS:
            raise CSGOSeen10DatasetError("manifest.counts.seen must list exactly the fixed Seen-10 maps in order")
        self._counts: dict[str, dict[str, int]] = {}
        for map_name in SEEN_MAPS:
            map_counts = _require_object(counts.get(map_name), f"manifest.counts.seen.{map_name}")
            self._counts[map_name] = {
                key: _require_integer(map_counts.get(key), f"counts.seen.{map_name}.{key}", minimum=1)
                for key in _COUNT_KEYS
            }

        continuous_protocol = _require_object(
            manifest.get("continuous_protocol"), "manifest.continuous_protocol"
        )
        self.frames_per_clip = _require_integer(
            continuous_protocol.get("frames_per_clip"), "continuous_protocol.frames_per_clip", minimum=1
        )
        self.max_frame_gap = _require_integer(
            continuous_protocol.get("max_frame_gap"), "continuous_protocol.max_frame_gap", minimum=1
        )
        for map_name, map_counts in self._counts.items():
            expected_frames = map_counts["continuous_clips"] * self.frames_per_clip
            if map_counts["continuous_frames"] != expected_frames:
                raise CSGOSeen10DatasetError(
                    f"Manifest continuous counts disagree for {map_name}: "
                    f"{map_counts['continuous_clips']} clips x {self.frames_per_clip} frames "
                    f"!= {map_counts['continuous_frames']} frames"
                )

        image_by_map = _require_object(images.get("by_map"), "report.images.by_map")
        total_reported_images = 0
        for map_name, item in image_by_map.items():
            map_image_info = _require_object(item, f"report.images.by_map.{map_name}")
            total_reported_images += _require_integer(
                map_image_info.get("count"), f"report.images.by_map.{map_name}.count", minimum=0
            )
        reported_total = _require_integer(images.get("count"), "report.images.count", minimum=0)
        if reported_total != total_reported_images:
            raise CSGOSeen10DatasetError(
                f"report.images.count mismatch: expected sum {total_reported_images}, got {reported_total}"
            )
        for map_name, map_counts in self._counts.items():
            map_image_info = _require_object(
                image_by_map.get(map_name), f"report.images.by_map.{map_name}"
            )
            expected_image_count = sum(
                map_counts[key] for key in ("train", "validation", "discrete_test", "continuous_frames")
            )
            actual_image_count = _require_integer(
                map_image_info.get("count"), f"report.images.by_map.{map_name}.count", minimum=0
            )
            if actual_image_count != expected_image_count:
                raise CSGOSeen10DatasetError(
                    f"Image count mismatch for {map_name}: manifest splits total {expected_image_count}, "
                    f"report says {actual_image_count}"
                )

    def _load_z_ranges(self) -> dict[str, dict[str, float]]:
        calibration_meta = _require_object(self.manifest.get("calibration"), "manifest.calibration")
        relative_path = calibration_meta.get("file")
        if relative_path != "calibration/z_calibration.json":
            raise CSGOSeen10DatasetError(
                f"manifest.calibration.file must be calibration/z_calibration.json, got {relative_path!r}"
            )
        calibration_path = (self.data_root / relative_path).resolve()
        if self.data_root not in calibration_path.parents:
            raise CSGOSeen10DatasetError(f"Calibration path escapes data root: {relative_path}")
        raw_bytes = calibration_path.read_bytes()
        actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        if calibration_meta.get("sha256") != actual_sha256:
            raise CSGOSeen10DatasetError("Manifest and z_calibration.json SHA-256 differ")

        calibration = _require_object(_read_json(calibration_path), "z_calibration.json")
        if calibration.get("benchmark_id") != _BENCHMARK_ID or calibration.get("schema_version") != 1:
            raise CSGOSeen10DatasetError("z_calibration.json benchmark id/schema_version mismatch")
        fingerprint = calibration_meta.get("fingerprint")
        if not fingerprint or calibration.get("calibration_sha256") != fingerprint:
            raise CSGOSeen10DatasetError("Manifest and z_calibration.json fingerprints differ")
        if calibration.get("decision_review_status") != "approved":
            raise CSGOSeen10DatasetError("z_calibration.json decision_review_status must be approved")

        manifest_ranges = _require_object(calibration_meta.get("z_ranges"), "manifest.calibration.z_ranges")
        calibration_ranges = _require_object(calibration.get("z_ranges"), "calibration.z_ranges")
        ranges: dict[str, dict[str, float]] = {}
        for map_name in SEEN_MAPS:
            manifest_range = _require_object(
                manifest_ranges.get(map_name), f"manifest.calibration.z_ranges.{map_name}"
            )
            calibration_range = _require_object(
                calibration_ranges.get(map_name), f"calibration.z_ranges.{map_name}"
            )
            z_min = _finite_number(manifest_range.get("z_min"), f"{map_name}.z_min")
            z_max = _finite_number(manifest_range.get("z_max"), f"{map_name}.z_max")
            file_z_min = _finite_number(calibration_range.get("z_min"), f"calibration.{map_name}.z_min")
            file_z_max = _finite_number(calibration_range.get("z_max"), f"calibration.{map_name}.z_max")
            if (z_min, z_max) != (file_z_min, file_z_max):
                raise CSGOSeen10DatasetError(f"Manifest and z_calibration.json ranges differ for {map_name}")
            if not z_max > z_min:
                raise CSGOSeen10DatasetError(f"Invalid z range for {map_name}: {z_min}, {z_max}")
            ranges[map_name] = {"z_min": z_min, "z_max": z_max}
        return ranges

    def _load_radar_paths(self) -> dict[str, Path]:
        radar_info = _require_object(self.report.get("radars"), "report.radars")
        entries = radar_info.get("entries")
        if not isinstance(entries, list):
            raise CSGOSeen10DatasetError("report.radars.entries must be a list")
        targets: dict[str, str] = {}
        for index, entry in enumerate(entries):
            radar_entry = _require_object(entry, f"report.radars.entries[{index}]")
            map_name = radar_entry.get("map")
            target = radar_entry.get("target")
            if not isinstance(map_name, str) or not isinstance(target, str):
                raise CSGOSeen10DatasetError(f"Radar entry {index} requires string map and target")
            if map_name in targets:
                raise CSGOSeen10DatasetError(f"Duplicate radar mapping for {map_name}")
            path_parts = PurePosixPath(target)
            if (
                target.startswith("/")
                or "\\" in target
                or len(path_parts.parts) != 2
                or path_parts.parts[0] != map_name
                or any(part in ("", ".", "..") for part in path_parts.parts)
            ):
                raise CSGOSeen10DatasetError(
                    f"Radar target must be a relative <map>/<filename> path, got {target!r}"
                )
            targets[map_name] = target

        missing = [map_name for map_name in SEEN_MAPS if map_name not in targets]
        if missing:
            raise CSGOSeen10DatasetError(f"Minimal dataset report is missing Seen-10 radar mappings: {missing}")

        result: dict[str, Path] = {}
        radar_root = (self.data_root / "radars").resolve()
        for map_name in SEEN_MAPS:
            path = (radar_root / targets[map_name]).resolve()
            if radar_root not in path.parents:
                raise CSGOSeen10DatasetError(f"Radar target escapes radar root for {map_name}: {targets[map_name]}")
            if not path.is_file():
                raise FileNotFoundError(f"Missing radar for {map_name}: {path}")
            result[map_name] = path
        return result

    def _load_split_rows(self) -> list[dict[str, Any]]:
        count_key, filename = _SPLITS[self.split]
        result: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for map_name in SEEN_MAPS:
            split_path = self.data_root / "splits" / "seen" / map_name / filename
            payload = _read_json(split_path)
            if self.split == "seen_continuous":
                map_rows = self._load_continuous_map(map_name, payload)
            else:
                if not isinstance(payload, list):
                    raise CSGOSeen10DatasetError(f"Expected JSON array in {split_path}")
                expected_count = self._counts[map_name][count_key]
                if len(payload) != expected_count:
                    raise CSGOSeen10DatasetError(
                        f"Manifest count mismatch for {map_name}/{count_key}: "
                        f"expected {expected_count}, got {len(payload)}"
                    )
                map_rows = [
                    self._normalise_row(map_name, raw, clip_id=None, frame_index=None)
                    for raw in payload
                ]

            for row in map_rows:
                if row["sample_id"] in seen_ids:
                    raise CSGOSeen10DatasetError(f"Duplicate sample identity in {self.split}: {row['sample_id']}")
                seen_ids.add(row["sample_id"])
                result.append(row)
        return result

    def _load_continuous_map(self, map_name: str, payload: Any) -> list[dict[str, Any]]:
        continuous = _require_object(payload, f"continuous_clips.json for {map_name}")
        if continuous.get("benchmark_id") != _BENCHMARK_ID:
            raise CSGOSeen10DatasetError(f"Continuous split benchmark id mismatch for {map_name}")
        clips = continuous.get("clips")
        if not isinstance(clips, list):
            raise CSGOSeen10DatasetError(f"Continuous split clips must be a list for {map_name}")
        expected_clips = self._counts[map_name]["continuous_clips"]
        expected_frames = self._counts[map_name]["continuous_frames"]
        if len(clips) != expected_clips:
            raise CSGOSeen10DatasetError(
                f"Manifest clip count mismatch for {map_name}: expected {expected_clips}, got {len(clips)}"
            )

        rows: list[dict[str, Any]] = []
        clip_ids: set[str] = set()
        for clip in clips:
            clip_object = _require_object(clip, f"continuous clip for {map_name}")
            clip_id = clip_object.get("clip_id")
            frames = clip_object.get("frames")
            if not isinstance(clip_id, str) or not clip_id:
                raise CSGOSeen10DatasetError(f"Continuous clip for {map_name} has no clip_id")
            if clip_id in clip_ids:
                raise CSGOSeen10DatasetError(f"Duplicate continuous clip id for {map_name}: {clip_id}")
            clip_ids.add(clip_id)
            if not isinstance(frames, list):
                raise CSGOSeen10DatasetError(f"Continuous clip {clip_id} frames must be a list")
            if len(frames) != self.frames_per_clip:
                raise CSGOSeen10DatasetError(
                    f"Clip length mismatch for {clip_id}: expected {self.frames_per_clip}, got {len(frames)}"
                )

            clip_rows = [
                self._normalise_row(map_name, frame, clip_id=clip_id, frame_index=index)
                for index, frame in enumerate(frames)
            ]
            self._validate_frame_order(clip_id, clip_rows)
            rows.extend(clip_rows)

        if len(rows) != expected_frames:
            raise CSGOSeen10DatasetError(
                f"Manifest continuous frame count mismatch for {map_name}: expected {expected_frames}, got {len(rows)}"
            )
        return rows

    def _normalise_row(
        self,
        map_name: str,
        row: Any,
        *,
        clip_id: str | None,
        frame_index: int | None,
    ) -> dict[str, Any]:
        row_object = _require_object(row, f"split row for {map_name}")
        if row_object.get("map") != map_name:
            raise CSGOSeen10DatasetError(
                f"Row map mismatch: expected {map_name}, got {row_object.get('map')!r}"
            )
        file_frame = row_object.get("file_frame")
        if not isinstance(file_frame, str) or _FILE_FRAME_RE.fullmatch(file_frame) is None:
            raise CSGOSeen10DatasetError(f"Invalid file_frame identity {file_frame!r}")

        x = _finite_number(row_object.get("x"), f"{map_name}/{file_frame}.x")
        y = _finite_number(row_object.get("y"), f"{map_name}/{file_frame}.y")
        z = _finite_number(row_object.get("z"), f"{map_name}/{file_frame}.z")
        angle_h = _finite_number(row_object.get("angle_h"), f"{map_name}/{file_frame}.angle_h")
        angle_v = _finite_number(row_object.get("angle_v"), f"{map_name}/{file_frame}.angle_v")

        z_min = self.z_ranges[map_name]["z_min"]
        z_max = self.z_ranges[map_name]["z_max"]
        pose_values = (
            x / 1024.0,
            y / 1024.0,
            (z - z_min) / (z_max - z_min),
            angle_v / (2.0 * math.pi),
            angle_h / (2.0 * math.pi),
        )
        image_relative = self.report["images"]["target_template"].format(
            map=map_name, file_frame=file_frame
        )
        image_path = (self.data_root / image_relative).resolve()
        if self.data_root not in image_path.parents:
            raise CSGOSeen10DatasetError(f"Image path escapes data root: {image_relative}")
        radar_path = self.radar_paths[map_name]
        parsed = _FILE_FRAME_RE.fullmatch(file_frame)
        if parsed is None:
            raise CSGOSeen10DatasetError(f"Invalid file_frame identity {file_frame!r}")
        sample_id = f"{map_name}/{file_frame}"
        raw_pose = {
            "x": x,
            "y": y,
            "z": z,
            "angle_h": angle_h,
            "angle_v": angle_v,
            "pitch": angle_v,
            "yaw": angle_h,
        }
        metadata = {
            "benchmark_id": _BENCHMARK_ID,
            "split": self.split,
            "sample_id": sample_id,
            "file_frame": file_frame,
            "map_name": map_name,
            "clip_id": clip_id,
            "frame_index": frame_index,
            "image_path": str(image_path),
            "radar_path": str(radar_path),
            "raw_pose": raw_pose,
        }
        return {
            "sample_id": sample_id,
            "map_name": map_name,
            "file_frame": file_frame,
            "image_path": image_path,
            "radar_path": radar_path,
            "pose_values": pose_values,
            "raw_pose": raw_pose,
            "clip_id": clip_id,
            "frame_index": frame_index,
            "file_num": int(parsed.group("file_num")),
            "frame_id": int(parsed.group("frame_id")),
            "metadata": metadata,
        }

    def _validate_frame_order(self, clip_id: str, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            raise CSGOSeen10DatasetError(f"Empty continuous clip: {clip_id}")
        file_num = rows[0]["file_num"]
        previous_frame = rows[0]["frame_id"]
        for row in rows[1:]:
            if row["file_num"] != file_num:
                raise CSGOSeen10DatasetError(f"Clip {clip_id} changes source file")
            frame_gap = row["frame_id"] - previous_frame
            if frame_gap <= 0:
                raise CSGOSeen10DatasetError(f"Clip {clip_id} frame order is not strictly increasing")
            if frame_gap > self.max_frame_gap:
                raise CSGOSeen10DatasetError(
                    f"Clip {clip_id} exceeds max_frame_gap={self.max_frame_gap}"
                )
            previous_frame = row["frame_id"]

    @property
    def image_processor(self) -> Any:
        if self._image_processor is None:
            from ..pipelines.image_processor import OmniGen2ImageProcessor

            self._image_processor = OmniGen2ImageProcessor(vae_scale_factor=16, do_resize=True)
        return self._image_processor

    @staticmethod
    def _open_resized_rgb(path: Path) -> Image.Image:
        try:
            with Image.open(path) as image:
                return image.convert("RGB").resize(
                    (_IMAGE_SIZE, _IMAGE_SIZE), resample=_BICUBIC
                )
        except OSError as exc:
            raise OSError(f"Cannot open CSGO Benchmark image {path}: {exc}") from exc

    def _preprocess(self, image: Image.Image, *, max_pixels: int) -> torch.Tensor:
        processed = self.image_processor.preprocess(
            image,
            max_pixels=max_pixels,
            max_side_length=self.max_side_length,
        )
        if not isinstance(processed, torch.Tensor):
            processed = torch.as_tensor(processed)
        if processed.ndim == 4:
            if processed.shape[0] != 1:
                raise CSGOSeen10DatasetError(
                    f"Image processor returned batch size {processed.shape[0]} for a single image"
                )
            processed = processed[0]
        if tuple(processed.shape) != (3, _IMAGE_SIZE, _IMAGE_SIZE):
            raise CSGOSeen10DatasetError(
                "OmniGen2 image processor must preserve each image as 3x448x448; "
                f"got {tuple(processed.shape)}"
            )
        return processed

    def _instruction(self, row: Mapping[str, Any], *, drop_prompt: bool = False) -> str:
        if drop_prompt:
            if not self.use_chat_template:
                return ""
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT_DROP},
                {"role": "user", "content": ""},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        pose = ", ".join(f"{value:.8g}" for value in row["pose_values"])
        user_text = (
            "Generate a first-person Counter-Strike: Global Offensive gameplay screenshot "
            f"for map {row['map_name']}. Use the provided radar image as the map reference. "
            f"The normalized player pose in [x, y, z, pitch, yaw] order is [{pose}]."
        )
        if not self.use_chat_template:
            return user_text
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        drop_prompt = random.random() < self.prompt_dropout_prob
        drop_ref_img = drop_prompt and random.random() < self.ref_img_dropout_prob
        radar_pil = None
        radar_tensor = None
        if not drop_ref_img:
            radar_pil = self._open_resized_rgb(row["radar_path"])
            radar_tensor = self._preprocess(radar_pil, max_pixels=self.max_input_pixels)
            if tuple(radar_pil.size) != (_IMAGE_SIZE, _IMAGE_SIZE):
                raise CSGOSeen10DatasetError("PIL radar resize did not produce 448x448")

        output_image = None
        if self.load_target:
            target_pil = self._open_resized_rgb(row["image_path"])
            output_image = self._preprocess(target_pil, max_pixels=self.max_output_pixels)

        metadata = dict(row["metadata"])
        return {
            "task_type": "csgo_generation",
            "instruction": self._instruction(row, drop_prompt=drop_prompt),
            "input_images": [radar_tensor] if radar_tensor is not None else None,
            "input_images_path": [str(row["radar_path"])] if radar_tensor is not None else None,
            "input_images_pil": [radar_pil] if radar_pil is not None else None,
            "target_img_size": (_IMAGE_SIZE, _IMAGE_SIZE),
            "output_image": output_image,
            "output_image_path": str(row["image_path"]) if self.load_target else None,
            "pose_values": torch.tensor(row["pose_values"], dtype=torch.float32),
            "metadata": metadata,
            # train.py's existing reward path expects JSON strings in meta_data.
            "meta_data": json.dumps(metadata, ensure_ascii=False),
            "sample_id": row["sample_id"],
            "map_name": row["map_name"],
            "file_frame": row["file_frame"],
            "clip_id": row["clip_id"],
            "frame_index": row["frame_index"],
        }

    def __len__(self) -> int:
        return len(self.rows)


class CSGOSeen10Collator:
    """Batch CSGO samples using the field names consumed by OmniGen2 train.py."""

    def __init__(self, tokenizer: Any, max_token_len: int = 888):
        if tokenizer is None:
            raise ValueError("CSGOSeen10Collator requires a tokenizer")
        self.tokenizer = tokenizer
        self.max_token_len = _require_integer(max_token_len, "max_token_len", minimum=1)

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not batch:
            raise ValueError("Cannot collate an empty batch")
        instructions = [str(sample["instruction"]) for sample in batch]
        text_inputs = self.tokenizer(
            instructions,
            padding="longest",
            max_length=self.max_token_len,
            truncation=True,
            return_tensors="pt",
        )
        try:
            text_ids = text_inputs["input_ids"]
            text_mask = text_inputs["attention_mask"]
        except (KeyError, TypeError):
            text_ids = text_inputs.input_ids
            text_mask = text_inputs.attention_mask

        output_images = [sample.get("output_image") for sample in batch]
        if all(image is None for image in output_images):
            output_image = None
        elif any(image is None for image in output_images):
            raise ValueError("A batch cannot mix samples with and without loaded targets")
        else:
            # train.py encodes each sample separately through the VAE.
            output_image = output_images

        pose_values = torch.stack(
            [torch.as_tensor(sample["pose_values"], dtype=torch.float32).reshape(5) for sample in batch]
        )
        metadata = [dict(sample["metadata"]) for sample in batch]
        meta_data = [
            str(sample.get("meta_data") or json.dumps(item, ensure_ascii=False))
            for sample, item in zip(batch, metadata)
        ]

        return {
            "task_type": [sample.get("task_type", "csgo_generation") for sample in batch],
            "instruction": instructions,
            "text_ids": text_ids,
            "text_mask": text_mask,
            # Keep one-list-per-sample, matching OmniGen2's reference-image API.
            "input_images": [sample.get("input_images") for sample in batch],
            "input_images_path": [sample.get("input_images_path") for sample in batch],
            "input_images_pil": [sample.get("input_images_pil") for sample in batch],
            "target_img_size": [sample["target_img_size"] for sample in batch],
            "output_image": output_image,
            "output_image_path": [sample.get("output_image_path") for sample in batch],
            "pose_values": pose_values,
            "metadata": metadata,
            "meta_data": meta_data,
            "sample_id": [sample["sample_id"] for sample in batch],
            "map_name": [sample["map_name"] for sample in batch],
            "file_frame": [sample["file_frame"] for sample in batch],
            "clip_id": [sample["clip_id"] for sample in batch],
            "frame_index": [sample["frame_index"] for sample in batch],
        }


__all__ = ["CSGOSeen10Dataset", "CSGOSeen10Collator", "CSGOSeen10DatasetError", "SEEN_MAPS"]
