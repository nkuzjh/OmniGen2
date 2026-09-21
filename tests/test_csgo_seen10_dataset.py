import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from omnigen2.dataset import csgo_seen10_dataset as dataset_module
from omnigen2.dataset.csgo_seen10_dataset import (
    CSGOSeen10Collator,
    CSGOSeen10Dataset,
    CSGOSeen10DatasetError,
    SEEN_MAPS,
)


class FakeImageProcessor:
    def __init__(self):
        self.seen = []

    def preprocess(self, image, *, max_pixels, max_side_length):
        self.seen.append((image.size, image.mode, max_pixels, max_side_length))
        return torch.zeros((1, 3, image.height, image.width), dtype=torch.float32)


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        width = max(1, max(len(text.split()) for text in texts))
        ids = torch.zeros((len(texts), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, text in enumerate(texts):
            length = len(text.split())
            ids[row, :length] = torch.arange(1, length + 1)
            mask[row, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        return "\n".join(message["content"] for message in messages)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_image(path: Path, size=(32, 24), color=(90, 130, 170)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG")


def _row(map_name: str, file_num: int, frame_id: int, *, x=512, y=256, z=-10):
    return {
        "map": map_name,
        "file_frame": f"file_num{file_num}_frame_{frame_id:04d}",
        "x": x,
        "y": y,
        "z": z,
        "angle_h": math.pi,
        "angle_v": math.pi / 2,
    }


def _make_bundle(root: Path):
    z_ranges = {map_name: {"z_min": -20.0, "z_max": 20.0} for map_name in SEEN_MAPS}
    radar_entries = []
    counts = {}
    image_counts = {}

    for map_index, map_name in enumerate(SEEN_MAPS):
        radar_target = f"{map_name}/{map_name}_radar.png"
        radar_path = root / "radars" / radar_target
        radar_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 16), (40, 70, 100)).save(radar_path, format="PNG")
        radar_entries.append({"map": map_name, "target": radar_target})

        train_rows = [
            _row(map_name, map_index + 1, 1, x=2048, y=-256, z=40),
            _row(map_name, map_index + 1, 2),
        ]
        validation_rows = [_row(map_name, map_index + 2, 3)]
        test_rows = [_row(map_name, map_index + 3, 4)]
        continuous_frames = [
            _row(map_name, map_index + 4, 10),
            _row(map_name, map_index + 4, 11),
        ]
        all_rows = train_rows + validation_rows + test_rows + continuous_frames
        counts[map_name] = {
            "train": len(train_rows),
            "validation": len(validation_rows),
            "discrete_test": len(test_rows),
            "continuous_clips": 1,
            "continuous_frames": len(continuous_frames),
        }
        image_counts[map_name] = {"count": len(all_rows), "bytes": 0}

        split_dir = root / "splits" / "seen" / map_name
        _write_json(split_dir / "train.json", train_rows)
        _write_json(split_dir / "validation.json", validation_rows)
        _write_json(split_dir / "discrete_test.json", test_rows)
        _write_json(
            split_dir / "continuous_clips.json",
            {
                "benchmark_id": "csgo_benchmark_v2",
                "clips": [
                    {
                        "clip_id": f"{map_name}_clip_0",
                        "frames": continuous_frames,
                    }
                ],
            },
        )
        for row in all_rows:
            _write_image(root / "images" / map_name / f"{row['file_frame']}.jpg")

    fingerprint = "test-calibration-fingerprint"
    calibration = {
        "schema_version": 1,
        "benchmark_id": "csgo_benchmark_v2",
        "calibration_sha256": fingerprint,
        "decision_review_status": "approved",
        "z_ranges": z_ranges,
    }
    calibration_path = root / "calibration" / "z_calibration.json"
    _write_json(calibration_path, calibration)
    calibration_sha256 = hashlib.sha256(calibration_path.read_bytes()).hexdigest()

    manifest = {
        "schema_version": 1,
        "benchmark_id": "csgo_benchmark_v2",
        "benchmark": {"id": "csgo_benchmark_v2", "version": "2.0.0", "strict_protocol": True},
        "protocol": {"seen_maps": list(SEEN_MAPS), "seen_splits": ["train", "validation", "discrete_test"]},
        "counts": {"seen": counts},
        "continuous_protocol": {"frames_per_clip": 2, "max_frame_gap": 2},
        "calibration": {
            "file": "calibration/z_calibration.json",
            "sha256": calibration_sha256,
            "fingerprint": fingerprint,
            "z_ranges": z_ranges,
        },
    }
    report = {
        "schema_version": 1,
        "benchmark_id": "csgo_benchmark_v2",
        "status": "verified",
        "maps": list(SEEN_MAPS),
        "images": {
            "root": "images",
            "status": "verified",
            "target_template": "images/{map}/{file_frame}.jpg",
            "count": sum(item["count"] for item in image_counts.values()),
            "by_map": image_counts,
        },
        "radars": {
            "root": "radars",
            "status": "verified",
            "entries": radar_entries,
        },
    }
    _write_json(root / "benchmark_manifest.json", manifest)
    _write_json(root / "minimal_dataset_report.json", report)
    return root


class TestCSGOSeen10Dataset(unittest.TestCase):
    def setUp(self):
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)

    def test_seen_train_order_pose_processor_and_collator(self):
        root = _make_bundle(self.root / "bundle")
        processor = FakeImageProcessor()
        tokenizer = FakeTokenizer()
        dataset = CSGOSeen10Dataset(
            root,
            "seen_train",
            tokenizer=tokenizer,
            use_chat_template=True,
            image_processor=processor,
        )

        self.assertEqual(len(dataset), len(SEEN_MAPS) * 2)
        samples = [dataset[index] for index in range(2)]
        self.assertEqual([dataset[index]["map_name"] for index in (0, 2, 4)], list(SEEN_MAPS[:3]))
        self.assertEqual([seen[:2] for seen in processor.seen[:4]], [((448, 448), "RGB")] * 4)
        for actual, expected in zip(
            samples[0]["pose_values"].tolist(), [2.0, -0.25, 1.5, 0.25, 0.5]
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(tuple(samples[0]["input_images"][0].shape), (3, 448, 448))
        self.assertEqual(tuple(samples[0]["output_image"].shape), (3, 448, 448))
        self.assertEqual(samples[0]["metadata"]["sample_id"], f"{SEEN_MAPS[0]}/file_num1_frame_0001")
        self.assertEqual(samples[0]["metadata"]["raw_pose"]["x"], 2048.0)
        self.assertIn("[x, y, z, pitch, yaw]", samples[0]["instruction"])

        batch = CSGOSeen10Collator(tokenizer, max_token_len=64)(samples)
        self.assertEqual(batch["text_ids"].shape[0], 2)
        self.assertEqual(tuple(batch["text_mask"].shape), tuple(batch["text_ids"].shape))
        self.assertEqual(tuple(batch["input_images"][0][0].shape), (3, 448, 448))
        self.assertEqual(len(batch["output_image"]), 2)
        self.assertEqual(tuple(batch["output_image"][0].shape), (3, 448, 448))
        self.assertEqual(tuple(batch["pose_values"].shape), (2, 5))
        self.assertEqual(batch["metadata"][0]["file_frame"], "file_num1_frame_0001")
        self.assertEqual(json.loads(batch["meta_data"][0])["map_name"], SEEN_MAPS[0])

    def test_continuous_keeps_clip_and_frame_order(self):
        root = _make_bundle(self.root / "bundle")
        dataset = CSGOSeen10Dataset(
            root,
            "seen_continuous",
            load_target=False,
            image_processor=FakeImageProcessor(),
        )
        first_map_rows = [row for row in dataset.rows if row["map_name"] == SEEN_MAPS[0]]
        self.assertEqual([row["frame_index"] for row in first_map_rows], [0, 1])
        self.assertEqual([row["clip_id"] for row in first_map_rows], [f"{SEEN_MAPS[0]}_clip_0"] * 2)
        self.assertEqual(
            [row["file_frame"] for row in first_map_rows],
            ["file_num4_frame_0010", "file_num4_frame_0011"],
        )

    def test_inference_never_opens_target(self):
        root = _make_bundle(self.root / "bundle")
        processor = FakeImageProcessor()
        dataset = CSGOSeen10Dataset(
            root,
            "seen_discrete_test",
            load_target=False,
            image_processor=processor,
        )
        target_path = dataset.rows[0]["image_path"].resolve()
        original_open = Image.open

        def guarded_open(path, *args, **kwargs):
            if Path(path).resolve() == target_path:
                raise AssertionError("inference opened the target image")
            return original_open(path, *args, **kwargs)

        with patch.object(dataset_module.Image, "open", side_effect=guarded_open):
            sample = dataset[0]
        self.assertIsNone(sample["output_image"])
        self.assertIsNone(sample["output_image_path"])
        self.assertEqual(len(processor.seen), 1)
        self.assertEqual(sample["metadata"]["image_path"], str(target_path))

    def test_inference_item_skips_tensor_preprocess_and_shares_radar_pil_cache(self):
        root = _make_bundle(self.root / "bundle")
        processor = FakeImageProcessor()
        radar_cache = {}
        train_dataset = CSGOSeen10Dataset(
            root,
            "seen_train",
            load_target=False,
            image_processor=processor,
            inference_radar_cache=radar_cache,
        )
        test_dataset = CSGOSeen10Dataset(
            root,
            "seen_discrete_test",
            load_target=False,
            image_processor=processor,
            inference_radar_cache=radar_cache,
        )

        first = train_dataset.get_inference_item(0)
        second = train_dataset.get_inference_item(1)
        across_split = test_dataset.get_inference_item(0)

        self.assertIs(first["input_images_pil"][0], second["input_images_pil"][0])
        self.assertIs(first["input_images_pil"][0], across_split["input_images_pil"][0])
        self.assertEqual(first["input_images_pil"][0].size, (448, 448))
        self.assertEqual(tuple(first["pose_values"].shape), (5,))
        self.assertIsNone(first["output_image"])
        self.assertEqual(processor.seen, [])
        self.assertEqual(set(radar_cache), {SEEN_MAPS[0]})

    def test_strict_status_and_split_count_validation(self):
        root = _make_bundle(self.root / "bundle")
        report_path = root / "minimal_dataset_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["status"] = "incomplete"
        _write_json(report_path, report)
        with self.assertRaisesRegex(CSGOSeen10DatasetError, "status must be verified"):
            CSGOSeen10Dataset(root, "seen_train", image_processor=FakeImageProcessor())

        root = _make_bundle(self.root / "count_bundle")
        split_path = root / "splits" / "seen" / SEEN_MAPS[0] / "discrete_test.json"
        _write_json(split_path, [])
        with self.assertRaisesRegex(CSGOSeen10DatasetError, "count mismatch"):
            CSGOSeen10Dataset(root, "seen_discrete_test", image_processor=FakeImageProcessor())

    def test_calibration_hash_and_seen_order_are_strict(self):
        root = _make_bundle(self.root / "bundle")
        manifest_path = root / "benchmark_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["calibration"]["sha256"] = "0" * 64
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(CSGOSeen10DatasetError, "SHA-256 differ"):
            CSGOSeen10Dataset(root, "seen_train", image_processor=FakeImageProcessor())

        root = _make_bundle(self.root / "order_bundle")
        manifest_path = root / "benchmark_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["protocol"]["seen_maps"] = list(reversed(SEEN_MAPS))
        _write_json(manifest_path, manifest)
        with self.assertRaisesRegex(CSGOSeen10DatasetError, "Seen-10 map order"):
            CSGOSeen10Dataset(root, "seen_train", image_processor=FakeImageProcessor())


if __name__ == "__main__":
    unittest.main()
