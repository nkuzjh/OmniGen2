#!/usr/bin/env python3
"""CPU-only, real-manifest audit. Never opens test target FPVs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from transformers import AutoTokenizer
from omnigen2.dataset.csgo_seen10_dataset import CSGOSeen10Dataset, SEEN_MAPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/jiahao/task/UniLIP/data/csgo_benchmark_v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", local_files_only=True)
    expected = dict(seen_train=50000, seen_validation=5000, seen_discrete_test=20000, seen_continuous=12800)
    report = {"smoke_only": True, "splits": {}, "test_target_open_count": 0}
    original_open = Image.open
    data_root = Path(args.data_root).resolve()
    for split, count in expected.items():
        test = split in ("seen_discrete_test", "seen_continuous")
        dataset = CSGOSeen10Dataset(data_root, split, tokenizer=tokenizer, use_chat_template=True,
                                   load_target=not test, reference_image_size=224, target_image_size=448,
                                   max_input_pixels=224**2, max_output_pixels=448**2, max_side_length=448)
        assert len(dataset) == count
        maps = Counter(row["map_name"] for row in dataset.rows)
        assert tuple(maps) == SEEN_MAPS and set(maps.values()) == {count // 10}
        max_tokens = 0
        for start in range(0, len(dataset), 512):
            prompts = [dataset._instruction(row) for row in dataset.rows[start:start+512]]
            tokens = tokenizer(prompts, truncation=False, padding=False)["input_ids"]
            max_tokens = max(max_tokens, *(len(row) for row in tokens))
        assert max_tokens <= 888

        def guarded_open(path, *positional, **kwargs):
            if test and data_root / "images" in Path(path).resolve().parents:
                raise AssertionError(f"Inference attempted to read target: {path}")
            return original_open(path, *positional, **kwargs)

        with patch("PIL.Image.open", side_effect=guarded_open):
            for index in range(0, len(dataset), count // 10):
                item = dataset[index]
                assert tuple(item["input_images"][0].shape) == (3, 224, 224)
                if test:
                    assert item["output_image"] is None
                    assert dataset.get_inference_item(index)["input_images_pil"][0].size == (224, 224)
                else:
                    assert tuple(item["output_image"].shape) == (3, 448, 448)
        entry = {"count": len(dataset), "per_map": dict(maps), "max_prompt_tokens": max_tokens,
                 "reference_size": [224, 224], "target_size": [448, 448], "load_target": not test}
        if split == "seen_continuous":
            clips = Counter((row["map_name"], row["clip_id"]) for row in dataset.rows)
            assert len(clips) == 200 and set(clips.values()) == {64}
            entry.update(clips=200, frames_per_clip=64)
        report["splits"][split] = entry
    report["protocol_sha256"] = {
        str(path.relative_to(data_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (data_root / "benchmark_manifest.json", data_root / "calibration/z_calibration.json")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
