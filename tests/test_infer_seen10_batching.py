import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

import infer_seen10


def _identities(count):
    return [
        infer_seen10.SampleIdentity(
            index=index,
            sample_id=f"cs_agency/file_num1_frame_{index:04d}",
            map_name="cs_agency",
            file_frame=f"file_num1_frame_{index:04d}",
            clip_id="clip_0",
            frame_index=index,
        )
        for index in range(count)
    ]


def _inputs(identities):
    image = Image.new("RGB", (448, 448), (20, 30, 40))
    return [
        (
            f"prompt for {identity.sample_id}",
            [image],
            torch.full((1, 5), float(index), dtype=torch.float32),
        )
        for index, identity in enumerate(identities)
    ]


class RecordingPipeline:
    _execution_device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            images=[Image.new("RGB", (448, 448), (index, 0, 0)) for index, _ in enumerate(kwargs["prompt"])]
        )


class TestSeen10BatchedInference(unittest.TestCase):
    def test_pending_marker_archives_script_only_change_before_any_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "discrete"
            old = {"task": "discrete", "seed": 42, "inference_script_sha256": "a" * 64}
            current = {**old, "inference_script_sha256": "b" * 64}
            plan = infer_seen10.TaskPlan(
                task="discrete", split="seen_discrete_test", dataset=object(),
                task_root=root, identities=[], base_manifest=current,
            )
            infer_seen10._atomic_create_json(plan.pending_manifest_path, old)
            infer_seen10._prepare_partial_resume(plan)
            self.assertEqual(json.loads(plan.pending_manifest_path.read_text()), current)
            archived = root / f".inference_manifest.pending.{'a' * 64}.superseded.json"
            self.assertEqual(json.loads(archived.read_text()), old)

    def test_pending_marker_keeps_rejecting_changes_with_existing_jpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "discrete"
            old = {"task": "discrete", "seed": 42, "inference_script_sha256": "a" * 64}
            current = {**old, "inference_script_sha256": "b" * 64}
            plan = infer_seen10.TaskPlan(
                task="discrete", split="seen_discrete_test", dataset=object(),
                task_root=root, identities=[], base_manifest=current,
            )
            infer_seen10._atomic_create_json(plan.pending_manifest_path, old)
            generated = root / "gen_imgs" / "cs_agency" / "file_num1_frame_0000.jpg"
            infer_seen10._atomic_create_jpeg(Image.new("RGB", (448, 448)), generated)
            with self.assertRaisesRegex(FileExistsError, "refusing to mix outputs"):
                infer_seen10._prepare_partial_resume(plan)

    def test_aligned_pipeline_loads_named_local_safetensors_and_slow_processor(self):
        from diffusers import AutoencoderKL
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
        from omnigen2.pipelines.omnigen2 import pipeline_omnigen2
        from omnigen2.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            (adapter / "pytorch_lora_weights.safetensors").write_bytes(b"local weights")
            args = SimpleNamespace(
                experiment=infer_seen10.ALIGNED_EXPERIMENT,
                dtype="bf16", offload=False, fuse_lora=False,
                model_path="OmniGen2/OmniGen2", adapter_path=str(adapter),
                aligned_sources={
                    "pretrained_model_path": Path("/cached/omnigen"),
                    "pretrained_vae_model_name_or_path": Path("/cached/flux"),
                    "pretrained_text_encoder_model_name_or_path": Path("/cached/qwen"),
                },
            )
            transformer = SimpleNamespace(config=SimpleNamespace(pose_conditioning=False), pose_adapter=None)
            pipeline = MagicMock()
            with (
                patch.object(torch.cuda, "is_available", return_value=False),
                patch.object(OmniGen2Transformer2DModel, "from_pretrained", return_value=transformer),
                patch.object(AutoencoderKL, "from_pretrained", return_value=object()),
                patch.object(Qwen2_5_VLForConditionalGeneration, "from_pretrained", return_value=object()),
                patch.object(AutoProcessor, "from_pretrained", return_value=object()) as load_processor,
                patch.object(FlowMatchEulerDiscreteScheduler, "from_pretrained", return_value=object()),
                patch.object(pipeline_omnigen2, "OmniGen2Pipeline", return_value=pipeline),
            ):
                self.assertIs(infer_seen10._load_pipeline(args), pipeline)
            pipeline.load_lora_weights.assert_called_once_with(
                str(adapter), weight_name="pytorch_lora_weights.safetensors", local_files_only=True,
            )
            load_processor.assert_called_once_with("/cached/qwen", use_fast=False)

    def test_aligned_generation_uses_small_reference_and_no_numeric_pose_argument(self):
        identity = _identities(1)
        sample_inputs = [("pose in text", [Image.new("RGB", (224, 224))], torch.zeros((1, 5)))]
        pipeline = RecordingPipeline()
        with patch.object(
            infer_seen10, "_aligned_empty_negative_prompt",
            return_value=(torch.zeros((1, 2, 4)), torch.ones((1, 2))),
        ):
            infer_seen10._generate_batch(
                pipeline, identity, sample_inputs, task="discrete", base_seed=42,
                steps=28, vae_decode_batch_size=1,
                experiment=infer_seen10.ALIGNED_EXPERIMENT,
            )
        call = pipeline.calls[0]
        self.assertNotIn("pose_values", call)
        self.assertEqual(call["max_input_image_side_length"], 224)
        self.assertEqual(call["max_sequence_length"], 888)
        self.assertIn("negative_prompt_embeds", call)
        self.assertFalse(call["align_res"])

    def test_aligned_negative_prompt_uses_training_dropout_template(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return "formatted empty prompt"

        class Pipeline:
            def __init__(self):
                self.processor = SimpleNamespace(tokenizer=Tokenizer())
                self.mllm = SimpleNamespace(dtype=torch.bfloat16)
                self.calls = 0

            def _get_qwen2_prompt_embeds(self, **kwargs):
                self.calls += 1
                self.prompt = kwargs["prompt"]
                return torch.zeros((1, 2, 4)), torch.ones((1, 2))

        pipeline = Pipeline()
        embeddings, mask = infer_seen10._aligned_empty_negative_prompt(pipeline, 3, torch.device("cpu"))
        infer_seen10._aligned_empty_negative_prompt(pipeline, 2, torch.device("cpu"))
        self.assertEqual(pipeline.processor.tokenizer.messages[0]["content"],
                         "You are a helpful assistant that generates images.")
        self.assertEqual(pipeline.processor.tokenizer.messages[1]["content"], "")
        self.assertEqual(pipeline.prompt, ["formatted empty prompt"])
        self.assertEqual(tuple(embeddings.shape), (3, 2, 4))
        self.assertEqual(tuple(mask.shape), (3, 2))
        self.assertEqual(pipeline.calls, 1)

    def test_aligned_sources_require_matching_official_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            revision = "a" * 40
            identities = {
                "pretrained_model_path": "OmniGen2/OmniGen2",
                "pretrained_vae_model_name_or_path": "black-forest-labs/FLUX.1-dev",
                "pretrained_text_encoder_model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",
            }
            snapshots = {}
            for key, repo_id in identities.items():
                snapshot = root / ("models--" + repo_id.replace("/", "--")) / "snapshots" / revision
                snapshot.mkdir(parents=True)
                snapshots[key] = snapshot
            args = SimpleNamespace(
                model_path=str(snapshots["pretrained_model_path"]),
                vae_model_path=str(snapshots["pretrained_vae_model_name_or_path"]),
                text_encoder_model_path=str(snapshots["pretrained_text_encoder_model_name_or_path"]),
            )
            config = {"contract_identity": {"base_sources": {
                key: {"repo_id": repo_id, "revision": revision}
                for key, repo_id in identities.items()
            }}}
            self.assertEqual(infer_seen10._validated_aligned_sources(args, config), snapshots)
            config["contract_identity"]["base_sources"]["pretrained_model_path"]["revision"] = "b" * 40
            with self.assertRaisesRegex(ValueError, "does not match checkpoint revision"):
                infer_seen10._validated_aligned_sources(args, config)

    def test_legacy_cli_arguments_keep_working_with_batched_defaults(self):
        with patch.object(
            sys,
            "argv",
            [
                "infer_seen10.py",
                "--task",
                "all",
                "--seed",
                "0",
                "--model-path",
                "model",
                "--adapter-path",
                "adapter",
            ],
        ):
            args = infer_seen10.parse_args()

        self.assertEqual(args.task, "all")
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.vae_decode_batch_size, 1)
        self.assertTrue(args.oom_fallback)
        self.assertTrue(args.fuse_lora)

    def test_pipeline_load_fuses_transformer_lora_once_then_unloads_adapter(self):
        import convert_ckpt_to_hf_format
        from omnigen2.models.transformers.transformer_omnigen2 import (
            OmniGen2Transformer2DModel,
        )
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline

        transformer = object()
        pipeline = MagicMock()
        args = SimpleNamespace(
            dtype="bf16",
            offload=False,
            model_path="model",
            adapter_path="adapter",
            fuse_lora=True,
        )
        with (
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(
                OmniGen2Transformer2DModel,
                "from_pretrained",
                return_value=transformer,
            ),
            patch.object(convert_ckpt_to_hf_format, "load_pose_adapter") as load_pose,
            patch.object(
                OmniGen2Pipeline,
                "from_pretrained",
                return_value=pipeline,
            ),
        ):
            loaded = infer_seen10._load_pipeline(args)

        self.assertIs(loaded, pipeline)
        load_pose.assert_called_once_with(transformer, "adapter")
        pipeline.load_lora_weights.assert_called_once_with("adapter")
        pipeline.fuse_lora.assert_called_once_with(
            safe_fusing=True, components=["transformer"]
        )
        pipeline.unload_lora_weights.assert_called_once_with()
        pipeline.set_progress_bar_config.assert_called_once_with(disable=True)
        pipeline.to.assert_called_once_with(torch.device("cpu"))

    def test_batch_call_passes_aligned_lists_cache_keys_and_independent_seeds(self):
        identities = _identities(3)
        sample_inputs = _inputs(identities)
        pipeline = RecordingPipeline()
        outputs = infer_seen10._generate_batch(
            pipeline,
            identities,
            sample_inputs,
            task="continuous",
            base_seed=17,
            steps=28,
            vae_decode_batch_size=2,
        )

        self.assertEqual(len(outputs), len(identities))
        call = pipeline.calls[0]
        self.assertEqual(call["prompt"], [f"prompt for {item.sample_id}" for item in identities])
        self.assertEqual(call["input_images"], [sample[1] for sample in sample_inputs])
        self.assertEqual(tuple(call["pose_values"].shape), (3, 5))
        self.assertEqual(call["input_image_cache_keys"], [[item.map_name] for item in identities])
        self.assertEqual(call["vae_decode_batch_size"], 2)
        self.assertEqual(len(call["generator"]), 3)
        self.assertEqual(len(call["reference_generator"]), 3)
        self.assertEqual(
            [item.initial_seed() for item in call["generator"]],
            [infer_seen10._derive_sample_seed(17, "continuous", item.sample_id) for item in identities],
        )
        self.assertEqual(
            [item.initial_seed() for item in call["reference_generator"]],
            [infer_seen10._derive_reference_seed(17, "continuous", item.sample_id) for item in identities],
        )

    def test_single_sample_fallback_preserves_pipeline_b1_image_shape(self):
        identity = _identities(1)
        sample_inputs = _inputs(identity)
        pipeline = RecordingPipeline()

        infer_seen10._generate_batch(
            pipeline,
            identity,
            sample_inputs,
            task="discrete",
            base_seed=0,
            steps=28,
            vae_decode_batch_size=1,
        )

        self.assertEqual(pipeline.calls[0]["input_images"], sample_inputs[0][1])
        self.assertEqual(pipeline.calls[0]["input_image_cache_keys"], ["cs_agency"])
        self.assertIsInstance(pipeline.calls[0]["prompt"], list)

    def test_oom_fallback_retries_ordered_subbatches_with_fresh_generators(self):
        class OOMOncePipeline(RecordingPipeline):
            def __init__(self):
                super().__init__()
                self.noise_draws = []
                self.reference_draws = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                self.noise_draws.append(
                    [torch.rand((), generator=generator).item() for generator in kwargs["generator"]]
                )
                self.reference_draws.append(
                    [
                        torch.rand((), generator=generator).item()
                        for generator in kwargs["reference_generator"]
                    ]
                )
                if len(kwargs["prompt"]) > 2:
                    raise RuntimeError("CUDA out of memory while allocating tensor")
                return SimpleNamespace(
                    images=[Image.new("RGB", (448, 448)) for _ in kwargs["prompt"]]
                )

        identities = _identities(5)
        inputs = _inputs(identities)
        pipeline = OOMOncePipeline()
        result = infer_seen10._generate_batch_with_fallback(
            pipeline,
            identities,
            inputs,
            task="continuous",
            base_seed=23,
            steps=28,
            vae_decode_batch_size=1,
            oom_fallback=True,
        )

        self.assertEqual(len(result.images), 5)
        self.assertEqual(result.effective_batch_size, 2)
        self.assertTrue(result.had_oom_fallback)
        self.assertEqual([len(call["prompt"]) for call in pipeline.calls], [5, 2, 2, 1])
        first_call_seeds = [generator.initial_seed() for generator in pipeline.calls[0]["generator"]]
        retry_seeds = [
            seed
            for call in pipeline.calls[1:]
            for seed in (generator.initial_seed() for generator in call["generator"])
        ]
        expected_seeds = [
            infer_seen10._derive_sample_seed(23, "continuous", identity.sample_id)
            for identity in identities
        ]
        self.assertEqual(first_call_seeds, expected_seeds)
        self.assertEqual(retry_seeds, expected_seeds)

        first_reference_seeds = [
            generator.initial_seed()
            for generator in pipeline.calls[0]["reference_generator"]
        ]
        retry_reference_seeds = [
            seed
            for call in pipeline.calls[1:]
            for seed in (
                generator.initial_seed()
                for generator in call["reference_generator"]
            )
        ]
        expected_reference_seeds = [
            infer_seen10._derive_reference_seed(
                23, "continuous", identity.sample_id
            )
            for identity in identities
        ]
        self.assertEqual(first_reference_seeds, expected_reference_seeds)
        self.assertEqual(retry_reference_seeds, expected_reference_seeds)
        self.assertEqual(
            pipeline.noise_draws[0],
            [value for batch in pipeline.noise_draws[1:] for value in batch],
        )
        self.assertEqual(
            pipeline.reference_draws[0],
            [value for batch in pipeline.reference_draws[1:] for value in batch],
        )

    def test_oom_fallback_can_be_disabled(self):
        class AlwaysOOMPipeline(RecordingPipeline):
            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                raise RuntimeError("CUDA out of memory")

        identities = _identities(2)
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            infer_seen10._generate_batch_with_fallback(
                AlwaysOOMPipeline(),
                identities,
                _inputs(identities),
                task="discrete",
                base_seed=0,
                steps=28,
                vae_decode_batch_size=1,
                oom_fallback=False,
            )

    def test_oom_halving_locks_successful_size_for_rest_of_current_batch(self):
        class OOMAboveFourPipeline(RecordingPipeline):
            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                if len(kwargs["prompt"]) > 4:
                    raise RuntimeError("CUDA out of memory")
                return SimpleNamespace(
                    images=[Image.new("RGB", (448, 448)) for _ in kwargs["prompt"]]
                )

        identities = _identities(8)
        pipeline = OOMAboveFourPipeline()
        result = infer_seen10._generate_batch_with_fallback(
            pipeline,
            identities,
            _inputs(identities),
            task="continuous",
            base_seed=23,
            steps=28,
            vae_decode_batch_size=1,
            oom_fallback=True,
        )

        self.assertEqual(len(result.images), 8)
        self.assertEqual(result.effective_batch_size, 4)
        self.assertTrue(result.had_oom_fallback)
        self.assertEqual([len(call["prompt"]) for call in pipeline.calls], [8, 4, 4])

    def test_task_keeps_reduced_batch_size_after_first_oom(self):
        class OOMAboveTwoPipeline(RecordingPipeline):
            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                if len(kwargs["prompt"]) > 2:
                    for generator in kwargs["generator"]:
                        torch.rand((), generator=generator)
                    raise RuntimeError("CUDA out of memory")
                return SimpleNamespace(
                    images=[Image.new("RGB", (448, 448)) for _ in kwargs["prompt"]]
                )

        identities = _identities(7)

        class Dataset:
            load_target = False

            def get_inference_item(self, index):
                identity = identities[index]
                return {
                    "sample_id": identity.sample_id,
                    "map_name": identity.map_name,
                    "file_frame": identity.file_frame,
                    "clip_id": identity.clip_id,
                    "frame_index": identity.frame_index,
                    "instruction": f"prompt for {identity.sample_id}",
                    "input_images_pil": [Image.new("RGB", (448, 448))],
                    "pose_values": torch.zeros(5),
                    "output_image": None,
                }

        with tempfile.TemporaryDirectory() as temporary_directory:
            task_root = Path(temporary_directory) / "continuous"
            base_manifest = {"manifest_version": 2, "task": "continuous", "seed": 23}
            plan = infer_seen10.TaskPlan(
                task="continuous",
                split="seen_continuous",
                dataset=Dataset(),
                task_root=task_root,
                identities=identities,
                base_manifest=base_manifest,
            )
            infer_seen10._atomic_create_json(plan.pending_manifest_path, base_manifest)
            existing_output = infer_seen10._output_path(task_root, identities[-1])
            infer_seen10._atomic_create_jpeg(Image.new("RGB", (448, 448), (10, 20, 30)), existing_output)
            existing_digest = infer_seen10._validate_existing_jpeg(existing_output)
            infer_seen10._prepare_partial_resume(plan)
            pipeline = OOMAboveTwoPipeline()
            args = SimpleNamespace(
                seed=23,
                batch_size=4,
                vae_decode_batch_size=1,
                oom_fallback=True,
                num_inference_steps=28,
            )

            infer_seen10._run_task(plan, pipeline, args, ("cs_agency",))

            final_manifest = json.loads(plan.final_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual([len(call["prompt"]) for call in pipeline.calls], [4, 2, 2, 2])
            self.assertEqual(
                [entry["sample_id"] for entry in final_manifest["output_hashes"]],
                [identity.sample_id for identity in identities],
            )
            self.assertEqual(final_manifest["output_hashes"][-1]["sha256"], existing_digest)
            self.assertFalse(plan.pending_manifest_path.exists())

    def test_main_carries_reduced_batch_size_into_next_task(self):
        plans = [
            SimpleNamespace(task="discrete", identities=[object()], complete=False),
            SimpleNamespace(task="continuous", identities=[object()], complete=False),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = SimpleNamespace(
                task="all",
                seed=0,
                data_root=temporary_directory,
                output_root=str(Path(temporary_directory) / "outputs"),
                model_path="model",
                adapter_path="adapter",
                num_inference_steps=28,
                batch_size=16,
                vae_decode_batch_size=1,
                oom_fallback=True,
                fuse_lora=True,
                max_samples=None,
                dtype="bf16",
                offload=False,
            )
            with (
                patch.object(infer_seen10, "_checkpoint_provenance", return_value={}),
                patch.object(infer_seen10, "_make_task_plan", side_effect=plans),
                patch.object(infer_seen10, "_validate_complete_manifest", return_value=False),
                patch.object(infer_seen10, "_prepare_partial_resume"),
                patch.object(infer_seen10, "_load_pipeline", return_value=object()),
                patch.object(infer_seen10, "_run_task", side_effect=[8, 8]) as run_task,
            ):
                infer_seen10.main(args)

        self.assertEqual(run_task.call_count, 2)
        self.assertEqual(run_task.call_args_list[0].kwargs["initial_batch_size"], 16)
        self.assertEqual(run_task.call_args_list[1].kwargs["initial_batch_size"], 8)


if __name__ == "__main__":
    unittest.main()
