import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from omnigen2.models.transformers.repo import OmniGen2RotaryPosEmbed
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline


class _FakePosterior:
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.deterministic = deterministic

    def sample(self, generator=None):
        noise = torch.randn(
            self.parameters.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        return self.parameters + noise


class _FakeVae:
    def __init__(self):
        self.dtype = torch.float32
        self.training = False
        self.config = SimpleNamespace(shift_factor=None, scaling_factor=None)
        self.encode_calls = 0
        self.decode_batch_sizes = []

    def encode(self, image_tensor):
        self.encode_calls += 1
        parameters = image_tensor[:, :1].clone()
        return SimpleNamespace(latent_dist=_FakePosterior(parameters))

    def decode(self, latents, return_dict=False):
        self.decode_batch_sizes.append(latents.shape[0])
        return (latents.clone(),)


class _FakeImageProcessor:
    def preprocess(self, image, max_pixels=None, max_side_length=None):
        value = image.getpixel((0, 0))
        if isinstance(value, tuple):
            value = value[0]
        return torch.full((1, 1, 2, 2), float(value), dtype=torch.float32)


class _ProgressBar:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def update(self, *_args):
        pass


class _EmptyScheduler:
    order = 1

    def __init__(self):
        self.timesteps = []

    def set_timesteps(self, num_inference_steps=None, device=None, timesteps=None, **_kwargs):
        self.timesteps = [] if timesteps is None else timesteps


class _PoseAdapter:
    output_dim = 4

    def __call__(self, pose_values):
        return pose_values[:, : self.output_dim]


def _bare_pipeline():
    pipeline = object.__new__(OmniGen2Pipeline)
    pipeline._vae_posterior_cache = {}
    pipeline._empty_negative_prompt_cache = {}
    pipeline._base_rope_freqs_cache = {}
    pipeline.vae = _FakeVae()
    pipeline.vae_scale_factor = 8
    pipeline.image_processor = _FakeImageProcessor()
    return pipeline


def _generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


class PipelineInferenceCacheTests(unittest.TestCase):
    def test_b1_legacy_and_batched_nested_input_normalization(self):
        image0 = Image.new("L", (2, 2), color=10)
        image1 = Image.new("L", (2, 2), color=20)

        self.assertEqual(
            OmniGen2Pipeline._normalize_input_image_batch([image0], 1),
            [[image0]],
        )
        self.assertEqual(
            OmniGen2Pipeline._normalize_input_image_batch([[image0]], 1),
            [[image0]],
        )
        rows = OmniGen2Pipeline._normalize_input_image_batch([[image0], [image1]], 2)
        self.assertEqual(rows, [[image0], [image1]])
        self.assertEqual(
            OmniGen2Pipeline._normalize_input_image_cache_keys(["map-a"], [[image0]]),
            [["map-a"]],
        )
        self.assertEqual(
            OmniGen2Pipeline._normalize_input_image_cache_keys(
                (("map-a",),), [[image0]]
            ),
            [["map-a"]],
        )
        self.assertEqual(
            OmniGen2Pipeline._normalize_input_image_cache_keys(
                [["map-a"], ["map-b"]], rows
            ),
            [["map-a"], ["map-b"]],
        )

        with self.assertRaises(ValueError):
            OmniGen2Pipeline._normalize_input_image_cache_keys([["map-a"]], rows)
        with self.assertRaises(ValueError):
            OmniGen2Pipeline._normalize_input_image_cache_keys(
                [["map-a"], ["map-b", "extra"]], rows
            )

    def test_posterior_cache_resamples_with_per_sample_generators(self):
        pipeline = _bare_pipeline()
        radar = Image.new("L", (2, 2), color=37)

        def prepare(seeds):
            return pipeline.prepare_image(
                images=[[radar] for _ in seeds],
                batch_size=len(seeds),
                num_images_per_prompt=1,
                max_pixels=16,
                max_side_length=4,
                device=torch.device("cpu"),
                dtype=torch.float32,
                reference_generator=[_generator(seed) for seed in seeds],
                input_image_cache_keys=[["map-a"] for _ in seeds],
            )

        first = prepare([31, 37])
        repeated = prepare([31, 37])
        different = prepare([32, 38])

        self.assertEqual(pipeline.vae.encode_calls, 1)
        self.assertTrue(torch.equal(first[0][0], repeated[0][0]))
        self.assertTrue(torch.equal(first[1][0], repeated[1][0]))
        self.assertFalse(torch.equal(first[0][0], first[1][0]))
        self.assertFalse(torch.equal(first[0][0], different[0][0]))

        single = pipeline.prepare_image(
            images=[radar],
            batch_size=1,
            num_images_per_prompt=1,
            max_pixels=16,
            max_side_length=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            reference_generator=_generator(31),
            input_image_cache_keys=["map-a"],
        )
        self.assertTrue(torch.equal(first[0][0], single[0][0]))

        with self.assertRaisesRegex(ValueError, "different image content"):
            pipeline.prepare_image(
                images=[Image.new("L", (2, 2), color=38)],
                batch_size=1,
                num_images_per_prompt=1,
                max_pixels=16,
                max_side_length=4,
                device=torch.device("cpu"),
                dtype=torch.float32,
                reference_generator=_generator(41),
                input_image_cache_keys=["map-a"],
            )

    def test_reference_generator_list_length_is_validated(self):
        pipeline = _bare_pipeline()
        radar = Image.new("L", (2, 2), color=9)
        with self.assertRaisesRegex(ValueError, "one generator per output sample"):
            pipeline.prepare_image(
                images=[[radar], [radar]],
                batch_size=2,
                num_images_per_prompt=1,
                max_pixels=16,
                max_side_length=4,
                device=torch.device("cpu"),
                dtype=torch.float32,
                reference_generator=[_generator(1)],
                input_image_cache_keys=[["map"], ["map"]],
            )

    def test_tensor_image_fingerprint_supports_bfloat16(self):
        tensor = torch.arange(4, dtype=torch.bfloat16).view(1, 1, 2, 2)
        fingerprint = OmniGen2Pipeline._image_fingerprint(tensor)
        self.assertEqual(fingerprint, OmniGen2Pipeline._image_fingerprint(tensor.clone()))
        self.assertNotEqual(
            fingerprint,
            OmniGen2Pipeline._image_fingerprint(tensor + torch.tensor(1, dtype=torch.bfloat16)),
        )

    def test_vae_decode_microbatches_keep_sample_order(self):
        pipeline = _bare_pipeline()
        pipeline.scheduler = _EmptyScheduler()
        pipeline.transformer = SimpleNamespace(enable_teacache=False)
        pipeline.progress_bar = lambda total: _ProgressBar()
        pipeline._text_guidance_scale = 1.0
        pipeline._image_guidance_scale = 1.0
        pipeline._cfg_range = (0.0, 1.0)

        latents = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1)
        decoded = pipeline.processing(
            latents=latents,
            ref_latents=[None] * len(latents),
            prompt_embeds=torch.empty(5, 1, 1),
            freqs_cis=(),
            negative_prompt_embeds=None,
            prompt_attention_mask=torch.ones(5, 1, dtype=torch.long),
            negative_prompt_attention_mask=None,
            num_inference_steps=0,
            timesteps=[],
            device=torch.device("cpu"),
            dtype=torch.float32,
            verbose=False,
            vae_decode_batch_size=2,
        )

        self.assertEqual(pipeline.vae.decode_batch_sizes, [2, 2, 1])
        self.assertTrue(torch.equal(decoded, latents))

    def test_empty_negative_prompt_and_rope_base_are_cached_once(self):
        pipeline = _bare_pipeline()
        pipeline.mllm = SimpleNamespace(dtype=torch.float32, training=False)
        embedding = torch.arange(6, dtype=torch.float32).view(1, 2, 3)
        mask = torch.tensor([[1, 1]])
        pipeline._apply_chat_template = lambda prompt: f"chat:{prompt}"
        pipeline._get_qwen2_prompt_embeds = Mock(return_value=(embedding, mask))

        first_negative = pipeline._get_cached_empty_negative_prompt(torch.device("cpu"), 32)
        second_negative = pipeline._get_cached_empty_negative_prompt(torch.device("cpu"), 32)
        self.assertIs(first_negative, second_negative)
        pipeline._get_qwen2_prompt_embeds.assert_called_once()

        pipeline.transformer = SimpleNamespace(
            config=SimpleNamespace(axes_dim_rope=(2, 2, 2), axes_lens=(4, 5, 6))
        )
        with patch.object(
            OmniGen2RotaryPosEmbed,
            "get_freqs_cis",
            return_value=[torch.ones(4, 1), torch.ones(5, 1), torch.ones(6, 1)],
        ) as get_freqs:
            first_freqs = pipeline._get_base_rope_freqs(torch.device("cpu"))
            second_freqs = pipeline._get_base_rope_freqs(torch.device("cpu"))

        self.assertIs(first_freqs, second_freqs)
        get_freqs.assert_called_once()
        self.assertTrue(all(freq.device.type == "cpu" for freq in first_freqs))

    def test_transformer_direct_pose_validation_remains_enabled_by_default(self):
        transformer_stub = SimpleNamespace(pose_adapter=_PoseAdapter())
        hidden_states = torch.zeros(1, 2, 4)
        right_padded_mask = torch.tensor([[1, 0]])
        pose_values = torch.zeros(1, 5)

        with self.assertRaisesRegex(ValueError, "finite values"):
            OmniGen2Transformer2DModel._append_pose_condition(
                transformer_stub,
                hidden_states,
                right_padded_mask,
                torch.tensor([[float("nan"), 0, 0, 0, 0]]),
                1,
            )

        with self.assertRaisesRegex(ValueError, "right padded"):
            OmniGen2Transformer2DModel._append_pose_condition(
                transformer_stub,
                hidden_states,
                torch.tensor([[0, 1]]),
                pose_values,
                1,
            )


if __name__ == "__main__":
    unittest.main()
