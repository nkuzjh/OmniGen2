import tempfile
import unittest

import torch

from convert_ckpt_to_hf_format import load_pose_adapter, save_pose_adapter
from omnigen2.models.transformers.repo import OmniGen2RotaryPosEmbed
from omnigen2.models.transformers import block_lumina2, transformer_omnigen2
from omnigen2.models.transformers.components import swiglu as torch_swiglu
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import prepare_pose_values


# The repository prefers Triton RMSNorm whenever Triton is installed, including
# in CPU-only environments where those kernels cannot launch.
block_lumina2.RMSNorm = torch.nn.RMSNorm
transformer_omnigen2.RMSNorm = torch.nn.RMSNorm
block_lumina2.swiglu = torch_swiglu


def _disable_flash_attention_for_cpu_tests():
    raise ImportError("FlashAttention is not used by this CPU-only unit test")


transformer_omnigen2.OmniGen2AttnProcessorFlash2Varlen = _disable_flash_attention_for_cpu_tests


def make_tiny_transformer(pose_conditioning=True):
    return OmniGen2Transformer2DModel(
        patch_size=2,
        in_channels=2,
        hidden_size=32,
        num_layers=1,
        num_refiner_layers=0,
        num_attention_heads=4,
        num_kv_heads=2,
        multiple_of=8,
        norm_eps=1e-5,
        axes_dim_rope=(2, 2, 4),
        axes_lens=(32, 32, 32),
        text_feat_dim=16,
        timestep_scale=1.0,
        pose_conditioning=pose_conditioning,
        pose_input_dim=5,
        pose_hidden_dim=8,
    )


class PoseConditioningTests(unittest.TestCase):
    def test_transformer_forward_backward_and_pose_validation(self):
        model = make_tiny_transformer()
        batch_size = 2
        text_hidden_states = torch.randn(batch_size, 4, 16)
        text_attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
        pose_values = torch.tensor(
            [[0.1, -0.2, 0.3, 0.4, -0.5], [0.5, 0.4, 0.3, 0.2, 0.1]],
            dtype=torch.float32,
        )

        conditioned_text, conditioned_mask = model._append_pose_condition(
            text_hidden_states, text_attention_mask, pose_values, batch_size
        )
        self.assertEqual(tuple(conditioned_text.shape), (batch_size, 5, 16))
        self.assertTrue(torch.equal(conditioned_mask, torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])))
        self.assertTrue(torch.allclose(conditioned_text[0, 2], model.pose_adapter(pose_values[:1])[0]))

        # The released Transformer initializes several modulation/output
        # projections to zero. Make this tiny test model non-degenerate so the
        # generated image path has a gradient back to the pose token.
        with torch.no_grad():
            for block in model.layers:
                block.norm1.linear.weight.normal_(mean=0.0, std=0.02)
                block.norm1.linear.bias.normal_(mean=0.0, std=0.02)
            model.norm_out.linear_1.weight.normal_(mean=0.0, std=0.02)
            model.norm_out.linear_1.bias.normal_(mean=0.0, std=0.02)
            model.norm_out.linear_2.weight.normal_(mean=0.0, std=0.02)

        hidden_states = torch.randn(batch_size, 2, 4, 4)
        freqs_cis = OmniGen2RotaryPosEmbed.get_freqs_cis(
            model.config.axes_dim_rope,
            model.config.axes_lens,
            theta=10000,
        )
        output = model(
            hidden_states=hidden_states,
            timestep=torch.ones(batch_size),
            text_hidden_states=text_hidden_states,
            freqs_cis=freqs_cis,
            text_attention_mask=text_attention_mask,
            pose_values=pose_values,
        )
        self.assertEqual(tuple(output.shape), tuple(hidden_states.shape))
        output.square().mean().backward()
        first_layer_grad = model.pose_adapter.network[0].weight.grad
        self.assertIsNotNone(first_layer_grad)
        self.assertGreater(first_layer_grad.abs().sum().item(), 0.0)

        with self.assertRaises(ValueError):
            model._append_pose_condition(
                text_hidden_states, text_attention_mask, torch.zeros(batch_size, 4), batch_size
            )
        with self.assertRaises(ValueError):
            model._append_pose_condition(
                text_hidden_states,
                text_attention_mask,
                torch.tensor([[float("nan")] * 5] * batch_size),
                batch_size,
            )

    def test_pipeline_batch_expansion_and_legacy_mode(self):
        pose_values = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.float32)
        expanded = prepare_pose_values(
            pose_values=pose_values,
            batch_size=2,
            num_images_per_prompt=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
            pose_conditioning_enabled=True,
        )
        self.assertEqual(tuple(expanded.shape), (6, 5))
        self.assertTrue(torch.equal(expanded, pose_values.repeat_interleave(3, dim=0)))
        self.assertIsNone(
            prepare_pose_values(
                pose_values=None,
                batch_size=1,
                num_images_per_prompt=1,
                device=torch.device("cpu"),
                dtype=torch.float32,
                pose_conditioning_enabled=False,
            )
        )
        with self.assertRaises(ValueError):
            prepare_pose_values(
                pose_values=pose_values[:1],
                batch_size=1,
                num_images_per_prompt=1,
                device=torch.device("cpu"),
                dtype=torch.float32,
                pose_conditioning_enabled=False,
            )

    def test_pose_adapter_sidecar_round_trip(self):
        source = make_tiny_transformer()
        with torch.no_grad():
            for parameter in source.pose_adapter.parameters():
                parameter.fill_(0.125)

        target = make_tiny_transformer(pose_conditioning=False)
        with tempfile.TemporaryDirectory() as directory:
            save_pose_adapter(source, directory)
            loaded_adapter = load_pose_adapter(target, directory)

        self.assertIsNotNone(loaded_adapter)
        self.assertTrue(target.config.pose_conditioning)
        for source_parameter, target_parameter in zip(
            source.pose_adapter.parameters(), target.pose_adapter.parameters()
        ):
            self.assertTrue(torch.equal(source_parameter, target_parameter))


if __name__ == "__main__":
    unittest.main()
