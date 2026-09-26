"""Exercise the actual local LoRA path used by aligned inference on CPU."""

from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from omnigen2.pipelines.lora_pipeline import OmniGen2LoraLoaderMixin


def _tiny_transformer():
    return OmniGen2Transformer2DModel(
        hidden_size=120,
        num_layers=1,
        num_refiner_layers=1,
        num_attention_heads=1,
        num_kv_heads=1,
        axes_dim_rope=(40, 40, 40),
        axes_lens=(16, 16, 16),
        text_feat_dim=32,
    )


class TestLoraLoaderCompat(unittest.TestCase):
    def test_legacy_dict_fetch_result_remains_supported(self):
        state = {"transformer.layers.0.attn.to_q.lora_A.weight": torch.zeros((8, 120))}
        with patch("omnigen2.pipelines.lora_pipeline._fetch_state_dict", return_value=state):
            self.assertIs(OmniGen2LoraLoaderMixin.lora_state_dict("unused"), state)

    def test_local_safetensors_load_and_fuse_on_real_transformer(self):
        source = _tiny_transformer()
        source.add_adapter(LoraConfig(
            r=8, lora_alpha=8, lora_dropout=0.0,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        ))
        with torch.no_grad():
            for name, parameter in source.named_parameters():
                if "lora_A" in name:
                    parameter.fill_(0.02)
                elif "lora_B" in name:
                    parameter.fill_(0.03)
        state = get_peft_model_state_dict(source, adapter_name="default")
        self.assertTrue(state)

        with TemporaryDirectory() as directory:
            OmniGen2Pipeline.save_lora_weights(
                save_directory=directory, transformer_lora_layers=state,
            )
            target = _tiny_transformer()
            pipeline = OmniGen2Pipeline(
                transformer=target, vae=None, scheduler=None, mllm=None, processor=None,
            )
            pipeline.load_lora_weights(
                directory, weight_name="pytorch_lora_weights.safetensors",
                local_files_only=True,
            )
            adapter_name = next(iter(target.peft_config))
            loaded = get_peft_model_state_dict(target, adapter_name=adapter_name)
            self.assertEqual(set(loaded), set(state))
            self.assertTrue(all(torch.equal(loaded[key], state[key]) for key in state))
            base_before = target.noise_refiner[0].attn.to_q.base_layer.weight.detach().clone()
            pipeline.fuse_lora(safe_fusing=True, components=["transformer"])
            base_after = target.noise_refiner[0].attn.to_q.base_layer.weight.detach()
            self.assertFalse(torch.equal(base_before, base_after))
            pipeline.unload_lora_weights()


if __name__ == "__main__":
    unittest.main()
