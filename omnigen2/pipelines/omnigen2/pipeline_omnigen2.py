"""
OmniGen2 Diffusion Pipeline

Copyright 2025 BAAI, The OmniGen2 Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import hashlib
import inspect
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple, Union

import math

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from transformers import Qwen2_5_VLForConditionalGeneration

from diffusers.models.autoencoders import AutoencoderKL
from ...models.transformers import OmniGen2Transformer2DModel
from ...models.transformers.repo import OmniGen2RotaryPosEmbed
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from dataclasses import dataclass

import PIL.Image

from diffusers.utils import BaseOutput

from omnigen2.pipelines.image_processor import OmniGen2ImageProcessor

from omnigen2.utils.teacache_util import TeaCacheParams

from ..lora_pipeline import OmniGen2LoraLoaderMixin


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

from ...cache_functions import cache_init 

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def prepare_pose_values(
    pose_values: Optional[torch.Tensor],
    batch_size: int,
    num_images_per_prompt: int,
    device: torch.device,
    dtype: torch.dtype,
    pose_conditioning_enabled: bool,
) -> Optional[torch.Tensor]:
    """Validate and expand one normalized [x, y, z, pitch, yaw] row per prompt."""
    if num_images_per_prompt <= 0:
        raise ValueError("num_images_per_prompt must be a positive integer")
    if pose_values is None:
        if pose_conditioning_enabled:
            raise ValueError("pose_values are required when pose_conditioning=True")
        return None
    if not pose_conditioning_enabled:
        raise ValueError("pose_values were supplied to a transformer without pose conditioning")
    if not torch.is_tensor(pose_values):
        raise TypeError("pose_values must be a torch.Tensor with shape [B, 5]")
    if pose_values.ndim != 2 or tuple(pose_values.shape) != (batch_size, 5):
        raise ValueError(
            f"pose_values must have shape [{batch_size}, 5], got {tuple(pose_values.shape)}"
        )
    if not pose_values.is_floating_point():
        raise TypeError("pose_values must use a floating-point dtype")
    if not torch.isfinite(pose_values).all().item():
        raise ValueError("pose_values must contain only finite values")

    return pose_values.to(device=device, dtype=dtype).repeat_interleave(
        num_images_per_prompt, dim=0
    )


def _validate_right_padded_attention_mask(
    attention_mask: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    name: str,
) -> None:
    """Validate a text mask once before the denoising loop."""
    if attention_mask is None:
        raise ValueError(f"{name} is required when pose conditioning is enabled")
    if attention_mask.ndim != 2 or tuple(attention_mask.shape) != tuple(hidden_states.shape[:2]):
        raise ValueError(f"{name} must match the [B, sequence] text dimensions")

    valid_mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
    lengths = valid_mask.sum(dim=1)
    positions = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
    expected_mask = positions < lengths.unsqueeze(1)
    if not torch.equal(valid_mask, expected_mask):
        raise ValueError(f"{name} must be right padded with a contiguous valid prefix")


@dataclass
class FMPipelineOutput(BaseOutput):
    """
    Output class for OmniGen2 pipeline.

    Args:
        images (Union[List[PIL.Image.Image], np.ndarray]): 
            List of denoised PIL images of length `batch_size` or numpy array of shape 
            `(batch_size, height, width, num_channels)`. Contains the generated images.
    """
    images: Union[List[PIL.Image.Image], np.ndarray]


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class OmniGen2Pipeline(DiffusionPipeline, OmniGen2LoraLoaderMixin):
    """
    Pipeline for text-to-image generation using OmniGen2.

    This pipeline implements a text-to-image generation model that uses:
    - Qwen2.5-VL for text encoding
    - A custom transformer architecture for image generation
    - VAE for image encoding/decoding
    - FlowMatchEulerDiscreteScheduler for noise scheduling

    Args:
        transformer (OmniGen2Transformer2DModel): The transformer model for image generation.
        vae (AutoencoderKL): The VAE model for image encoding/decoding.
        scheduler (FlowMatchEulerDiscreteScheduler): The scheduler for noise scheduling.
        text_encoder (Qwen2_5_VLModel): The text encoder model.
        tokenizer (Union[Qwen2Tokenizer, Qwen2TokenizerFast]): The tokenizer for text processing.
    """

    model_cpu_offload_seq = "mllm->transformer->vae"

    def __init__(
        self,
        transformer: OmniGen2Transformer2DModel,
        vae: AutoencoderKL,
        scheduler: FlowMatchEulerDiscreteScheduler,
        mllm: Qwen2_5_VLForConditionalGeneration,
        processor,
    ) -> None:
        """
        Initialize the OmniGen2 pipeline.

        Args:
            transformer: The transformer model for image generation.
            vae: The VAE model for image encoding/decoding.
            scheduler: The scheduler for noise scheduling.
            text_encoder: The text encoder model.
            tokenizer: The tokenizer for text processing.
        """
        super().__init__()

        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            mllm=mllm,
            processor=processor
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = OmniGen2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_resize=True)
        self.default_sample_size = 128

        # These inference caches only contain deterministic model outputs or
        # posterior parameters. In particular, the random VAE samples are
        # never retained, so each sample can keep using its own RNG seed.
        self._vae_posterior_cache: Dict[Any, Any] = {}
        self._empty_negative_prompt_cache: Dict[Any, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._base_rope_freqs_cache: Dict[Any, Tuple[torch.Tensor, ...]] = {}
        self._transformer_forward_parameters = frozenset(
            inspect.signature(self.transformer.forward).parameters.keys()
        )

    def clear_inference_caches(self) -> None:
        """Release deterministic inference caches after changing models/devices."""
        self._vae_posterior_cache.clear()
        self._empty_negative_prompt_cache.clear()
        self._base_rope_freqs_cache.clear()

    @staticmethod
    def _normalize_input_image_batch(images: Any, batch_size: int) -> List[List[Any]]:
        """Normalize legacy B=1 images and batched per-sample reference images."""
        supported_image_types = (PIL.Image.Image, np.ndarray, torch.Tensor)

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if images is None or (isinstance(images, (list, tuple)) and len(images) == 0):
            return [[] for _ in range(batch_size)]

        if isinstance(images, supported_image_types):
            if batch_size != 1:
                raise ValueError(
                    "a single input image is only valid for batch_size=1; "
                    "pass one image row per batch sample"
                )
            rows = [[images]]
        elif not isinstance(images, (list, tuple)):
            raise TypeError("input_images must be an image or a list/tuple of per-sample images")
        elif batch_size == 1:
            # The established B=1 API is [PIL.Image]. Also accept [[PIL.Image]]
            # so the same batch-building code can be used for B=1 and B>1.
            if len(images) == 1 and images[0] is None:
                rows = [[]]
            elif len(images) == 1 and isinstance(images[0], (list, tuple)):
                rows = [list(images[0])]
            else:
                rows = [list(images)]
        else:
            if len(images) != batch_size:
                raise ValueError(
                    f"input_images must contain one row per prompt ({batch_size}), got {len(images)}"
                )
            rows = []
            for index, row in enumerate(images):
                if row is None:
                    rows.append([])
                elif isinstance(row, supported_image_types):
                    rows.append([row])
                elif isinstance(row, (list, tuple)):
                    rows.append(list(row))
                else:
                    raise TypeError(
                        f"input_images[{index}] must be an image, None, or a list/tuple of images"
                    )

        for sample_index, row in enumerate(rows):
            for image_index, image in enumerate(row):
                if not isinstance(image, supported_image_types):
                    raise TypeError(
                        f"input_images[{sample_index}][{image_index}] must be a PIL image, "
                        "NumPy array, or torch.Tensor"
                    )
        return rows

    @staticmethod
    def _normalize_input_image_cache_keys(
        cache_keys: Any,
        image_rows: List[List[Any]],
    ) -> Optional[List[List[Optional[Hashable]]]]:
        """Validate cache-key rows against normalized reference-image rows."""
        if cache_keys is None:
            return None
        batch_size = len(image_rows)
        if not isinstance(cache_keys, (list, tuple)):
            raise TypeError("input_image_cache_keys must be a list/tuple matching input_images")

        if batch_size == 1:
            if (
                len(cache_keys) == 1
                and isinstance(cache_keys[0], (list, tuple))
                and len(cache_keys[0]) == len(image_rows[0])
            ):
                key_rows = [list(cache_keys[0])]
            else:
                # B=1 shorthand: [key] for the legacy [image] input shape.
                key_rows = [list(cache_keys)]
        else:
            if len(cache_keys) != batch_size:
                raise ValueError(
                    "input_image_cache_keys must contain one key row per prompt "
                    f"({batch_size}), got {len(cache_keys)}"
                )
            key_rows = []
            for sample_index, row in enumerate(cache_keys):
                if not isinstance(row, (list, tuple)):
                    raise TypeError(
                        f"input_image_cache_keys[{sample_index}] must be a list/tuple of keys"
                    )
                key_rows.append(list(row))

        if len(key_rows) != batch_size:
            raise ValueError("input_image_cache_keys batch dimension does not match input_images")
        for sample_index, (keys, images) in enumerate(zip(key_rows, image_rows)):
            if len(keys) != len(images):
                raise ValueError(
                    f"input_image_cache_keys[{sample_index}] has {len(keys)} entries for "
                    f"{len(images)} input images"
                )
            for image_index, key in enumerate(keys):
                if key is not None:
                    try:
                        hash(key)
                    except TypeError as error:
                        raise TypeError(
                            f"input_image_cache_keys[{sample_index}][{image_index}] must be hashable"
                        ) from error
        return key_rows

    @staticmethod
    def _normalize_generators(
        generator: Optional[Union[torch.Generator, List[torch.Generator]]],
        expected_length: int,
        name: str,
    ) -> Optional[Union[torch.Generator, List[torch.Generator]]]:
        if generator is None or isinstance(generator, torch.Generator):
            return generator
        if not isinstance(generator, (list, tuple)):
            raise TypeError(f"{name} must be a torch.Generator or a list/tuple of generators")
        if len(generator) != expected_length:
            raise ValueError(
                f"{name} must contain one generator per output sample "
                f"({expected_length}), got {len(generator)}"
            )
        if any(not isinstance(item, torch.Generator) for item in generator):
            raise TypeError(f"every item in {name} must be a torch.Generator")
        return list(generator)

    @staticmethod
    def _image_fingerprint(image: Any) -> str:
        digest = hashlib.sha256()
        if isinstance(image, PIL.Image.Image):
            digest.update(image.mode.encode("utf-8"))
            digest.update(repr(image.size).encode("ascii"))
            digest.update(image.tobytes())
        elif isinstance(image, np.ndarray):
            contiguous = np.ascontiguousarray(image)
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(repr(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
        elif torch.is_tensor(image):
            tensor = image.detach().to(device="cpu").contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        else:
            raise TypeError("cached reference images must be PIL images, NumPy arrays, or torch.Tensor")
        return digest.hexdigest()

    def _get_vae_posterior(
        self,
        image: Any,
        cache_key: Optional[Hashable],
        max_pixels: int,
        max_side_length: int,
        device: torch.device,
    ):
        cache_identity = None
        fingerprint = None
        cached = None
        if cache_key is not None and not self.vae.training:
            cache_identity = (
                cache_key,
                max_pixels,
                max_side_length,
                self.vae_scale_factor,
                str(device),
                str(self.vae.dtype),
            )
            cached = self._vae_posterior_cache.get(cache_identity)
            if cached is not None:
                # Keyed images are an immutable-input contract. The Seen-10
                # runner shares read-only PIL objects, so an identity hit can
                # avoid re-hashing every radar for every generated frame.
                if cached["source_image"] is not image:
                    fingerprint = self._image_fingerprint(image)
                if fingerprint is not None and cached["fingerprint"] != fingerprint:
                    raise ValueError(
                        f"input image cache key {cache_key!r} was reused for different image content"
                    )
                return cached["distribution_class"](
                    cached["parameters"], deterministic=cached["deterministic"]
                )
            fingerprint = self._image_fingerprint(image)

        image_tensor = self.image_processor.preprocess(
            image, max_pixels=max_pixels, max_side_length=max_side_length
        ).to(device=device)
        posterior = self.vae.encode(image_tensor.to(dtype=self.vae.dtype)).latent_dist
        if cache_identity is not None:
            parameters = posterior.parameters.detach().clone()
            deterministic = bool(getattr(posterior, "deterministic", False))
            distribution_class = posterior.__class__
            self._vae_posterior_cache[cache_identity] = {
                "fingerprint": fingerprint,
                "source_image": image,
                "parameters": parameters,
                "deterministic": deterministic,
                "distribution_class": distribution_class,
            }
            posterior = distribution_class(parameters, deterministic=deterministic)
        return posterior

    def _get_cached_empty_negative_prompt(
        self,
        device: torch.device,
        max_sequence_length: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mllm_dtype = self.mllm.dtype
        cache_key = (str(device), str(mllm_dtype), max_sequence_length)
        cached = self._empty_negative_prompt_cache.get(cache_key)
        if cached is None:
            empty_prompt = self._apply_chat_template("")
            cached = self._get_qwen2_prompt_embeds(
                [empty_prompt],
                device=device,
                max_sequence_length=256 if max_sequence_length is None else max_sequence_length,
            )
            cached = (cached[0].detach(), cached[1].detach())
            self._empty_negative_prompt_cache[cache_key] = cached
        return cached

    def _get_base_rope_freqs(self, device: torch.device) -> Tuple[torch.Tensor, ...]:
        axes_dim = tuple(self.transformer.config.axes_dim_rope)
        axes_lens = tuple(self.transformer.config.axes_lens)
        cache_key = (axes_dim, axes_lens, 10000, device.type, device.index)
        freqs = self._base_rope_freqs_cache.get(cache_key)
        if freqs is None:
            freqs = tuple(
                freq.to(device=device)
                for freq in OmniGen2RotaryPosEmbed.get_freqs_cis(
                    axes_dim, axes_lens, theta=10000
                )
            )
            self._base_rope_freqs_cache[cache_key] = freqs
        return freqs

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
        latents: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Prepare the initial latents for the diffusion process.

        Args:
            batch_size: The number of images to generate.
            num_channels_latents: The number of channels in the latent space.
            height: The height of the generated image.
            width: The width of the generated image.
            dtype: The data type of the latents.
            device: The device to place the latents on.
            generator: The random number generator to use.
            latents: Optional pre-computed latents to use instead of random initialization.

        Returns:
            torch.FloatTensor: The prepared latents tensor.
        """
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        return latents

    def encode_vae(
        self,
        img: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.FloatTensor:
        """
        Encode an image into the VAE latent space.

        Args:
            img: The input image tensor to encode.

        Returns:
            torch.FloatTensor: The encoded latent representation.
        """
        z0 = self.vae.encode(img.to(dtype=self.vae.dtype)).latent_dist.sample(generator=generator)
        if self.vae.config.shift_factor is not None:
            z0 = z0 - self.vae.config.shift_factor
        if self.vae.config.scaling_factor is not None:
            z0 = z0 * self.vae.config.scaling_factor
        z0 = z0.to(dtype=self.vae.dtype)
        return z0

    def prepare_image(
        self,
        images: Any,
        batch_size: int,
        num_images_per_prompt: int,
        max_pixels: int,
        max_side_length: int,
        device: torch.device,
        dtype: torch.dtype,
        reference_generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        input_image_cache_keys: Any = None,
    ) -> List[Optional[torch.FloatTensor]]:
        """
        Prepare input images for processing by encoding them into the VAE latent space.

        Args:
            images: Single image or list of images to process.
            batch_size: The number of images to generate per prompt.
            num_images_per_prompt: The number of images to generate for each prompt.
            device: The device to place the encoded latents on.
            dtype: The data type of the encoded latents.

        Returns:
            List[Optional[torch.FloatTensor]]: List of encoded latent representations for each image.
        """
        if num_images_per_prompt <= 0:
            raise ValueError("num_images_per_prompt must be a positive integer")
        image_rows = self._normalize_input_image_batch(images, batch_size)
        cache_key_rows = self._normalize_input_image_cache_keys(
            input_image_cache_keys, image_rows
        )
        generators = self._normalize_generators(
            reference_generator,
            batch_size * num_images_per_prompt,
            "reference_generator",
        )

        output_latents: List[Optional[List[torch.FloatTensor]]] = []
        for sample_index, row in enumerate(image_rows):
            for output_index in range(num_images_per_prompt):
                output_sample_index = sample_index * num_images_per_prompt + output_index
                if isinstance(generators, list):
                    sample_generator = generators[output_sample_index]
                else:
                    sample_generator = generators

                if not row:
                    output_latents.append(None)
                    continue

                ref_latents = []
                for image_index, image in enumerate(row):
                    cache_key = (
                        cache_key_rows[sample_index][image_index]
                        if cache_key_rows is not None
                        else None
                    )
                    posterior = self._get_vae_posterior(
                        image=image,
                        cache_key=cache_key,
                        max_pixels=max_pixels,
                        max_side_length=max_side_length,
                        device=device,
                    )
                    latent = posterior.sample(generator=sample_generator)
                    if self.vae.config.shift_factor is not None:
                        latent = latent - self.vae.config.shift_factor
                    if self.vae.config.scaling_factor is not None:
                        latent = latent * self.vae.config.scaling_factor
                    ref_latents.append(latent.to(dtype=dtype).squeeze(0))
                output_latents.append(ref_latents)

        return output_latents
    
    def _get_qwen2_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        max_sequence_length: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prompt embeddings from the Qwen2 text encoder.

        Args:
            prompt: The prompt or list of prompts to encode.
            device: The device to place the embeddings on. If None, uses the pipeline's device.
            max_sequence_length: Maximum sequence length for tokenization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The prompt embeddings tensor
                - The attention mask tensor

        Raises:
            Warning: If the input text is truncated due to sequence length limitations.
        """
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        # text_inputs = self.processor.tokenizer(
        #     prompt,
        #     padding="max_length",
        #     max_length=max_sequence_length,
        #     truncation=True,
        #     return_tensors="pt",
        # )
        text_inputs = self.processor.tokenizer(
            prompt,
            padding="longest",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        untruncated_ids = self.processor.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids.to(device)

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.processor.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because Gemma can only handle sequences up to"
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_attention_mask = text_inputs.attention_mask.to(device)
        prompt_embeds = self.mllm(
            text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-1]

        if self.mllm is not None:
            dtype = self.mllm.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, prompt_attention_mask
    
    def _apply_chat_template(self, prompt: str):
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates high-quality images based on user instructions.",
            },
            {"role": "user", "content": prompt},
        ]
        prompt = self.processor.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        return prompt

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                Lumina-T2I, this should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. For Lumina-T2I, it's should be the embeddings of the "" string.
            max_sequence_length (`int`, defaults to `256`):
                Maximum sequence length to use for the prompt.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [self._apply_chat_template(_prompt) for _prompt in prompt]

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_qwen2_prompt_embeds(
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)
        prompt_attention_mask = prompt_attention_mask.view(batch_size * num_images_per_prompt, -1)

        # Get negative embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt if negative_prompt is not None else ""

            # Normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt = [self._apply_chat_template(_negative_prompt) for _negative_prompt in negative_prompt]

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            negative_prompt_embeds, negative_prompt_attention_mask = self._get_qwen2_prompt_embeds(
                prompt=negative_prompt,
                device=device,
                max_sequence_length=max_sequence_length,
            )

            batch_size, seq_len, _ = negative_prompt_embeds.shape
            # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(
                batch_size * num_images_per_prompt, -1
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask
    
    @property
    def num_timesteps(self):
        return self._num_timesteps
    
    @property
    def text_guidance_scale(self):
        return self._text_guidance_scale
    
    @property
    def image_guidance_scale(self):
        return self._image_guidance_scale
    
    @property
    def cfg_range(self):
        return self._cfg_range
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.LongTensor] = None,
        negative_prompt_attention_mask: Optional[torch.LongTensor] = None,
        max_sequence_length: Optional[int] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
        input_images: Optional[List[PIL.Image.Image]] = None,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_pixels: int = 1024 * 1024,
        max_input_image_side_length: int = 1024,
        align_res: bool = True,
        num_inference_steps: int = 28,
        text_guidance_scale: float = 4.0,
        image_guidance_scale: float = 1.0,
        cfg_range: Tuple[float, float] = (0.0, 1.0),
        attention_kwargs: Optional[Dict[str, Any]] = None,
        timesteps: List[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        verbose: bool = False,
        step_func=None,
        pose_values: Optional[torch.Tensor] = None,
        reference_generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        input_image_cache_keys: Optional[Any] = None,
        vae_decode_batch_size: Optional[int] = None,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if not isinstance(num_images_per_prompt, int) or isinstance(num_images_per_prompt, bool) or num_images_per_prompt <= 0:
            raise ValueError("num_images_per_prompt must be a positive integer")
        if vae_decode_batch_size is not None and (
            not isinstance(vae_decode_batch_size, int)
            or isinstance(vae_decode_batch_size, bool)
            or vae_decode_batch_size <= 0
        ):
            raise ValueError("vae_decode_batch_size must be a positive integer or None")

        if prompt is None:
            if prompt_embeds is None:
                raise ValueError("either prompt or prompt_embeds must be provided")
            if not torch.is_tensor(prompt_embeds) or prompt_embeds.ndim != 3:
                raise ValueError("prompt_embeds must be a tensor with shape [B, sequence, hidden_dim]")
            batch_size = prompt_embeds.shape[0]
        elif isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            if not prompt:
                raise ValueError("prompt list must contain at least one prompt")
            if any(not isinstance(item, str) for item in prompt):
                raise TypeError("every item in prompt must be a string")
            batch_size = len(prompt)
        else:
            raise TypeError("prompt must be a string, a list of strings, or None with prompt_embeds")

        if prompt_embeds is not None:
            if not torch.is_tensor(prompt_embeds) or prompt_embeds.ndim != 3:
                raise ValueError("prompt_embeds must be a tensor with shape [B, sequence, hidden_dim]")
            if prompt_embeds.shape[0] != batch_size:
                raise ValueError(
                    f"prompt_embeds batch dimension ({prompt_embeds.shape[0]}) must match "
                    f"prompt batch size ({batch_size})"
                )

        image_rows = self._normalize_input_image_batch(input_images, batch_size)
        cache_key_rows = self._normalize_input_image_cache_keys(
            input_image_cache_keys, image_rows
        )
        effective_batch_size = batch_size * num_images_per_prompt
        generator = self._normalize_generators(generator, effective_batch_size, "generator")
        reference_generator = self._normalize_generators(
            reference_generator, effective_batch_size, "reference_generator"
        )

        self._text_guidance_scale = text_guidance_scale
        self._image_guidance_scale = image_guidance_scale
        self._cfg_range = cfg_range
        self._attention_kwargs = attention_kwargs

        device = self._execution_device

        # 3. Encode input prompt
        use_cached_empty_negative_prompt = (
            self.text_guidance_scale > 1.0
            and not self.mllm.training
            and negative_prompt_embeds is None
            and negative_prompt_attention_mask is None
            and (
                negative_prompt is None
                or negative_prompt == ""
                or (
                    isinstance(negative_prompt, list)
                    and len(negative_prompt) == batch_size
                    and all(item == "" for item in negative_prompt)
                )
            )
        )
        if use_cached_empty_negative_prompt:
            cached_negative_embeds, cached_negative_mask = self._get_cached_empty_negative_prompt(
                device=device,
                max_sequence_length=max_sequence_length,
            )
            negative_prompt_embeds = cached_negative_embeds.expand(batch_size, -1, -1)
            negative_prompt_attention_mask = cached_negative_mask.expand(batch_size, -1)
            if num_images_per_prompt > 1:
                negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(
                    num_images_per_prompt, dim=0
                )
                negative_prompt_attention_mask = negative_prompt_attention_mask.repeat_interleave(
                    num_images_per_prompt, dim=0
                )

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            self.text_guidance_scale > 1.0,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
        )

        pose_conditioning_enabled = bool(
            getattr(self.transformer.config, "pose_conditioning", False)
        )
        pose_values = prepare_pose_values(
            pose_values=pose_values,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=prompt_embeds.dtype,
            pose_conditioning_enabled=pose_conditioning_enabled,
        )

        if pose_conditioning_enabled:
            _validate_right_padded_attention_mask(
                prompt_attention_mask, prompt_embeds, "prompt_attention_mask"
            )
            if negative_prompt_embeds is not None:
                _validate_right_padded_attention_mask(
                    negative_prompt_attention_mask,
                    negative_prompt_embeds,
                    "negative_prompt_attention_mask",
                )

        dtype = self.vae.dtype
        # 3. Prepare control image
        ref_latents = self.prepare_image(
            images=image_rows,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            max_pixels=max_pixels,
            max_side_length=max_input_image_side_length,
            device=device,
            dtype=dtype,
            reference_generator=reference_generator,
            input_image_cache_keys=cache_key_rows,
        )

        has_input_images = any(bool(row) for row in image_rows)
        if batch_size == 1 and len(image_rows[0]) == 1 and align_res:
            width, height = ref_latents[0][0].shape[-1] * self.vae_scale_factor, ref_latents[0][0].shape[-2] * self.vae_scale_factor
            ori_width, ori_height = width, height
        else:
            ori_width, ori_height = width, height

            cur_pixels = height * width
            ratio = (max_pixels / cur_pixels) ** 0.5
            ratio = min(ratio, 1.0)

            height, width = int(height * ratio) // 16 * 16, int(width * ratio) // 16 * 16
        
        if not has_input_images:
            self._image_guidance_scale = 1

        # 4. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        freqs_cis = self._get_base_rope_freqs(device)
        
        image = self.processing(
            latents=latents,
            ref_latents=ref_latents,
            prompt_embeds=prompt_embeds,
            freqs_cis=freqs_cis,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            device=device,
            dtype=dtype,
            verbose=verbose,
            step_func=step_func,
            pose_values=pose_values,
            vae_decode_batch_size=vae_decode_batch_size,
            inputs_validated=True,
        )

        image = F.interpolate(image, size=(ori_height, ori_width), mode='bilinear')

        image = self.image_processor.postprocess(image, output_type=output_type)
        
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return image
        else:
            return FMPipelineOutput(images=image)

    def processing(
        self,
        latents,
        ref_latents,
        prompt_embeds,
        freqs_cis,
        negative_prompt_embeds,
        prompt_attention_mask,
        negative_prompt_attention_mask,
        num_inference_steps,
        timesteps,
        device,
        dtype,
        verbose,
        step_func=None,
        pose_values=None,
        vae_decode_batch_size=None,
        inputs_validated=False,
    ):
        batch_size = latents.shape[0]
        if vae_decode_batch_size is not None and (
            not isinstance(vae_decode_batch_size, int)
            or isinstance(vae_decode_batch_size, bool)
            or vae_decode_batch_size <= 0
        ):
            raise ValueError("vae_decode_batch_size must be a positive integer or None")

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            num_tokens=latents.shape[-2] * latents.shape[-1]
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        enable_taylorseer = getattr(self, "enable_taylorseer", False)
        if enable_taylorseer:
            model_pred_cache_dic, model_pred_current = cache_init(self, num_inference_steps)
            model_pred_ref_cache_dic, model_pred_ref_current = cache_init(self, num_inference_steps)
            model_pred_uncond_cache_dic, model_pred_uncond_current = cache_init(self, num_inference_steps)
            self.transformer.enable_taylorseer = True
        elif self.transformer.enable_teacache:
            # Use different TeaCacheParams for different conditions
            teacache_params = TeaCacheParams()
            teacache_params_uncond = TeaCacheParams()
            teacache_params_ref = TeaCacheParams()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if enable_taylorseer:
                    self.transformer.cache_dic = model_pred_cache_dic
                    self.transformer.current = model_pred_current
                elif self.transformer.enable_teacache:
                    teacache_params.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                    self.transformer.teacache_params = teacache_params

                model_pred = self.predict(
                    t=t,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    freqs_cis=freqs_cis,
                    prompt_attention_mask=prompt_attention_mask,
                    ref_image_hidden_states=ref_latents,
                    pose_values=pose_values,
                    inputs_validated=inputs_validated,
                )
                text_guidance_scale = self.text_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0
                image_guidance_scale = self.image_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0
                
                if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_ref_cache_dic
                        self.transformer.current = model_pred_ref_current
                    elif self.transformer.enable_teacache:
                        teacache_params_ref.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_ref

                    model_pred_ref = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=ref_latents,
                        pose_values=pose_values,
                        inputs_validated=inputs_validated,
                    )

                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_uncond_cache_dic
                        self.transformer.current = model_pred_uncond_current
                    elif self.transformer.enable_teacache:
                        teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_uncond

                    model_pred_uncond = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=None,
                        pose_values=pose_values,
                        inputs_validated=inputs_validated,
                    )

                    model_pred = model_pred_uncond + image_guidance_scale * (model_pred_ref - model_pred_uncond) + \
                        text_guidance_scale * (model_pred - model_pred_ref)
                elif text_guidance_scale > 1.0:
                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_uncond_cache_dic
                        self.transformer.current = model_pred_uncond_current
                    elif self.transformer.enable_teacache:
                        teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_uncond

                    model_pred_uncond = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=None,
                        pose_values=pose_values,
                        inputs_validated=inputs_validated,
                    )
                    model_pred = model_pred_uncond + text_guidance_scale * (model_pred - model_pred_uncond)

                latents = self.scheduler.step(model_pred, t, latents, return_dict=False)[0]

                latents = latents.to(dtype=dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                
                if step_func is not None:
                    step_func(i, self._num_timesteps)

        if enable_taylorseer:
            del model_pred_cache_dic, model_pred_ref_cache_dic, model_pred_uncond_cache_dic
            del model_pred_current, model_pred_ref_current, model_pred_uncond_current

        latents = latents.to(dtype=dtype)
        if self.vae.config.scaling_factor is not None:
            latents = latents / self.vae.config.scaling_factor
        if self.vae.config.shift_factor is not None:
            latents = latents + self.vae.config.shift_factor
        if vae_decode_batch_size is None or vae_decode_batch_size >= batch_size:
            image = self.vae.decode(latents, return_dict=False)[0]
        else:
            decoded = []
            for start in range(0, batch_size, vae_decode_batch_size):
                decoded.append(
                    self.vae.decode(
                        latents[start : start + vae_decode_batch_size], return_dict=False
                    )[0]
                )
            image = torch.cat(decoded, dim=0)
        
        return image

    def predict(
        self,
        t,
        latents,
        prompt_embeds,
        freqs_cis,
        prompt_attention_mask,
        ref_image_hidden_states,
        pose_values=None,
        inputs_validated=False,
    ):
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        batch_size, num_channels_latents, height, width = latents.shape
        
        optional_kwargs = {}
        if 'ref_image_hidden_states' in self._transformer_forward_parameters:
            optional_kwargs['ref_image_hidden_states'] = ref_image_hidden_states
        if pose_values is not None:
            if 'pose_values' not in self._transformer_forward_parameters:
                raise ValueError("the loaded transformer does not support pose_values")
            optional_kwargs['pose_values'] = pose_values
        if inputs_validated and '_inputs_validated' in self._transformer_forward_parameters:
            optional_kwargs['_inputs_validated'] = True
        
        model_pred = self.transformer(
            latents,
            timestep,
            prompt_embeds,
            freqs_cis,
            prompt_attention_mask,
            **optional_kwargs
        )
        return model_pred
