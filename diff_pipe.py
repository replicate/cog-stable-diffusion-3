# Copyright 2023 The HuggingFace Team / exx8 / Replicate. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified from https://github.com/exx8/differential-diffusion/blob/main/SD2/diff_pipe.py

import inspect
from typing import Callable, List, Optional, Union

import numpy as np
from PIL import Image, ImageOps
import torch
import torchvision.transforms
from transformers import (
    T5EncoderModel,
    T5TokenizerFast,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import PIL_INTERPOLATION, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img import (
    StableDiffusion3Img2ImgPipeline,
    retrieve_timesteps,
    retrieve_latents,
)
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3PipelineOutput


logger = logging.get_logger(__name__)


def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, Image.Image):
        image = [image]

    if isinstance(image[0], Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 8, (w, h))

        image = [
            np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :]
            for i in image
        ]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


class StableDiffusion3DiffImg2ImgPipeline(StableDiffusion3Img2ImgPipeline):
    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
    ):
        super().__init__(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            text_encoder_3=text_encoder_3,
            tokenizer_3=tokenizer_3,
        )

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def get_timesteps(self, num_inference_steps, strength, device):
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]
        return timesteps, num_inference_steps - t_start

    def prepare_latents(
        self,
        image,
        num_inference_steps,
        strength,
        dtype,
        device,
        generator=None,
    ):
        # Get noised images for every time step.
        # Only tested with FlowMatchEulerDiscreteScheduler.

        image = image.to(device=device, dtype=dtype)

        latents = retrieve_latents(self.vae.encode(image))
        latents = (
            latents - self.vae.config.shift_factor
        ) * self.vae.config.scaling_factor
        latents = torch.cat([latents], dim=0)
        shape = latents.shape
        noise = randn_tensor(shape, generator=None, device=device, dtype=dtype)
        latents_list = []
        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))

        for t in range(t_start, num_inference_steps):
            sigma = self.scheduler.sigmas[t]
            latents_list.append((sigma * noise + (1.0 - sigma) * latents)[0])
        latents = torch.stack(latents_list)

        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, Image.Image] = None,
        strength: float = 1,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        mask: Image.Image = None,
    ):
        image = ensure_image_size_multiple_of_eight(image)
        width, height = image.size
        inverted_mask = ImageOps.invert(
            mask.convert("RGB")
        )  # consistent with existing inpainting models
        resized_mask = inverted_mask.resize([width // 8, sixteen // 8])
        map = torchvision.transforms.ToTensor()(resized_mask)[0, :, :].to("cuda")

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt=negative_prompt,
            negative_prompt_2=None,
            negative_prompt_3=None,
            do_classifier_free_guidance=do_classifier_free_guidance,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds]
            )

        image = preprocess(image)

        map = torchvision.transforms.Resize(
            tuple(s // self.vae_scale_factor for s in image.shape[2:]), antialias=None
        )(map)

        assert batch_size == 1
        assert num_images_per_prompt == 1

        self.scheduler.set_timesteps(num_inference_steps)
        original_with_noise = self.prepare_latents(
            image,
            num_inference_steps,
            strength,
            prompt_embeds.dtype,
            device,
            generator,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, None
        )
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, device
        )

        thresholds = torch.arange(len(timesteps), dtype=map.dtype) / len(timesteps)
        thresholds = thresholds.unsqueeze(1).unsqueeze(1).to(device)
        masks = map > thresholds

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i == 0:
                    latents = original_with_noise[:1]
                else:
                    mask = masks[i].unsqueeze(0)
                    mask = mask.to(latents.dtype)
                    mask = mask.unsqueeze(1)
                    latents = original_with_noise[i] * mask + latents * (1 - mask)

                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=t.expand(latent_model_input.shape[0]),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if i == len(timesteps) - 1 or (
                    (i + 1)
                    > len(timesteps) - num_inference_steps * self.scheduler.order
                    and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)


def ensure_image_size_multiple_of_sixteen(image):
    width, height = image.size
    if (width // 16) * 16 != width or (height // 16) * 16 != height:
        print(
            "Width or height of input image are not multiples of 16. Slightly cropping..."
        )

    new_width = (width // 16) * 16
    new_height = (height // 16) * 16

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    return image.crop((left, top, right, bottom))
