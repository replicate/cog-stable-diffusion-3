import os
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusion3Pipeline,
    StableDiffusion3Img2ImgPipeline
)
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPImageProcessor
from PIL import ImageOps

from weights import WeightsDownloadCache


SD3_MODEL_CACHE = "./sd3-cache"
SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "./feature-extractor"
SD3_URL = "https://weights.replicate.delivery/default/sd3/sd3-fp16.tar"
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def load_trained_weights(self, weights):
        weights = str(weights)
        if weights == self.loaded_weights:
            print("weights already loaded, no-op")
            return
        if self.loaded_weights is not None:
            self.unload_trained_weights()

        local_weights_path = self.weights_cache.ensure(weights)
        self.txt2img_pipe.load_lora_weights(local_weights_path)
        self.loaded_weights = weights
        return

    def unload_trained_weights(self):
        self.txt2img_pipe.unload_lora_weights()
        self.loaded_weights = None
        return

    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""

        start = time.time()

        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        if not os.path.exists(SD3_MODEL_CACHE):
            download_weights(SD3_URL, SD3_MODEL_CACHE)

        print("Loading sd3 txt2img pipeline...")
        self.txt2img_pipe = StableDiffusion3Pipeline.from_pretrained(
            SD3_MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

        self.weights_cache = WeightsDownloadCache()
        self.loaded_weights = None
        if str(weights) == "weights":
            weights = None
        if weights:
            self.load_trained_weights(weights)

        self.txt2img_pipe.to("cuda")

        print("Loading sd3 img2img pipeline...")
        self.img2img_pipe = StableDiffusion3Img2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            text_encoder_3=self.txt2img_pipe.text_encoder_3,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            tokenizer_3=self.txt2img_pipe.tokenizer_3,
            transformer=self.txt2img_pipe.transformer,
            scheduler=self.txt2img_pipe.scheduler,
        )

        # fix for img2img
        self.img2img_pipe.image_processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=self.img2img_pipe.vae.config.latent_channels)
        self.img2img_pipe.to("cuda")

    
        print("setup took: ", time.time() - start)

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        tmp_img = load_image("/tmp/image.png").convert("RGB")
        return ImageOps.contain(tmp_img, (1024, 1024))

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept
    
    def aspect_ratio_to_width_height(self, aspect_ratio: str):
        aspect_ratios = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "21:9": (1536, 640),
            "3:2": (1216, 832),
            "2:3": (832, 1216),
            "4:5": (896, 1088),
            "5:4": (1088, 896),
            "9:16": (768, 1344),
            "9:21": (640, 1536),
        }
        return aspect_ratios.get(aspect_ratio)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="",
        ),
        negative_prompt: str = Input(
            description="Input negative prompt",
            default="",
        ),
        image: Path = Input(
            description="Input image for img2img mode",
            default=None,
        ),
        aspect_ratio: str = Input(
            description="Aspect ratio for the generated image",
            choices=["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
            default="1:1",
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=0, le=50, default=4.5
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        output_format: str = Input(
            description="Format of the output images",
            choices=["webp", "jpg", "png"],
            default="webp",
        ),
        output_quality: int = Input(
            description="Quality when saving the output images, from 0 to 100. 100 is best quality, 0 is lowest quality. Not relevant for .png outputs",
            default=80,
            ge=0,
            le=100,
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images. This feature is only available through the API. See [https://replicate.com/docs/how-does-replicate-work#safety](https://replicate.com/docs/how-does-replicate-work#safety)",
            default=False,
        ),
        replicate_weights: str = Input(
            description="Replicate LoRA weights to use. Leave blank to use the default weights.",
            default=None,
        )
    ) -> List[Path]:
        """Run a single prediction on the model."""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if replicate_weights:
            self.load_trained_weights(replicate_weights)
        elif self.loaded_weights is not None:
            self.unload_trained_weights()

        width, height = self.aspect_ratio_to_width_height(aspect_ratio)

        num_inference_steps = 28

        sd3_kwargs = {}
        print(f"Prompt: {prompt}")
        if image:
            print("img2img mode")
            sd3_kwargs["image"] = self.load_image(image)
            sd3_kwargs["strength"] = prompt_strength
            pipe = self.img2img_pipe
        else:
            print("txt2img mode")
            sd3_kwargs["width"] = width
            sd3_kwargs["height"] = height
            pipe = self.txt2img_pipe

        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        output = pipe(**common_args, **sd3_kwargs)

        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(output.images)

        output_paths = []
        for i, image in enumerate(output.images):
            if not disable_safety_checker:
                if has_nsfw_content[i]:
                    print(f"NSFW content detected in image {i}")
                    continue
            output_path = f"/tmp/out-{i}.{output_format}"
            if output_format != 'png':
                image.save(output_path, quality=output_quality, optimize=True)
            else:
                image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
