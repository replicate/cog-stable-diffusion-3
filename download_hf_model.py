import argparse
from diffusers import StableDiffusion3Pipeline
import torch
from huggingface_hub import login, snapshot_download
import os

def main():
    token = os.environ['HUGGING_FACE_HUB_TOKEN']
    login(token)
    pipe = StableDiffusion3Pipeline.from_pretrained('stabilityai/stable-diffusion-3-medium-diffusers', torch_dtype=torch.float16)
    pipe = pipe.to('cuda')

    pipe('a cool dog', height=1024, width=1024, num_inference_steps=28)

    pipe.save_pretrained("sd3-cache/", safe_serialization=True)

if __name__ == '__main__':
    main()