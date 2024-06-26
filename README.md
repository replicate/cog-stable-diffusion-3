# Cog template for Stable Diffusion 3

[![Replicate demo and cloud API](https://replicate.com/stability-ai/stable-diffusion-3/badge)](https://replicate.com/stability-ai/stable-diffusion-3)

This is an implementation of Stability AI's [Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3-medium) as a [Cog](https://github.com/replicate/cog) model.

Stable Diffusion 3 has [a non-commercial license](https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE). [stability-ai/stable-diffusion-3](https://replicate.com/stability-ai/stable-diffusion-3) on Replicate is licensed and can be used for commercial work, but if you want to use custom versions built from this repository for commercial work, you'll need to buy [a commercial license](https://stability.ai/license).

## Run Stable Diffusion 3 yourself

Use this repository with the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of Stable Diffusion 3 to [Replicate](https://replicate.com).

To pull a copy of the Stable Diffusion 3 weights from HuggingFace, first get access by filling out the form on the [weights page](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/tree/main). Then, set your `HUGGING_FACE_HUB_TOKEN` as an env var and run `download_hf_model.py`. 


## Local Usage

For a local prediction, run:

```bash
cog predict -i prompt="a photo of a cool dog"
```

Learn more about how to deploy a cog model [here](https://github.com/replicate/cog). 
