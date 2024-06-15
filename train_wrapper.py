import mimetypes
import os
import shutil
import sys
import tarfile
from typing import Optional
from cog import Input, BaseModel, Path

import subprocess

from predict import SD3_MODEL_CACHE, SD3_URL, download_weights

OUTPUT_DIR = '/src/sd3-lora-out'
IMAGE_DIR = '/src/lora-dataset'
CAPTION_DIR = '/src/captions'

class TrainingOutput(BaseModel):
    weights: Path
    logs: Optional[Path]


def train(
        input_images: Path = Input(description='''a .tar file containing the images you'll use for fine tuning. You can add a csv of captions (captions.csv) to provide per-image captions; default image & caption columns are "image" and "caption". You can specify alternatives with the --image_column and --caption_column arguments.'''),
        instance_prompt: str = Input(description="The single caption to be used for all of your training images. Ignored if you pass a csv of captions with input images"),
        validation_prompt: str = Input(description="The prompt to run for validation if you want to run validation. This does not get parsed properly in the inputs field, so it's here"),
        inputs: str = Input(description="All the params you want to pass to the training script, separated by space. e.g. --resolution 1024 --train_batch_size 1", default=""),
        return_logs: bool = Input(description="If true, return tensorboard logs from training. NOT FULLY TESTED YET", default=False)
        ) -> TrainingOutput:
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    if os.path.exists(IMAGE_DIR):
        shutil.rmtree(IMAGE_DIR)
    if os.path.exists(CAPTION_DIR):
        shutil.rmtree(CAPTION_DIR)

    assert str(input_images).endswith(
        ".tar"
    ), "files must be a tar file if not zip"
    caption_csv = None
    with tarfile.open(input_images, "r") as tar_ref:
        for tar_info in tar_ref:
            if tar_info.name[-1] == "/" or tar_info.name.startswith("__MACOSX"):
                continue
            mt = mimetypes.guess_type(tar_info.name)
            if mt and mt[0] and mt[0].startswith("image/"):
                tar_info.name = os.path.basename(tar_info.name)
                tar_ref.extract(tar_info, IMAGE_DIR)
            if mt and mt[0] and mt[0] == "text/csv":
                tar_info.name = os.path.basename(tar_info.name)
                tar_ref.extract(tar_info, CAPTION_DIR)
                caption_csv = os.path.join(CAPTION_DIR, tar_info.name)

    if not os.path.exists(SD3_MODEL_CACHE):
        download_weights(SD3_URL, SD3_MODEL_CACHE)
    
    inputs = inputs.replace('"', '') # quotes don't pass well to the actual script
    args = inputs.split()
    to_run = ['./train.sh', f"--instance_prompt={instance_prompt}", f"--output_dir={OUTPUT_DIR}"] + args

    if caption_csv:
        to_run.append(f'--caption_csv={caption_csv}')
    
    if validation_prompt:
        to_run.append(f"--validation_prompt={validation_prompt}")

    print("training with command", to_run)
    proc = subprocess.Popen(to_run, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in proc.stdout:
        sys.stdout.write(line)
        
    return_code = proc.wait()

    if return_code == 0:
        print("successful training, packaging up output")

        os.system(f"tar -cvf lora_out.tar -C {OUTPUT_DIR}/ pytorch_lora_weights.safetensors")
        logs = None
        if return_logs:
            os.system("tar -cvf tensorboard_logs.tar -C {OUTPUT_DIR}/ logs/")
            logs = Path('/src/tensorboard_logs.tar')

        return TrainingOutput(weights=Path("/src/lora_out.tar"), logs=logs)
    else:
        print("training error")
