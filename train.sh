#!/bin/bash

if ! command -v accelerate &> /dev/null
then
    echo "accelerate cli is not installed. Installing now..."
    pip uninstall accelerate -y
    pip install accelerate
fi

accelerate launch train.py \
  --pretrained_model_name_or_path=/src/sd3-cache \
  --instance_data_dir=/src/datasets/monstertoy \
  --output_dir=/src/sd3-dreambooth \
  --mixed_precision="fp16" \
  --instance_prompt="a photo of a sks toy" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --weighting_scheme="logit_normal" \
  --validation_prompt="A photo of a sks toy on the beach" \
  --validation_epochs=50 \
  --seed="0"