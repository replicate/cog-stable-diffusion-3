#!/bin/bash

if ! command -v accelerate &> /dev/null
then
    echo "accelerate cli is not installed. Installing now..."
    pip uninstall accelerate -y
    pip install accelerate
fi

accelerate launch trainer.py \
  --pretrained_model_name_or_path=/src/sd3-cache \
  --instance_data_dir=/src/datasets/monstertoy \
  "$@"