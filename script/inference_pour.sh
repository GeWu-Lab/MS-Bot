#!/bin/bash

python -u inference_pour.py --model MSBot \
    --seq_len 200 \
    --model_dir checkpoints/pour/MSBot/model_10_acc.pth \
    --inference_weight 120