#!/bin/bash

python -u inference_peg.py --model MSBot \
    --seq_len 200 \
    --model_dir checkpoints/peg/MSBot/model_10_acc.pth