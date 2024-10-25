#!/bin/bash

python -u train_peg.py --model MSBot \
    --num_workers 16 \
    --seq_len 200 \
    --blur_p 0.25 \
    --beta 0.5 \
    --gamma 15 \
    --penalty_intensity 10.0