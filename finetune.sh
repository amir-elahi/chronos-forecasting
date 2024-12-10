#!/bin/bash

torchrun \
  --nproc-per-node=1 \
  scripts/training/train.py \
  --config ./scripts/training/configs/chronos-t5-tiny.yaml \
  --model-id amazon/chronos-t5-tiny \
  --no-random-init \
  --learning-rate 0.001 \
  --max-steps 400000