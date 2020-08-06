#!/usr/bin/env bash
export DISPLAY=':0.0'
CUDA_VISIBLE_DEVICES=7 python3 ./second/pytorch/train.py train --config_path=./second/configs/pointpillars/my_train.config --model_dir=./model_save --pretrained_path=./old_best_pointpillars.pth
