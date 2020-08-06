#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python3 ./second/pytorch/pointpillars_train.py train --config_path=./second/configs/pointpillars/my_train.config --model_dir=./model_save --pretrained_path=./best_pointpillars.pth
