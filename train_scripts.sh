#!/bin/bash

# 确保使用 conda 的 shell hook
source /opt/conda/etc/profile.d/conda.sh
conda activate SiT

# 进入项目目录
cd /inspire/hdd/project/embodied-multimodality/gongjingjing-25039/ytchen/SiT

# 设置 HF_HOME, HF_LEROBOT_HOME, TORCH_HOME 环境变量
export WANDB_KEY="3317e96ffed42e6f81ba5dcbb29c17be5d4e7d13"
export ENTITY="yitongchen719-ustc"
export PROJECT="SiT"
export HF_HUB_OFFLINE=1
export WANDB_MODE="offline"
export HF_HOME=./data

torchrun --nnodes=1 --nproc_per_node=4 train.py --model SiT-XL/2 --data-path /inspire/hdd/global_public/public_datas/imagenet/ILSVRC/Data/CLS-LOC/train/ --wandb --num-workers=64 --global-batch-size=256