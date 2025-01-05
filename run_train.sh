#!/bin/bash

# 启用调试模式和错误即刻退出
set -xe

# 设置 Python 无缓冲模式
export PYTHONUNBUFFERED=1

# 移除或注释掉不支持的 CUDA 内存分配配置
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 初始化 conda，确保路径正确
source /home/lpya/miniconda3/etc/profile.d/conda.sh

# 激活 'vit' 环境
conda activate vit

# 检查是否成功激活环境
echo "当前 conda 环境: $(conda info --envs | grep '*' | awk '{print $1}')"

# 设置使用的 CUDA 设备（所有 8 个 GPU）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 设置 OpenMP 线程数，避免系统过载
export OMP_NUM_THREADS=1

# 设置分布式训练的环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=8
export NPROC_PER_NODE=8

echo "开始使用 torchrun 进行分布式训练..."

# 使用 torchrun 进行分布式训练，并将输出和错误日志保存到 train.log
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         src/train.py \
         --data-dir /home/lpya/projects/datasets \
         --batch-size 128 \
         --epochs 100 \
         --lr 3e-4 \
         --weight-decay 5e-2 \
         --accumulation-steps 4 \
         --checkpoint-dir ./checkpoints > train.log 2>&1

echo "训练脚本已完成。"
