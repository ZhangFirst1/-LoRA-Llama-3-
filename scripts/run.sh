#!/bin/bash

# 创建输出目录
mkdir -p results
mkdir -p model

echo "================================================="
echo "   Start Training: Llama-3 + LoRA on AdGen       "
echo "================================================="

# 设置显卡ID
export CUDA_VISIBLE_DEVICES=0

# 运行训练脚本
python src/train.py \
    --epoch 5 \
    --lr 2e-4 \
    --batch_size 2

echo "Training finished. Starting Evaluation..."

# 运行评估脚本
python src/evaluate.py

echo "Pipeline Completed!"