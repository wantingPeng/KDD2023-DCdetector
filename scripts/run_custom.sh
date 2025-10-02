#!/bin/bash
# 使用自定义数据集运行DCdetector的脚本

# 激活虚拟环境
source venv/bin/activate

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:."

# 运行训练和测试
python run_custom_dataset.py \
    --dataset Custom \
    --data_path "dataset/pca/pc_features_20_contact.parquet" \
    --win_size 60 \
    --input_c 20 \
    --output_c 20 \
    --batch_size 64 \
    --lr 1e-4 \
    --num_epochs 1 \
    --patience 3 \
    --anormly_ratio 2 \
    --mode train \
    --index 0 \
    --patch_size 3,6,10 \
    --n_heads 8 \
    --d_model 256 \
    --e_layers 3 \
    --d_ff 512 \
    --dropout 0.0 \
    --activation gelu \
    --output_attention True

