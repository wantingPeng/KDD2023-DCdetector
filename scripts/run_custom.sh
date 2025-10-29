#!/bin/bash
# 使用自定义数据集运行DCdetector的脚本

# 激活虚拟环境
source venv/bin/activate

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:."

# 仅禁用 numba/stumpy 的 CUDA 使用，避免导入时显存错误；不影响 PyTorch 使用 GPU
export NUMBA_DISABLE_CUDA=1

# 运行训练和测试
python run_custom_dataset.py \
    --dataset Custom \
    --data_path "dataset/downsampleData_scratch_1minut/pcb/pcb_cleaned_1minut_20250928_161509.parquet" \
    --win_size 30 \
    --input_c 31 \
    --output_c 31 \
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
    --activation gelu \
    --output_attention True \
    #--checkpoint_dir "checkpoints/Ring_cleaned_1minut_20250928_170147_checkpoint_2025-10-28_09-46-19"
