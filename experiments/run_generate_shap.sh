#!/bin/bash
# 运行SHAP分析脚本
# 使用方法: bash run_generate_shap.sh

cd /home/wanting/KDD2023-DCdetector
source venv/bin/activate
cd experiments
python generate_shap_analysis.py
