"""
重新可视化SHAP值 - 使用归一化以解决数量级差异问题
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import json
import os

# 读取最新的SHAP分析结果
shap_dir = "shap_analysis_20251216_132051"

# 加载SHAP值
shap_values = np.load(os.path.join(shap_dir, "shap_values.npy"))
print(f"SHAP values shape: {shap_values.shape}")

# 加载分析信息
with open(os.path.join(shap_dir, "analysis_info.json"), 'r') as f:
    info = json.load(f)

feature_names = info['feature_names']
print(f"Features: {len(feature_names)}")

# 处理SHAP值 - 与generate_shap_analysis.py中相同的逻辑
n_samples, n_outputs, n_features, win_size = shap_values.shape

# 对所有输出和时间聚合，使用最大值
shap_reshaped = shap_values.transpose(0, 2, 1, 3).reshape(
    shap_values.shape[0], shap_values.shape[2], -1
)

max_abs_indices = np.argmax(np.abs(shap_reshaped), axis=2)
shap_values_by_feature_signed = np.take_along_axis(
    shap_reshaped, 
    max_abs_indices[:, :, np.newaxis], 
    axis=2
).squeeze(-1)

print(f"SHAP by feature shape: {shap_values_by_feature_signed.shape}")
print(f"SHAP value range: [{shap_values_by_feature_signed.min():.2e}, {shap_values_by_feature_signed.max():.2e}]")

# 加载测试数据（从分析信息中获取）
# 注意：这里我们使用虚拟数据，因为原始测试数据没有保存
# 实际上beeswarm图主要看SHAP值分布，测试数据只用于颜色编码
test_data_dummy = np.random.randn(n_samples, n_features)

# ========== 方法1: 标准化归一化（Z-score） ==========
print("\n生成标准化归一化的SHAP图...")
shap_standardized = np.copy(shap_values_by_feature_signed)
for i in range(shap_standardized.shape[1]):
    std = np.std(shap_standardized[:, i])
    mean = np.mean(shap_standardized[:, i])
    if std > 1e-8:
        shap_standardized[:, i] = (shap_standardized[:, i] - mean) / std

plt.figure(figsize=(14, 10))
shap.summary_plot(
    shap_standardized,
    test_data_dummy,
    feature_names=feature_names,
    max_display=20,
    show=False
)
plt.title("SHAP Summary - Standardized (Z-score)\nEach feature's SHAP values standardized to mean=0, std=1", 
          fontsize=12, pad=15)
plt.tight_layout()
save_path = os.path.join(shap_dir, "shap_beeswarm_standardized.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"保存标准化SHAP图: {save_path}")
plt.close()

# ========== 方法2: MinMax归一化到[-1, 1] ==========
print("\n生成MinMax归一化的SHAP图...")
shap_minmax = np.copy(shap_values_by_feature_signed)
for i in range(shap_minmax.shape[1]):
    min_val = shap_minmax[:, i].min()
    max_val = shap_minmax[:, i].max()
    range_val = max_val - min_val
    if range_val > 1e-8:
        shap_minmax[:, i] = 2 * (shap_minmax[:, i] - min_val) / range_val - 1

plt.figure(figsize=(14, 10))
shap.summary_plot(
    shap_minmax,
    test_data_dummy,
    feature_names=feature_names,
    max_display=20,
    show=False
)
plt.title("SHAP Summary - MinMax Normalized to [-1, 1]\nEach feature's SHAP values scaled independently", 
          fontsize=12, pad=15)
plt.tight_layout()
save_path = os.path.join(shap_dir, "shap_beeswarm_minmax.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"保存MinMax归一化SHAP图: {save_path}")
plt.close()

# ========== 方法3: 仅除以各自的标准差（保留符号和相对大小）==========
print("\n生成除以标准差的SHAP图...")
shap_div_std = np.copy(shap_values_by_feature_signed)
for i in range(shap_div_std.shape[1]):
    std = np.std(np.abs(shap_div_std[:, i]))
    if std > 1e-8:
        shap_div_std[:, i] = shap_div_std[:, i] / std

plt.figure(figsize=(14, 10))
shap.summary_plot(
    shap_div_std,
    test_data_dummy,
    feature_names=feature_names,
    max_display=20,
    show=False
)
plt.title("SHAP Summary - Divided by Feature Std\nEach feature's SHAP values divided by its own standard deviation", 
          fontsize=12, pad=15)
plt.tight_layout()
save_path = os.path.join(shap_dir, "shap_beeswarm_div_std.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"保存除以标准差SHAP图: {save_path}")
plt.close()

# ========== 输出统计信息 ==========
print("\n" + "="*60)
print("SHAP值统计分析")
print("="*60)

# 计算每个特征的统计信息
feature_stats = []
for i, feat_name in enumerate(feature_names):
    shap_feat = shap_values_by_feature_signed[:, i]
    feature_stats.append({
        'feature': feat_name,
        'mean_abs': np.mean(np.abs(shap_feat)),
        'std': np.std(shap_feat),
        'min': shap_feat.min(),
        'max': shap_feat.max(),
        'range': shap_feat.max() - shap_feat.min()
    })

df_stats = pd.DataFrame(feature_stats).sort_values('mean_abs', ascending=False)
print("\n特征SHAP值统计 (Top 15):")
print(df_stats.head(15).to_string(index=False))

# 保存统计信息
stats_path = os.path.join(shap_dir, "shap_statistics_detailed.csv")
df_stats.to_csv(stats_path, index=False)
print(f"\n详细统计信息已保存: {stats_path}")

print("\n完成！现在查看生成的三个归一化图像：")
print(f"  1. {shap_dir}/shap_beeswarm_standardized.png (Z-score标准化)")
print(f"  2. {shap_dir}/shap_beeswarm_minmax.png (MinMax归一化)")
print(f"  3. {shap_dir}/shap_beeswarm_div_std.png (除以标准差)")
