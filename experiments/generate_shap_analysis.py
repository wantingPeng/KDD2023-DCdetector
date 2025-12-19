"""
SHAP Analysis Script for DCdetector
生成DCdetector模型的SHAP可解释性分析图
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import json
import matplotlib.pyplot as plt
from datetime import datetime
import random
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from model.DCdetector import DCdetector
from data_factory.data_loader import get_loader_segment

# 尝试导入SHAP
try:
    import shap
except ImportError:
    print("SHAP库未安装，正在尝试安装...")
    os.system("pip install shap")
    import shap


class ModelWrapper(nn.Module):
    """
    包装DCdetector模型以便SHAP分析
    将模型输出转换为异常分数
    """
    def __init__(self, model, win_size, temperature=50):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.win_size = win_size
        self.temperature = temperature
        
    def my_kl_loss(self, p, q):
        """计算KL散度"""
        res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
        return torch.mean(torch.sum(res, dim=-1), dim=1)
    
    def forward(self, x):
        """
        前向传播，返回每个时间步的异常分数（per-timestep）
        输入: x [B, L, D] - batch_size, window_size, features
        输出: anomaly_score [B, L] - 每个时间步的异常分数
        
        注意：DCdetector返回series和prior（没有重构输出）
        """
        # 获取模型输出
        series, prior = self.model(x)
        
        # 计算association discrepancy
        # 保留.detach()以符合原始测试逻辑
        series_loss = 0.0
        prior_loss = 0.0
        
        for u in range(len(prior)):
            # 归一化prior
            prior_sum = torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)
            prior_normalized = prior[u] / (prior_sum + 1e-8)

            if u == 0:
                # 注意：移除.detach()以保持梯度流，这对SHAP分析至关重要
                # 原始训练代码使用detach是为了避免某些梯度循环，但SHAP分析需要完整的梯度
                series_loss = self.my_kl_loss(series[u], prior_normalized.detach()) * self.temperature
                prior_loss  = self.my_kl_loss(prior_normalized, series[u].detach()) * self.temperature
            else:
                series_loss += self.my_kl_loss(series[u], prior_normalized.detach()) * self.temperature
                prior_loss  += self.my_kl_loss(prior_normalized, series[u].detach()) * self.temperature
        
        # 计算每个时间步的异常分数
        # print(f"series_loss: {series_loss}")
        # print(f"prior_loss: {prior_loss}")
        #DCdetector的异常分数 = series_loss + prior_loss (loss越大，异常程度越高)
        # 注意：不使用softmax，因为loss数值太大会导致softmax输出极端化(0或1)，使得SHAP值失效
        anomaly_score =torch.softmax(( -series_loss - prior_loss ), dim=-1) # [B, L]
        # print(f"anomaly_score (loss): {anomaly_score}")
        # print(f"anomaly_score stats - min: {anomaly_score.min():.2f}, max: {anomaly_score.max():.2f}, mean: {anomaly_score.mean():.2f}")
        # # 返回每个时间步的异常分数，不做平均
        return anomaly_score  # [B, L]


def load_model_and_config(model_path, config_path):
    """加载模型和配置"""
    # 读取配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 创建模型（使用配置文件中的参数）
    model = DCdetector(
        win_size=config['win_size'],
        enc_in=config['input_c'],
        c_out=config['output_c'],
        n_heads=config.get('n_heads', 1),
        d_model=config.get('d_model', 256),
        e_layers=config.get('e_layers', 3),
        patch_size=config.get('patch_size', [3, 5, 7]),
        channel=config['input_c']
    )
    
    # 加载权重
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"模型加载成功: {model_path}")
    print(f"配置: win_size={config['win_size']}, input_c={config['input_c']}, output_c={config['output_c']}")
    print(f"  patch_size={config.get('patch_size', [3, 5, 7])}")
    
    return model, config, device


def prepare_background_data(data_loader, n_samples=100, seed=42):
    """
    准备背景数据集用于SHAP分析
    使用固定随机种子确保可重复性
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    background_samples = []
    count = 0
    
    for input_data, _ in data_loader:
        background_samples.append(input_data)
        count += input_data.shape[0]
        if count >= n_samples:
            break
    
    background_data = torch.cat(background_samples, dim=0)[:n_samples]
    # 确保数据需要梯度（SHAP需要）
    background_data = background_data.requires_grad_(True)
    print(f"背景数据集准备完成: {background_data.shape}")
    
    return background_data


def generate_shap_values(wrapped_model, background_data, test_data, device, batch_size=100):
    """
    生成SHAP值
    使用批处理以避免内存问题
    """
    print("\n开始计算SHAP值...")
    print(f"背景数据: {background_data.shape}")
    print(f"测试数据: {test_data.shape}")
    
    # 将数据移到设备上，并确保需要梯度（SHAP需要）
    background_data = background_data.to(device).float()
    test_data = test_data.to(device).float()
    
    # 创建SHAP解释器 (使用DeepExplainer，适用于深度学习模型)
    print("初始化SHAP解释器...")
    explainer = shap.DeepExplainer(wrapped_model, background_data)
    
    # 批处理计算SHAP值以避免内存问题
    # 注意：不能使用torch.no_grad()，因为SHAP需要梯度计算
    n_samples = test_data.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"分 {n_batches} 批计算SHAP值 (每批 {batch_size} 个样本)...")
    shap_values_list = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch_data = test_data[start_idx:end_idx]
        
        print(f"  处理批次 {i+1}/{n_batches} (样本 {start_idx}-{end_idx-1})...")
        
        # SHAP内部会处理梯度计算，我们不需要手动设置requires_grad
        # 禁用 additivity 检查，因为复杂模型（如 DCdetector）可能不完全满足 additivity
        batch_shap = explainer.shap_values(batch_data, check_additivity=False)
        
        # 处理SHAP返回值（可能是列表或数组）
        if isinstance(batch_shap, list):
            # 如果是列表，取第一个元素（通常是输出）
            batch_shap = batch_shap[0]
        
        # 确保是numpy数组
        if isinstance(batch_shap, torch.Tensor):
            batch_shap = batch_shap.detach().cpu().numpy()
        
        shap_values_list.append(batch_shap)
    
    # 合并所有批次的SHAP值
    shap_values = np.concatenate(shap_values_list, axis=0)
    
    print(f"SHAP值计算完成: {shap_values.shape}")
    
    return shap_values, explainer


def plot_shap_summary(shap_values, test_data, feature_names, save_dir):
    """
    绘制SHAP汇总图（保持时间×特征结构，不flatten）
    
    shap_values: [n_samples, n_outputs, n_features, win_size]
                 对于每个输出时间步，有对应的SHAP值
    """
    # SHAP返回的形状: [n_samples, n_outputs, n_features, win_size]
    # 其中 n_outputs = win_size （每个时间步一个输出）
    n_samples, n_outputs, n_features, win_size = shap_values.shape
    
    print(f"\nSHAP值形状: {shap_values.shape}")
    print(f"  n_samples={n_samples}, n_outputs={n_outputs}, n_features={n_features}, win_size={win_size}")
    
    # ========== 方法1: 对所有输出和时间聚合，得到每个特征的总贡献 ==========
    # 对于异常检测，使用最大值更合理：异常往往发生在特定时间点，使用最大值能捕捉关键时刻的影响
    # 先reshape以便处理：[n_samples, n_outputs, n_features, time] -> [n_samples, n_features, n_outputs*time]
    shap_reshaped = shap_values.transpose(0, 2, 1, 3).reshape(
        shap_values.shape[0], shap_values.shape[2], -1
    )
    
    # 对每个样本的每个特征，找到在所有output和time中绝对值最大的SHAP值（保留符号）
    max_abs_indices = np.argmax(np.abs(shap_reshaped), axis=2)  # [n_samples, n_features]
    shap_values_by_feature_signed = np.take_along_axis(
        shap_reshaped, 
        max_abs_indices[:, :, np.newaxis], 
        axis=2
    ).squeeze(-1)  # [n_samples, n_features] 保留符号用于beeswarm
    
    shap_values_by_feature = np.abs(shap_values_by_feature_signed)  # [n_samples, n_features] 用于重要性排序
    
    print(f"特征维度SHAP值: {shap_values_by_feature.shape}")
    
    # 准备测试数据用于beeswarm图
    # test_data形状: [n_samples, win_size, n_features]
    # 对时间维度取平均，得到每个特征的平均值
    if len(test_data.shape) == 3:
        test_data_by_feature = np.mean(test_data, axis=1)  # [n_samples, n_features]
    else:
        # 如果已经是2D，直接使用
        test_data_by_feature = test_data
    

    # 1b. SHAP Summary Plot (beeswarm) - Z-score标准化SHAP值
    # 对每个特征的SHAP值进行标准化，使得不同数量级的特征可以在同一尺度下比较
    # 使用Z-score: (x - mean) / std，这样每个特征的SHAP值都有mean=0, std=1
    shap_standardized = np.copy(shap_values_by_feature_signed)
    for i in range(shap_standardized.shape[1]):
        mean_val = np.mean(shap_standardized[:, i])
        std_val = np.std(shap_standardized[:, i])
        if std_val > 1e-8:  # 避免除以0
            shap_standardized[:, i] = (shap_standardized[:, i] - mean_val) / std_val
    
    plt.figure(figsize=(14, 10))
    shap.summary_plot(
        shap_standardized,  # 使用Z-score标准化的SHAP值
        test_data_by_feature,
        feature_names=feature_names,
        max_display=20,
        show=False
    )
    plt.title("SHAP Summary Beeswarm ", 
              fontsize=13, pad=20)
    plt.xlabel("SHAP value", fontsize=11)
    plt.tight_layout()
    save_path = os.path.join(save_dir, "shap_summary_beeswarm.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"保存SHAP汇总图: {save_path}")
    plt.close()
    

    # 3. 保存特征重要性数值
    mean_abs_shap = np.mean(shap_values_by_feature, axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    csv_path = os.path.join(save_dir, "feature_importance.csv")
    feature_importance_df.to_csv(csv_path, index=False)
    print(f"保存特征重要性数据: {csv_path}")
    
    print("\n特征重要性排名 (Top 10):")
    print(feature_importance_df.head(10).to_string(index=False))
    
    # 4. 绘制SHAP值热图 - 特征 × 时间
    plot_shap_heatmap(shap_values, feature_names, save_dir, top_k=15)


def plot_shap_heatmap(shap_values, feature_names, save_dir, top_k=15):
    """
    绘制SHAP值热图，显示特征在时间维度上的重要性分布
    
    参数:
        shap_values: [n_samples, n_outputs, n_features, win_size] - 原始SHAP值
        feature_names: list - 特征名称列表
        save_dir: str - 保存目录
        top_k: int - 显示前K个最重要的特征
    """
    print(f"\n生成SHAP热图...")
    
    n_samples, n_outputs, n_features, win_size = shap_values.shape
    
    # ========== 方法1: 对所有样本取平均，得到 [n_features, win_size] 的SHAP值 ==========
    # 对output维度取平均（因为n_outputs通常等于win_size，表示每个时间步的输出）
    # 对样本维度取平均
    shap_avg = np.mean(np.abs(shap_values), axis=(0, 1))  # [n_features, win_size]
    
    print(f"平均SHAP值形状: {shap_avg.shape}")
    
    # 计算每个特征的总重要性（用于排序）
    feature_importance = np.mean(shap_avg, axis=1)
    
    # 选择top_k个最重要的特征
    top_indices = np.argsort(feature_importance)[-top_k:][::-1]
    top_feature_names = [feature_names[i] for i in top_indices]
    shap_heatmap_data = shap_avg[top_indices, :]
    
    # 绘制热图
    plt.figure(figsize=(16, max(8, top_k * 0.4)))
    
    # 使用seaborn绘制热图（如果可用）
    # ========== 方法2: 归一化热图 - 每个特征按行归一化 ==========
    # 这样可以看出每个特征在哪个时间步最重要（相对于该特征自己）
    shap_normalized = np.copy(shap_heatmap_data)
    for i in range(shap_normalized.shape[0]):
        max_val = shap_normalized[i, :].max()
        if max_val > 1e-8:
            shap_normalized[i, :] = shap_normalized[i, :] / max_val
    
    plt.figure(figsize=(16, max(8, top_k * 0.4)))
    
    try:
        import seaborn as sns
        sns.heatmap(
            shap_normalized,
            xticklabels=[f't-{win_size-i-1}' if i < win_size else f't+{i-win_size+1}' 
                         for i in range(win_size)],
            yticklabels=top_feature_names,
            cmap='YlOrRd',  # 黄-橙-红配色
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'SHAP (0-1)'},
            linewidths=0.5,
            linecolor='gray'
        )
        plt.title(f"SHAP Feature-Time Heatmap ",
                  fontsize=14, pad=20)
    except ImportError:
        im = plt.imshow(shap_normalized, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        plt.colorbar(im, label='SHAP (0-1)')
        plt.yticks(range(top_k), top_feature_names)
        plt.xticks(range(0, win_size, max(1, win_size//10)), 
                   [f't-{win_size-i-1}' for i in range(0, win_size, max(1, win_size//10))])
        plt.title(f"SHAP Feature-Time Heatmap", fontsize=14, pad=20)
    
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "shap_feature_time_heatmap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"保存SHAP特征-时间热图: {save_path}")
    plt.close()
    
    # ========== 保存热图数据 ==========
    heatmap_df = pd.DataFrame(
        shap_heatmap_data,
        index=top_feature_names,
        columns=[f't-{win_size-i-1}' for i in range(win_size)]
    )
    csv_path = os.path.join(save_dir, "shap_heatmap_data.csv")
    heatmap_df.to_csv(csv_path)
    print(f"保存热图数据: {csv_path}")


def main():
    """主函数"""
    # ==================== 配置参数 ====================
    # TODO: 修改为你的模型和数据路径
    MODEL_PATH = "checkpoint_original/checkpoints_d_model_oiginalDatasets/ring_d_model256_20251122_100023/Ring_cleaned_1minut_20250928_170147_checkpoint_2025-11-22_10-06-09/model.pth"
    CONFIG_PATH = "checkpoint_original/checkpoints_d_model_oiginalDatasets/ring_d_model256_20251122_100023/Ring_cleaned_1minut_20250928_170147_checkpoint_2025-11-22_10-06-09/config.json"
    DATA_PATH = "dataset/downsampleData_scratch_1minut/ring/Ring_cleaned_1minut_20250928_170147.parquet"
   
    # MODEL_PATH = "checkpoint_original/checkpoints_e_layers_oiginalDatasets/contact_e_layers7_20251121_181504/contact_cleaned_1minut_20250928_172122_checkpoint_2025-11-21_18-25-11/model.pth"
    # CONFIG_PATH = "checkpoint_original/checkpoints_e_layers_oiginalDatasets/contact_e_layers7_20251121_181504/contact_cleaned_1minut_20250928_172122_checkpoint_2025-11-21_18-25-11/config.json"
    # DATA_PATH = "dataset/downsampleData_scratch_1minut/contact/contact_cleaned_1minut_20250928_172122.parquet"
    
    # MODEL_PATH = "checkpoint_original/checkpoints_patch_size_oiginalDatasets/pcb_patch_size15_20251122_180712/pcb_cleaned_1minut_20250928_161509_checkpoint_2025-11-22_18-16-23/model.pth"
    # CONFIG_PATH = "checkpoint_original/checkpoints_patch_size_oiginalDatasets/pcb_patch_size15_20251122_180712/pcb_cleaned_1minut_20250928_161509_checkpoint_2025-11-22_18-16-23/config.json"
    # DATA_PATH = "dataset/downsampleData_scratch_1minut/pcb/pcb_cleaned_1minut_20250928_161509.parquet"

    INDEX = None  # DCdetector 需要的 index 参数
    
    # SHAP参数
    N_BACKGROUND_SAMPLES = 200  # 背景数据集样本数
    N_TEST_SAMPLES = 50  # 用于计算SHAP值的测试样本数
    
    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"experiments/shap_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}\n")
    
    # 设置随机种子确保可重复性
    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    # ==================== 1. 加载模型 ====================
    model, config, device = load_model_and_config(MODEL_PATH, CONFIG_PATH)
    
    # 创建包装模型
    wrapped_model = ModelWrapper(
        model=model,
        win_size=config['win_size']
    )
    wrapped_model.to(device)
    wrapped_model.eval()
    
    # ==================== 2. 加载数据 ====================
    print("\n加载数据...")
    
    # 加载训练数据作为背景数据
    train_loader = get_loader_segment(
        INDEX,
        DATA_PATH,
        batch_size=32,
        win_size=config['win_size'],
        mode='train',
        dataset='Custom'
    )
    
    # 加载测试数据
    test_loader = get_loader_segment(
        INDEX,
        DATA_PATH,
        batch_size=32,
        win_size=config['win_size'],
        mode='test',
        dataset='Custom'
    )
    
    # 准备背景数据
    background_data = prepare_background_data(train_loader, n_samples=N_BACKGROUND_SAMPLES, seed=RANDOM_SEED)
    
    # 准备测试数据 - 使用分层采样确保包含异常样本
    print("\n准备测试数据（分层采样）...")
    all_test_samples = []
    all_test_labels = []
    
    # 首先收集所有测试数据
    for input_data, labels in test_loader:
        all_test_samples.append(input_data)
        all_test_labels.append(labels)
    
    all_test_data = torch.cat(all_test_samples, dim=0)
    all_test_labels = torch.cat(all_test_labels, dim=0)
    
    # 分离正常和异常样本
    # 检查每个窗口是否包含异常（至少有一个时间点是异常）
    anomaly_mask = (all_test_labels.sum(dim=1) > 27)  # [N] bool tensor
    normal_indices = torch.where(~anomaly_mask)[0]
    anomaly_indices = torch.where(anomaly_mask)[0]
    
    print(f"测试集总样本数: {len(all_test_data)}")
    print(f"  正常样本数: {len(normal_indices)} ({100*len(normal_indices)/len(all_test_data):.1f}%)")
    print(f"  异常样本数: {len(anomaly_indices)} ({100*len(anomaly_indices)/len(all_test_data):.1f}%)")
    
    # 分层采样：确保包含一定比例的异常样本
    n_anomaly_samples = min(int(N_TEST_SAMPLES * 0.5), len(anomaly_indices))  # 至少30%异常样本
    n_normal_samples = N_TEST_SAMPLES - n_anomaly_samples
    
    # 如果异常样本不够，调整比例
    if n_anomaly_samples > len(anomaly_indices):
        n_anomaly_samples = len(anomaly_indices)
        n_normal_samples = N_TEST_SAMPLES - n_anomaly_samples
    
    # 随机选择样本（使用固定种子）
    torch.manual_seed(RANDOM_SEED)
    selected_anomaly_idx = anomaly_indices[torch.randperm(len(anomaly_indices))[:n_anomaly_samples]]
    selected_normal_idx = normal_indices[torch.randperm(len(normal_indices))[:n_normal_samples]]
    
    # 合并并打乱
    selected_indices = torch.cat([selected_anomaly_idx, selected_normal_idx])
    selected_indices = selected_indices[torch.randperm(len(selected_indices))]
    
    # 提取选中的样本
    test_data = all_test_data[selected_indices]
    test_labels_array = all_test_labels[selected_indices]
    
    # 确保测试数据需要梯度（SHAP需要）
    test_data = test_data.requires_grad_(True)
    
    print(f"\n最终测试数据: {test_data.shape}")
    print(f"  包含异常样本: {n_anomaly_samples} ({100*n_anomaly_samples/N_TEST_SAMPLES:.1f}%)")
    print(f"  包含正常样本: {n_normal_samples} ({100*n_normal_samples/N_TEST_SAMPLES:.1f}%)")
    
    # 获取特征名称
    df = pd.read_parquet(DATA_PATH)
    feature_cols = [col for col in df.columns if col not in ['TimeStamp', 'anomaly_label']]
    print(f"\n特征数量: {len(feature_cols)}")
    print(f"特征列表: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"特征列表: {feature_cols}")
    
    # ==================== 3. 计算SHAP值 ====================
    # 使用较小的批处理大小以避免内存问题
    shap_batch_size = min(50, N_TEST_SAMPLES)  # 每批最多50个样本
    shap_values, explainer = generate_shap_values(
        wrapped_model,
        background_data,
        test_data,
        device,
        batch_size=shap_batch_size
    )
    
    # 处理SHAP值形状
    # ModelWrapper返回 [B, L]，SHAP会对每个输出单独计算SHAP值
    # 实际返回形状: [n_samples, n_outputs, n_features, win_size]
    # 其中 n_outputs = win_size（每个时间步一个输出）
    print(f"SHAP值原始形状: {shap_values.shape}")
    
    # SHAP对于多输出模型，返回: [n_samples, n_outputs, ...input_shape]
    # 我们的情况：输入 [B, L, D]，输出 [B, L]
    # SHAP返回: [n_samples, n_outputs, n_features, win_size]
    # 表示：第i个输出对第j个特征的每个时间步的SHAP值
    
    if len(shap_values.shape) != 4:
        raise ValueError(f"期望SHAP值形状为4D，但得到: {shap_values.shape}")
    
    n_samples, n_outputs, dim1, dim2 = shap_values.shape
    # 根据实际形状判断：应该是 [n_samples, n_outputs, n_features, win_size]
    # 其中 n_features=27, win_size=30
    if dim1 == len(feature_cols):
        # dim1是特征数
        print(f"SHAP值形状: [n_samples={n_samples}, n_outputs={n_outputs}, n_features={dim1}, win_size={dim2}]")
    else:
        # 可能是 [n_samples, n_outputs, win_size, n_features]
        print(f"SHAP值形状: [n_samples={n_samples}, n_outputs={n_outputs}, dim1={dim1}, dim2={dim2}]")
        print(f"  注意：需要根据实际维度调整代码")
    
    # 转换测试数据为numpy（需要先detach，因为数据设置了requires_grad）
    if isinstance(test_data, torch.Tensor):
        test_data_np = test_data.detach().cpu().numpy()
    else:
        test_data_np = test_data
    
    # ==================== 4. 生成可视化 ====================
    print("\n生成SHAP可视化图...")
    
    # 新的可视化函数已经包含多种分析方法
    plot_shap_summary(shap_values, test_data_np, feature_cols, output_dir)
    
    # ==================== 5. 保存SHAP值 ====================
    shap_save_path = os.path.join(output_dir, "shap_values.npy")
    np.save(shap_save_path, shap_values)
    print(f"\n保存SHAP值: {shap_save_path}")
    
    # 保存配置信息
    info = {
        "model_path": MODEL_PATH,
        "data_path": DATA_PATH,
        "n_background_samples": N_BACKGROUND_SAMPLES,
        "n_test_samples": N_TEST_SAMPLES,
        "win_size": config['win_size'],
        "input_c": config['input_c'],
        "output_c": config['output_c'],
        "patch_size": config.get('patch_size', [3, 5, 7]),
        "feature_names": feature_cols,
        "timestamp": timestamp
    }
    
    info_path = os.path.join(output_dir, "analysis_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"保存分析信息: {info_path}")
    
    print(f"\n✅ SHAP分析完成！")
    print(f"所有结果保存在: {output_dir}")
    print("\n生成的文件:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")


if __name__ == "__main__":
    main()

