#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理Parquet格式的时间序列数据集
将数据按时间顺序分割为训练集、验证集和测试集
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def process_parquet_to_splits(
    input_file_path, 
    output_dir,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    time_column='TimeStamp'
):
    """
    将Parquet文件按时间顺序分割为训练、验证、测试集
    
    Args:
        input_file_path (str): 输入Parquet文件路径
        output_dir (str): 输出目录路径
        train_ratio (float): 训练集比例，默认0.7
        val_ratio (float): 验证集比例，默认0.15
        test_ratio (float): 测试集比例，默认0.15
        time_column (str): 时间戳列名，默认'TimeStamp'
    """
    
    print("🔄 开始处理Parquet数据集...")
    
    # 检查比例是否合理
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"训练、验证、测试集比例之和必须为1.0，当前为{train_ratio + val_ratio + test_ratio}")
    
    # 读取数据
    print(f"📖 正在读取数据: {input_file_path}")
    df = pd.read_parquet(input_file_path)
    
    print(f"✅ 数据读取成功!")
    print(f"   - 数据形状: {df.shape}")
    print(f"   - 总样本数: {len(df):,}")
    print(f"   - 特征数: {len(df.columns)-1} (除去时间戳列)")
    
    # 检查时间戳列是否存在
    if time_column not in df.columns:
        raise ValueError(f"时间戳列 '{time_column}' 不存在于数据中")
    
    # 按时间戳排序
    print(f"🔄 按时间戳 '{time_column}' 排序...")
    df_sorted = df.sort_values(by=time_column).reset_index(drop=True)
    
    # 显示时间范围
    start_time = df_sorted[time_column].iloc[0]
    end_time = df_sorted[time_column].iloc[-1]
    print(f"   - 时间范围: {start_time} 到 {end_time}")
    
    # 计算分割点
    total_samples = len(df_sorted)
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    print(f"📊 数据分割信息:")
    print(f"   - 训练集: 0 到 {train_end-1} ({train_end:,} 样本, {train_ratio*100:.1f}%)")
    print(f"   - 验证集: {train_end} 到 {val_end-1} ({val_end-train_end:,} 样本, {val_ratio*100:.1f}%)")
    print(f"   - 测试集: {val_end} 到 {total_samples-1} ({total_samples-val_end:,} 样本, {test_ratio*100:.1f}%)")
    
    # 分割数据
    train_data = df_sorted.iloc[:train_end].copy()
    val_data = df_sorted.iloc[train_end:val_end].copy()
    test_data = df_sorted.iloc[val_end:].copy()
    
    # 移除时间戳列
    print(f"🗑️  移除时间戳列 '{time_column}'...")
    train_data = train_data.drop(columns=[time_column])
    val_data = val_data.drop(columns=[time_column])
    test_data = test_data.drop(columns=[time_column])
    
    # 显示异常标签统计
    print(f"📈 异常标签统计:")
    if 'anomaly_label' in train_data.columns:
        train_anomaly = train_data['anomaly_label'].sum()
        val_anomaly = val_data['anomaly_label'].sum()
        test_anomaly = test_data['anomaly_label'].sum()
        
        print(f"   - 训练集异常样本: {train_anomaly:,} / {len(train_data):,} ({train_anomaly/len(train_data)*100:.2f}%)")
        print(f"   - 验证集异常样本: {val_anomaly:,} / {len(val_data):,} ({val_anomaly/len(val_data)*100:.2f}%)")
        print(f"   - 测试集异常样本: {test_anomaly:,} / {len(test_data):,} ({test_anomaly/len(test_data)*100:.2f}%)")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存文件
    train_file = os.path.join(output_dir, 'train_processed.csv')
    val_file = os.path.join(output_dir, 'val_processed.csv')
    test_file = os.path.join(output_dir, 'test_processed.csv')
    
    print(f"💾 保存文件到 {output_dir}...")
    train_data.to_csv(train_file, index=False)
    val_data.to_csv(val_file, index=False)
    test_data.to_csv(test_file, index=False)
    
    print(f"✅ 处理完成!")
    print(f"   - 训练集: {train_file}")
    print(f"   - 验证集: {val_file}")
    print(f"   - 测试集: {test_file}")
    
    # 创建处理记录
    record_file = os.path.join(output_dir, 'processing_record.md')
    with open(record_file, 'w', encoding='utf-8') as f:
        f.write(f"# 数据处理记录\n\n")
        f.write(f"**处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**输入文件**: {input_file_path}\n\n")
        f.write(f"**原始数据**:\n")
        f.write(f"- 总样本数: {total_samples:,}\n")
        f.write(f"- 特征数: {len(df.columns)-1}\n")
        f.write(f"- 时间范围: {start_time} 到 {end_time}\n\n")
        f.write(f"**数据分割**:\n")
        f.write(f"- 训练集: {len(train_data):,} 样本 ({train_ratio*100:.1f}%)\n")
        f.write(f"- 验证集: {len(val_data):,} 样本 ({val_ratio*100:.1f}%)\n")
        f.write(f"- 测试集: {len(test_data):,} 样本 ({test_ratio*100:.1f}%)\n\n")
        if 'anomaly_label' in df.columns:
            f.write(f"**异常标签统计**:\n")
            f.write(f"- 训练集异常: {train_data['anomaly_label'].sum():,} / {len(train_data):,}\n")
            f.write(f"- 验证集异常: {val_data['anomaly_label'].sum():,} / {len(val_data):,}\n")
            f.write(f"- 测试集异常: {test_data['anomaly_label'].sum():,} / {len(test_data):,}\n")
    
    print(f"📝 处理记录保存到: {record_file}")
    
    return train_data, val_data, test_data

def main():
    """主函数"""
    # 设置文件路径
    input_file = 'dataset/ALLcontact_noSegment/ALL_noSement_Contacting_cleaned_1minut_20250802_170647'
    output_dir = 'dataset/ALLcontact_noSegment'
    
    try:
        # 处理数据
        train_data, val_data, test_data = process_parquet_to_splits(
            input_file_path=input_file,
            output_dir=output_dir,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        print("\n🎉 数据处理成功完成!")
        
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
