"""
专门用于绘制 patch_size 参数分析结果的脚本
处理分类型参数（如 "3", "3,6", "3,6,10"）的横坐标显示问题
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Plot styling (与 parameter_analysis.py 保持一致)
PLOT_CONFIG = {
    'colors': {'contact': 'red', 'ring': 'green', 'pcb': 'blue'},
    'markers': {'contact': 'o', 'ring': 's', 'pcb': '^'},
    'labels': {'contact': 'Contact', 'ring': 'Ring', 'pcb': 'PCB'}
}

PLOT_STYLE = 'seaborn-v0_8-darkgrid'


def plot_patch_size_results(results_path, output_dir=None):
    """
    绘制 patch_size 参数分析结果
    
    Args:
        results_path: JSON 结果文件路径
        output_dir: 输出目录（如果为 None，则使用结果文件所在目录）
    """
    # 加载结果
    print(f"正在加载结果: {results_path}")
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    if output_dir is None:
        output_dir = os.path.dirname(results_path)
    
    # 设置绘图样式
    plt.figure(figsize=(12, 6))
    plt.style.use(PLOT_STYLE)
    
    colors = PLOT_CONFIG['colors']
    markers = PLOT_CONFIG['markers']
    labels = PLOT_CONFIG['labels']
    
    # 定义 patch_size 的顺序（从简单到复杂）
    # 这样可以保证横坐标是有意义的顺序，而不是字母序
    patch_size_order = []
    all_patch_sizes = set()
    
    # 收集所有的 patch_size 值
    for dataset, dataset_results in results.items():
        for patch_size in dataset_results.keys():
            all_patch_sizes.add(patch_size)
    
    # 按照 patch_size 的复杂度排序（逗号数量）
    patch_size_order = sorted(all_patch_sizes, 
                              key=lambda x: (len(x.split(',')), x))
    
    print(f"Patch size 顺序: {patch_size_order}")
    
    # 为每个 patch_size 分配一个数值位置（用于绘图）
    x_positions = {ps: i for i, ps in enumerate(patch_size_order)}
    
    # 绘制每个数据集的结果
    for dataset, dataset_results in results.items():
        if not dataset_results:
            continue
        
        # 提取数据
        x_vals = []
        f1_scores = []
        
        for patch_size in patch_size_order:
            if patch_size in dataset_results:
                x_vals.append(x_positions[patch_size])
                f1_scores.append(dataset_results[patch_size]['f_score'] * 100)
        
        if not x_vals:
            continue
        
        # 绘制线条和标记
        plt.plot(x_vals, f1_scores, 
                marker=markers[dataset], 
                color=colors[dataset], 
                linewidth=2.5, 
                markersize=10,
                label=labels[dataset],
                alpha=0.9)
        
        # 添加数值标注
        for x_val, f1_score, patch_size in zip(x_vals, f1_scores, 
                                                [ps for ps in patch_size_order if ps in dataset_results]):
            plt.annotate(f'{f1_score:.2f}', 
                        xy=(x_val, f1_score),
                        xytext=(0, 8), 
                        textcoords='offset points',
                        fontsize=9,
                        color=colors[dataset],
                        weight='bold',
                        ha='center')
    
    # 设置横坐标标签
    plt.xticks(range(len(patch_size_order)), patch_size_order, fontsize=11)
    
    # 自定义图表
    plt.xlabel('Patch Sizes (Multi-Scale Configuration)', fontsize=13, weight='bold')
    plt.ylabel('F1 Score (%)', fontsize=13, weight='bold')
    plt.title('F1 Score vs Patch Size Configuration for Different Datasets', 
             fontsize=14, weight='bold', pad=15)
    
    # 图例
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # 网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Y轴范围（自适应）
    all_f1_scores = []
    for dataset_results in results.values():
        for metrics in dataset_results.values():
            all_f1_scores.append(metrics['f_score'] * 100)
    
    if all_f1_scores:
        y_min = max(0, min(all_f1_scores) - 5)
        y_max = min(100, max(all_f1_scores) + 5)
        plt.ylim(y_min, y_max)
    
    # X轴范围（留出边距）
    plt.xlim(-0.5, len(patch_size_order) - 0.5)
    
    # 保存图表
    os.makedirs(output_dir, exist_ok=True)
    
    png_path = os.path.join(output_dir, 'patch_size_f1_scores.png')
    pdf_path = os.path.join(output_dir, 'patch_size_f1_scores.pdf')
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    print(f"\n{'='*80}")
    print(f"图表已保存:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")
    print(f"{'='*80}\n")
    
    plt.show()
    
    # 打印详细结果表格
    print_results_table(results, patch_size_order)


def print_results_table(results, patch_size_order):
    """
    打印结果表格
    
    Args:
        results: 结果字典
        patch_size_order: patch_size 的顺序
    """
    print("\n" + "="*80)
    print("详细结果表格")
    print("="*80)
    
    # 表头
    header = f"{'Patch Size':<20} | {'Contact F1':<12} | {'Ring F1':<12} | {'PCB F1':<12}"
    print(header)
    print("-" * len(header))
    
    # 每一行
    for patch_size in patch_size_order:
        row = f"{patch_size:<20} | "
        
        for dataset in ['contact', 'ring', 'pcb']:
            if patch_size in results.get(dataset, {}):
                f1_score = results[dataset][patch_size]['f_score'] * 100
                row += f"{f1_score:>10.2f}% | "
            else:
                row += f"{'N/A':>10} | "
        
        print(row)
    
    print("="*80 + "\n")
    
    # 找出每个数据集的最佳配置
    print("最佳配置:")
    print("-" * 40)
    for dataset in ['contact', 'ring', 'pcb']:
        if dataset in results and results[dataset]:
            best_patch_size = max(results[dataset].items(), 
                                 key=lambda x: x[1]['f_score'])
            print(f"  {dataset.capitalize():>8}: {best_patch_size[0]:<15} "
                  f"(F1 = {best_patch_size[1]['f_score']*100:.2f}%)")
    print("="*80 + "\n")


if __name__ == "__main__":
    # 结果文件路径
    results_path = 'experiments/results_pca/patch_size_results.json'
    
    # 输出目录
    output_dir = '/home/wanting/KDD2023-DCdetector/experiments/results_pca'
    
    print("="*80)
    print("Patch Size 参数分析结果可视化")
    print("="*80)
    print(f"结果文件: {results_path}")
    print(f"输出目录: {output_dir}")
    print("="*80 + "\n")
    
    # 生成图表
    plot_patch_size_results(results_path, output_dir)

