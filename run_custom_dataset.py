#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用自定义数据集运行DCdetector异常检测模型的示例脚本
"""
import argparse
import os
from solver import Solver
# 使用Python标准logging替代自定义logger
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    # 设置参数
    parser = argparse.ArgumentParser(description='DCdetector with Custom Dataset')
    
    # 数据集相关参数
    parser.add_argument('--dataset', type=str, default='Custom', 
                       help='数据集名称，使用Custom加载您的数据')
    parser.add_argument('--data_path', type=str, default='', 
                       help='数据集路径，留空使用默认dataset/目录')
    
    # 模型参数
    parser.add_argument('--win_size', type=int, default=100, 
                       help='滑动窗口大小')
    parser.add_argument('--input_c', type=int, default=27, 
                       help='输入特征维度，根据您的数据集调整')
    parser.add_argument('--output_c', type=int, default=27, 
                       help='输出特征维度，通常与input_c相同')
    
    # 训练参数  
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, 
                       help='学习率')
    parser.add_argument('--num_epochs', type=int, default=3, 
                       help='训练轮数')
    parser.add_argument('--patience', type=int, default=3, 
                       help='早停耐心值')
    parser.add_argument('--anormly_ratio', type=float, default=1.0, 
                       help='异常比例，用于阈值计算')
    
    # 模型结构参数
    parser.add_argument('--n_heads', type=int, default=1, 
                       help='注意力头数')
    parser.add_argument('--d_model', type=int, default=256, 
                       help='模型维度')
    parser.add_argument('--e_layers', type=int, default=3, 
                       help='编码器层数')
    def parse_list(arg):
        try:
            if arg.startswith('[') and arg.endswith(']'):
                # 移除方括号并按逗号分割
                return [int(item.strip()) for item in arg[1:-1].split(',')]
            else:
                # 尝试按逗号分割
                return [int(item.strip()) for item in arg.split(',')]
        except:
            return [3, 4, 5]  # 默认值
    
    parser.add_argument('--patch_size', type=parse_list, default=[3,4,5], 
                       help='补丁大小列表，格式为[3,4,5]或3,4,5')
    parser.add_argument('--d_ff', type=int, default=512, 
                       help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.0, 
                       help='dropout率')
    parser.add_argument('--activation', type=str, default='gelu', 
                       help='激活函数')
    parser.add_argument('--output_attention', type=bool, default=True, 
                       help='是否输出注意力权重')
    parser.add_argument('--loss_fuc', type=str, default='MSE', 
                       choices=['MSE', 'MAE'], help='损失函数')
    
    # 运行参数
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'test'], help='运行模式')
    parser.add_argument('--index', type=int, default=0, 
                       help='数据集索引')
    parser.add_argument('--model_save_path', type=str, default='checkpoints/', 
                       help='模型保存路径')
    parser.add_argument('--checkpoint_dir', type=str, default='',
                       help='仅测试模式下使用：已保存模型的目录路径，目录内应包含model.pth/config.json')
    
    args = parser.parse_args()
    
    logger.info("开始使用自定义数据集运行DCdetector...")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"输入特征维度: {args.input_c}")
    logger.info(f"窗口大小: {args.win_size}")
    logger.info(f"批次大小: {args.batch_size}")
    
    # 创建Solver实例
    solver = Solver(vars(args))
    
    if args.mode == 'train':
        logger.info("开始训练模型...")
        solver.train()
        logger.info("训练完成!")
        
        logger.info("开始测试模型...")
        solver.test()
        logger.info("测试完成!")
    else:
        logger.info("开始测试模型...")
        # 若提供了checkpoint目录，则优先使用
        if getattr(args, 'checkpoint_dir', ''):
            solver.latest_checkpoint_dir = args.checkpoint_dir
        solver.test()
        logger.info("测试完成!")

if __name__ == '__main__':
    main()
