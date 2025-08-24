# 使用自定义数据集运行DCdetector异常检测

## 概述
我已经为您创建了一个自定义的数据加载器，可以直接使用您的CSV格式数据集运行DCdetector异常检测模型。

## 数据集格式
您的数据集应该包含以下文件，位于 `dataset/` 目录下：
- `train_processed.csv` - 训练数据
- `val_processed.csv` - 验证数据  
- `test_processed.csv` - 测试数据

每个CSV文件的格式：
- 前27列：特征数据（rTotalActivePower, rTotalReactivePower, 等等）
- 最后一列：`anomaly_label` - 异常标签（0=正常，1=异常）

## 数据统计
根据测试结果，您的数据集包含：
- 训练集：376,939 个样本
- 验证集：80,633 个样本
- 测试集：80,579 个样本
- 特征维度：27

## 使用方法

### 方法1：使用脚本运行
```bash
# 激活虚拟环境
source venv/bin/activate

# 运行训练和测试
./scripts/run_custom.sh
```

### 方法2：直接运行Python文件
```bash
source venv/bin/activate
python run_custom_dataset.py \
    --dataset Custom \
    --win_size 100 \
    --input_c 27 \
    --output_c 27 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_epochs 3 \
    --mode train
```

### 方法3：修改现有的main.py
在 `main.py` 中设置以下参数：
```python
args.dataset = 'Custom'
args.input_c = 27  # 特征维度
args.output_c = 27
```

## 关键参数说明

### 数据相关参数
- `--dataset Custom`: 使用自定义数据集
- `--input_c 27`: 输入特征维度（根据您的数据）
- `--output_c 27`: 输出特征维度
- `--win_size 100`: 滑动窗口大小

### 训练参数
- `--batch_size 32`: 批次大小
- `--lr 1e-4`: 学习率
- `--num_epochs 3`: 训练轮数
- `--anormly_ratio 1.0`: 异常比例，用于阈值计算

## CustomSegLoader特性

我创建的 `CustomSegLoader` 类具有以下特性：

1. **自动数据预处理**：
   - 使用 StandardScaler 进行特征标准化
   - 自动处理 NaN 值
   - 基于训练数据拟合标准化器

2. **灵活的模式支持**：
   - `train`: 训练模式，返回训练数据和虚拟标签
   - `val`: 验证模式，返回验证数据和真实标签
   - `test`: 测试模式，返回测试数据和真实标签
   - `thre`: 阈值计算模式，用于异常检测阈值确定

3. **滑动窗口**：
   - 支持可配置的窗口大小和步长
   - 自动生成时间序列窗口

## 测试数据加载器
运行以下命令测试数据加载器是否正常工作：
```bash
source venv/bin/activate
python test_custom_dataloader.py
```

## 输出结果
模型训练完成后，会输出以下评估指标：
- Precision（精确度）
- Recall（召回率）
- F1-Score
- AUC
- 其他异常检测指标

## 注意事项
1. 确保数据集文件位于正确的路径（`dataset/`目录）
2. 确保CSV文件格式正确，最后一列为异常标签
3. 模型会自动处理数据标准化，无需手动预处理
4. 根据您的硬件配置调整 `batch_size` 参数

## 故障排除
如果遇到问题：
1. 检查数据文件路径是否正确
2. 确认CSV文件格式是否符合要求
3. 查看日志文件了解详细错误信息
4. 运行测试脚本验证数据加载器
