# Pythia70mModel_dropout 使用说明

## 概述

我已经成功为您的项目添加了一个新的 `Pythia70mModel_dropout` 模型类，它是原有 `Pythia70mModel` 的带 dropout 版本。这个新模型完全独立于原有模型，不会影响现有功能。

## 新增功能

### 1. 新模型类：`Pythia70mModel_dropout`

- **位置**：`model/DNN.py`
- **功能**：带 dropout 的 Pythia-70m 模型包装器
- **特性**：
  - 支持多层 dropout（隐藏层、注意力层、分类器层）
  - 动态调整 dropout 率
  - 完全兼容现有的训练和推理流程

### 2. 模型选择：`pythia70m_dropout`

- **使用方法**：在参数中设置 `net_name='pythia70m_dropout'`
- **Dropout 配置**：通过 `dropout_rate` 参数设置（默认 0.1）

### 3. 参数支持

- **新增参数**：`--dropout_rate`，默认值 0.1
- **参数说明**：控制 dropout 率，范围 0.0-1.0

## 使用方法

### 方法一：直接使用模型类

```python
from model.DNN import Pythia70mModel_dropout

# 创建带 dropout 的模型
model = Pythia70mModel_dropout(
    num_classes=5,
    max_length=128,
    dropout_rate=0.3  # 设置 30% 的 dropout 率
)

# 动态调整 dropout 率
model.set_dropout_rate(0.5)
```

### 方法二：通过 DNN 类使用

```python
from model.DNN import DNN

args = {
    'dataset_name': 'sst5',
    'net_name': 'pythia70m_dropout',  # 使用新的 dropout 模型
    'dropout_rate': 0.3,               # 设置 dropout 率
    'device': 'cuda',
    'lr': 2e-5,
    'num_epochs': 10,
    'optim': 'Adam',
    'is_dp_defense': False,
    'max_norm': 0
}

dnn = DNN(args)
```

### 方法三：命令行使用

```bash
python main.py \
    --dataset_name sst5 \
    --net_name pythia70m_dropout \
    --dropout_rate 0.3 \
    --lr 2e-5 \
    --num_epochs 10
```

## 技术细节

### Dropout 配置

该模型支持三层 dropout 配置：

1. **hidden_dropout**：隐藏层 dropout
2. **attention_dropout**：注意力机制 dropout  
3. **classifier_dropout**：分类器 dropout
4. **额外输出层 dropout**：在最终输出上应用

### 模型架构差异

- **原始模型** (`pythia70m`)：无 dropout，适用于推理和标准训练
- **Dropout 模型** (`pythia70m_dropout`)：带 dropout，适用于需要正则化的训练

### 参数对应关系

由于 Pythia 模型基于 GPTNeoX 架构，使用以下 dropout 参数：

- `hidden_dropout`：隐藏层 dropout（而非 `hidden_dropout_prob`）
- `attention_dropout`：注意力 dropout（而非 `attention_probs_dropout_prob`）
- `classifier_dropout`：分类器 dropout

## 推荐 Dropout 率

根据不同任务推荐以下 dropout 率：

- **文本分类**：0.1 - 0.3
- **防止过拟合**：0.2 - 0.5
- **训练初期**：0.3 - 0.5（较高 dropout）
- **训练后期**：0.1 - 0.2（较低 dropout）

## 测试

运行测试脚本验证功能：

```bash
python test_pythia_dropout_model.py
```

## 注意事项

1. **参数命名**：确保使用正确的 GPTNeoX dropout 参数名称
2. **模型大小**：Dropout 模型与原模型大小相同，只是配置不同
3. **训练/评估模式**：训练时启用 dropout，评估时自动禁用
4. **兼容性**：完全兼容现有的数据加载和训练流程

## 故障排除

如果遇到参数不匹配错误，请确保：

1. 使用正确的模型名称：`pythia70m_dropout`
2. 设置合理的 dropout 率：0.0-1.0 范围内
3. 检查 transformers 库版本兼容性

## 总结

现在您有两个 Pythia-70m 模型选项：

- `pythia70m`：原始模型，无 dropout
- `pythia70m_dropout`：新模型，带 dropout 功能

两个模型可以并行使用，互不影响，为您提供更多的实验选择。