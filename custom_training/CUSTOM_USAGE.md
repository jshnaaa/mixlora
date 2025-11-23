# MixLoRA 文化数据集自定义训练指南

本指南详细说明了如何使用定制的MixLoRA训练脚本在文化数据集上进行训练和评估。

## 🎯 主要特性

### ✅ 完全满足您的需求
- ✅ 支持LLaMA3.1-8B-Instruct模型
- ✅ 支持Qwen2等多种模型架构
- ✅ 数据集ID参数化配置（DATA_ID=2/3）
- ✅ 自动8:1:1数据集分割（随机打乱）
- ✅ 保存训练参数权重（不保存完整模型）
- ✅ 基于验证集准确率的最佳模型保存
- ✅ 自动测试集评估
- ✅ 智能答案提取（从生成文本中提取阿拉伯数字）
- ✅ 完整评估指标（准确率、精确率、召回率、F1）
- ✅ 生成答案保存（包含原始问题/正确答案/预测答案/是否正确）

## 📁 文件结构

```
custom_training/
├── train_mixlora_custom.py      # 定制训练脚本（主要）
├── inference_custom.py          # 定制推理脚本
├── run_custom_training.sh       # 训练启动脚本
├── run_custom_inference.sh      # 推理启动脚本
├── dataset.py                   # 数据集处理模块
└── CUSTOM_USAGE.md             # 本说明文档
```

## 🚀 快速开始

### 1. 训练模型

```bash
# 使用默认参数训练（llama + culturalbench）
./run_custom_training.sh

# 训练不同组合
./run_custom_training.sh llama 2    # LLaMA + culturalbench
./run_custom_training.sh qwen 2     # Qwen + culturalbench
./run_custom_training.sh llama 3    # LLaMA + normad
./run_custom_training.sh qwen 3     # Qwen + normad
```

### 2. 推理评估

```bash
# 自动找到最新训练的模型并评估（自动检测backbone）
./run_custom_inference.sh --adapter_path auto --dataset_path /path/to/external_test.json

# 自动找到最新的qwen模型并评估
./run_custom_inference.sh --adapter_path auto --backbone qwen --dataset_path /path/to/test.json

# 指定特定模型进行评估
./run_custom_inference.sh --adapter_path /root/autodl-fs/data/mixlora/culturalbench_qwen_20241122_1430/best_model --dataset_path /path/to/test.json

# 交互式推理（自动检测最新模型）
./run_custom_inference.sh --adapter_path auto --interactive

# 交互式推理（指定backbone）
./run_custom_inference.sh --adapter_path auto --backbone qwen --interactive
```

## 📊 数据集配置

### 支持的数据集
| DATA_ID | 数据集名称 | 路径 | DATASET_TAG |
|---------|------------|------|-------------|
| 2 (默认) | CulturalBench | `/root/autodl-fs/CulturalBench_merge_gen.json` | `culturalbench` |
| 3 | NorMaD | `/root/autodl-fs/normad_merge_gen.json` | `normad` |

### 数据格式要求
```json
{
    "instruction": "### Question: Give me the answer from 1 to 4: ...\n### Answer: ",
    "instruction_mask": "### Question: Give me the answer from 1 to 4: ...\n### Answer: ",
    "input": "",
    "output": "1",
    "label": "1"
}
```

## 🔧 训练配置

### 支持的模型架构
| BACKBONE | 模型路径 | 支持状态 |
|----------|----------|----------|
| llama | `/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct` | ✅ 完全支持 |
| qwen | `/root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Qwen-2.5-7B-Instruct` | ✅ 完全支持 |

### 模型参数
```bash
BACKBONE="llama"       # 模型架构：llama 或 qwen
NUM_EXPERTS=8          # MoE专家数量
TOP_K=2               # 路由选择的专家数量
LORA_R=8              # LoRA秩
LORA_ALPHA=16         # LoRA alpha参数
```

### 训练参数
```bash
BATCH_SIZE=4                    # 批次大小
GRADIENT_ACCUMULATION_STEPS=4   # 梯度累积步数
LEARNING_RATE=1e-4             # 学习率
NUM_EPOCHS=3                   # 训练轮数
EVAL_INTERVAL=1                # 每轮都进行验证评估
```

## 📈 训练流程

### 自动化训练流程
1. **数据加载**: 根据DATA_ID自动加载对应数据集
2. **数据分割**: 随机打乱后按8:1:1分割训练/验证/测试集
3. **模型初始化**: 加载基础模型并注入MixLoRA适配器
4. **训练循环**:
   - 每个epoch后在验证集上评估
   - 保存验证集准确率最高的模型
   - 记录所有验证结果
5. **最终评估**: 使用最佳模型在测试集上评估
6. **结果保存**: 保存所有评估结果和生成答案

### 输出目录结构
```
/root/autodl-fs/data/mixlora/${DATASET_TAG}_${BACKBONE}_YYYYMMDD_HHMM/
├── best_model/
│   ├── adapter_config.json     # 适配器配置
│   └── adapter_model.bin       # 适配器权重
├── training_config.json        # 训练配置
├── validation_results.json     # 验证集结果（每轮）
├── generated_answers.json      # 验证集生成答案
└── test_results.json          # 测试集最终结果
```

**示例目录名称**：
- `culturalbench_llama_20241122_1430/` - LLaMA在CulturalBench上的训练
- `culturalbench_qwen_20241122_1430/` - Qwen在CulturalBench上的训练
- `normad_llama_20241122_1430/` - LLaMA在NorMaD上的训练

## 🎯 评估指标

### 生成答案格式（generated_answers.json）
```json
{
    "metrics": {
        "accuracy": 0.8532,
        "precision": 0.8421,
        "recall": 0.8398,
        "f1": 0.8409,
        "total_samples": 1000,
        "valid_predictions": 987
    },
    "predictions": [
        {
            "instruction": "### Question: ...",  // 原始问题
            "target": "2",                       // 正确答案
            "predicted": "2",                    // 预测答案
            "correct": true,                     // 是否正确
            "generated_text": "2"               // 生成的原始文本
        }
    ]
}
```

### 答案提取逻辑
系统会从模型生成的文本中智能提取阿拉伯数字：
1. 首先尝试完全匹配
2. 查找文本开头的选择
3. 使用正则表达式提取所有数字
4. 选择第一个有效的选择数字
5. 如果没找到有效答案，标记为错误

## 🔄 推理模式

### 1. 外部数据集评估
```bash
# 自动检测backbone
./run_custom_inference.sh \
    --adapter_path /path/to/best_model \
    --dataset_path /path/to/external_test.json

# 明确指定backbone
./run_custom_inference.sh \
    --adapter_path /path/to/best_model \
    --backbone qwen \
    --dataset_path /path/to/external_test.json
```

### 2. 交互式推理
```bash
# 自动找到最新模型（任意backbone）
./run_custom_inference.sh --adapter_path auto --interactive

# 自动找到最新的qwen模型
./run_custom_inference.sh --adapter_path auto --backbone qwen --interactive
```

### 3. 智能模型检测
- **自动路径检测**: 使用 `--adapter_path auto` 自动找到最新训练的模型
- **智能backbone检测**: 从路径中自动识别模型架构（llama/qwen）
- **灵活筛选**: 可以指定backbone来筛选特定架构的模型

### 4. 多模型管理
```bash
# 列出所有训练的模型
ls -la /root/autodl-fs/data/mixlora/

# 示例输出：
# culturalbench_llama_20241122_1430/
# culturalbench_qwen_20241122_1435/
# normad_llama_20241122_1440/
# normad_qwen_20241122_1445/

# 自动选择最新的llama模型
./run_custom_inference.sh --adapter_path auto --backbone llama --interactive

# 自动选择最新的qwen模型
./run_custom_inference.sh --adapter_path auto --backbone qwen --interactive
```

## 🎛️ 高级配置

### 修改训练参数
编辑 `run_custom_training.sh` 文件中的参数：

```bash
# 增加专家数量
NUM_EXPERTS=16

# 调整学习率
LEARNING_RATE=5e-5

# 更频繁的评估
EVAL_INTERVAL=1  # 每轮评估一次

# 增加训练轮数
NUM_EPOCHS=5
```

### 修改数据集
在 `train_mixlora_custom.py` 中添加新的数据集配置：

```python
configs = {
    2: {
        "path": "/root/autodl-fs/CulturalBench_merge_gen.json",
        "tag": "culturalbench"
    },
    3: {
        "path": "/root/autodl-fs/normad_merge_gen.json",
        "tag": "normad"
    },
    4: {  # 新数据集
        "path": "/path/to/new_dataset.json",
        "tag": "newdataset"
    }
}
```

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批次大小
   BATCH_SIZE=2
   # 增加梯度累积
   GRADIENT_ACCUMULATION_STEPS=8
   ```

2. **模型加载失败**
   ```bash
   # 检查基础模型路径
   ls -la /root/autodl-tmp/CultureMoE/Culture_Alignment/Meta-Llama-3.1-8B-Instruct
   ```

3. **数据集路径错误**
   ```bash
   # 检查数据集文件
   ls -la /root/autodl-fs/CulturalBench_merge_gen.json
   ```

4. **权限问题**
   ```bash
   chmod +x run_custom_training.sh
   chmod +x run_custom_inference.sh
   ```

### 日志查看
训练过程中的详细日志会显示：
- 数据集加载信息
- 模型配置信息
- 训练进度
- 验证结果
- 最佳模型保存信息

## 📝 使用示例

### 完整训练和评估流程
```bash
# 1. 训练LLaMA模型（culturalbench数据集）
./run_custom_training.sh llama 2

# 2. 训练Qwen模型（同一数据集）
./run_custom_training.sh qwen 2

# 3. 查看训练结果
ls -la /root/autodl-fs/data/mixlora/culturalbench_*/

# 4. 比较不同模型在外部测试集上的表现
./run_custom_inference.sh --adapter_path auto --backbone llama --dataset_path /path/to/test.json --output_file llama_results.json
./run_custom_inference.sh --adapter_path auto --backbone qwen --dataset_path /path/to/test.json --output_file qwen_results.json

# 5. 查看和比较评估结果
echo "LLaMA Results:" && cat llama_results.json | grep '"accuracy"'
echo "Qwen Results:" && cat qwen_results.json | grep '"accuracy"'
```

### 训练normad数据集
```bash
# 训练LLaMA在normad数据集上
./run_custom_training.sh llama 3

# 训练Qwen在normad数据集上
./run_custom_training.sh qwen 3

# 查看结果
ls -la /root/autodl-fs/data/mixlora/normad_*/
```

### 模型架构对比实验
```bash
# 1. 在同一数据集上训练不同架构
./run_custom_training.sh llama 2  # LLaMA + CulturalBench
./run_custom_training.sh qwen 2   # Qwen + CulturalBench

# 2. 在不同数据集上训练同一架构
./run_custom_training.sh llama 2  # LLaMA + CulturalBench
./run_custom_training.sh llama 3  # LLaMA + NorMaD

# 3. 全矩阵实验
./run_custom_training.sh llama 2  # LLaMA + CulturalBench
./run_custom_training.sh llama 3  # LLaMA + NorMaD
./run_custom_training.sh qwen 2   # Qwen + CulturalBench
./run_custom_training.sh qwen 3   # Qwen + NorMaD
```

这套定制的训练系统完全满足您的所有需求，提供了完整的训练、评估和推理流程！