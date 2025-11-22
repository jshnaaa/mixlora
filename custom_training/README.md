# MixLoRA 自定义训练 - 选择题数据集

本目录包含了在选择题格式数据集上训练 MixLoRA 模型的完整解决方案。

## 文件结构

```
custom_training/
├── dataset.py              # 数据集处理模块
├── train_mixlora.py        # 训练脚本
├── inference.py            # 推理和评估脚本
├── run_training.sh         # 训练启动脚本
├── run_inference.sh        # 推理启动脚本
├── configs/
│   └── train_config_example.json  # 配置文件示例
└── README.md              # 本说明文件
```

## 数据格式

您的数据集应该是 JSON 格式，每条数据包含以下字段：

```json
{
    "instruction": "### Question: Give me the answer from 1 to 4: In the Netherlands, which of the following is an unusual common public practice? 1. Tipping generously 2. Using deodorant 3. Talking loudly on the phone. 4. Cycle everywhere. This question is for a country or language that is Netherlands. You can only choose one option.\n ### Answer: ",
    "instruction_mask": "### Question: Give me the answer from 1 to 4: In the [MASK], which of the following is an unusual common public practice? 1. Tipping generously 2. Using deodorant 3. Talking loudly on the phone. 4. Cycle everywhere. This question is for a country or language that is [MASK]. You can only choose one option.\n ### Answer: ",
    "input": "",
    "output": "1",
    "label": "1"
}
```

**重要说明：**
- `instruction` 字段包含完整的问题和提示
- `input` 字段通常为空，可以忽略
- `output` 字段是正确答案（用于训练标签）
- `label` 字段不用于训练，可以忽略
- 模型将学习根据 `instruction` + `input` 预测 `output`

## 快速开始

### 1. 准备数据

确保您的数据集符合上述格式，保存为 JSON 文件（支持 `.json` 和 `.jsonl` 格式）。

### 2. 配置训练参数

编辑 `run_training.sh` 文件，修改以下参数：

```bash
BASE_MODEL="meta-llama/Llama-2-7b-hf"  # 基础模型
DATASET_PATH="path/to/your/train_dataset.json"  # 训练数据集路径
VALIDATION_DATASET_PATH="path/to/your/val_dataset.json"  # 验证数据集路径（可选）
OUTPUT_DIR="./mixlora_choice_model"  # 输出目录
```

### 3. 开始训练

```bash
# 给脚本执行权限
chmod +x run_training.sh

# 开始训练
./run_training.sh
```

### 4. 运行推理

训练完成后，可以进行推理和评估：

```bash
# 给脚本执行权限
chmod +x run_inference.sh

# 交互式推理
./run_inference.sh interactive

# 在测试集上评估
./run_inference.sh eval
```

## 详细使用说明

### 训练参数说明

#### MixLoRA 参数
- `num_experts`: 专家数量（默认：8）
- `top_k`: 路由选择的专家数量（默认：2）
- `routing_strategy`: 路由策略（默认："mixlora"）
- `router_aux_loss_coef`: 路由辅助损失系数（默认：0.01）
- `router_init_range`: 路由器初始化范围（默认：0.02）
- `jitter_noise`: 抖动噪声（默认：0.0）

#### LoRA 参数
- `lora_r`: LoRA 秩（默认：8）
- `lora_alpha`: LoRA alpha 参数（默认：16）
- `lora_dropout`: LoRA dropout（默认：0.05）
- `use_dora`: 是否使用 DoRA（默认：False）
- `use_rslora`: 是否使用 RSLoRA（默认：False）

#### 训练参数
- `max_length`: 最大序列长度（默认：512）
- `batch_size`: 批次大小（默认：4）
- `gradient_accumulation_steps`: 梯度累积步数（默认：4）
- `learning_rate`: 学习率（默认：1e-4）
- `num_epochs`: 训练轮数（默认：3）
- `warmup_ratio`: 预热比例（默认：0.1）
- `weight_decay`: 权重衰减（默认：0.01）

### 直接使用 Python 脚本

如果您不想使用 shell 脚本，可以直接调用 Python 脚本：

#### 训练

```bash
python train_mixlora.py \
    --base_model meta-llama/Llama-2-7b-hf \
    --dataset_path path/to/your/dataset.json \
    --output_dir ./mixlora_output \
    --num_experts 8 \
    --top_k 2 \
    --lora_r 8 \
    --lora_alpha 16 \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4
```

#### 推理

```bash
# 交互式推理
python inference.py \
    --model_path ./mixlora_output \
    --interactive

# 评估
python inference.py \
    --model_path ./mixlora_output \
    --dataset_path path/to/test_dataset.json \
    --output_file predictions.json
```

## 支持的模型架构

当前支持以下模型架构：
- LLaMA/LLaMA2
- Mistral
- Gemma/Gemma2
- Qwen2
- Phi
- Phi3

## 评估指标

评估脚本会计算以下指标：
- **准确率 (Accuracy)**: 整体预测准确率
- **宏平均精确率 (Macro Precision)**: 各选择项精确率的平均值
- **宏平均召回率 (Macro Recall)**: 各选择项召回率的平均值
- **宏平均 F1 分数 (Macro F1)**: 各选择项 F1 分数的平均值
- **每个选择项的详细指标**: 精确率、召回率、F1 分数、支持样本数

## 注意事项

1. **GPU 内存**: MixLoRA 训练需要足够的 GPU 内存。如果遇到内存不足，可以：
   - 减少 `batch_size`
   - 增加 `gradient_accumulation_steps`
   - 减少 `max_length`
   - 减少 `num_experts`

2. **选择范围自动检测**: 如果没有指定 `choice_range`，系统会自动从数据集中检测有效的选择范围。

3. **数据验证**: 系统会自动验证数据格式，移除不符合选择范围的无效样本。

4. **断点续训**: 训练脚本支持从检查点继续训练。

5. **实验追踪**: 支持 Weights & Biases 进行实验追踪，设置 `wandb_project` 参数即可启用。

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   - 减少批次大小或序列长度
   - 使用梯度检查点（在训练参数中添加 `gradient_checkpointing=True`）

2. **模型加载失败**
   - 检查基础模型路径是否正确
   - 确保有足够的磁盘空间下载模型

3. **数据格式错误**
   - 检查 JSON 格式是否正确
   - 确保所有必需字段都存在

4. **推理结果不理想**
   - 调整生成参数（temperature、max_new_tokens）
   - 检查训练数据质量
   - 增加训练轮数或调整学习率

## 联系和支持

如有问题，请查看原始 MixLoRA 项目的文档或提交 issue。