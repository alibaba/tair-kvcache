# Trace Converter - 统一的Trace格式转换工具

将各种trace格式转换为Optimizer标准的Get+Write格式或推理引擎的DialogTurn格式。

## ✨ 特性

### 自动发现 Converter

系统会自动扫描并发现所有可用的 converter：
- **内置 converter**: `converters/` 目录下的标准格式
- **自动发现**: 通过路径推断自动加载额外的 converter
- **用户扩展**: 支持通过参数添加自定义 converter

```bash
# 使用内置 converter
python3 trace_converter.py -i input.jsonl -o output.jsonl -f qwen_bailian

# 使用自定义 converter
python3 trace_converter.py -i input.jsonl -o output.jsonl -f custom \
    --converter-module /path/to/custom_converter.py
```

---

## 安装

```bash
cd kv_cache_manager/optimizer/tools/trace_converter

# 方式1: 直接使用 (推荐)
pip install -r requirements.txt

# 方式2: 安装为Python包 (可选)
pip install -e .
```

## 快速开始

**自动输出文件名**: 如果不指定`-o`参数,会自动在输入文件同目录生成输出文件:
- Optimizer模式: `<input_name>_optimizer.jsonl`
- Inference模式: `<input_name>_inference.jsonl`

### Publisher Log转换

**自动识别所有instance** (从日志的`source`字段提取)

```bash
# 最简单用法 (自动生成输出文件)
python trace_converter.py \
    -i /path/to/publisher.log \
    -f publisher_log \
    --mode optimizer
# → 输出: /path/to/publisher_optimizer.jsonl

# 指定输出文件
python trace_converter.py \
    -i /path/to/publisher.log \
    -o /path/to/optimizer_trace.jsonl \
    -f publisher_log \
    --mode optimizer \
    --instance-block-sizes "instance1:16,instance2:32"

# 输出示例:
# 📌 Discovered instance: instance1 (block_size=16)
# 📌 Discovered instance: instance2 (block_size=32)
# ✅ Discovered 2 instance(s):
#    - instance1: block_size=16, traces=150
#    - instance2: block_size=32, traces=200

# 使用默认block_size=16
python trace_converter.py \
    -i /path/to/publisher.log \
    -o /path/to/optimizer_trace.jsonl \
    -f publisher_log \
    --mode optimizer

# Inference模式 (DialogTurn格式)
python trace_converter.py \
    -i /path/to/publisher.log \
    -o /path/to/inference_trace.jsonl \
    -f publisher_log \
    --mode inference
```

### Qwen Bailian数据集转换

**单instance** (数据集本身没有instance概念,使用`--instance-id`指定)

```bash
# Optimizer模式 - 指定block_size
python trace_converter.py \
    -i /path/to/qwen_trace.jsonl \
    -o /path/to/optimizer_trace.jsonl \
    -f qwen_bailian \
    --mode optimizer \
    --instance-id my_instance \
    --instance-block-sizes "my_instance:32"

# 使用默认block_size=16
python trace_converter.py \
    -i /path/to/qwen_trace.jsonl \
    -o /path/to/optimizer_trace.jsonl \
    -f qwen_bailian \
    --instance-id my_instance
```

### 文本对话转换

**单instance** (文本trace本身没有instance概念,使用`--instance-id`指定)

```bash
# Optimizer模式 - 使用本地tokenizer
python trace_converter.py \
    -i /path/to/text_conversations.jsonl \
    -o /path/to/optimizer_trace.jsonl \
    -f text \
    --mode optimizer \
    --tokenizer-path /path/to/model/deepseek-v3 \
    --time-field time \
    --content-field prompt_messages \
    --instance-id my_text_instance \
    --instance-block-sizes "my_text_instance:32"

# 使用HuggingFace模型ID (自动下载)
python trace_converter.py \
    -i /path/to/text_conversations.jsonl \
    -o /path/to/optimizer_trace.jsonl \
    -f text \
    --mode optimizer \
    --tokenizer-path "Qwen/Qwen2.5-7B-Instruct" \
    --instance-id my_instance

# 优先使用本地缓存 (如果./model/deepseek-v3存在)
python trace_converter.py \
    -i /path/to/text_conversations.jsonl \
    -o /path/to/optimizer_trace.jsonl \
    -f text \
    --mode optimizer \
    --tokenizer-path "deepseek-ai/DeepSeek-V3" \
    --model-name deepseek-v3 \
    --instance-id my_instance

# 仅使用model-name (如果./model/qwen存在)
python trace_converter.py \
    -i /path/to/text_conversations.jsonl \
    -o /path/to/optimizer_trace.jsonl \
    -f text \
    --mode optimizer \
    --model-name qwen \
    --instance-id my_instance

# 使用默认block_size=16
python trace_converter.py \
    -i /path/to/text_conversations.jsonl \
    -o /path/to/optimizer_trace.jsonl \
    -f text \
    --tokenizer-path /path/to/model/deepseek-v3 \
    --instance-id my_text_instance
```

**Tokenizer获取方式**:

1. **本地tokenizer** (推荐,所有模型共享):
   ```bash
   --tokenizer-path /path/to/local/tokenizer  # 或 HF模型ID
   --model-name Qwen2.5-7B-Instruct          # 优先使用 ./model/{model_name}
   ```

2. **自动从HF拉取** (需要模型映射):
   - 默认映射: `GLM-4.6 → zai-org/GLM-4.6`
   - 自定义映射: `--model-mapping "Qwen2.5:Qwen/Qwen2.5-7B-Instruct,CustomModel:org/custom-model"`

**说明**:
- 如果指定了`--tokenizer-path`或`--model-name`,所有模型将共享同一个tokenizer


## 输入格式

### Publisher Log格式

JSON格式,每行一个事件:

```json
{
  "type": "GetCacheLocation",
  "source": "instance_01",
  "trigger_time_us": 1704110400000000,
  "keys": [123, 456, 789],
  "query_type": "prefix_match"
}
```

### Qwen Bailian格式

```json
{
  "timestamp": 1704110400.0,
  "hash_ids": [123, 456, 789],  // 只包含input部分的hash
  "input_length": 48,
  "output_length": 32            // 仅用于统计,无对应KV Cache
}
```

**注意**: `hash_ids`只包含input部分,`output_length`仅供推理引擎模拟器使用。

### 文本格式

```json
{
  "time": "2024-01-01 12:00:00.000000",
  "prompt_messages": "用户输入的文本"
}
```

或对话格式:

```json
{
  "time": "2024-01-01 12:00:00.000000",
  "prompt_messages": {
    "messages": [
      {"role": "user", "content": "你好"},
      {"role": "assistant", "content": "你好!"}
    ]
  }
}
```

## 输出格式

### Optimizer模式 (Get+Write)

**GetLocationSchemaTrace** (读操作):
```json
{
  "instance_id": "instance",
  "trace_id": "trace_instance_1704110400000000",
  "timestamp_us": 1704110400000000,
  "tokens": [],
  "keys": [123, 456, 789],
  "query_type": "prefix_match",
  "block_mask": [],
  "sw_size": 0,
  "location_spec_names": []
}
```

**WriteCacheSchemaTrace** (写操作):
```json
{
  "instance_id": "instance",
  "trace_id": "trace_instance_1704110400000001",
  "timestamp_us": 1704110400000001,
  "tokens": [],
  "keys": [123, 456, 789, 1011]
}
```

### Inference模式 (DialogTurn)

```json
{
  "instance_id": "instance",
  "trace_id": "trace_instance_1704110400000000",
  "timestamp_us": 1704110400000000,
  "tokens": [],
  "keys": [123, 456, 789],
  "query_type": "prefix_match",
  "block_mask": [],
  "sw_size": 0,
  "location_spec_names": [],
  "input_len": 48,
  "output_len": 32,
  "total_keys": [123, 456, 789, 1011]
}
```

## 命令行参数

### 通用参数

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `-i, --input` | 是 | 输入文件路径 | - |
| `-o, --output` | 否 | 输出文件路径 | 自动生成: `<input>_optimizer.jsonl` 或 `<input>_inference.jsonl` |
| `-f, --format` | 是 | 输入格式: `publisher_log`, `qwen_bailian`, `text` | - |
| `--mode` | 否 | 输出模式: `optimizer` (Get+Write) 或 `inference` (DialogTurn) | `optimizer` |
| `--instance-id` | 否 | 默认实例ID (仅当格式无instance信息时使用) | `instance` |
| `--instance-block-sizes` | 否 | 每个instance的block_size (格式: `inst1:16,inst2:32`) | 未指定使用默认16 |

### Tokenizer参数 (text格式需要)

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--tokenizer-path` | 否* | Tokenizer路径 (本地路径或HuggingFace模型ID) | - |
| `--model-name` | 否 | 模型名称 (如果指定且`./model/{model_name}`存在,优先使用) | - |
| `--model-mapping` | 否 | 模型名称映射表 (格式见下方) | `GLM-4.6:zai-org/GLM-4.6` |

**环境变量**:
- `HF_ENDPOINT`: HuggingFace镜像地址 (默认: `https://hf-mirror.com`)
  - 设置为空字符串可使用官方源: `export HF_ENDPOINT=""`
  - 自定义镜像: `export HF_ENDPOINT=https://your-mirror.com`

### 格式特定参数

| 参数 | 适用格式 | 说明 | 默认值 |
|------|---------|------|--------|
| `--time-field` | text | 时间字段名 | `time` |
| `--content-field` | text | 内容字段名 | `prompt_messages` |

### 使用说明

**Tokenizer参数** :
- `--tokenizer-path`: tokenizer路径 (本地路径或HuggingFace模型ID)
  - 本地路径: `model/deepseek-v3`, `/path/to/model`
  - HF模型ID: `deepseek-ai/DeepSeek-V3`, `Qwen/Qwen2.5-7B-Instruct`
  - 与`--model-name`二选一,或配合使用
- `--model-name`: 模型名称 (如果`./model/{model_name}`存在,优先使用)
  - 可单独使用 (如果本地已有tokenizer)
  - 或作为`--tokenizer-path`的本地缓存优先级

**加载逻辑** :
1. 如果指定`--model-name`且`./model/{model_name}`存在 → 使用本地缓存
2. 否则使用`--tokenizer-path` → 传给`AutoTokenizer.from_pretrained()`
3. `AutoTokenizer`自动判断是本地路径还是HF模型ID

**Instance参数**:
- `--instance-block-sizes`: 格式 `instance1:16,instance2:32,instance3:8`
- 未指定的instance使用默认值16
- `publisher_log`格式会自动识别所有instance
- `qwen_bailian`和`text`格式使用`--instance-id`参数

## 与Optimizer集成

转换后的trace文件可直接用于Optimizer:

```json
{
    "trace_file_path": "/path/to/optimizer_trace.jsonl",
    "output_result_path": "/path/to/output",
    "instance_groups": [
        {
            "group_name": "instance_group_01",
            "quota_capacity": 12000,
            "instances": [
                {
                    "instance_id": "instance",
                    "block_size": 16,
                    "eviction_policy_type": "lru"
                }
            ]
        }
    ]
}
```

```bash
bazel run //kv_cache_manager/optimizer:optimizer_main -- config.json
```

## 多Instance支持

### 自动识别 (Publisher Log)

Publisher Log格式会自动从日志的`source`字段识别所有instance,可以为每个instance指定不同的block_size:

```bash
# 为每个instance指定block_size
python trace_converter.py \
    -i publisher.log \
    -o output.jsonl \
    -f publisher_log \
    --instance-block-sizes "instance1:16,instance2:32,instance3:8"

# 输出:
# 📌 Discovered instance: instance1 (block_size=16)
# 📌 Discovered instance: instance2 (block_size=32)
# 📌 Discovered instance: instance3 (block_size=8)
# ✅ Discovered 3 instance(s):
#    - instance1: block_size=16, traces=150
#    - instance2: block_size=32, traces=200
#    - instance3: block_size=8, traces=100

# 使用默认block_size=16
python trace_converter.py \
    -i publisher.log \
    -o output.jsonl \
    -f publisher_log
```

生成的trace文件会包含所有instance的数据,每个trace的`instance_id`字段保留原始值。

### 单Instance (Qwen/Text)

Qwen Bailian和Text格式本身没有instance概念,使用`--instance-id`指定instance ID,使用`--instance-block-sizes`指定block_size:

```bash
# 指定block_size
python trace_converter.py \
    -i qwen.jsonl \
    -o output.jsonl \
    -f qwen_bailian \
    --instance-id my_instance \
    --instance-block-sizes "my_instance:32"

# 使用默认block_size=16
python trace_converter.py \
    -i qwen.jsonl \
    -o output.jsonl \
    -f qwen_bailian \
    --instance-id my_instance
```

---

## 前缀哈希算法

本工具使用前缀依赖的block ID生成算法,与C++ Optimizer内部的`hashInt64Func`保持一致:

```python
def hash_int64_func(prev_hash: int, current_value: int) -> int:
    combined = f"{prev_hash}_{current_value}"
    return hash(combined) & 0x7FFFFFFFFFFFFFFF

# 应用前缀哈希
hash_value = 0
for hash_id in hash_ids:
    hash_value = hash_int64_func(hash_value, hash_id)
    block_keys.append(hash_value)
```

## 合成时间戳策略

对于不能区分读写的trace (如Qwen Bailian, 文本对话),在Optimizer模式下:

```
原始时间戳: T

生成策略:
- GetLocationSchemaTrace.timestamp_us = T
- WriteCacheSchemaTrace.timestamp_us = T + 1 (微秒)

理由: 保证Get在Write之前 (prefill先于decode)
```

时间戳冲突自动处理:
- 如果T或T+1已被使用,自动递增直到找到可用时间戳
- 确保所有trace的时间戳唯一且有序

## 扩展：添加自定义 Converter

### 步骤1: 创建 Converter 类

创建一个继承 `BaseConverter` 的类：

```python
# my_converter.py
from converters.base import BaseConverter
import json

class MyConverter(BaseConverter):
    """自定义格式转换器"""
    
    def __init__(self, default_instance_id='instance',
                 instance_block_sizes=None, mode='optimizer', **kwargs):
        super().__init__(default_instance_id, instance_block_sizes, mode)
    
    def convert(self, input_file: str, output_file: str) -> int:
        """转换逻辑实现"""
        pass
```

**命名约定**：
- 类名以 `Converter` 结尾
- 系统自动推断 format 名称：`MyConverter` → `my`

### 步骤2: 使用自定义 Converter

```bash
# 方式A: 显式注册文件
python3 trace_converter.py -i input.jsonl -o output.jsonl -f my \
    --converter-module /path/to/my_converter.py

# 方式B: 扫描目录
python3 trace_converter.py -i input.jsonl -o output.jsonl -f my \
    --converter-dir /path/to/converters_dir

# 方式C: 放到 converters/ 目录（自动发现）
cp my_converter.py converters/
python3 trace_converter.py -i input.jsonl -o output.jsonl -f my
```

### BaseConverter 提供的辅助方法

```python
# 创建 Get trace
self._create_get_trace(
    timestamp_us=timestamp_us,
    keys=block_keys,
    instance_id=instance_id,
    tokens=token_ids  # 可选
)

# 创建 Write trace
self._create_write_trace(
    timestamp_us=timestamp_us,
    keys=block_keys,
    instance_id=instance_id,
    tokens=token_ids  # 可选
)

# 创建 DialogTurn trace
self._create_dialog_trace(
    timestamp_us=timestamp_us,
    keys=prefill_keys,
    input_len=input_token_count,
    output_len=output_token_count,
    total_keys=all_keys,
    instance_id=instance_id,
    tokens=token_ids  # 可选
)

# 获取 instance 的 block_size
block_size = self.get_block_size(instance_id)
```

### 完整示例

查看现有 converter 的实现：
- `converters/publisher_log.py` - 日志解析示例
- `converters/qwen_bailian.py` - 简单数据集转换
- `converters/text_trace.py` - 复杂的 tokenization 处理

---

## 相关文档

- [Optimizer README](../../README.md) - Optimizer主文档
- [BaseConverter API](converters/base.py) - 基类接口说明
