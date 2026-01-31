# Trace Anonymous Tool - 使用指南

## 概述

`trace_anonymous_tool` 将通用文本 trace 转换为 Optimizer 标准的 **DialogTurnSchemaTrace JSON** 格式，可直接导入 Optimizer 使用。

## 工作流程

```
文本 trace (JSONL)
    ↓
[tokenizer.py] Token 化
    ↓
tokenids_*.jsonl
    ↓
[optimizer_schema_anonymizer.py] 匿名化 + 格式转换
    ↓
optimizer_trace_*.jsonl (DialogTurnSchemaTrace JSON)
    ↓
Optimizer 直接使用
```

---

## 工具说明

### 1. tokenizer.py - Token 化工具

将文本 trace 转换为 token IDs。

**功能**：
- 支持灵活的输入格式（自定义字段名）
- 支持多种 tokenizer（deepseek-v3、Qwen3-Coder 等）
- 支持工具调用（tool calls）场景
- 自动清洗和验证消息格式

**输入要求**：
- JSONL 格式（每行一个 JSON 对象）
- 包含时间戳字段（格式：`YYYY-MM-DD HH:MM:SS.ffffff`）
- 包含文本内容字段

**输出**：`result/not_yet_anonymous_files/tokenids_*.jsonl`

### 2. optimizer_schema_anonymizer.py - 匿名化和格式转换工具

将 token IDs 转换为符合 Optimizer 标准的 DialogTurnSchemaTrace 格式。

**功能**：
- 将 token IDs 转换为 block IDs（带前缀依赖）
- 输出符合 DialogTurnSchemaTrace 标准的 JSON 格式
- 支持自定义 instance_id
- 支持多种 block_mask 模式

**输入**：`tokenids_*.jsonl`

**输出**：`result/anonymous_files/optimizer_trace_*.jsonl`

---

## 快速开始

### 最简示例

假设你的 trace 文件格式为：
```jsonl
{"time": "2024-01-01 12:00:00.000000", "text": "用户输入内容"}
```

**处理步骤**：

```bash
# Step 1: Token 化
python tokenizer.py \
    --file_path your_trace.jsonl \
    --time_field time \
    --content_field text

# Step 2: 匿名化并转换为 Optimizer 标准格式
python optimizer_schema_anonymizer.py \
    --file_path result/not_yet_anonymous_files/tokenids_your_trace.jsonl \
    --block_size 16 \
    --instance-id instance

# 输出: result/anonymous_files/optimizer_trace_your_trace.jsonl
```

---

## 输入格式

### tokenizer.py 支持的格式

#### 格式 1: 纯文本
```jsonl
{"time": "2024-01-01 12:00:00.000000", "content": "这是一段文本"}
```

#### 格式 2: 对话消息
```jsonl
{"timestamp": "2024-01-01 12:00:00.000000", "messages": [{"role": "user", "content": "你好"}]}
```

#### 格式 3: 带工具的对话
```jsonl
{
    "time": "2024-01-01 12:00:00.000000",
    "conversation": {
        "messages": [{"role": "user", "content": "查询天气"}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {...}}}]
    }
}
```

**字段名可自定义**，通过命令行参数指定：
- `--time_field` 指定时间戳字段名
- `--content_field` 指定内容字段名

---

## 输出格式

### DialogTurnSchemaTrace JSON

```json
{
    "instance_id": "instance",
    "trace_id": "trace_instance_1704110400000000",
    "timestamp_us": 1704110400000000,
    "tokens": [],
    "keys": [1234567890, 2345678901, 3456789012],
    "query_type": "prefix_match",
    "block_mask": [],
    "sw_size": 0,
    "location_spec_names": [],
    "input_len": 48,
    "output_len": 0,
    "total_keys": [1234567890, 2345678901, 3456789012]
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `instance_id` | string | 实例标识符 |
| `trace_id` | string | Trace 唯一 ID（格式：`trace_{instance_id}_{timestamp_us}`） |
| `timestamp_us` | int | 微秒时间戳 |
| `tokens` | array | Token IDs（当前为空数组以减小文件大小） |
| `keys` | array[int] | Block keys（包含前缀依赖的匿名化 block IDs） |
| `query_type` | string | 查询类型（默认 `"prefix_match"`） |
| `block_mask` | array\|int | Block mask：`[]`（空数组）、`0`（整数）或 `[true, false, ...]`（布尔数组） |
| `sw_size` | int | 滑动窗口大小（默认 0） |
| `location_spec_names` | array | Location spec 名称（默认空数组） |
| `input_len` | int | 输入 token 数量（= `len(keys) × block_size`） |
| `output_len` | int | 输出 token 数量（chat template 场景下为 0） |
| `total_keys` | array[int] | 总的 输入输出keys（这里等于 keys，表示只有 prefill） |

---

## 命令行参数

### tokenizer.py

```bash
python tokenizer.py \
    --file_path <input_file>              # 必需：输入文件路径
    --time_field <field_name>             # 可选：时间戳字段名（默认: time）
    --content_field <field_name>          # 可选：内容字段名（默认: prompt_messages）
    --model_name <model_name>             # 可选：模型名称（deepseek-v3 等）
    --tokenizer_path <path>               # 可选：tokenizer 路径
```

**支持的 tokenizer**：
- 任何 HuggingFace transformers 兼容的 tokenizer

### optimizer_schema_anonymizer.py

```bash
python optimizer_schema_anonymizer.py \
    --file_path <input_file>              # 必需：输入文件路径（tokenids_*.jsonl）
    --block_size <size>                   # 可选：Block 大小（默认: 16）,这里的block需要与 optimizer 配置中同一个instance的block size对应
    --truncate                            # 可选：是否截断不完整的 block
    --instance-id <id>                    # 可选：实例 ID（默认: instance）
```


---

## 核心设计

### 1. 前缀依赖算法

每个 block 的 key 都依赖于前一个 block，确保前缀匹配的正确性：

```python
key_tuple = (prev_hash, tuple(block_tokens))  # 包含前缀依赖
block_hash = hashlib.md5(str(key_tuple).encode()).hexdigest()
block_id = get_or_create_id(block_hash)
```

这与 Optimizer 中的 `ApplyPrefixHash` 逻辑完全一致。

### 2. input_len 和 output_len

- **`input_len`**：用户输入的 token 数量
  ```python
  input_len = len(keys) × block_size
  ```

- **`output_len`**：模型输出的 token 数量
  - 在 chat template 场景下固定为 `0`
  - 原因：处理的是用户的**输入请求**，此时模型还没有生成输出
  - 表示"只有 prefill，没有 decode"

### 3. 与 Optimizer 的兼容性

输出格式完全符合 `optimizer_schema_trace.h` 中定义的 `DialogTurnSchemaTrace` 结构：
- ✅ 可直接作为 trace 文件被 Optimizer 读取和处理

---

## 常见问题

### Q1: 如何处理不同的时间戳格式？

A: tokenizer.py 支持标准格式 `YYYY-MM-DD HH:MM:SS.ffffff`。如果你的格式不同，需要先预处理或修改 `extract_timestamp` 函数。

### Q2: 如何处理多个实例？

A: 对不同实例的 trace 分别运行 anonymizer.py，使用不同的 `--instance-id`：

```bash
python optimizer_schema_anonymizer.py --file_path trace1.jsonl --instance-id instance1
python optimizer_schema_anonymizer.py --file_path trace2.jsonl --instance-id instance2
```

---

## 完整示例

### 示例

**输入文件** (`user_queries.jsonl`):
```jsonl
{"ts": "2024-01-01 10:00:00.000000", "query": "如何学习Python"}
{"ts": "2024-01-01 10:01:30.000000", "query": "推荐一些Python书籍"}
```

**处理命令**:
```bash
# Step 1: Token 化
python tokenizer.py \
    --file_path user_queries.jsonl \
    --time_field ts \
    --content_field query \
    --model_name deepseek-v3

# Step 2: 匿名化
python optimizer_schema_anonymizer.py \
    --file_path result/not_yet_anonymous_files/tokenids_user_queries.jsonl \
    --block_size 16 \
    --instance-id my_instance
```

**输出**: `result/anonymous_files/optimizer_trace_user_queries.jsonl`


- 初始版本，输出自定义格式
