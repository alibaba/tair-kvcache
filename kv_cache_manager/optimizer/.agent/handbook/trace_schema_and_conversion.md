# Trace Schema 和转换

Optimizer 只接受标准 JSONL。每一行是一条事件。外部 trace 必须先转换后才能回放。

## 标准类型

| `type` | 含义 | 必填字段 |
|---|---|---|
| `get` | 读 / prefill 查询 | `instance_id`、`timestamp_ns`、`keys`、`input_len` |
| `write` | cache 写入 | `instance_id`、`timestamp_ns`、`keys` |
| `request` | 一条请求，内部拆成 get 和 delayed write | `instance_id`、`timestamp_ns`、`keys`、`input_len` |

完整语义见 [../../docs/strategy_config.md](../../docs/strategy_config.md)。这里是转换检查清单。

## 字段清单

| 字段 | 类型 | 谁使用 | 说明 |
|---|---|---|---|
| `type` | string | loader | `get`、`write` 或 `request` |
| `instance_id` | string | 路由 / cache 归属 | 必须匹配 config；preserve-trace 模式下表示 infer/pod id |
| `trace_id` | string | 调试 / 分析 | 可选但建议保留 |
| `timestamp_ns` | int64 | 回放排序 | 必须是整数 ns；不要经过 float 转换 |
| `keys` | int64/uint64 array | cache index | 只包含完整 block key |
| `input_len` | int64 | token 命中率分母 | `get` 和 `request` 必填 |
| `query_type` | string | 读语义 | `prefix_match` 或 `batch_get`，默认 `prefix_match` |
| `block_mask` | bool array 或 int | 进入 KVCM/L3 前已有的本地命中 | 直接分析请求时通常为空 |
| `sw_size` | int32 | 兼容字段 | 通常为 `0` |
| `location_spec_names` | string array | 兼容字段 | 通常为空 |
| `ttl_us` | int64 | write/request TTL | `0` 使用 group 默认，`-1` 禁用 TTL |

## 完整 block 规则

`keys` 不能包含不足 `block_size` 的尾部 block。

例如 `block_size=2048`、`input_len=5000`：

```text
floor(5000 / 2048) = 2 个完整 block
keys 长度 <= 2
尾部 904 个 token 只保留在 input_len 中
```

这也是 token hit rate 可能低于 block hit rate 的原因：尾部 token 进入 `InputTokens`，但不能成为完整 block 命中。

## Prefix Hash 规则

从原始模型 trace 转换时：

- 基于完整 token prefix 计算稳定的 block key。
- 如果要跨小时、跨文件观察复用，同一服务必须使用相同 hash 函数和 salt。
- 时间戳转换到整数 ns，不能经过 `float`。
- 输出 JSONL 按 `timestamp_ns` 排序；同一时间戳下需要稳定 tie-break。
- 只有 cache 归属确实需要时，才按 service 或 pod 拆文件。

## 时间戳转换

不要这样做：

```python
timestamp_ns = int(float(timestamp_s) * 1_000_000_000)
```

优先用整数或 decimal：

```python
from decimal import Decimal

timestamp_ns = int(Decimal(timestamp_s_text) * Decimal(1_000_000_000))
```

如果源时间戳是微秒：

```python
timestamp_ns = timestamp_us * 1000
```

## 转换校验

运行 optimizer 前检查：

- 每一行都能解析为 JSON。
- 每种 `type` 的必填字段都存在。
- `timestamp_ns > 0`。
- 文件按 `timestamp_ns` 排序。
- `get/request` 的 `input_len > 0`。
- `get/request` 满足 `len(keys) <= input_len // block_size`。
- `instance_id` 存在于 config 或 hierarchical cluster 中。
- 拆分后的 service/pod 数量符合预期。
