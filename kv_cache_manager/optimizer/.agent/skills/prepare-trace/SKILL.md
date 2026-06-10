# 准备 Optimizer Trace Skill

当需要把外部日志转换成 optimizer JSONL，或校验已有 optimizer trace 时，使用这个 skill。

## 需要确认的输入

- 源 trace 格式和时间戳单位。
- 请求路由信息是否已经存在。
- `block_size`。
- prefix hash 规则，以及 hash 是否需要跨小时/跨文件稳定。
- 输出应该按 service、pod 拆分，还是生成一个全局文件。

## 步骤

1. 阅读 [../../handbook/trace_schema_and_conversion.md](../../handbook/trace_schema_and_conversion.md)。
2. 把源时间戳转换为整数 `timestamp_ns`，不要经过 float。
3. 生成稳定 prefix block key。
4. 丢弃不足 `block_size` 的尾部 key，但把尾部 token 保留在 `input_len`。
5. 写出 `type=get/write/request` 的标准 JSONL。
6. 按 `timestamp_ns` 排序；必要时使用稳定 tie-break。
7. 按 cache 归属拆文件：
   - 全局理论池：一个 service 文件，一个逻辑 instance。
   - per-pod replay：每个 pod 一个文件。
   - hierarchical preserve-trace：保留路由后的 `instance_id`。

## 校验

- 每一行都能解析为 JSON。
- 必填字段存在。
- `timestamp_ns > 0`。
- `get/request` 的 `input_len > 0`。
- `len(keys) <= input_len // block_size`。
- 文件已排序。
- 如果预期跨小时复用，需要抽样确认 prefix 可复用。

## 回复内容

报告：

- 输出文件树
- service/pod 数量
- 时间范围
- 请求数
- 校验失败项，如有
