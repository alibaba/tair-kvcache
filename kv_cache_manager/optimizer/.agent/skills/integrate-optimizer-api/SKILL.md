# 接入 Optimizer API Skill

当其他 C++ 或 Python 组件希望直接调用 Optimizer，而不是通过 CLI 工具启动时，使用这个 skill。

## 需要确认的输入

- 需要的回放模式：普通 `OptimizerManager` 还是 `HierarchicalReplayManager`。
- 接入方需要直接事件调用，还是完整 trace 回放。
- 需要的输出：CSV、内存结果、radix tree export、lifecycle 或图。
- 线程模型和对象所有权预期。

## 步骤

1. 阅读 [../../handbook/internal_integration.md](../../handbook/internal_integration.md)。
2. 优先使用 config-driven 构造，保证 CLI 和嵌入式行为一致。
3. 普通回放使用 `OptimizerManager`。
4. 涉及 storage pool、P2P、active window 或 cache drop 时，使用 `HierarchicalReplayManager`。
5. 直接 API 调用必须等价于标准 trace event：
   - 读请求传入 `input_len`。
   - 只传完整 block key。
   - 时间戳使用整数 ns。
6. 只有 C++ API 稳定后，才新增或更新 pybind。

## 校验

- 增加测试，对比嵌入式 API 行为和等价 JSONL replay 行为。
- 校验 token hit-rate 计数，不只看 block 计数。
- 校验非法 instance id 按预期失败或返回失败。

## 回复内容

报告：

- 使用的 API
- 语义等价性测试
- public method 变化
- 如果对用户可见，说明更新了哪些文档或 skill
