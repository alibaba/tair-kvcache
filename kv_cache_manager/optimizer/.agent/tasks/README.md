# Optimizer 需求记录

在 `kv_cache_manager/optimizer/` 下处理的每个新需求，都必须在这里先记录，再开始实现。

## 文件命名

使用：

```text
YYYY-MM-DD-short-title.md
```

标题要稳定、具体，例如：

```text
2026-06-10-cache-drop-events.md
```

## 必填内容

从 [TEMPLATE.md](TEMPLATE.md) 开始。task 记录必须包含：

- 原始需求或忠实摘要
- 范围和非目标
- owner/status
- 需要人确认的决策
- 链接到 `../plans/` 下的 plan
- 完成前需要的校验

## 状态

使用以下状态之一：

- `proposed`
- `in_progress`
- `completed`
- `blocked`

不要删除已完成 task；它们也是 optimizer 工作的轻量 changelog。
