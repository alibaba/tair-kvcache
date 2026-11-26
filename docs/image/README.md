# 架构图生成脚本

`generate_architecture.py` 生成 README 与设计文档使用的中英双语架构图：

| 输出文件 | 内容 |
|---|---|
| `kvcm-deployment-{zh,en}` | 部署视图 |
| `kvcm-architecture-overview-{zh,en}` | 模块视图 |
| `kvcm-server-components-{zh,en}` | 服务端组件协作 |

## 使用

```bash
python3 docs/image/generate_architecture.py            # SVG + 4K PNG
python3 docs/image/generate_architecture.py --svg-only # 仅生成 SVG
```

- 生成 SVG 仅依赖 Python 标准库。
- 安装 `rsvg-convert`（macOS 可执行 `brew install librsvg`）后会额外导出 3840×2160 PNG。
- 建议安装 Alibaba PuHuiTi 2.0 或 Noto Sans CJK SC，保证中文渲染一致。

## 修改文案

文案和组件标签集中在脚本的 `COPY`、`COMPONENT_COPY`、`DEPLOYMENT_COPY` 及各 `render` 方法中。`text(..., max_width=...)` 会按字号估算文本宽度，超过卡片宽度时生成失败并报出具体文案，避免改字后静默溢出。

改完脚本后重新生成图片，并同步 README 与 [模块架构与关联关系](../design/module_architecture.md)。
