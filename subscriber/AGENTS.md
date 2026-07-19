# AGENTS.md — subscriber

与推理引擎同机部署的 Python 进程。代码在 `subscriber/`（本目录），独立管理依赖，不依赖父项目构建系统。

## 开发命令

所有命令在本目录（`subscriber/`）下执行，使用 `uv` 管理环境。

```bash
# 初始化 / 同步依赖
uv sync --dev

# 运行测试
uv run pytest
uv run pytest -v                              # 详细输出
uv run pytest tests/test_vllm_adapter.py      # 单文件

# Lint（ruff，line-length=88，规则集 E/F/I/UP/B）
uv run ruff check subscriber/ tests/
uv run ruff check --fix subscriber/ tests/    # 自动修复

# 格式化
uv run ruff format subscriber/ tests/
uv run ruff format --check subscriber/ tests/ # 仅检查，不修改

# 类型检查（mypy strict）
uv run mypy subscriber/

# 启动进程（默认连接本机 vLLM）
uv run python -m subscriber

# 指定参数启动
uv run python -m subscriber \
  --zmq-pub-endpoint tcp://localhost:5557 \
  --zmq-replay-endpoint tcp://localhost:5558 \
  --kvcm-addr 10.0.0.1:50051 \
  --engine-type vllm

# 使用配置文件
uv run python -m subscriber --config config.yaml
```

## 约束

- **asyncio 全异步**：所有 IO 必须使用 `zmq.asyncio.Socket`，禁止在 async 函数中调用同步 ZMQ socket 方法。
- **引擎适配器隔离**：ZMQ 订阅和 replay 逻辑封装在 `AbstractEngineAdapter` 子类中，`kv_event_loop` 不感知具体引擎实现。
- **引擎存活探测隔离**：底层连接状态（如 ZMQ socket monitor）由 `AbstractEngineAdapter.watch_liveness()` 暴露为引擎无关事件，判死、epoch、reset 策略集中在 health coordinator。
- **冷启动 reset 语义**：事件流 adapter（vLLM）首次连接前不发送 `AllBlocksCleared`；RTP 在首个有效快照前不得注册节点或发送心跳，全量 adapter 默认在首个稳定快照执行一次 `RegisterInstance → HOST_DOWN → NODE_REGISTER`，用于清除 Subscriber 重启期间遗留的旧 key，再上报完整 add 集合。
- **发送门控**：向 kvcm 转发实时或 replay batch 前必须通过 `EngineHealthCoordinator.wait_ready_epoch()` 获取当前可发送 epoch。
- **Replay 语义**：检测到 seq gap 时，先聚合所有 replay batch 一次性 yield，再 yield 当前实时消息。replay 使用 DEALER socket（非 REQ），避免 async 下严格一问一答的阻塞问题。
- **RTP 全量语义**：RTP adapter 只接受所有配置 endpoint 都成功返回的完整快照；diff baseline 仅在 KVCM 确认发送成功后推进，删除需连续快照确认。
- **端口语义**：RTP `GetCacheStatus` 走 rank gRPC 端口（默认 `START_PORT + 1`），不是前端 HTTP 推理端口；多 DP/多机地址由启动器显式传入。
- **配置优先级**：CLI 显式参数 > `--config yaml` 文件 > dataclass 默认值。
- **Lint/类型检查禁用规则**：禁止通过内联注释（如 `# noqa`、`# type: ignore`）禁用 ruff 或 mypy 检查。自动生成的代码（如 protobuf 生成文件）应在 `pyproject.toml` 或 `ruff.toml` 中通过 `exclude` / `per-file-ignores` 排除整个文件；只有确实无法通过修改代码解决的极少数情况，才允许使用注释禁用，且必须在注释中说明原因。

## 添加新引擎适配器（如 SGLang）

1. 新建 `subscriber/engine/sglang.py`，实现 `AbstractEngineAdapter`，加注册装饰器：

```python
from subscriber.engine.base import AbstractEngineAdapter

@AbstractEngineAdapter.register("sglang")
class SglangAdapter(AbstractEngineAdapter):
    async def subscribe_kv_events(self):
        ...  # 实现事件订阅逻辑

    async def watch_liveness(self):
        ...  # 实现引擎无关存活事件流
```

2. 启动时传入 `--engine-type sglang`，`AbstractEngineAdapter.create()` 会自动 lazy import 并实例化。

无需修改 `main.py`、`kv_event_loop` 或任何 client 代码。

## Stub 待接入项

| 编号 | 内容 | 文件 | 阶段 |
|---|---|---|---|
| S-4 | subscriber 不可用恢复策略：kvcm SDK 心跳、TTL、session/checkpoint | kvcm SDK / kvcm 服务端 / subscriber | TODO |
