# AGENTS.md — subscriber

与 vLLM 同机部署的 Python 进程。源码包在仓库根目录下的 `subscriber/`，独立管理依赖，不依赖父项目构建系统。

## 项目概览

subscriber 订阅推理引擎的 KV cache 事件，转发给 KVCM，并与同机 DashServing 绑定健康状态
（同生同死）。

**技术栈**：Python ≥ 3.11（asyncio 全异步）、ZeroMQ（`pyzmq`）、gRPC（`grpcio` 1.75.1 +
`protobuf` 3.20.3，pb 手工维护）、`msgspec`（msgpack）、`httpx`、`pyyaml`、dashlog（可选依赖）。
工具链：`uv` + `ruff` + `mypy --strict` + `pytest`。

## 架构

| 模块 | 职责 |
|---|---|
| `subscriber/cli.py`、`main.py` | 入口与 `SubscriberLifecycle`：启动 7 步、serving 监督、关停 6 步 |
| `subscriber/config.py` | `SubscriberConfig`：字段、CLI 注册、校验、派生属性 |
| `subscriber/forwarding.py`、`pipeline/` | producer / sender 协程与无状态 per-batch 阶段 |
| `subscriber/engine/` | 引擎适配层（`AbstractEngineAdapter` + vllm / sglang + gRPC client） |
| `subscriber/health/` | 引擎存活协调器（epoch 门控）+ DashServing 状态上报 |
| `subscriber/kvcm/` | KVCM 客户端栈（domain client + HTTP transport + 服务发现） |
| `subscriber/metrics/` | 指标唯一出口（主路径 telemetry + lifecycle 点事件） |
| `subscriber/proto/` | `engine_service_rpc` pb 绑定（手工维护，见下文） |

分层依赖方向：`main` → `forwarding` → (`engine`, `health`, `kvcm`, `pipeline`, `metrics`)。
`engine/` 与 `kvcm/` 互不依赖（`KvCacheDescriptor` 由 `main` 在启动时注入），禁止加反向 import。

**稳态架构文档在 `docs/architecture/`，改架构前先读、改完回来更新**：

| 文档 | 内容 |
|---|---|
| [00-overview](docs/architecture/00-overview.md) | 模块划分、启动 / 退出时序、数据流总览、核心不变量 |
| [01-forwarding-pipeline](docs/architecture/01-forwarding-pipeline.md) | 双 pipeline 数据流、凑批、门控、丢弃语义 |
| [02-engine-adapter](docs/architecture/02-engine-adapter.md) | adapter 契约、代际语义、vLLM 实现、replay、快照信号 |
| [03-health-and-liveness](docs/architecture/03-health-and-liveness.md) | epoch、判死、HostDown、同生同死探针 |
| [04-kvcm-client](docs/architecture/04-kvcm-client.md) | 传输、服务发现、两步注册、心跳、location spec |
| [05-observability](docs/architecture/05-observability.md) | span 阶段、指标出口、日志、trace id |
| [06-configuration](docs/architecture/06-configuration.md) | 配置优先级、字段分组、环境变量、校验 |

## 开发命令

所有命令在仓库根目录（`pyproject.toml` 所在目录）下执行，使用 `uv` 管理环境。

```bash
# 初始化 / 同步依赖
uv sync --dev

# 运行测试
uv run pytest
uv run pytest -v                              # 详细输出
uv run pytest tests/engine/vllm/test_incremental.py  # 单文件

# Lint（ruff，line-length=88，规则集 E/F/I/UP/B）
uv run ruff check subscriber/ tests/ harness/
uv run ruff check --fix subscriber/ tests/ harness/    # 自动修复

# 格式化
uv run ruff format subscriber/ tests/ harness/
uv run ruff format --check subscriber/ tests/ harness/ # 仅检查，不修改

# 类型检查（mypy strict）
uv run mypy subscriber/

# Manifest 驱动的跨仓 harness（原子记录写入已忽略的 harness/records/runs/）
harness/run_local_checks.sh baseline all
harness/run_local_checks.sh protocol all
harness/run_local_checks.sh quality all
uv run python harness/loop.py review

# 启动进程（SPECTRUM_DEPLOYMENT_NAME、--kvcm-base-url 与 --host-port 为启动必填）
export SPECTRUM_DEPLOYMENT_NAME=my-deployment
uv run python -m subscriber --kvcm-base-url grpc://10.0.0.1:6381 --host-port 8080

# 指定参数启动
SPECTRUM_DEPLOYMENT_NAME=my-deployment uv run python -m subscriber \
  --kvcm-base-url spectrum://vs-example:6381 \
  --host-port 8080 \
  --engine-type vllm

# 使用配置文件（kvcm_base_url 与 host_port 可写在 yaml 中）
SPECTRUM_DEPLOYMENT_NAME=my-deployment uv run python -m subscriber --config config.yaml
```

默认 gRPC 时，裸 `host:port`、`grpc://` 和 `http(s)://` 都是直接 channel target；只有
`static://` / `spectrum://` 会经 service discovery 解析。端口必须为 KVCM `meta_rpc_port`；
HTTP 回退时才改为 `meta_http_port`。

## 质量门禁（Quality Gates）

- **pre-commit 钩子（本地强制）**：`.pre-commit-config.yaml` 按暂存文件路由执行，通过
  `uv run pre-commit install` 安装到 `.git/hooks/pre-commit` 后在每次 commit 时强制执行。
  clone 仓库后必须先执行一次 `uv run pre-commit install`。检查项：
  - `subscriber/`、`tests/` 或 `harness/` 的 Python 文件暂存时：ruff check、ruff format --check；
    subscriber/config/toolchain 变更另触发 mypy 与全量 `pytest --cov`
    （覆盖率门禁 `fail_under = 90`）；
  - `subscriber/proto/` 文件暂存时额外执行：全部 pb2/pb2_grpc import（protobuf 3.20.3
    兼容）、authoritative `.proto` / runtime / `.pyi` parity，以及
    `tests/proto/test_engine_service_rpc.py`、`tests/engine/test_grpc_clients.py` 和
    KVCM gRPC client 测试；
  - `docs/metrics.json` 或 metrics 源码/测试变更时执行 metrics catalog parity；
  - 每次 commit 检查 staged whitespace/conflict marker；修改 hook 配置时先验证配置本身。
- **pytest（开发中随手跑）**：commit 时 pre-commit 已强制全量测试，
  开发过程中仍应随手 `uv run pytest` 获得快速反馈（见「约束」一节的
  「修改后必须通过检查」）。测试未通过的修改视为未完成，不得提交/推送。
- **AoneCI（MR 触发）**：MR opened 时触发两条模板流水线——「Python单元测试」
  （Python 3.12，`pip install '.[dev]'` 后跑 pytest，失败钉钉通知）与
  「代码质量扫描（多语言）」。dev 依赖的唯一权威来源是
  `[project.optional-dependencies].dev`（uv 的 dev group 引用该 extra），
  新增 dev 依赖写在 extras 里，否则 CI 装不上。CI 状态用 `a1 ci run get` 查询。
- **e2e（显式 opt-in）**：`tests/e2e/` 的每个真实运行时用例都用 `pytest.mark.skipif`
  检查各自前置条件：DashServing 用例要求 `DSV_BINARY`（已构建的 `dashservingd`），KVCM
  gRPC 用例要求 `KVCM_REAL_GRPC_TARGET`（可选 `KVCM_REAL_ADMIN_HTTP_URL` 做隔离 setup）。
  默认全部跳过，保证默认 `uv run pytest` 在任何机器上行为一致。

## Git 约定

### 分支命名

- 常规开发：`feat/<description>`，如 `feat/full-kvcache`
- 缺陷修复：`fix/<description>`，如 `fix/zmq-replay-seq`
- 个人长期特性分支：`feature/<花名>/<description>`
- 临时验证：`worktree-<description>`（配合 git worktree，用完删掉）

不要直接在 `master` 上开发；不要 force push `master`。

### Commit Message

沿用仓库历史的 conventional commits：`<type>: <description>`，type ∈
`feat` / `fix` / `refactor` / `chore` / `docs` / `test` / `perf`。

示例：`feat: add use_eagle_pop and mamba_cache_mode in get_kv_cache_descriptor`、
`fix: fix zmq replay seq and frame`、`docs: update README/AGENTS.md`

- 一个 commit 只做一件事；重构与行为变更不要混在同一个 commit。
- **禁止**写入 `Co-Authored-By` 行。
- 不得使用 `--no-verify` 跳过 pre-commit 钩子；钩子失败要修根因。钩子失败意味着 commit 没有
  发生，此时**新建 commit**，不要 `--amend`。

### MR / 代码评审

- MR 标题与 commit message 保持一致。
- MR 描述必须包含：背景与动机、改动点、验证方式（贴 `uv run pytest` / `ruff` / `mypy` 的实际
  结论）、影响面与回滚方式。
- **跨仓库改动**（dashllm gRPC server、KVCM 服务端、`.proto` wire schema）必须在描述里列出对
  应仓库的 MR 链接，并说明上线顺序与兼容窗口。

## 开发流程

按以下 7 步执行。**每完成一个步骤后必须停下来等用户确认，不要自动继续下一步。**

1. **明确需求与范围** — 复述需求、边界、不做什么；不清楚的地方先问，不要猜。
2. **分析代码** — 先读 `docs/architecture/` 对应文档，再读源码，输出数据流 / 关键调用路径 /
   受影响的不变量；发现文档与代码不一致，先报告再动手。
3. **确定方案** — 写入 `docs/specs/YYYY-MM-DD_slug.md`（含方案、被影响的不变量、测试策略、
   风险），等用户确认后再写代码。
4. **实施修改** — 从最新 `master` 建分支；测试先行（改行为先改/加测试）；小步提交。
5. **质量门禁** — `uv run pytest`、`uv run ruff check subscriber/ tests/ harness/`、
   `uv run mypy subscriber/` 全绿，并把实际输出结论贴出来。不许用「应该没问题」代替执行。
6. **提交与评审** — commit → push 分支 → 创建 MR，指定评审人。
7. **回顾与更新** — 检查并更新 `docs/architecture/`、`AGENTS.md`、`README.md`、
   `docs/metrics.json`；把落地方案与设计文档的差异回写到 spec。

## Protobuf 兼容代码维护

生产环境固定使用 `protobuf==3.20.3`。当前 `grpc-tools` 生成的 Python
代码会导入 `google.protobuf.runtime_version` / `_builder` 并校验较新的
protobuf runtime，无法在生产环境加载。因此 `subscriber/proto/` 下的
Python pb 文件必须 hand-write / 手工维护，**禁止运行
`grpc_tools.protoc` 后直接提交生成结果**。

`subscriber/proto/engine_service_rpc.proto` 仍是 authoritative wire
schema。修改 `.proto` 后：

1. 手工同步 `engine_service_rpc_pb2.py` 中的 descriptor
   message/field/service 定义，继续使用兼容 protobuf 3.20.3 的
   `descriptor_pb2` / `message_factory` 实现；不得引入
   `runtime_version` 或 `_builder`。
2. 手工同步 `engine_service_rpc_pb2.pyi` 的字段、类型和构造函数。
3. 只有 RPC service、method 或 request/response type 发生变化时，才
   手工同步 `engine_service_rpc_pb2_grpc.py`；仅新增 message field
   时该文件不应产生 diff。
4. 保持 authoritative field number、无 `package` 的 service identity 和
   完整 RPC path 不变，并增加 descriptor field-number、raw wire parse 和
   RPC path 测试。

上述 pb 文件均提交到仓库，构建时不会自动生成。修改后至少运行对应
proto/client 测试和 `uv run mypy subscriber/`，确认 protobuf 3.20.3
环境可以正常 import。

## 约束

- **asyncio 全异步**：所有 IO 必须使用 `zmq.asyncio.Socket`，禁止在 async 函数中调用同步 ZMQ socket 方法。
- **引擎适配器隔离**：ZMQ 订阅和 replay 逻辑封装在 `AbstractEngineAdapter` 子类中，`kv_event_loop` 不感知具体引擎实现。
- **引擎存活探测隔离**：底层连接状态（如 ZMQ socket monitor）由 `AbstractEngineAdapter.watch_liveness()` 暴露为引擎无关事件，判死、epoch、reset 策略集中在 health coordinator。
- **冷启动 reset 语义**：subscriber 首次连接引擎前不发送 `AllBlocksCleared`；只有曾经进入可发送代际后再断线并达到重试阈值，才发送 reset。
- **发送门控**：向 kvcm 转发实时或 replay batch 前必须通过 `EngineHealthCoordinator.wait_ready_epoch()` 获取当前可发送 epoch。
- **分阶段 span 与主路径指标**：adapter 创建并 yield `EngineEventBatch(batches, telemetry, trace_id)`；同一个 `BatchTelemetry` 经 `PipelineContext` 贯穿到 KVCM client。adapter 标记 `decode` / `replay_fetch` / `snapshot_*`，forwarding 标记 `queue_wait` / `engine_gate_wait` / `block_filter`，KVCM client 标记 `expand` / `kvcm_send`。`telemetry.mark("stage")` 只记录内存状态；`count` / `gauge` 可带 `tags`，空标签分别保持 counter 聚合与 gauge 最后值，非空标签保留独立 observation。仅 `MetricsReporter` 后台任务 flush 到 dashlog；`BatchTelemetry` 与 reporter 的观测路径均 best-effort、永不阻断转发主路径。
- **Metrics 上报规范**：所有指标通过 `subscriber/metrics/` 包（`_base.py` 中的 `_dashlog_counter` / `_dashlog_gauge`）统一出口上报，自动添加 `kvcache_subscriber_` 前缀。维度信息（endpoint、status、reason 等）通过 `tags` 参数传递（对应 Prometheus labels），不要编码进 metric name。dashlog 本地不可编译时所有 report 函数为 no-op（`try: import dashlog / except ImportError` fallback）。新增或修改指标后必须同步更新 `docs/metrics.json` 指标目录。`_dashlog_counter` 按 `DS_EAS_USE_OTEL` 双路径切换（判定规则与 dashlog `EASOTelClient::Enabled()` 一致）：ASI-EAS 部署由 manager 注入该变量，直调 dashlog 原生 `Counter` 走 OTel 增量语义；未设置（PAI-EAS）时走 Gauge 累积绕行，因为 legacy `AddCounters` 在 OTel 关闭时会静默丢数据。
- **Replay 语义**：检测到 seq gap 时，先聚合 replay batch 一次性 yield，再 yield 当前实时消息。vLLM replay 端会返回所有 `seq >= gap_start_seq` 的缓冲消息（包含触发 gap 的当前实时消息及更新消息），必须过滤只转发缺失区间 `[gap_start_seq, current_seq - 1]`（其余会以实时消息到达，直接转发会重复）；过滤后仍要 drain 到 END marker。replay 成功完成（返回列表，含空列表）后立即把 `_last_seq` 推进到 `current_seq - 1`，与实时消息 decode 成败解耦，避免同一 gap 被重复 replay。replay 中途 abort（超时、异常、帧格式错误、decode 失败）必须调用 `_replace_replay_socket()` 更换 DEALER socket，防止残留帧污染下一次 replay。replay 使用 DEALER socket（非 REQ），避免 async 下严格一问一答的阻塞问题。
- **配置优先级**：CLI 显式参数 > `--config yaml` 文件 > dataclass 默认值；`kvcm_base_url` 为必填项，
  可为 CLI / 配置文件 / `KVS_KVCM_ENDPOINT` 环境变量提供（优先级 CLI > YAML > 环境变量），
  全部为空时在 `validate()` 阶段直接报错。
- **启动必需的身份环境变量**：`SPECTRUM_DEPLOYMENT_NAME` 必须为非空唯一部署标识；`KvcmClient.start()` 会校验，缺失/空白直接失败（fatal），因为空值会退化为 `_<block_size>` 这类被无关实例共享的 instance_id，破坏"KVCache 仅在同一 instance_id 内复用"的隔离不变量。
- **Lint/类型检查禁用规则**：禁止通过内联注释（如 `# noqa`、`# type: ignore`）禁用 ruff 或 mypy 检查。自动生成的代码（如 protobuf 生成文件）应在 `pyproject.toml` 或 `ruff.toml` 中通过 `exclude` / `per-file-ignores` 排除整个文件；只有确实无法通过修改代码解决的极少数情况，才允许使用注释禁用，且必须在注释中说明原因。
- **Docstring 规范**：通用接口/模块定义（后续要实现某个接口的 placeholder）、抽象基类的 public 方法、复杂实现逻辑必须有 docstring，说明契约、默认行为和关键不变量。简单的内部 helper 和自解释代码不需要。
- **文档落盘目录**：`docs/architecture/` 放**稳态架构文档**（当前代码长什么样，编号命名如
  `03-health-and-liveness.md`）；`docs/specs/` 放**变更设计文档**（某次改动怎么做，命名
  `YYYY-MM-DD_slug.md`，如 `2026-07-21_engine-node-metadata-rpc.md`）；`docs/plans/` 放实施计划，
  `docs/reviews/` 放评审与测试报告。不要在项目根目录或其他位置创建 `specs/` 目录。**改了架构必须
  回写 `docs/architecture/` 对应文档，不能只留一篇 spec。**
- **仓库根目录不落运行时日志**：运行时日志不得写入仓库根目录。stdlib 日志后端默认只输出 stderr，需要文件日志时设置 `SUBSCRIBER_LOG_FILE=<path>`（指向仓库外或部署目录）；review / 报告类产物归档到 `docs/reviews/`。
- **修改后必须通过检查**：所有代码修改完成后，必须确保 `uv run pytest`、`uv run ruff check subscriber/ tests/ harness/` 和 `uv run mypy subscriber/` 全部通过，否则修改视为未完成。四项均由 pre-commit 钩子在 commit 时强制（含全量 pytest --cov）；开发中仍应随手运行以获得快速反馈（见「质量门禁」一节）。

## 代码风格

### 类型与命名

- `mypy --strict`：所有函数（含测试 helper 之外的内部函数）必须有完整注解；异步生成器标注
  `AsyncGenerator[T, None]`。
- 文件首行 `from __future__ import annotations`。
- 模块私有用 `_` 前缀；只在包 `__init__.py` 里 re-export 对外 API。
- **单位与语义后缀**：秒 `_s`、毫秒 `_ms`、字节 `_bytes`、计数 `_count`、上限 `_maxsize`、
  间隔 `_interval_s`、超时 `_timeout_s`。新字段沿用既有后缀，不要自造。
- 不可变优先：跨模块传递的数据载体用 `@dataclass(frozen=True)`（如 `EngineEventBatch`、
  `MergedBatch`、`KvCacheDescriptor`）；只有需要在 pipeline 中累积状态的才可变
  （如 `PipelineContext`）。

### 日志

- 结构化调用：`logger.info("message", step="<stage>", tags={...})`。消息本身是固定短语，
  **不要用 f-string 把变量拼进 message**——变量放 `tags`。
- debug 级诊断先判 `logger.is_debug_enabled()` 再拼装，避免热路径做无用工作。
- 高频失败必须限流（首次详细 + 周期汇总），参考 `KvcmDropTracker` 与 state reporter 的写法。

### 异常处理

- 只捕获预期异常类型，禁止裸 `except:`；`asyncio.CancelledError` 必须原样重新抛出。
- 观测 / 上报路径（metrics、span、日志、best-effort 状态上报）永不向主路径抛异常。
- 业务不可用与业务拒绝要区分类型（`KvcmUnavailableError` vs `KvcmReportRejectedError`），
  不要用同一个异常表达「可重试」和「永久失败」。
- 编程错误不要吞：非预期异常让它冒到 `_serve_until_shutdown`，由生命周期统一处理。

### 测试

- `tests/` 目录结构镜像 `subscriber/`；跨模块行为放顶层（`test_forwarding.py` /
  `test_main.py`）。
- 真实 ZMQ socket 用例打 `@pytest.mark.integration`，需要真实 `dashservingd` 的打
  `@pytest.mark.e2e`（显式 opt-in：设置 `DSV_BINARY` 指向已构建的二进制才运行，
  否则跳过）。
- 覆盖率门禁 `fail_under = 90`：只能往上调，不许为了让改动通过而下调。

## Health Integration 开发原则

subscriber 与同机 DashServing "同生同死"：subscriber 未 active 时 DashServing 不导流，subscriber 异常时 DashServing 暴露健康失败。详细 spec 见 `docs/specs/2026-07-22_subscriber-health-dashserving-integration.md`。

**核心不变量（修改代码前必须理解）：**

- **引擎健康只由 `GetWorkerStatus.alive` 驱动**：不用 HTTP `/readiness` 做引擎健康判断，不用 DashServing 探针反馈 engine DEAD。
- **KVCM 是 lossy 旁路，不是门控（运行期）**：进入 serving 后，KVCM 不可用不阻塞转发、不影响探针、不改变 subscriber 状态。发送失败丢弃并计数，恢复后自动续传。不要为 KVCM 加 generation/snapshot/condition 等复杂机制。**启动期例外（有意设计）**：`_graceful_startup` 第 3 步会等待 KVCM 注册成功（`_wait_kvcm_registered`）才启动 pipeline 并上报 active——启动时需要 KVCM 注册成功才能够启动，避免在从未注册的情况下导流；等待期间 subscriber 保持 starting（readiness 503），并周期性打印 "still waiting for kvcm registration" 警告。不要把这个启动等待当作 bug"修掉"。
- **seq_id 线性化**：状态上报靠单调递增 seq_id 防乱序。低 seq 的 active 不能复活已接受的 inactive。不要加 session_id 除非需要支持独立重启 subscriber。
- **HostDown 幂等**：`AllBlocksCleared` 每个 sendable epoch 最多发一次。冷启动失败不发。
- **同生同死探针语义**：starting → readiness 503 / liveness 200（startup grace）；active → 200/200；inactive/failed/TTL expired → 503/503。
- **环境变量**：`DS_LLM_LAUNCH_KV_EVENT_SUBSCRIBER=1` 时 DashServing 启用 subscriber 健康检查，否则探针行为不变。

**不要做的事：**

- 不要为引擎重启加 metadata 重新验证（拓扑不变，Pod 替换兜底）
- 不要轮询 DashServing readiness 来控制转发（`wait_ready_epoch()` 已是正确门控）
- 不要在 KVCM transport 失败时改 `is_registered`（heartbeat loop 自己管重连）
- 不要把 learn mode 加回来（已删除）
- 不要把 transient 启动错误报为 failed（只有 protocol/unsupported 才是 fatal）

## 添加新引擎适配器（参考现有 SGLang 实现）

1. 参考现有 `subscriber/engine/sglang/` 包布局：`adapter.py` 实现 `AbstractEngineAdapter` 并加注册装饰器，`__init__.py` re-export 适配器类：

```python
# subscriber/engine/<engine>/adapter.py
from subscriber.engine.base import AbstractEngineAdapter


@AbstractEngineAdapter.register("sglang")
class SGLangAdapter(AbstractEngineAdapter):
    async def subscribe_kv_events(self): ...  # 实现事件订阅逻辑

    async def watch_liveness(self): ...  # 实现引擎无关存活事件流
```

完整可运行示例见 `subscriber/engine/sglang/adapter.py`（placeholder 适配器，含全部必须实现的方法）。

2. 启动时传入 `--engine-type sglang`，`AbstractEngineAdapter.create()` 会自动 lazy import 并实例化。

无需修改 `main.py`、`kv_event_loop` 或任何 client 代码。

## Stub 待接入项

| 编号 | 内容 | 文件 | 阶段 |
|---|---|---|---|
| S-4 | subscriber 不可用恢复策略：kvcm SDK 心跳、TTL、session/checkpoint | kvcm SDK / kvcm 服务端 / subscriber | DONE（heartbeat + TTL + state reporter 已实现；session/checkpoint 见 TODO(independent-restart)） |
| S-5 | 多 DP（bootstrap `runtime_topology.data_parallel_size > 1`）事件订阅：当前 adapter 只支持一个 engine-discovered transport，接受 bootstrap 后会 fail closed；rank 1+ 尚未聚合 | subscriber/engine/vllm/adapter.py | TODO（实现 per-rank bootstrap/transport 聚合前不要放开 `data_parallel_size == 1` 校验） |
