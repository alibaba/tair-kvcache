# Client SDK I/O 契约（静态超时预算 / 内部取消 / Buffer 生命周期）

## 1. 核心契约（一句话）

**调用接口不携带任何时间参数；SdkWrapper 在 Init 阶段把自身配置的静态超时预算
（`sdk_config.timeout_config` 的 get/put_timeout_ms）注入每个后端的
`SdkBackendConfig::timeout_config()`，后端从自身任务起点起算 deadline，
预算内完成或内部取消 —— 到点后不得再触碰调用方的 local buffers。**

各后端按自身能力履约（见 §3 矩阵）。做不到"绝不碰"的后端必须如实声明为 soft 级。

## 2. 机制分层

| 层 | 职责 |
|---|---|
| SdkWrapper | 等待上限 = 入口 + 静态预算；任务从线程池被拾起时做**准入检查**（已过预算的任务拒绝发起 I/O）；Init 时注入预算给后端 |
| 存储后端 | 从**自身任务起点**起算 deadline（= 起点 + 注入预算）；组级/逐 block/逐 key 准入；超时时执行内部取消 |

两个起算点的关系：后端起点晚于 wrapper 入口，差值为 wrapper 线程池的排队等待 W_q。
**排序不变量**：`W_q + 后端内部耗时 ≤ wrapper 预算`。同步调用方（排队为零）
自然满足；异步调用方依赖排队有界（池线程数配置）。后端内部超时取值应给 W_q
留出余量 —— PACE 的实践是内部 10s 对 wrapper 15s（5s 余量）。

## 3. 后端履约矩阵

| 后端 | 有界性 | 准入检查 | Buffer 级别 | 超时行为 |
|---|---|---|---|---|
| LocalFile / NFS | 逐 block | 组级 + 逐 block | hard | abort 路径 sync GPU stream（GpuStreamDrainGuard），真取消 |
| HF3FS | hf3fs_wait_for_ios(abs_timeout) | 逐 block | hard | 数据先落自有 shm iov；超时不执行 CopyIovs；泄漏 iov/IOR 而不 free（free = UAF） |
| Mooncake | 逐 key 前置检查 | 逐 key | soft | 上游无取消语义；超时路径输出可归因日志（key/buffer/elapsed） |
| TairMempool (PACE) | PACE 内部超时（10s） | PACE 自管理 | hard（staging + cancel_and_drain） | **不读取注入的预算**（自行管理内部超时）；要求内部超时 < wrapper 预算（10s < 15s） |

注：PACE 属跨仓库生产路径，本仓库不修改其实现；注入字段对它是纯增量（不读即无行为
变化），未来若希望显式对齐可读取 `timeout_config()` 校验自身配置。

## 4. Known Limitations

1. **W_q 残留窗口**：后端 deadline 锚定自身起点，排队等待 W_q 使后端可能晚于
   wrapper 预算完成（同步调用方无此问题；生产 v6d 路径为同步直调）。后端内部
   取消机制保证其自身 deadline 之后不再触碰 caller buffer。
2. **ManagerClient / RTPLLMClient 层**：走 TransferClient 的既有路径，自动获得
   wrapper 预算，无额外配置。
3. **超时路径不等在飞任务；普通错误路径有界等待**：RunWithTimeoutParallel 在
   预算到期时立即返回；在飞任务的 I/O 不被取消（hard 后端的逐块准入会尽快
   停下；soft 后端见 §3）。普通错误路径则不同：错误往往发生在预算之前，SDK 仍在
   契约允许的窗口内写 caller buffer，因此错误路径会先置 stop 拦截仍在排队的
   group（不再发起新 I/O），再有界等待在飞 peer 至多到预算才返回 —— 否则
   caller 拿到错误后立即复用/释放 buffer，与在飞 DMA 构成数据竞争。
4. **HF3FS 超时泄漏 iov/IOR**：当 DeadlineExpired 导致 WaitIos 超时时，不释放已
   提交的 I/O 的 iov 缓冲区和 IOR（释放会导致 UAF）。泄漏规模 = 一次超时的读写
   调用涉及的 iov 大小。线上应在 WARN 日志中观测泄漏频率，若高频则需引入 3FS
   取消机制。
5. **Mooncake 为 soft 级**：超时后在飞的 DMA 无法取消；调用方在收到超时后立即
   复用相应 buffer 是文档化的数据竞争。逐 key 准入把暴露面从整批降到 ≤1 个在飞
   key。
