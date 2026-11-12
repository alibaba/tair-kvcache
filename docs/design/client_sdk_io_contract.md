# Client SDK I/O 契约（超时预算 / 内部取消 / Buffer 生命周期）

## 1. 核心契约（一句话）

**旧读写接口不携带时间参数；SdkWrapper 在 Init 阶段把自身配置的静态超时预算
（`sdk_config.timeout_config` 的 get/put_timeout_ms）注入每个后端的
`SdkBackendConfig::timeout_config()`，后端从自身任务起点起算 deadline，
按各后端能力进行准入与超时收尾；超时返回不证明在飞 I/O 已停止访问 caller buffer。**

各后端按自身能力履约（见 §3 矩阵）。做不到"绝不碰"的后端必须如实声明为 soft 级。
hard/soft 描述后端已有能力，不能据此判断 wrapper 超时返回时 Buffer 是否安全。

`LoadKvCachesWithDeadline` / `SaveKvCachesWithDeadline` 接收本进程 `steady_clock` 绝对毫秒期限；
正值按 `min(调用方期限, 当前阶段起点 + get/put_timeout_ms)` 收紧预算，非正值沿用旧行为。
各分组继承该期限；自定义包装层须重写 WithDeadline 继续转发。
新 API 以 `KVCM_HAS_TRANSFER_DEADLINE` 标识，SDK 与调用方需使用配套头文件、动态库重编译。

## 2. 机制分层

| 层 | 职责 |
|---|---|
| SdkWrapper | 等待上限 = 入口 + 静态预算（有调用方期限时取较早者）；任务从线程池被拾起时做**准入检查**（已过预算的任务拒绝发起 I/O）；Init 时注入预算给后端 |
| 存储后端 | 从**自身任务起点**起算 deadline（有调用方期限时取较早者）；组级/逐 block/逐 key 准入；按后端能力处理超时 |

未传调用方期限时，后端起点晚于 wrapper 入口，差值为线程池排队等待 W_q。
预算配置应满足 `W_q + 后端内部耗时 ≤ wrapper 预算`，为排队等待留出余量。
传入期限时，各层继承较早的绝对期限，排队或分组不会重新起算这份预算。

## 3. 后端履约矩阵

| 后端 | 有界性 | 准入检查 | Buffer 级别 | 超时行为 |
|---|---|---|---|---|
| LocalFile / NFS | 逐 block | 组级 + 逐 block | hard | abort 路径 sync GPU stream（GpuStreamDrainGuard），排空已启动的 copy |
| HF3FS | hf3fs_wait_for_ios(abs_timeout) | 逐 block | hard | 数据先落自有 shm iov；超时不执行 CopyIovs；泄漏 iov/IOR 而不 free（free = UAF） |
| Mooncake | 逐 key 前置检查 | 逐 key | soft | 上游无取消语义；超时路径输出可归因日志（key/buffer/elapsed） |
| TairMempool (PACE) | 原有等待上限与调用方期限取较早者 | PACE 自管理 | 未确认完成时不保证安全 | 既有等待/重试消费期限，不强制取消在飞 DMA |

注：PACE 预算透传需配套提供 `PACE_HAS_TRANSFER_DEADLINE` 的新 client 库。
阻塞 UDS/CUDA 和安全排空仍可能超过期限，超时不能代替 Buffer 安全确认。

## 4. Known Limitations

1. **W_q 残留窗口（旧接口）**：未传调用方期限时，后端 deadline 仍从自身起点起算，
   排队可能使后端晚于 wrapper 预算完成。传入期限可收紧这些检查和等待，
   但不能强制中断已经启动的系统调用、GPU copy 或 DMA。
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
