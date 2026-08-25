# Client SDK I/O 契约（deadline / 取消 / Buffer 生命周期）

## 1. 核心契约（一句话）

**deadline（绝对 steady_clock 毫秒，0=无限制）到达后，实现方不得再触碰调用方的 local buffers。**

各后端按自身能力履约（见 §3 矩阵）。做不到"绝不碰"的后端必须如实声明为 soft 级。

## 2. 调用方用法

- 读/写路径均传真实 deadline：`deadline_ms_from_now(sdk_get/put_timeout_ms)`。
- 写路径的租约（write_timeout_seconds）不由 connector 计算或传递；SdkWrapper 内部对
  外部 deadline 与内部 timeout_config 取 min，**min 结果原样下发进 SDK 作为准入
  deadline**（caller 传 0 时即内部预算）——wrapper 的等待上限与 SDK 的准入依据
  始终是同一个时间点，内层自律不越过外层租约。
- deadline 在各进程内各自计算，不跨进程/跨 rank 传递（见 Known Limitations）。

## 3. 后端履约矩阵

| 后端 | 有界性 | 准入检查 | Buffer 级别 | 备注 |
|---|---|---|---|---|
| LocalFile / NFS | 逐 block | 逐 block | hard | abort 路径 sync GPU stream |
| HF3FS | hf3fs_wait_for_ios(abs_timeout) | 逐 block | hard | 数据先落我们自己的 shm；超时不执行 CopyIovs |
| Mooncake | 逐 key 前置检查 | 逐 key | soft | 上游无取消语义；超时路径输出可归因日志 |
| TairMempool (PACE) | PACE 内部超时 + deadline 透传 | PACE 无队列准入 | hard（默认 staging） | PACE 已有 cancel_and_drain + BufferUseGuard |

## 4. Known Limitations

1. **各进程自算 DDL 导致租约可能越界**：worker 自算起点晚于 scheduler 拿到租约的时刻，
   极端情况下写入可能越过 KVCM 租约。取舍理由：多机时间不一致在 happy path 上更危险，
   租约越界不在 happy path。
2. **写租约时间原点偏差**：KVCM 服务端在处理 StartWriteCache 时即开始计时
   （write_location_manager.cc:187），且会 cap（cache_manager.cc:1090
   kMaxWriteTimeoutSeconds=1800）。客户端算出的 DDL 可能晚于服务端真实 expire。
   这是 KVCM 服务端既有问题，本次不改。
3. **ManagerClient / RTPLLMClient 层无 DDL**：这两层公开接口保持传 0，走 SdkWrapper 兜底。
   不确定外部用户是谁，暂不加。
4. **超时路径不等在飞任务；普通错误路径有界等待**：RunWithTimeoutParallel 在
   deadline 到期时立即返回，在飞任务的 I/O 不被取消也不被等待（hard 后端的逐块
   准入会尽快停下；soft 后端见 §3）。普通错误路径则不同：错误往往发生在 deadline
   之前，SDK 仍在契约允许的窗口内写 caller buffer，因此错误路径会先置 stop 拦截
   仍在排队的 group（不再发起新 I/O），再有界等待在飞 peer 至多到 deadline 才返回
   ——否则 caller 拿到错误后立即复用/释放 buffer，与在飞 DMA 构成数据竞争。
5. **HF3FS 超时泄漏 iov/IOR**：当 DeadlineExpired 导致 WaitIos 超时时，不释放已提交
   的 I/O 的 iov 缓冲区和 IOR（释放会导致 UAF）。泄漏规模 = 一次超时的读写调用涉及的
   iov 大小。线上应在 WARN 日志中观测泄漏频率，若高频则需引入 3FS 取消机制。