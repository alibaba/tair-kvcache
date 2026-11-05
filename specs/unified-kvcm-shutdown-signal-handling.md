# 【需求】统一 KVCM 退出信号处理与优雅关闭流程 — UltraPlan Spec

## Requirements（需求与验收标准）

### User Stories
- 作为 KVCM 服务的运维/平台方，我希望向进程发送任一退出信号（SIGTERM/SIGINT/SIGUSR1/SIGUSR2）都能触发一次可预期、可完成的优雅关闭，以便 Pod 漂移和发布时不丢已接受请求、不留脏的选主状态。
- 作为 KVCM 开发者，我希望关闭流程只在一个固定的、非组件线程的执行上下文里运行，以便不再出现自连接、重入死锁和排空循环等待这类只在特定信号落点才复现的偶发故障。
- 作为 KVCM 开发者，我希望有可重复、可定向的信号回归测试，以便后续改动关闭链路时能在 CI 里立刻发现退化。
- 作为运维方，我希望在优雅关闭卡住时可以用第二次信号强制结束进程，以便不必依赖外部 SIGKILL 才能恢复调度。

### 验收标准（EARS）

- **AC-1**: THE SYSTEM SHALL 在创建任何其它线程之前阻塞 SIGTERM/SIGINT/SIGUSR1/SIGUSR2，并由一个固定的控制线程在普通执行上下文（非信号处理器）中消费这些信号。
  - 进程内不再存在退出信号的 signal handler；信号上下文不执行加锁、日志与清理。
  - 四个退出信号语义保持一致，均为「请求优雅关闭」；`SIGPIPE` 仍被忽略；SIGSEGV/SIGABRT/SIGBUS/SIGFPE/SIGILL 仍由 `CrashHandler` 独立处理，不受阻塞掩码影响。
  - 退出信号集合与今天完全相同：SIGTERM/SIGINT/SIGUSR1/SIGUSR2。其中 SIGUSR1/SIGUSR2 是 POSIX 留给应用自定义语义的信号，KVCM 继续把它们定义为优雅退出，不新增也不移除任何退出信号。
  - 在掩码设置之后创建的线程一律继承该掩码（含日志库、gRPC、HTTP、选主、Metrics 线程）。

- **AC-2**: WHEN 控制线程收到首个退出信号 THE SYSTEM SHALL 只记录该信号并唤醒主线程，由主线程按既有关闭顺序执行完整关闭流程：撤销选主资格 → 关闭业务门禁 → 等待已开始的恢复/清理与已接受请求排空 → 停止 gRPC/HTTP/Metrics → 销毁所依赖的服务与 manager → 销毁 logger。
  - 控制线程自身不执行任何组件的 Stop/join/清理。
  - 关闭期间的客户端可见行为与今天一致：门禁先于传输层关闭，新的 leader-only 请求仍拿到门禁拒绝码而不是传输错误。

- **AC-3**: WHILE 优雅关闭正在进行 IF 再次收到任一退出信号 THEN THE SYSTEM SHALL 记录强制退出原因与被丢弃的关闭进度，并以退出码 `128 + 信号号` 立即终止进程。

- **AC-4**: THE SYSTEM SHALL 保证任何组件线程（LeaderLock、LeaseCheck、状态转换、Metrics、正在处理业务请求的 gRPC/HTTP 线程）都不会成为关闭流程的执行者，从而不发生 self-join 抛出 `std::system_error`、持锁重入死锁与请求排空循环等待。
  - `LoopThread::Stop()` 在检测到「被自身循环线程调用」时记录错误日志并置可观测的检测标志，但**不跳过 join、不 detach**；线上行为与今天一致，问题在测试与日志中暴露而非被掩盖。
  - 组件线程收到定向信号时继续执行自身逻辑，不被打断进入关闭路径。

- **AC-5**: IF 关闭流程被重复或并发请求（多次 `Server::Stop()`、启动失败路径与信号路径同时触发）THEN THE SYSTEM SHALL 只执行一次完整关闭，其余请求等待该次关闭完成后返回，不产生并发的组件 Stop/join。

- **AC-6**: WHEN 进程经优雅关闭正常退出 THE SYSTEM SHALL 返回退出码 0，并在销毁所依赖的服务与 manager 之前完成选主撤销、请求排空与生命周期清理；控制线程必须在 logger 销毁之前完成 join。

- **AC-7**: WHERE 定向信号回归测试启用 THE SYSTEM SHALL 在带超时上限的测试中覆盖以下组合并通过普通与 ASAN CI：
  - 信号落点：选主线程 / 租约检查线程 / 状态转换线程 / Metrics 线程 / 正在处理业务请求的线程。
  - 时序：线程持锁中被信号打断、恢复进行中、降级清理进行中、请求排空期间、重复信号、并发退出请求。
  - 端到端：真实子进程收到 SIGTERM 后在超时上限内以退出码 0 结束；连续两次 SIGTERM 后以 `128+SIGTERM` 结束。

### 非目标（Out of Scope）
- 不调整 `Server::Stop()` 内部的组件关闭顺序，不改 `Server::OnNoLongerLeader()` 的排空与清理步骤。
- 不引入优雅关闭超时配置项与看门狗线程（本次只以「重复信号强制退出」作为兜底）。
- 不改造 `CacheManager` 内部各后台线程的停止方式（仍由析构驱动），不改造 optimizer 服务已有的 `RequestShutdown` 实现。
- 不新增 Admin/Debug RPC 或健康检查触发关闭的能力。
- 不重构 `CommandArgs::ParseCmdLine` 的 `exit(1)`、`CrashHandler` 的 `_exit` 兜底、启动失败路径的返回码语义。
- 不引入 `signalfd`/`epoll` 等 Linux 专有机制，保持 `sigtimedwait` 的可移植性。
- 不改变退出信号集合，也不把 SIGUSR1/SIGUSR2 重新定义为诊断用途（如日志级别切换、dump 触发）。已全仓确认这两个信号当前无任何调用方，但按「保持各信号原有语义」原样保留；若将来要复用它们，需单独立项。

## Design（技术方案）

### 架构决策：信号线程只通知，主线程是唯一执行者与唯一销毁者

被拒方案与理由（存在显著取舍，故记录）：
- **控制线程直接执行完整 `Server::Stop()`**：控制线程进入 Stop() 后不再 `sigwait`，重复信号只能挂起到 Stop() 返回才被消费，AC-3 无法实现；要实现就得再加一个 `sigtimedwait` 轮询看护线程，且主线程必须先 join 控制线程才能 `server_.reset()`，多一处销毁时序耦合。
- **控制线程只做最小唤醒（提前 `rpc_server_->Shutdown()`）**：唤醒手段会在门禁关闭前先关传输层，新的 leader-only 请求将拿到传输错误而非门禁拒绝码，违反 AC-2 与既有关闭约束。这是 `optimizer/service/online_optimizer_server.cc:189-196` 现有范式不适用于 KVCM 主服务的原因。

采用方案：控制线程只 `sigtimedwait` + 置状态 + `notify`；主线程把阻塞点从 `Server::Wait()` 换成 `ShutdownController::WaitForShutdownRequest()`，被唤醒后自己跑 `Stop()` → `Wait()` → 销毁。可行性依据：gRPC 由自身线程池服务，`rpc_server_->Wait()`（`server.cc:251`）只是栅栏，主线程不阻塞在那里不影响服务；`command_line.cc:138-139` 的启动失败路径本就是 `Stop()` 再 `Wait()` 这个顺序。

### 数据流（关闭时序）

```
启动：Run() → BlockExitSignals()（在任何线程创建前）→ StartDaemon/InitLogger
      → CrashHandler::Install() + SIGPIPE 忽略 → controller.Start()（sigwait 线程）
      → Server Init/Start → controller.WaitForShutdownRequest()   ← 主线程阻塞点

关闭：kill -TERM（落到任意线程，因全局阻塞而成为进程 pending）
      → 控制线程 sigtimedwait 返回 → state: kIdle→kRequested + cv.notify_all（继续循环）
      → 主线程 WaitForShutdownRequest() 返回
      → Server::Stop()（call_once）：leader_elector_->Stop()（含 OnNoLongerLeader 排空/清理）
         → gRPC Shutdown → HTTP Stop → Metrics 线程 Stop
      → Server::Wait()（join HTTP 线程；gRPC Wait 立即返回）
      → controller.Stop()（join sigwait 线程，必须早于 DestroyLogger）
      → server_.reset()（销毁 service/manager，此时其依赖仍有效）→ DestroyLogger → return 0

强退：kRequested 状态下再次收到任一退出信号 → KVCM_LOG_WARN + _exit(128+sig)
```

### 接口与实现改动（精确到文件/字段）

**新增 `kv_cache_manager/common/shutdown_controller.h` / `.cc`**（与 `crash_handler`、`loop_thread` 同层，属进程级基础设施）：
```cpp
class ShutdownController {
public:
    // 仅阻塞掩码，不启线程；必须在任何其它线程创建之前调用。
    static bool BlockExitSignals();          // pthread_sigmask(SIG_BLOCK, {SIGTERM,SIGINT,SIGUSR1,SIGUSR2})
    static const sigset_t &ExitSignalSet();
    bool Start();                            // 启动 sigwait 线程；失败时解除阻塞以恢复默认终止语义（fail-safe）
    bool WaitForShutdownRequest();           // 阻塞至 kRequested；返回是否为信号驱动
    void Stop();                             // 置 stop_ + join，幂等
    bool IsShutdownRequested() const;
    // 测试可观测（配合 -fno-access-control 亦可直接访问私有成员）
    int  ReceivedSignal() const;             // 首个退出信号号
    int  ForceExitSignal() const;            // 触发强退的信号号（fork 子进程用）
private:
    void SignalLoop();                       // sigtimedwait(&set, &info, kPollTimeout)
    enum class State { kIdle, kRequested };
    static constexpr int kPollTimeoutMs = 100;   // Stop() 的 join 延迟上限
    std::thread thread_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    State state_ = State::kIdle;
    std::atomic<bool> stop_{false};
    int received_signal_ = 0;
};
```
`SignalLoop()` 逻辑：`sigtimedwait` 超时（`EAGAIN`）→ 检查 `stop_` 后继续；取到信号且 `state_ == kIdle` → 记录 `received_signal_`、置 `kRequested`、`cv_.notify_all()`、**继续循环**（这是 AC-3 能立即捕获重复信号的关键）；取到信号且 `state_ == kRequested` → `KVCM_LOG_WARN` 后 `_exit(128 + sig)`（与 `crash_handler.cc:367,439` 的 `_exit` 兜底一致，跳过 atexit 与静态析构）。

**`kv_cache_manager/common/BUILD`**：新增 `cc_library(name = "shutdown_controller", srcs = ["shutdown_controller.cc"], hdrs = ["shutdown_controller.h"], deps = [":logger"])`（参照 `crash_handler` 目标，`common/BUILD:96-110`）。

**`kv_cache_manager/service/command_line.h`**：删除 `CommandLine::GetNext()`、`CommandLine::Terminate()`、成员 `next_`；新增成员 `ShutdownController controller_;`；`RegisterSignal()` 更名/收敛为 `InstallSignalPolicy()`（只做 `CrashHandler::Install()` + `signal(SIGPIPE, SIG_IGN)` + `controller_.Start()`）。

**`kv_cache_manager/service/command_line.cc`**：
- 删除 `static std::mutex G_CMD_LOCK`（:15）、`static CommandLine *G_CMD_LIST`（:16）、`Run()` 里的链表注册（:104-108）、`Terminate()`（:148-151）、`__sign_handler__`（:166-180）、`RegisterSignal()` 中四个 `signal()` 注册（:194-197）。
- `Run()` 重排为单一退出路径：`ParseCmdLine` → `config.Parse` → **`ShutdownController::BlockExitSignals()`**（必须早于 `StartDaemon()` 与 `InitLogger()`，因为 alog 会创建 flush 线程）→ `StartDaemon` → `InitLogger/UpdateLogLevel/InitLogLevelFromEnv` → `InstallSignalPolicy()` → 若 `!controller_.IsShutdownRequested()` 才 `new Server` / `Init` / `Start`（启动中收到信号则跳过 Start，直接进入关闭，避免在已请求退出时还跑冗长的 recovery）→ `controller_.WaitForShutdownRequest()` → `server_->Stop(); server_->Wait();` → `controller_.Stop()` → `server_.reset()` → `LoggerBroker::DestroyLogger()` → `return 0`。
- 启动失败路径（今天的 :132-141）保持返回 -1，但必须先 `controller_.Stop()` 再返回，避免 sigwait 线程在 logger 销毁后仍存活；建议用一个 scope guard / 私有 `Finish(int code)` 收敛所有 return 分支。

**`kv_cache_manager/service/server.h` / `server.cc`**：`std::atomic<bool> stop_{false}`（`server.h:60`）替换为 `std::once_flag stop_once_;`；`Server::Stop()`（`server.cc:538-575`）改为 `std::call_once(stop_once_, [this]{ DoStop(); })`，把现有函数体整体移入私有 `DoStop()`，顺序与日志一字不改（参照 `online_optimizer_server.cc:185-187` 的 `call_once` 写法）。`call_once` 天然满足 AC-5：并发调用者等待同一次关闭完成。`Server::Wait()`、`Server::OnNoLongerLeader()` 不改。

**`kv_cache_manager/common/loop_thread.h` / `.cc`**：`Stop()`（`loop_thread.cc:31-44`）在 `thread_.joinable()` 分支前加自连接检测：`thread_.get_id() == std::this_thread::get_id()` 时置 `self_join_detected_ = true` 并 `KVCM_LOG_ERROR("LoopThread[%s] Stop() called from its own loop thread", name_.c_str())`，随后**照常 `join()`**（保持 release 行为不变）。新增 `bool self_join_detected_ = false;` 与 `#include "kv_cache_manager/common/logger.h"`；`common/BUILD` 的 `loop_thread` 目标（:227-237）`deps` 加 `":logger"`。`~LoopThread()`（:9）复用同一逻辑。

### 与现有代码的集成点
- `service/command_line.cc:191-198` `RegisterSignal()`：唯一信号注册点，改造后不再注册退出信号 handler。
- `service/server.cc:538-575` `Stop()`：唯一完整关闭入口；`leader_elector_->Stop()` 仍在最前，保证「撤销选主资格 + 关闭门禁」先于传输层关闭。
- `config/leader_elector.cc:64-101` `Stop()`：`LoseLeaseLocked` 触发 `leadership_lost_handler_`（`server.cc:501-504` 关门禁）→ 停两个 LoopThread → `ReleaseLock()` → join 状态转换线程；join 前状态线程会跑完 `OnNoLongerLeader()`（`server.cc:175-203`）的排空与清理。不改逻辑，改由主线程调用后自连接与循环等待自动消失。
- `common/loop_thread.cc:31-44` `Stop()`：LeaderLock/LeaseCheck/Metrics/GC/Reclaimer 等所有 LoopThread 的 join 点。
- `service/service_impl_base.cc:17-22` `WaitForAllLeaderOnlyRequestsToComplete()`：排空等待点，是循环等待的另一半；改造后调用者恒为主线程（经状态转换线程），不与请求处理线程重合。
- `integration_test/client_test/service_process_controller.h:64-75` `StopService()`：当前是无超时死循环轮询，关闭卡住会永久挂起 CI；本次给它加超时上限并失败时报错。
- 信号来源现状（已全仓核实）：运维与测试侧**只发 SIGTERM**——`integration_test/client_test/service_process_controller.h:67`、`integration_test/testlib/module_base.py:78`（force 时用 SIGKILL）、`client_java/src/integrationTest/.../KvcmServerManager.java:110`（SIGTERM → SIGKILL）、`docs/design/ha_leader_elector.md:175`；SIGINT 只出现在客户端与压测工具（`tools/bench_meta_service/bench_meta_service.cpp:354`、`optimizer/service/online_optimizer_server_main.cc:83`、`py_connector/sglang/test*.py`）；SIGUSR1/SIGUSR2 除 `command_line.cc:194-195` 的注册处外无任何调用方。因此 AC-6 的「退出码 0 且关闭必须真正结束」主要由 SIGTERM 路径承载：端到端测试用 SIGTERM，其余三个信号在单测层用 `pthread_kill` 覆盖语义一致性。

### 错误处理与边界
- **`Start()` 创建线程失败**：记录错误并 `pthread_sigmask(SIG_UNBLOCK, ...)` 解除阻塞，恢复「SIGTERM 直接终止进程」的默认语义，避免进程变成只能被 SIGKILL 杀掉。
- **启动阶段收到信号**：信号在掩码下保持 pending，`controller_.Start()` 后立即被消费；`Run()` 在 `Start()` 前检查 `IsShutdownRequested()`，已请求则跳过 Server 启动直接走关闭并返回 0。
- **信号落在组件线程上**：因全线程阻塞，信号不会在组件线程内被处理，只成为进程 pending 并由控制线程消费；组件线程的阻塞系统调用可能被 `EINTR` 打断，各组件沿用现有错误处理（不引入 `SA_RESTART` 语义变化）。
- **`Stop()` 的 join 延迟**：`sigtimedwait` 轮询上限 100ms，故 `controller_.Stop()` 最多引入 100ms 关闭延迟。
- **强退时的日志**：`controller_.Stop()` 早于 `LoggerBroker::DestroyLogger()`，故控制线程在 `_exit` 前调用 `KVCM_LOG_WARN` 是安全的；这一顺序作为不变量写进代码注释与测试。
- **多实例**：删除 `G_CMD_LIST` 后一个进程只支持一个 `CommandLine` 实例，与 `main.cpp:3-6` 的现状一致。
- **与 CrashHandler 的边界**：`BlockExitSignals()` 只在 `CommandLine::Run()` 中调用，作用域限于 server 进程；`CrashHandler` 接管的 SIGSEGV/SIGABRT/SIGBUS/SIGFPE/SIGILL 不在阻塞集合内，崩溃路径不受影响。注意 `common/test/crash_handler_test.cc:160-168` 依赖「SIGUSR1 默认应终止进程」来验证 CrashHandler 不接管非崩溃信号——该用例是独立子进程、自行 `signal(SIGUSR1, SIG_DFL)` 且不调用 `BlockExitSignals()`，因此现状不受本次改动影响；但新增测试若在同一进程内既阻塞了退出信号又需要靠 `raise()` 立即终止，必须先 `pthread_sigmask(SIG_UNBLOCK, ...)` 再 raise，否则信号会挂起导致用例超时。这条约束写进 `shutdown_controller.h` 的注释与 T2 的测试实现。

## Tasks（任务拆解）

- **T1**: 新增 `common/shutdown_controller.{h,cc}` 与 `common/BUILD` 目标，实现掩码阻塞、`sigtimedwait` 轮询线程、kIdle/kRequested 状态机、重复信号 `_exit(128+sig)`、`Start()` 失败时解除阻塞的 fail-safe。（对应: AC-1, AC-3, AC-5；依赖: 无；完成判据: 目标可编译；`Start()/WaitForShutdownRequest()/Stop()` 具备幂等性；`Stop()` 在无信号时能在 ≤200ms 内返回。）
- **T2**: 新增 `common/test/shutdown_controller_test.cc` 与 `ShutdownControllerTest` bazel 目标（`copts = ["-fno-access-control"]`，参照 `common/test/BUILD:49-62`）。（对应: AC-1, AC-3, AC-7；依赖: T1；完成判据: 覆盖①掩码继承断言②对 SIGTERM/SIGINT/SIGUSR1/SIGUSR2 分别 `pthread_kill` 到指定工作线程后，工作线程未执行任何关闭逻辑且 `WaitForShutdownRequest()` 在超时上限内返回③fork 子进程验证重复信号退出码 == `128+sig`④`Stop()` 幂等；全部等待使用带超时的 `wait_for`，无裸 `wait()`。）
- **T3**: `Server::Stop()` 幂等化：`server.h:60` 的 `stop_` 换成 `std::once_flag stop_once_`，`server.cc:538-575` 改为 `call_once` + 私有 `DoStop()`，关闭顺序与日志不变。（对应: AC-5, AC-2；依赖: 无；完成判据: 两次及并发调用 `Stop()` 只产生一份 "server stopping..." / "kvcm server stopped, goodbye!" 日志；现有 service 测试全绿。）
- **T4**: 改造 `service/command_line.{h,cc}`：删除 `G_CMD_LOCK`/`G_CMD_LIST`/`GetNext`/`Terminate`/`__sign_handler__` 与四个 `signal()` 注册；`Run()` 按 Design 重排（早阻塞、单一退出路径、`WaitForShutdownRequest()` 取代 `server_->Wait()` 作为阻塞点、启动失败与启动中收到信号的分支）。（对应: AC-1, AC-2, AC-6；依赖: T1, T3；完成判据: 二进制启动后 `kill -TERM` 能在超时上限内以退出码 0 结束并打出完整关闭日志序列；`-h`/`-v` 行为不变；无 handler 残留（grep `__sign_handler__` 为空）。）
- **T5**: `LoopThread::Stop()` 自连接检测：加 `self_join_detected_` 标志与 `KVCM_LOG_ERROR`，不跳过 join、不 detach；`loop_thread.cc` 引入 logger 头、`common/BUILD` 的 `loop_thread` 目标 deps 加 `:logger`；扩展 `common/test/loop_thread_test.cc`。（对应: AC-4；依赖: 无；完成判据: 在自身 loop 回调中调用 `Stop()` 时标志置真并抛出/传播 `std::system_error`（用 `EXPECT_THROW` 断言），正常路径标志为假；`LoopThreadTest` 全绿。）
- **T6**: 选主与排空的定向信号回归测试：扩展 `config/test/leader_elector_test.cc`（或新增 `leader_elector_shutdown_test.cc` + BUILD 目标）与 `service/test/service_impl_base_test.cc`。（对应: AC-4, AC-7；依赖: T1, T4, T5；完成判据: 覆盖①信号分别定向投递到 LeaderLock/LeaseCheck/状态转换线程（用 `-fno-access-control` 取 `leader_lock_thread_ptr_->thread_`、`state_transition_thread_->get_id()`），关闭在超时上限内完成、无 `std::system_error`、所有 LoopThread 的 `self_join_detected_` 为假②handler 持 `transition_mutex_` / 恢复进行中 / 降级清理进行中三种 barrier 时序下关闭仍能完成且不跳过清理③线程 X 持有一个 in-flight leader-only 请求时被定向投递信号，主线程关门禁 + 排空，X 完成请求后排空在超时上限内结束。）
- **T7**: 端到端退出测试与 CI 兜底：新增 `service/test/command_line_shutdown_test.cc`（fork 子进程跑最小 Server，父进程 `kill(SIGTERM)` 校验退出码 0 与关闭日志；连续两次 SIGTERM 校验 `128+SIGTERM`），并给 `integration_test/client_test/service_process_controller.h:64-75` 的 `StopService()` 加超时上限。（对应: AC-3, AC-6, AC-7；依赖: T4, T3；完成判据: 两个用例在超时上限内稳定通过、重复运行 20 次无 flaky；`StopService()` 超时时返回失败而非死循环。若单测进程内无法构造最小 Server 配置，则退化为扩展 `integration_test/client_test` 的 SIGTERM 用例，并在 PR 正文说明原因。）
- **T8**: 全量验证：普通 CI 与 ASAN CI 通过，`LeaderElectorTest`/`LeaderElectorReconcileTest`/`ServiceImplBaseTest`/`LoopThreadTest`/`CrashHandlerTest`/新增测试全绿；手工验证 SIGINT/SIGUSR1/SIGUSR2 与 SIGTERM 行为一致、`SIGPIPE` 仍被忽略、崩溃信号仍产出 CrashHandler 日志。（对应: AC-1 ~ AC-7；依赖: T1-T7；完成判据: CI 两条流水线均绿；手工验证记录（信号 → 退出码 → 关键日志行）附在 PR 正文；PR 正文含「Spec 偏差」小节，逐条列出本次所有调整，无偏差则明确写「无」。）

## 执行约定：Spec 可被质疑，偏差必须上报

本 Spec 由需求方与设计者基于对默认分支代码的静态阅读 + 三轮确认写成，**未经编译与运行验证**。设计者可能出错：文中引用的行号可能已 drift，某个时序可能被漏想，某条 Design 决策可能在真实代码里根本行不通。凡此种种都属于 Spec 缺陷，不是执行者的问题。

**总原则**：AC 的意图优先于 Design 的字面描述。两者冲突、或 Design 无法实现时，按 AC 意图执行并上报；不要为了让代码「看起来能跑」而绕过问题。宁可停下来问，也不要让一个隐患被掩盖——本工作项的起因正是前一次改动把退出路径的问题掩盖到了线上。

### 必须停下来反馈并请求确认（不得自行决定）
- Spec 引用的文件/行号/函数签名与真实代码不符，或 Design 描述的调用关系不存在。
- 按 Design 实现会破坏某条 AC 的意图，尤其是：AC-2 的关闭顺序与客户端可见语义、AC-4 的「不掩盖」原则、AC-6 的销毁顺序不变量（控制线程 join 必须早于 logger 销毁）。
- 遇到 Spec 未覆盖的时序、线程等待关系或对象生命周期问题（包括第三方库在全局信号阻塞下的意外行为）。
- 需要削弱或删除任一 AC、缩小 AC-7 的时序矩阵覆盖范围。
- 需要改变退出码语义（0 / `128+sig`）、退出信号集合，或引入 detach、跳过 join、跳过清理、「尽力而为」式关闭。
- 需要把改动面扩大到非目标模块（`CacheManager` 内部线程停止方式、optimizer 服务、新增配置项）。
- 为了让测试通过而需要弱化断言、放宽超时上限、跳过用例，或用 sleep 掩盖 flaky。

### 允许结合实际情况自行调整（无需事先批准，但必须事后完整告知）
- 命名、文件位置、私有成员布局、日志文案、注释措辞。
- 等价技术替换：`sigtimedwait` 轮询间隔的具体取值、`std::call_once` 与等价幂等原语、测试目标名与用例组织方式、`-fno-access-control` 之外的等价私有成员访问手段。
- 为达成 AC 所必需的最小补充改动（例如发现还有别的组件线程需要纳入自连接检测）。
- 在依赖关系允许的前提下重排、合并或拆分 T1-T8，以及调整单个任务的提交粒度。
- T7 已预设的退化路径：若单测进程内无法构造最小 Server 配置，可退化为扩展 `integration_test/client_test` 的 SIGTERM 用例，但必须在 PR 正文说明原因。

### 告知方式（保证可追溯）
- 所有偏差写进 PR 正文的「Spec 偏差」小节，每条按 `偏差项 → Spec 原文位置（AC-x / Design 段 / T-y）→ 真实情况 → 采取的做法 → 影响面` 记录；无偏差则明确写「无」。
- 属于「必须停下来确认」的阻断性问题，在工作项评论中提出并等待需求方回复后再继续，不得先合入再补说明。
