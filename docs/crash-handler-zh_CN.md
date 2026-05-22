# Crash Stack Trace

KVCacheManager 在 `CommandLine::RegisterSignal` 中默认安装了一个
async-signal-safe 的 crash handler，用于在进程被以下信号终止时，
**在 coredump 之外额外**保留一份可读的堆栈：

`SIGSEGV` / `SIGABRT` / `SIGBUS` / `SIGFPE` / `SIGILL`

handler 不会替代 coredump；它在打完堆栈后把信号 reraise 给原 handler
（默认 → 内核生成 coredump；ASAN 拉起时 → ASAN 接管）。所以同时拥有
coredump 时，handler 输出只是辅助；coredump 因 ulimit / `core_pattern`
丢失时，handler 输出就是唯一线索。

## 输出在哪里

每次崩溃同时写两份：

| 路径 | 备注 |
|---|---|
| `stderr` | 始终输出。容器/systemd 会自动收集 |
| `logs/std_stack_trace.log` | 默认相对 cwd 的 `logs/`，与 alog 配置同目录；`O_APPEND` 追加，多次崩溃叠加 |

`logs/` 目录创建失败（权限、只读 fs）时优雅降级为只写 stderr，**不会**
阻塞 server 启动。

## 输出样例

```
*** KVCM CrashHandler caught signal 11 (SIGSEGV) at addr 0x..., pid 1234, tid 1234 ***
version: 0.0.1+20260518.9bfcf9d5 (commit 9bfcf9d5, built 2026-05-18 10:30:09)
si_code: 1 SEGV_MAPERR (address not mapped)
time(epoch_sec): 1779085225
Stack trace (12 frames, demangle with bin/decode_stack_trace.sh):
#0 ./bin/kv_cache_manager_bin(_ZN3foo3barEv+0x1f)[0x55d...]
#1 ./bin/kv_cache_manager_bin(_ZN3xyz4baz7processEPv+0x83)[0x55d...]
...
*** End of stack trace ***
```

字段含义：

| 字段 | 含义 |
|---|---|
| `signal N (NAME)` | POSIX 信号编号与名称 |
| `addr` | `siginfo_t::si_addr` —— 触发信号的访问地址（SIGSEGV 时即非法访问的地址） |
| `pid` / `tid` | 进程 ID / 触发崩溃线程的 LWP ID |
| `version` | binary 的编译版本与 git commit，用于事后定位代码版本 |
| `si_code` | 进一步细化的信号原因。常见取值：`SEGV_MAPERR`（访问未映射地址）、`SEGV_ACCERR`（权限错）、`BUS_ADRALN`（对齐错误）、`SI_USER`/`SI_TKILL`（用户态 raise/tkill）等 |
| `time(epoch_sec)` | 崩溃时刻 UNIX 时间戳，便于和上下游日志对齐 |
| `#N module(symbol+0xOFF)[0xABS]` | 第 N 帧。`module` 是二进制路径，`symbol+0xOFF` 是符号名+符号内偏移，`[0xABS]` 是运行时绝对地址 |

## 解析堆栈：bin/decode_stack_trace.sh

打包内置了一个离线解析脚本，用 `c++filt` demangle、`addr2line` 解
file:line（best-effort）：

```bash
# 默认从同机的 binary 路径解析（适合开发机）
bin/decode_stack_trace.sh logs/std_stack_trace.log

# 跨机分析：日志在 A 机产生，binary/动态库在 B 机分析
bin/decode_stack_trace.sh \
    -b /opt/kvcm/bin/kv_cache_manager_bin \
    -l /opt/kvcm/lib \
    crash.log

# 也可以从 stdin 读
cat logs/std_stack_trace.log | bin/decode_stack_trace.sh
```

输出形如：

```
Stack trace (12 frames, demangle with bin/decode_stack_trace.sh):
  #0   kv_cache_manager::foo::bar()  at kv_cache_manager/foo/bar.cc:42  (kv_cache_manager_bin +0x1f)
  #1   kv_cache_manager::xyz::baz::process(void*)  at kv_cache_manager/xyz/baz.cc:117  (kv_cache_manager_bin +0x83)
  ...
```

参数：

| 参数 | 用途 |
|---|---|
| `-b BINARY` | 指定主 binary 路径，覆盖日志中嵌入的路径（跨机分析常用） |
| `-l LIB_DIR` | 指定 .so 查找目录，按 basename 兜底 |
| `-h` | 帮助 |

依赖（不存在则优雅降级，不会失败）：`c++filt`、`addr2line`、`nm`。
绝大多数 Linux 发行版的 `binutils` / `gcc` 默认包就有。

## 与 coredump 的对比

| | CrashHandler 输出 | Linux coredump |
|---|---|---|
| 谁产生 | 当前进程（`backtrace_symbols_fd`） | 内核（受 ulimit -c / core_pattern 控制） |
| 格式 | 文本 | ELF core file |
| 容量 | KB 级 | 进程内存大小（GB 级） |
| 信息 | 一份调用栈 | 全部线程寄存器、栈、堆、mmap | 
| 工具 | 文本工具 + decode_stack_trace.sh | `gdb ./binary core` |

两者互补：CrashHandler 解决"文本能看到、能 grep、能告警"；coredump
解决"全栈、所有变量、可调试"。

## Worker 线程的栈溢出保护

`sigaltstack` 是**线程级** attribute：`Install()` 只给调用线程（通常是
主线程）装了 alternate signal stack。`pthread_create` 出来的 worker
线程默认**没有** alt stack。SIGSEGV 等 crash signal 注册时带了
`SA_ONSTACK`，kernel 在投递时如果当前线程没 alt stack 会 fallback
到当前线程自己的 stack —— 对绝大多数 SIGSEGV（nullptr 解引用、UAF
等）没影响，**只有 worker 线程发生栈溢出**时这个 fallback 是坏的：
handler 在已经爆掉的 stack 上跑不起来，堆栈输出会丢失。

如果你的 worker 线程有真正的栈溢出风险（深递归、超大 `alloca`/VLA
等），在线程入口调一次 `CrashHandler::InstallAltStackForCurrentThread()`：

```cpp
#include "kv_cache_manager/common/crash_handler.h"

std::thread worker([] {
    kv_cache_manager::CrashHandler::InstallAltStackForCurrentThread();
    // ... 业务逻辑 ...
});
```

特性：
- **幂等**：多次调用不会重复分配
- **自动释放**：每线程 ~64 KB，通过 `pthread_key` destructor 在线程退出时
  自动 `free`，不需要手动清理
- **失败 fail-safe**：分配失败时静默返回，worker 失去栈溢出保护，但
  线程本身和其他场景的 handler 都不受影响

栈溢出之外的崩溃**不需要**这个 API —— worker 自己的 stack 本身没坏，
handler 在 worker stack 上正常跑。仅在你确实关心栈溢出时再调。

## 配置

无需配置，自动启用。不存在 enable/disable 开关——这是兜底机制，
不应被关闭。如果对产出文件位置有特殊要求，可在调用处覆盖
`CrashHandler::Install("/your/log/dir")`，但默认值（`logs/`）已经和
alog 配置一致，无需额外改动。
