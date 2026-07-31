#include "kv_cache_manager/common/crash_handler.h"

#include <atomic>
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "kv_cache_manager/common/build_version.h"

namespace kv_cache_manager {

namespace {

constexpr int kMaxStackFrames = 128;
constexpr size_t kAltStackSize = 64 * 1024;

// 默认与 alog 配置（package/etc/default_logger_config.conf）使用同一个 logs/
// 相对目录，使运维只需要保证 cwd 下有 logs/ 即可同时收集 alog 输出和崩溃栈。
constexpr const char *kDefaultLogDir = "logs";
constexpr const char *kCrashLogFileName = "std_stack_trace.log";

constexpr int kCrashSignals[] = {SIGSEGV, SIGABRT, SIGBUS, SIGFPE, SIGILL};
constexpr size_t kNumSignals = sizeof(kCrashSignals) / sizeof(kCrashSignals[0]);

std::atomic<bool> g_installed{false};
std::atomic_flag g_in_handler = ATOMIC_FLAG_INIT;
int g_log_fd = -1;
char *g_alt_stack = nullptr;
struct sigaction g_old_actions[kNumSignals];

// 区分两类 handler 重入：
//   - 跨线程：并发崩溃，g_in_handler 拦住第 N 个，pause 等第一个刷完
//   - 同线程：user handler 内部又触发了一个 crash signal（典型：abort()
//     家族 idiom 不 reset disposition 直接 raise(SIGABRT)，因为 SIGABRT
//     还是我们接管，会再次落进 KvcmCrashSigaction）。这种情况绝不能去
//     pause —— 第一个"进入者"就是当前线程自己，没有别的线程会刷完，
//     进程直接 hang 死。
// 用 thread_local 标记进入态；同线程重入直接走 fail-safe 退出。
// thread_local 在已 init 完的线程上访问是 TLS slot 直读，async-signal-
// safe（首次访问可能调 __tls_get_addr，但 handler 装在已运行的线程上，
// 之前 C++ runtime 至少 access 过 TLS，slot 已经分配）。
thread_local bool t_in_handler = false;

// Per-thread alt stack 注册管理：用 pthread_key + destructor 保证 worker
// 线程退出时自动 free 分配的 alt stack，不依赖业务方手动清理。
pthread_key_t g_alt_stack_key;
pthread_once_t g_alt_stack_key_once = PTHREAD_ONCE_INIT;

void FreeThreadAltStack(void *p) {
    if (p != nullptr) {
        ::free(p);
    }
}

void CreateAltStackKey() { ::pthread_key_create(&g_alt_stack_key, FreeThreadAltStack); }

size_t WriteInt(char *buf, int64_t v) {
    size_t off = 0;
    uint64_t u;
    if (v < 0) {
        buf[off++] = '-';
        u = static_cast<uint64_t>(-(v + 1)) + 1;
    } else {
        u = static_cast<uint64_t>(v);
    }
    if (u == 0) {
        buf[off++] = '0';
        return off;
    }
    char tmp[24];
    size_t n = 0;
    while (u > 0 && n < sizeof(tmp)) {
        tmp[n++] = static_cast<char>('0' + (u % 10));
        u /= 10;
    }
    for (size_t i = 0; i < n; ++i) {
        buf[off + i] = tmp[n - 1 - i];
    }
    return off + n;
}

size_t WriteUint(char *buf, uint64_t v) {
    if (v == 0) {
        buf[0] = '0';
        return 1;
    }
    char tmp[24];
    size_t n = 0;
    while (v > 0 && n < sizeof(tmp)) {
        tmp[n++] = static_cast<char>('0' + (v % 10));
        v /= 10;
    }
    for (size_t i = 0; i < n; ++i) {
        buf[i] = tmp[n - 1 - i];
    }
    return n;
}

size_t WriteHex(char *buf, uint64_t v) {
    buf[0] = '0';
    buf[1] = 'x';
    if (v == 0) {
        buf[2] = '0';
        return 3;
    }
    char tmp[16];
    size_t n = 0;
    while (v > 0 && n < sizeof(tmp)) {
        int d = static_cast<int>(v & 0xf);
        tmp[n++] = static_cast<char>(d < 10 ? '0' + d : 'a' + d - 10);
        v >>= 4;
    }
    for (size_t i = 0; i < n; ++i) {
        buf[2 + i] = tmp[n - 1 - i];
    }
    return n + 2;
}

void WriteAll(int fd, const char *buf, size_t len) {
    if (fd < 0) {
        return;
    }
    while (len > 0) {
        ssize_t n = ::write(fd, buf, len);
        if (n > 0) {
            buf += n;
            len -= static_cast<size_t>(n);
        } else if (n == -1 && errno == EINTR) {
            continue;
        } else {
            break;
        }
    }
}

void WriteStr(int fd, const char *s) { WriteAll(fd, s, ::strlen(s)); }

const char *SignalName(int sig) {
    switch (sig) {
    case SIGSEGV:
        return "SIGSEGV";
    case SIGABRT:
        return "SIGABRT";
    case SIGBUS:
        return "SIGBUS";
    case SIGFPE:
        return "SIGFPE";
    case SIGILL:
        return "SIGILL";
    default:
        return "UNKNOWN";
    }
}

// 返回 si_code 的文字描述。优先匹配信号特定 code，再匹配通用 code。
const char *SignalCodeName(int sig, int code) {
    // 通用 code（kernel/headers 定义）
    switch (code) {
    case SI_USER:
        return "SI_USER";
    case SI_KERNEL:
        return "SI_KERNEL";
    case SI_QUEUE:
        return "SI_QUEUE";
    case SI_TIMER:
        return "SI_TIMER";
    case SI_MESGQ:
        return "SI_MESGQ";
    case SI_ASYNCIO:
        return "SI_ASYNCIO";
    case SI_TKILL:
        return "SI_TKILL";
    default:
        break;
    }
    if (sig == SIGSEGV) {
        switch (code) {
        case SEGV_MAPERR:
            return "SEGV_MAPERR (address not mapped)";
        case SEGV_ACCERR:
            return "SEGV_ACCERR (invalid permissions)";
#ifdef SEGV_BNDERR
        case SEGV_BNDERR:
            return "SEGV_BNDERR (bounds check failed)";
#endif
#ifdef SEGV_PKUERR
        case SEGV_PKUERR:
            return "SEGV_PKUERR (protection key)";
#endif
        default:
            break;
        }
    } else if (sig == SIGBUS) {
        switch (code) {
        case BUS_ADRALN:
            return "BUS_ADRALN (invalid address alignment)";
        case BUS_ADRERR:
            return "BUS_ADRERR (non-existent physical address)";
        case BUS_OBJERR:
            return "BUS_OBJERR (object-specific hw error)";
        default:
            break;
        }
    } else if (sig == SIGFPE) {
        switch (code) {
        case FPE_INTDIV:
            return "FPE_INTDIV (integer divide by zero)";
        case FPE_INTOVF:
            return "FPE_INTOVF (integer overflow)";
        case FPE_FLTDIV:
            return "FPE_FLTDIV (float divide by zero)";
        case FPE_FLTOVF:
            return "FPE_FLTOVF (float overflow)";
        case FPE_FLTUND:
            return "FPE_FLTUND (float underflow)";
        case FPE_FLTRES:
            return "FPE_FLTRES (float inexact result)";
        case FPE_FLTINV:
            return "FPE_FLTINV (float invalid op)";
        case FPE_FLTSUB:
            return "FPE_FLTSUB (subscript out of range)";
        default:
            break;
        }
    } else if (sig == SIGILL) {
        switch (code) {
        case ILL_ILLOPC:
            return "ILL_ILLOPC (illegal opcode)";
        case ILL_ILLOPN:
            return "ILL_ILLOPN (illegal operand)";
        case ILL_ILLADR:
            return "ILL_ILLADR (illegal addressing mode)";
        case ILL_ILLTRP:
            return "ILL_ILLTRP (illegal trap)";
        case ILL_PRVOPC:
            return "ILL_PRVOPC (privileged opcode)";
        case ILL_PRVREG:
            return "ILL_PRVREG (privileged register)";
        case ILL_COPROC:
            return "ILL_COPROC (coprocessor error)";
        case ILL_BADSTK:
            return "ILL_BADSTK (internal stack error)";
        default:
            break;
        }
    }
    return "UNKNOWN_CODE";
}

// 写入除堆栈外的所有头部字段（signal/version/si_code/time）。
// frames/depth 由调用方在 handler 入口处一次性捕获，再分别写到 stderr 和
// log 文件，避免在 handler 内多次调 ::backtrace —— 它会走
// dl_iterate_phdr 的 glibc 全局锁，重复持有会扩大死锁窗口，且两次调用
// 在极端优化路径下结果可能不一致。
void DumpToFd(int fd, int sig, siginfo_t *info, void *const *frames, int depth) {
    char buf[64];
    size_t n;

    WriteStr(fd, "*** KVCM CrashHandler caught signal ");
    n = WriteUint(buf, static_cast<uint64_t>(sig));
    WriteAll(fd, buf, n);
    WriteStr(fd, " (");
    WriteStr(fd, SignalName(sig));
    WriteStr(fd, ") at addr ");
    void *addr = info ? info->si_addr : nullptr;
    n = WriteHex(buf, reinterpret_cast<uint64_t>(addr));
    WriteAll(fd, buf, n);
    WriteStr(fd, ", pid ");
    n = WriteUint(buf, static_cast<uint64_t>(::getpid()));
    WriteAll(fd, buf, n);
    WriteStr(fd, ", tid ");
    n = WriteUint(buf, static_cast<uint64_t>(::syscall(SYS_gettid)));
    WriteAll(fd, buf, n);
    WriteStr(fd, " ***\n");

    WriteStr(fd, "version: ");
    WriteStr(fd, kKvcmFullVersion);
    WriteStr(fd, " (commit ");
    WriteStr(fd, kKvcmGitCommit);
    WriteStr(fd, ", built ");
    WriteStr(fd, kKvcmBuildTime);
    WriteStr(fd, ")\n");

    if (info) {
        WriteStr(fd, "si_code: ");
        n = WriteInt(buf, static_cast<int64_t>(info->si_code));
        WriteAll(fd, buf, n);
        WriteStr(fd, " ");
        WriteStr(fd, SignalCodeName(sig, info->si_code));
        WriteStr(fd, "\n");
    }

    struct timespec ts;
    if (::clock_gettime(CLOCK_REALTIME, &ts) == 0) {
        WriteStr(fd, "time(epoch_sec): ");
        n = WriteUint(buf, static_cast<uint64_t>(ts.tv_sec));
        WriteAll(fd, buf, n);
        WriteStr(fd, "\n");
    }

    WriteStr(fd, "Stack trace (");
    n = WriteUint(buf, static_cast<uint64_t>(depth));
    WriteAll(fd, buf, n);
    WriteStr(fd, " frames, demangle with bin/decode_stack_trace.sh):\n");
    // 逐帧调 backtrace_symbols_fd(.., 1, ..)，中间插 "#N " 前缀。
    // backtrace_symbols_fd 文档保证不调 malloc，async-signal-safe；
    // size=1 是合法用法（参考 glibc 实现：循环里就是逐帧 dladdr+write）。
    for (int i = 0; i < depth; ++i) {
        char prefix[16];
        size_t plen = 0;
        prefix[plen++] = '#';
        plen += WriteUint(prefix + plen, static_cast<uint64_t>(i));
        prefix[plen++] = ' ';
        WriteAll(fd, prefix, plen);
        ::backtrace_symbols_fd(frames + i, 1, fd);
    }
    WriteStr(fd, "*** End of stack trace ***\n");
}

int FindSignalIndex(int sig) {
    for (size_t i = 0; i < kNumSignals; ++i) {
        if (kCrashSignals[i] == sig) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// 调 user 旧 handler / 走 SIG_DFL（让内核默认行为接管 → terminate +
// coredump）路径前，先解除当前线程对全部 crash signal 的阻塞。POSIX 在
// handler 执行期间会把 sa_mask 里的信号加进当前线程 signal mask；我们
// Install() 时把 5 个 crash signal 都加进了 mask。
//
// 必须 unblock 全部 5 个、而不是只 unblock 当前 sig —— user handler 可
// 能用其他 crash signal 终止进程：例如 SIGSEGV handler 内部 abort()（=
// raise(SIGABRT)），或 sanitizer 在 SIGSEGV handler 里 raise(SIGABRT)
// 走自己的崩溃报告路径。如果只 unblock 当前 sig，其他 4 个 crash signal
// 仍被阻塞，raise 只 pending，user handler return 后被 fail-safe _exit
// 抢先终止，coredump / WIFSIGNALED 状态丢失。其他非 crash signal 的
// mask 不动。
void UnblockAllCrashSignalsInThread() {
    sigset_t unblock;
    ::sigemptyset(&unblock);
    for (int s : kCrashSignals) {
        ::sigaddset(&unblock, s);
    }
    ::pthread_sigmask(SIG_UNBLOCK, &unblock, nullptr);
}

extern "C" void KvcmCrashSigaction(int sig, siginfo_t *info, void *ucontext) {
    int saved_errno = errno;

    if (t_in_handler) {
        // 同一线程在 handler 里又触发了 crash signal（user handler 内部
        // raise 另一个 crash signal 又被我们接管）—— 不能进 pause()，没有
        // 别的线程来打破等待，进程会 hang。直接走 fail-safe 退出。
        ::_exit(128 + sig);
    }
    if (g_in_handler.test_and_set()) {
        // 已有另一线程在处理：阻塞当前线程，等第一个线程把堆栈刷完后
        // 再 raise 让整个进程退出，避免抢着 reraise 把第一个线程的
        // 输出截断（pause 是 async-signal-safe）。
        errno = saved_errno;
        while (true) {
            ::pause();
        }
    }
    t_in_handler = true;

    // 一次性捕获堆栈，避免在 handler 内重复持 dl_iterate_phdr 的 glibc 全局锁；
    // 同一份 frames 同步写到 stderr 和 log 文件，保证两边输出严格一致。
    void *frames[kMaxStackFrames];
    int depth = ::backtrace(frames, kMaxStackFrames);

    DumpToFd(STDERR_FILENO, sig, info, frames, depth);
    if (g_log_fd >= 0) {
        DumpToFd(g_log_fd, sig, info, frames, depth);
        ::fsync(g_log_fd);
    }

    // 链式调用旧 handler：
    //   1) SA_SIGINFO + 用户 handler  → 直接调，透传原始 (sig, info, ucontext)，
    //      让 ASAN / sanitizer / breakpad 等看到真实崩溃上下文，而不是
    //      raise(sig) 合成的 SI_TKILL 假上下文。
    //   2) sa_handler 是 SIG_DFL       → sigaction 还原后 raise，让内核走
    //      默认行为（生成 coredump）。
    //   3) sa_handler 是用户 handler   → 直接调（SI_USER 合成上下文，无
    //      办法回放原始 siginfo，但至少把崩溃通知给上层）。
    //   4) sa_handler 是 SIG_IGN       → 跳过，走兜底退出，不能让进程
    //      继续跑（synchronous fault 信号被 ignore 后会反复触发）。
    //
    // 所有"会进入 user 代码"的路径（1/2/3 + 末尾 fallback）都先 unblock 全
    // 部 5 个 crash signal。Install() 的 sa_mask 把它们都阻塞了；user
    // handler 内部经典 idiom "signal(sig, SIG_DFL); raise(sig);" 走 default
    // disposition 拿 coredump 要 unblock 当前 sig，而 user handler 也可能
    // 用其他 crash signal 终止进程（比如内部 abort() = raise(SIGABRT)），
    // 所以 unblock 整个 set 而不是只 unblock 当前 sig，否则 raise 只 pending、
    // 被 fail-safe _exit 抢先终止，coredump / WIFSIGNALED 状态丢失（参见
    // DefaultDispositionCausesSignalTermination /
    // ChainedUserHandlerCanReachDefaultDisposition /
    // ChainedHandlerCanTerminateViaDifferentCrashSignal）。
    // SIG_IGN 不需要 unblock —— 它直接跳到 fail-safe。
    int idx = FindSignalIndex(sig);
    if (idx >= 0) {
        const struct sigaction &old = g_old_actions[idx];
        errno = saved_errno;
        if ((old.sa_flags & SA_SIGINFO) != 0 && old.sa_sigaction != nullptr) {
            UnblockAllCrashSignalsInThread();
            old.sa_sigaction(sig, info, ucontext);
        } else if (old.sa_handler == SIG_DFL) {
            ::sigaction(sig, &old, nullptr);
            UnblockAllCrashSignalsInThread();
            ::raise(sig);
        } else if (old.sa_handler != SIG_IGN && old.sa_handler != nullptr) {
            UnblockAllCrashSignalsInThread();
            old.sa_handler(sig);
        }
    } else {
        ::signal(sig, SIG_DFL);
        errno = saved_errno;
        UnblockAllCrashSignalsInThread();
        ::raise(sig);
    }

    // 兜底：所有路径都没让进程退出（user handler return / SIG_IGN /
    // raise 被 ignore 等），强制终止。否则 g_in_handler 永久 set，后续
    // 任何崩溃都会卡死在上面的 pause() loop 里。退出码沿用 shell 惯例
    // 128 + signo。
    ::_exit(128 + sig);
}

} // namespace

void CrashHandler::Install(const std::string &log_dir) {
    bool expected = false;
    if (!g_installed.compare_exchange_strong(expected, true)) {
        return;
    }

    void *dummy[4];
    (void)::backtrace(dummy, 4);

    // 给主线程装 alt stack —— 如果别人（ASan/TSan/breakpad 等 crash
    // reporter，或用户自己）已经装过，跳过：覆盖会让对方在进程退出时
    // 按它记的旧地址 munmap 我们的 new 内存，触发 sanitizer 内部 CHECK
    // 失败 abort 进程。逻辑和 InstallAltStackForCurrentThread() 一致。
    // 保留 g_alt_stack = nullptr 表示我们没有分配，将来若有清理逻辑也
    // 不会去 delete 一块属于别人的内存。
    stack_t cur;
    if (::sigaltstack(nullptr, &cur) != 0 || (cur.ss_flags & SS_DISABLE) != 0) {
        g_alt_stack = new char[kAltStackSize];
        stack_t ss;
        ::memset(&ss, 0, sizeof(ss));
        ss.ss_sp = g_alt_stack;
        ss.ss_size = kAltStackSize;
        ss.ss_flags = 0;
        ::sigaltstack(&ss, nullptr);
    }

    // log_dir 为空时使用与 alog 一致的默认目录。stderr 始终打印一份；这里
    // 额外把 fd 准备好，handler 触发时同时落盘（如果 open 失败会优雅降级
    // 为只写 stderr）。
    {
        std::string dir = log_dir.empty() ? std::string(kDefaultLogDir) : log_dir;
        if (::mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
            // mkdir 失败（权限/磁盘等）：保留 g_log_fd = -1，handler 仅写 stderr
        } else {
            std::string path = dir;
            if (path.back() != '/') {
                path += '/';
            }
            path += kCrashLogFileName;
            g_log_fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        }
    }

    struct sigaction act;
    ::memset(&act, 0, sizeof(act));
    act.sa_sigaction = &KvcmCrashSigaction;
    ::sigemptyset(&act.sa_mask);
    for (int s : kCrashSignals) {
        ::sigaddset(&act.sa_mask, s);
    }
    // 不使用 SA_RESETHAND：在多线程场景下，SA_RESETHAND 会在第一个线程进入
    // handler 时立刻把 SIGSEGV 的进程级 disposition 重置为 SIG_DFL。其他线程
    // 同期 raise(SIGSEGV) 会让整个进程被默认行为 kill，第一个线程的输出会被
    // 截断。改为依赖 g_in_handler + pause() 拦住二次进入的线程，让第一个线程
    // 完整打完后再恢复 oldact 并 reraise。
    act.sa_flags = SA_SIGINFO | SA_ONSTACK;

    for (size_t i = 0; i < kNumSignals; ++i) {
        ::sigaction(kCrashSignals[i], &act, &g_old_actions[i]);
    }
}

bool CrashHandler::IsInstalled() { return g_installed.load(); }

void CrashHandler::InstallAltStackForCurrentThread() {
    ::pthread_once(&g_alt_stack_key_once, CreateAltStackKey);

    // 自己之前装过 → 幂等，直接返回。
    if (::pthread_getspecific(g_alt_stack_key) != nullptr) {
        return;
    }

    // 别人已经装了（ASan/TSan 在线程创建时会自动装 SIGSTKSZ*4 大小的；用
    // 户自己也可能装）→ 不要覆盖。sanitizer 在线程退出时会按自己的地址
    // munmap，被我们替换后会触发它内部 CHECK 失败把进程 abort，反不如
    // 当作 no-op，让别人继续管理。
    stack_t cur;
    if (::sigaltstack(nullptr, &cur) == 0 && (cur.ss_flags & SS_DISABLE) == 0) {
        return;
    }

    void *sp = ::malloc(kAltStackSize);
    if (sp == nullptr) {
        return; // 分配失败：worker 失去栈溢出保护，但其他场景的 handler 仍可用
    }
    // 用 pthread_setspecific 托管，线程退出时 destructor 自动 free。
    // 即使后面的 sigaltstack 失败也要登记，避免下次重复分配。
    ::pthread_setspecific(g_alt_stack_key, sp);

    stack_t ss;
    ::memset(&ss, 0, sizeof(ss));
    ss.ss_sp = sp;
    ss.ss_size = kAltStackSize;
    ss.ss_flags = 0;
    ::sigaltstack(&ss, nullptr);
}

} // namespace kv_cache_manager
