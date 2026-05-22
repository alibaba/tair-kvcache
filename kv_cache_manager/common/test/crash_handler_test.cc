#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <signal.h>
#include <sstream>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "kv_cache_manager/common/build_version.h"
#include "kv_cache_manager/common/crash_handler.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

namespace {

// 把 child_pipe_write 重定向到 stderr，确保 handler 写 stderr 时被父进程收到
void RedirectStderrTo(int fd) {
    dup2(fd, STDERR_FILENO);
    close(fd);
}

std::string ReadAllFromFd(int fd) {
    std::string out;
    char buf[4096];
    while (true) {
        ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n > 0) {
            out.append(buf, buf + n);
        } else if (n == 0) {
            break;
        } else if (errno == EINTR) {
            continue;
        } else {
            break;
        }
    }
    return out;
}

std::string SlurpFile(const std::string &path) {
    std::ifstream ifs(path);
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

struct ChildResult {
    int wstatus = 0;
    std::string stderr_output;
};

// 在子进程跑 body()，把它的 stderr 抓出来；body 中可以触发崩溃
template <typename F>
ChildResult RunInChild(F &&body) {
    int pipefd[2];
    if (::pipe(pipefd) != 0) {
        ADD_FAILURE() << "pipe() failed: " << strerror(errno);
        return {};
    }
    pid_t pid = ::fork();
    if (pid == 0) {
        ::close(pipefd[0]);
        RedirectStderrTo(pipefd[1]);
        body();
        // body 应该已经触发了崩溃，正常不会到这里
        _exit(0);
    }
    ::close(pipefd[1]);
    std::string out = ReadAllFromFd(pipefd[0]);
    ::close(pipefd[0]);
    int wstatus = 0;
    ::waitpid(pid, &wstatus, 0);
    return {wstatus, out};
}

bool ChildAbnormal(int wstatus) {
    if (WIFSIGNALED(wstatus)) {
        return true;
    }
    // ASAN 模式下子进程可能 _exit(非 0)
    if (WIFEXITED(wstatus) && WEXITSTATUS(wstatus) != 0) {
        return true;
    }
    return false;
}

} // namespace

class CrashHandlerTest : public TESTBASE {};

TEST_F(CrashHandlerTest, InstalledFlagBecomesTrue) {
    auto r = RunInChild([] {
        if (CrashHandler::IsInstalled()) {
            _exit(2); // 安装前不应为 true
        }
        CrashHandler::Install("");
        if (!CrashHandler::IsInstalled()) {
            _exit(3); // 安装后必须为 true
        }
        _exit(0);
    });
    ASSERT_TRUE(WIFEXITED(r.wstatus)) << "child terminated by signal";
    ASSERT_EQ(0, WEXITSTATUS(r.wstatus)) << "child exit=" << WEXITSTATUS(r.wstatus);
}

TEST_F(CrashHandlerTest, HandlesSigsegvWritesToStderr) {
    auto r = RunInChild([] {
        CrashHandler::Install("");
        // 主动触发：raise(SIGSEGV) 比解引用 nullptr 在 ASAN 下更可控
        ::raise(SIGSEGV);
        _exit(0);
    });
    EXPECT_TRUE(ChildAbnormal(r.wstatus)) << "child exited normally, wstatus=" << r.wstatus;
    EXPECT_NE(std::string::npos, r.stderr_output.find("KVCM CrashHandler")) << "no header in stderr:\n"
                                                                            << r.stderr_output;
    EXPECT_NE(std::string::npos, r.stderr_output.find("SIGSEGV")) << "no SIGSEGV in stderr:\n" << r.stderr_output;
    // backtrace_symbols_fd 输出形如 "./bin(func+0x..)[0x..]"，至少应包含 "0x"
    EXPECT_NE(std::string::npos, r.stderr_output.find("0x")) << "no stack frames in stderr:\n" << r.stderr_output;
}

TEST_F(CrashHandlerTest, HandlesSigabrt) {
    auto r = RunInChild([] {
        CrashHandler::Install("");
        ::raise(SIGABRT);
        _exit(0);
    });
    EXPECT_TRUE(ChildAbnormal(r.wstatus));
    EXPECT_NE(std::string::npos, r.stderr_output.find("SIGABRT")) << "stderr:\n" << r.stderr_output;
}

TEST_F(CrashHandlerTest, WritesToCrashLogFile) {
    std::string log_dir = GetPrivateTestRuntimeDataPath();
    std::string log_path = log_dir + "std_stack_trace.log";
    ::unlink(log_path.c_str());

    auto r = RunInChild([log_dir] {
        CrashHandler::Install(log_dir);
        ::raise(SIGSEGV);
        _exit(0);
    });
    EXPECT_TRUE(ChildAbnormal(r.wstatus));
    struct stat st;
    ASSERT_EQ(0, ::stat(log_path.c_str(), &st)) << "crash.log not found at " << log_path;
    ASSERT_GT(st.st_size, 0) << "crash.log is empty";
    std::string content = SlurpFile(log_path);
    EXPECT_NE(std::string::npos, content.find("SIGSEGV")) << "log content:\n" << content;
    EXPECT_NE(std::string::npos, content.find("KVCM CrashHandler")) << "log content:\n" << content;
}

TEST_F(CrashHandlerTest, NormalSignalsUnaffected) {
    // SIGINT/SIGTERM 等非 crash 信号不应被 CrashHandler 接管：默认 SIGUSR1 让进程死掉
    auto r = RunInChild([] {
        CrashHandler::Install("");
        ::signal(SIGUSR1, SIG_DFL);
        ::raise(SIGUSR1);
        _exit(0);
    });
    ASSERT_TRUE(WIFSIGNALED(r.wstatus));
    EXPECT_EQ(SIGUSR1, WTERMSIG(r.wstatus));
    // CrashHandler 不应该被触发
    EXPECT_EQ(std::string::npos, r.stderr_output.find("KVCM CrashHandler"))
        << "stderr unexpectedly contains crash header:\n"
        << r.stderr_output;
}

class CrashSignalParamTest : public TESTBASE, public ::testing::WithParamInterface<int> {};

TEST_P(CrashSignalParamTest, HandlesSignal) {
    int sig = GetParam();
    auto r = RunInChild([sig] {
        CrashHandler::Install("");
        ::raise(sig);
        _exit(0);
    });
    EXPECT_TRUE(ChildAbnormal(r.wstatus)) << "sig=" << sig << " wstatus=" << r.wstatus;
    EXPECT_NE(std::string::npos, r.stderr_output.find("KVCM CrashHandler")) << "sig=" << sig << " stderr:\n"
                                                                            << r.stderr_output;
}

INSTANTIATE_TEST_SUITE_P(AllCrashSignals,
                         CrashSignalParamTest,
                         ::testing::Values(SIGSEGV, SIGABRT, SIGBUS, SIGFPE, SIGILL));

TEST_F(CrashHandlerTest, InstallIsIdempotent) {
    // 多次 Install 不应崩溃，且仍然生效
    auto r = RunInChild([] {
        CrashHandler::Install("");
        CrashHandler::Install("");
        CrashHandler::Install("");
        if (!CrashHandler::IsInstalled()) {
            _exit(2);
        }
        ::raise(SIGSEGV);
        _exit(0);
    });
    EXPECT_TRUE(ChildAbnormal(r.wstatus));
    EXPECT_NE(std::string::npos, r.stderr_output.find("SIGSEGV")) << "stderr:\n" << r.stderr_output;
}

TEST_F(CrashHandlerTest, InstallDoesNotOverwriteExistingAltStack) {
    // ASan/TSan/breakpad 等在进程启动时会自动给主线程装 alt stack。Install
    // 必须复用而不是替换，否则对方在退出时按自己记的旧地址 munmap，发现
    // 地址被换 → sanitizer 内部 CHECK 失败 abort 进程。和
    // InstallAltStackForCurrentThread() 同款 no-op 策略。
    auto r = RunInChild([] {
        stack_t before;
        if (::sigaltstack(nullptr, &before) != 0) {
            _exit(10);
        }
        if ((before.ss_flags & SS_DISABLE) != 0) {
            // 没人预装（非 sanitizer 环境）：跳过断言，测试无意义
            _exit(0);
        }

        CrashHandler::Install("");

        stack_t after;
        if (::sigaltstack(nullptr, &after) != 0) {
            _exit(11);
        }
        if (before.ss_sp != after.ss_sp) {
            _exit(12); // Install 覆盖了
        }
        if (before.ss_size != after.ss_size) {
            _exit(13);
        }
        _exit(0);
    });
    ASSERT_TRUE(WIFEXITED(r.wstatus)) << "wstatus=" << r.wstatus;
    EXPECT_EQ(0, WEXITSTATUS(r.wstatus)) << "Install() overwrote pre-existing alt stack, code="
                                         << WEXITSTATUS(r.wstatus);
}

TEST_F(CrashHandlerTest, ConcurrentCrashesDoNotDeadlock) {
    // 多线程同时 raise SEGV，handler 不应死锁；至少一次完整堆栈被打印。
    // alarm(5) 是兜底：如果未来回归引入死锁，子进程 5 秒后被 SIGALRM 杀掉，
    // 而不是吃满 bazel test 的 5 分钟超时。
    auto r = RunInChild([] {
        ::alarm(5);
        CrashHandler::Install("");
        constexpr int kThreads = 8;
        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::vector<std::thread> ts;
        for (int i = 0; i < kThreads; ++i) {
            ts.emplace_back([&] {
                ready.fetch_add(1);
                while (!go.load()) {
                    std::this_thread::yield();
                }
                ::raise(SIGSEGV);
            });
        }
        while (ready.load() < kThreads) {
            std::this_thread::yield();
        }
        go.store(true);
        for (auto &t : ts) {
            t.join();
        }
        _exit(0);
    });
    EXPECT_TRUE(ChildAbnormal(r.wstatus)) << "wstatus=" << r.wstatus;
    EXPECT_NE(std::string::npos, r.stderr_output.find("KVCM CrashHandler")) << "stderr:\n" << r.stderr_output;
    EXPECT_NE(std::string::npos, r.stderr_output.find("End of stack trace"))
        << "stderr (truncated handler output may indicate deadlock):\n"
        << r.stderr_output;
}

TEST_F(CrashHandlerTest, OutputContainsBuildVersion) {
    auto r = RunInChild([] {
        CrashHandler::Install("");
        ::raise(SIGSEGV);
        _exit(0);
    });
    EXPECT_NE(std::string::npos, r.stderr_output.find("version:")) << "stderr:\n" << r.stderr_output;
    EXPECT_NE(std::string::npos, r.stderr_output.find(KVCM_GIT_COMMIT))
        << "no git commit in stderr (looking for " << KVCM_GIT_COMMIT << "):\n"
        << r.stderr_output;
}

TEST_F(CrashHandlerTest, SiCodeIsSignedNotHuge) {
    // SI_TKILL = -6；之前的 bug 是把负数当 uint64_t 打印成 18446744073709551610。
    auto r = RunInChild([] {
        CrashHandler::Install("");
        ::raise(SIGSEGV);
        _exit(0);
    });
    EXPECT_EQ(std::string::npos, r.stderr_output.find("18446744073709551")) << "si_code printed as unsigned overflow:\n"
                                                                            << r.stderr_output;
}

TEST_F(CrashHandlerTest, OutputContainsSiCode) {
    // raise(SIGSEGV) 由 tkill(2) 实现，si_code 是 SI_USER 或 SI_TKILL。
    // 实际触发的非法访问（kernel 发的）si_code 是 SEGV_MAPERR/SEGV_ACCERR。
    // 这里只断言"si_code:" 字段存在以及 _USER/_KERNEL/_TKILL 之一出现。
    auto r = RunInChild([] {
        CrashHandler::Install("");
        ::raise(SIGSEGV);
        _exit(0);
    });
    EXPECT_NE(std::string::npos, r.stderr_output.find("si_code:")) << "stderr:\n" << r.stderr_output;
    bool any =
        r.stderr_output.find("SI_USER") != std::string::npos || r.stderr_output.find("SI_TKILL") != std::string::npos ||
        r.stderr_output.find("SI_KERNEL") != std::string::npos || r.stderr_output.find("SEGV_") != std::string::npos;
    EXPECT_TRUE(any) << "no recognizable si_code name in stderr:\n" << r.stderr_output;
}

TEST_F(CrashHandlerTest, DefaultInstallWritesToLogsDir) {
    // 不传 log_dir：应该在子进程 cwd 下创建 logs/std_stack_trace.log，
    // 与 alog 配置文件 (default_logger_config.conf) 用的相对目录一致。
    std::string runtime = GetPrivateTestRuntimeDataPath();
    std::string expected_path = runtime + "logs/std_stack_trace.log";
    ::unlink(expected_path.c_str());

    auto r = RunInChild([runtime] {
        // 子进程切到测试 runtime 目录，使 logs/ 落到隔离的位置，避免污染
        // 外层 bazel test 的工作目录。
        if (::chdir(runtime.c_str()) != 0) {
            _exit(2);
        }
        CrashHandler::Install(); // 默认参数
        ::raise(SIGSEGV);
        _exit(0);
    });
    EXPECT_TRUE(ChildAbnormal(r.wstatus));
    struct stat st;
    ASSERT_EQ(0, ::stat(expected_path.c_str(), &st)) << "expected " << expected_path;
    ASSERT_GT(st.st_size, 0);
    std::string content = SlurpFile(expected_path);
    EXPECT_NE(std::string::npos, content.find("SIGSEGV"));
    EXPECT_NE(std::string::npos, content.find("KVCM CrashHandler"));
}

TEST_F(CrashHandlerTest, CrashLogStructuredFields) {
    std::string log_dir = GetPrivateTestRuntimeDataPath();
    std::string log_path = log_dir + "std_stack_trace.log";
    ::unlink(log_path.c_str());

    auto r = RunInChild([log_dir] {
        CrashHandler::Install(log_dir);
        ::raise(SIGSEGV);
        _exit(0);
    });
    EXPECT_TRUE(ChildAbnormal(r.wstatus));
    std::string content = SlurpFile(log_path);
    EXPECT_NE(std::string::npos, content.find("*** KVCM CrashHandler caught signal"));
    EXPECT_NE(std::string::npos, content.find("SIGSEGV"));
    EXPECT_NE(std::string::npos, content.find("version:"));
    EXPECT_NE(std::string::npos, content.find("si_code:"));
    EXPECT_NE(std::string::npos, content.find("time(epoch_sec):"));
    EXPECT_NE(std::string::npos, content.find("Stack trace ("));
    EXPECT_NE(std::string::npos, content.find("*** End of stack trace ***"));
}

namespace {

// 装在 SIGSEGV 上的 sa_handler：写个标记到 stderr 然后直接 return，
// 暴露"chained handler 返回"路径，验证 fail-safe 是否兜底退出。
void UserReturningSegvHandler(int /*sig*/) {
    const char msg[] = "USER_RETURNING_HANDLER_INVOKED\n";
    ::write(STDERR_FILENO, msg, sizeof(msg) - 1);
}

// 装在 SIGSEGV 上的 SA_SIGINFO handler：写标记 + 报告 si_code，验证
// CrashHandler 通过 sa_sigaction 路径链式调用旧 handler 时，原始
// siginfo 被透传过去（不是 raise(sig) 合成的 SI_TKILL/SI_USER 假上下文）。
void UserSigactionSegvHandler(int /*sig*/, siginfo_t *info, void * /*ucontext*/) {
    const char msg[] = "USER_SIGACTION_HANDLER_INVOKED\n";
    ::write(STDERR_FILENO, msg, sizeof(msg) - 1);
    if (info != nullptr) {
        const char tag[] = "USER_SIGACTION_GOT_SIGINFO\n";
        ::write(STDERR_FILENO, tag, sizeof(tag) - 1);
    }
}

// 经典 idiom：恢复 default 再 raise，期望被 kernel 默认行为杀掉生成
// coredump。验证 KvcmCrashSigaction 在调 user handler 前 unblock 了 sig
// —— 否则 raise 只让信号 pending，user handler return 后被 fail-safe
// _exit 抢先终止，进程以 exit(139) 退出而不是 WIFSIGNALED。
void UserHandlerChainsToDefault(int sig) {
    ::signal(sig, SIG_DFL);
    ::raise(sig);
    // 不应到达
}

void UserSigactionChainsToDefault(int sig, siginfo_t * /*info*/, void * /*ucontext*/) {
    ::signal(sig, SIG_DFL);
    ::raise(sig);
}

// SIGSEGV handler 里 abort()（= raise(SIGABRT)），模拟 sanitizer / 通用
// crash reporter 用另一个 crash signal 终止进程的典型路径。验证
// KvcmCrashSigaction unblock 的是全部 5 个 crash signal、而不是只解开当前
// sig —— 否则 SIGABRT 还在 sa_mask 里被阻塞，raise 只 pending，user
// handler return 后被 fail-safe _exit 抢先终止，子进程不会被 SIGABRT 杀死。
void UserSegvHandlerRaisesSigabrt(int /*sig*/) {
    ::signal(SIGABRT, SIG_DFL);
    ::raise(SIGABRT);
}

// 故意**不** reset SIGABRT 的 disposition 就 raise —— SIGABRT 仍是
// KvcmCrashSigaction，会再次落进来。验证同线程重入不死锁：第二次进入
// 检测到 t_in_handler 直接 fail-safe _exit(128+SIGABRT)，而不是进
// pause() 等永远不会到来的"第一个线程"。
void UserSegvHandlerRaisesSigabrtNoReset(int /*sig*/) { ::raise(SIGABRT); }

} // namespace

TEST_F(CrashHandlerTest, DefaultDispositionCausesSignalTermination) {
    // 装 CrashHandler 前把 SIGSEGV oldact 拉回 SIG_DFL（盖掉 ASan 等
    // runtime 注册的 handler），让链式调用走到 SIG_DFL 分支。
    //
    // 历史 bug：handler 装 sa_mask 时把全部 crash signal 都加进去，导致
    // 进入 handler 后 sig 被当前线程 mask 阻塞；SIG_DFL 路径裸 raise(sig)
    // 只能让信号 pending，立刻被后面的 _exit(128+sig) 抢先终止 —— 进程
    // exit code 是 139，但 WIFSIGNALED 为 false，kernel 也不生成 coredump，
    // docs 里承诺的"handler 不替代 coredump"实际失效。
    //
    // 修复后 SIG_DFL 路径会 pthread_sigmask 解阻塞再 raise，进程真正被
    // SIGSEGV 杀死，WIFSIGNALED 为 true。
    auto r = RunInChild([] {
        ::signal(SIGSEGV, SIG_DFL);
        CrashHandler::Install("");
        ::raise(SIGSEGV);
        _exit(0);
    });
    ASSERT_TRUE(WIFSIGNALED(r.wstatus)) << "child exited normally, kernel will not generate coredump. wstatus="
                                        << r.wstatus;
    EXPECT_EQ(SIGSEGV, WTERMSIG(r.wstatus));
}

TEST_F(CrashHandlerTest, FailSafeExitsWhenChainedHandlerReturns) {
    auto r = RunInChild([] {
        struct sigaction sa;
        ::memset(&sa, 0, sizeof(sa));
        sa.sa_handler = UserReturningSegvHandler;
        ::sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        ::sigaction(SIGSEGV, &sa, nullptr);

        CrashHandler::Install("");
        ::raise(SIGSEGV);
        _exit(0); // 不应到达：CrashHandler 兜底必须强制退出
    });
    ASSERT_TRUE(WIFEXITED(r.wstatus)) << "child killed by signal, wstatus=" << r.wstatus;
    EXPECT_EQ(128 + SIGSEGV, WEXITSTATUS(r.wstatus))
        << "fail-safe _exit(128+sig) not taken; chained user handler returned and "
           "process kept running. wstatus="
        << r.wstatus;
    EXPECT_NE(std::string::npos, r.stderr_output.find("USER_RETURNING_HANDLER_INVOKED")) << "old handler not chained:\n"
                                                                                         << r.stderr_output;
}

TEST_F(CrashHandlerTest, OldSigIgnDoesNotDeadlock) {
    auto r = RunInChild([] {
        // SIGABRT 是为数不多可以 SIG_IGN 的 crash 信号；之前的实现里
        // raise(SIGABRT) 会被 ignore 吞掉，进程继续跑，g_in_handler 永
        // 久 set，下次崩溃就卡 pause()。这里直接验证不会死循环。
        ::signal(SIGABRT, SIG_IGN);
        CrashHandler::Install("");
        ::raise(SIGABRT);
        _exit(0);
    });
    ASSERT_TRUE(WIFEXITED(r.wstatus)) << "child killed by signal, wstatus=" << r.wstatus;
    EXPECT_EQ(128 + SIGABRT, WEXITSTATUS(r.wstatus));
}

TEST_F(CrashHandlerTest, ChainedSigactionPathInvoked) {
    auto r = RunInChild([] {
        struct sigaction sa;
        ::memset(&sa, 0, sizeof(sa));
        sa.sa_sigaction = UserSigactionSegvHandler;
        ::sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_SIGINFO;
        ::sigaction(SIGSEGV, &sa, nullptr);

        CrashHandler::Install("");
        ::raise(SIGSEGV);
        _exit(0);
    });
    EXPECT_TRUE(ChildAbnormal(r.wstatus));
    // SA_SIGINFO 旧 handler 必须经 sa_sigaction 路径调用（而非 raise 合成）
    EXPECT_NE(std::string::npos, r.stderr_output.find("USER_SIGACTION_HANDLER_INVOKED"))
        << "SA_SIGINFO old handler not chained via sa_sigaction:\n"
        << r.stderr_output;
    EXPECT_NE(std::string::npos, r.stderr_output.find("USER_SIGACTION_GOT_SIGINFO"))
        << "siginfo not forwarded to chained handler:\n"
        << r.stderr_output;
}

TEST_F(CrashHandlerTest, ChainedUserHandlerCanReachDefaultDisposition) {
    // 装一个 user sa_handler，内部跑经典 idiom "signal(sig, SIG_DFL); raise(sig);"
    // 期望走 default disposition 杀掉进程生成 coredump。验证 KvcmCrashSigaction
    // 在调 user handler 前已经 unblock 了 sig —— 否则 raise 只让信号 pending，
    // user handler return 后被 fail-safe _exit(128+sig) 抢先终止，wstatus 是
    // exit(139) 而非 WIFSIGNALED，kernel 不生成 coredump。
    auto r = RunInChild([] {
        struct sigaction sa;
        ::memset(&sa, 0, sizeof(sa));
        sa.sa_handler = UserHandlerChainsToDefault;
        ::sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        ::sigaction(SIGSEGV, &sa, nullptr);

        CrashHandler::Install("");
        ::raise(SIGSEGV);
        _exit(0);
    });
    ASSERT_TRUE(WIFSIGNALED(r.wstatus)) << "user handler's chain-to-default got swallowed by "
                                           "fail-safe _exit (sig still blocked?). wstatus="
                                        << r.wstatus;
    EXPECT_EQ(SIGSEGV, WTERMSIG(r.wstatus));
}

TEST_F(CrashHandlerTest, ChainedHandlerCanTerminateViaDifferentCrashSignal) {
    // SIGSEGV user handler 内部 raise(SIGABRT)（abort 的实质路径）。
    // KvcmCrashSigaction 必须 unblock 全部 5 个 crash signal，而不是只解开
    // 当前 sig（SIGSEGV），否则 SIGABRT 还在 sa_mask 里阻塞，raise 只
    // pending，user handler return 后被 fail-safe _exit(128+SIGSEGV) 抢先
    // 终止；子进程是 _exit(139) 而不是 WIFSIGNALED == true, WTERMSIG == SIGABRT。
    auto r = RunInChild([] {
        struct sigaction sa;
        ::memset(&sa, 0, sizeof(sa));
        sa.sa_handler = UserSegvHandlerRaisesSigabrt;
        ::sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        ::sigaction(SIGSEGV, &sa, nullptr);

        CrashHandler::Install("");
        ::raise(SIGSEGV);
        _exit(0);
    });
    ASSERT_TRUE(WIFSIGNALED(r.wstatus)) << "child not killed by signal — SIGABRT got swallowed by fail-safe _exit "
                                           "(other crash signals still blocked by sa_mask?). wstatus="
                                        << r.wstatus;
    EXPECT_EQ(SIGABRT, WTERMSIG(r.wstatus));
}

TEST_F(CrashHandlerTest, ReentrantSameThreadHandlerDoesNotHang) {
    // SIGSEGV user handler 内部裸 raise(SIGABRT)（不 reset disposition）
    // 是个常见的 idiom 边角 case：SIGABRT 仍是 KvcmCrashSigaction，会再次
    // 落进来。修复前 g_in_handler 已 set 第二次进入会进 pause() 死循环，
    // 进程 hang 死；修复后 thread_local t_in_handler 标记当前线程已在
    // handler 里，第二次进入直接 _exit(128+SIGABRT) = 134。
    // alarm(3) 兜底：如果回归 hang 了，3 秒后被 SIGALRM 杀掉，避免吃满
    // bazel test 的 5 分钟超时。
    auto r = RunInChild([] {
        ::alarm(3);
        struct sigaction sa;
        ::memset(&sa, 0, sizeof(sa));
        sa.sa_handler = UserSegvHandlerRaisesSigabrtNoReset;
        ::sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;
        ::sigaction(SIGSEGV, &sa, nullptr);

        CrashHandler::Install("");
        ::raise(SIGSEGV);
        _exit(0);
    });
    // 关键：必须不是 SIGALRM 杀的（hang 的话才会走到 SIGALRM 兜底）。
    if (WIFSIGNALED(r.wstatus)) {
        ASSERT_NE(SIGALRM, WTERMSIG(r.wstatus))
            << "handler re-entered itself and hung in pause(); SIGALRM killed the child";
    }
    ASSERT_TRUE(WIFEXITED(r.wstatus)) << "wstatus=" << r.wstatus;
    EXPECT_EQ(128 + SIGABRT, WEXITSTATUS(r.wstatus));
}

TEST_F(CrashHandlerTest, ChainedSigactionHandlerCanReachDefaultDisposition) {
    // SA_SIGINFO 路径的对称覆盖：同样的 chain-to-default idiom 装在 sa_sigaction，
    // 验证 KvcmCrashSigaction 在调 SA_SIGINFO user handler 前也 unblock 了 sig。
    auto r = RunInChild([] {
        struct sigaction sa;
        ::memset(&sa, 0, sizeof(sa));
        sa.sa_sigaction = UserSigactionChainsToDefault;
        ::sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_SIGINFO;
        ::sigaction(SIGSEGV, &sa, nullptr);

        CrashHandler::Install("");
        ::raise(SIGSEGV);
        _exit(0);
    });
    ASSERT_TRUE(WIFSIGNALED(r.wstatus)) << "SA_SIGINFO user handler's chain-to-default got "
                                           "swallowed. wstatus="
                                        << r.wstatus;
    EXPECT_EQ(SIGSEGV, WTERMSIG(r.wstatus));
}

TEST_F(CrashHandlerTest, InstallAltStackForCurrentThreadRegistersAltStack) {
    // worker 线程上调 API 后，sigaltstack 查询应该返回非 disable 的 alt stack。
    // 用 exit code 编码失败原因：0 = OK，10+ = 各种失败点。
    auto r = RunInChild([] {
        int rc = 0;
        std::thread t([&] {
            CrashHandler::InstallAltStackForCurrentThread();
            stack_t cur;
            if (::sigaltstack(nullptr, &cur) != 0) {
                rc = 10;
                return;
            }
            if (cur.ss_sp == nullptr) {
                rc = 11;
                return;
            }
            if (cur.ss_size == 0) {
                rc = 12;
                return;
            }
            if ((cur.ss_flags & SS_DISABLE) != 0) {
                rc = 13;
                return;
            }
        });
        t.join();
        _exit(rc);
    });
    ASSERT_TRUE(WIFEXITED(r.wstatus)) << "child killed by signal, wstatus=" << r.wstatus << " stderr:\n"
                                      << r.stderr_output;
    EXPECT_EQ(0, WEXITSTATUS(r.wstatus)) << "alt stack check failed with code " << WEXITSTATUS(r.wstatus)
                                         << " stderr:\n"
                                         << r.stderr_output;
}

TEST_F(CrashHandlerTest, InstallAltStackForCurrentThreadIsIdempotent) {
    // 多次调用应不会重新分配 alt stack（sp 不变），避免泄漏。
    auto r = RunInChild([] {
        int rc = 0;
        std::thread t([&] {
            CrashHandler::InstallAltStackForCurrentThread();
            stack_t first;
            if (::sigaltstack(nullptr, &first) != 0) {
                rc = 20;
                return;
            }
            CrashHandler::InstallAltStackForCurrentThread();
            CrashHandler::InstallAltStackForCurrentThread();
            stack_t after;
            if (::sigaltstack(nullptr, &after) != 0) {
                rc = 21;
                return;
            }
            if (first.ss_sp != after.ss_sp) {
                rc = 22;
                return;
            }
            if (first.ss_size != after.ss_size) {
                rc = 23;
                return;
            }
        });
        t.join();
        _exit(rc);
    });
    ASSERT_TRUE(WIFEXITED(r.wstatus));
    EXPECT_EQ(0, WEXITSTATUS(r.wstatus)) << "idempotency check failed with code " << WEXITSTATUS(r.wstatus);
}

TEST_F(CrashHandlerTest, AltStackOnWorkerEnablesCrashHandler) {
    // 集成场景：worker 线程装好 alt stack 后触发 SIGSEGV，handler 仍能正常
    // 跑出完整堆栈到 stderr。验证 API 不会破坏链路。
    auto r = RunInChild([] {
        CrashHandler::Install("");
        std::thread t([] {
            CrashHandler::InstallAltStackForCurrentThread();
            ::raise(SIGSEGV);
        });
        t.join();
        _exit(0);
    });
    EXPECT_TRUE(ChildAbnormal(r.wstatus)) << "wstatus=" << r.wstatus;
    EXPECT_NE(std::string::npos, r.stderr_output.find("KVCM CrashHandler")) << "handler did not run on worker thread:\n"
                                                                            << r.stderr_output;
    EXPECT_NE(std::string::npos, r.stderr_output.find("SIGSEGV")) << r.stderr_output;
}
