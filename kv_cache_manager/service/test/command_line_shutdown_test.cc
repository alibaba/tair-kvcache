#include <gtest/gtest.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

namespace {

const int64_t kServerStartTimeoutMs = 60000;
const int64_t kShutdownTimeoutMs = 60000;

int GetFreePort() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    socklen_t len = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len) != 0) {
        close(fd);
        return -1;
    }
    int port = ntohs(addr.sin_port);
    close(fd);
    return port;
}

bool IsPortListening(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return false;
    }
    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    bool ok = connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0;
    close(fd);
    return ok;
}

bool IsProcessAlive(pid_t pid) { return kill(pid, 0) == 0; }

std::string ReadFile(const std::string &path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return "";
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

// 带超时地回收子进程；超时则 SIGKILL，避免用例把整个 CI 挂住。
bool WaitChildWithTimeout(pid_t pid, int &status, int64_t timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        pid_t ret = waitpid(pid, &status, WNOHANG);
        if (ret == pid) {
            return true;
        }
        if (ret < 0) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    return false;
}

// 真实 server 二进制的 runfiles 路径。用例必须 exec 这个独立进程，不能在 gtest 子进程里
// 直接调用 CommandLine::Run()：unittest.h 的 main() 已经初始化过 alog，fork 之后子进程
// 里 InitLogger 变成空操作（日志落到父进程 cwd），且 DestroyLogger 会在 alog 的 flush
// 线程缺失时崩溃。
std::string GetServerBinaryPath() {
    const char *src_dir = getenv("TEST_SRCDIR");
    const char *workspace = getenv("TEST_WORKSPACE");
    if (src_dir == nullptr || workspace == nullptr) {
        return "";
    }
    return std::string(src_dir) + "/" + workspace + "/kv_cache_manager/kv_cache_manager_bin";
}

} // namespace

// 端到端退出测试：子进程 exec 真实的 server 二进制（含真实 Server 启动），
// 父进程用信号驱动退出，校验退出码与关闭日志。
class CommandLineShutdownTest : public TESTBASE {
protected:
    void SetUp() override {
        work_dir_ = GetPrivateTestRuntimeDataPath() + "/command_line_shutdown";
        std::error_code ec;
        std::filesystem::remove_all(work_dir_, ec);
        std::filesystem::create_directories(work_dir_, ec);
        ASSERT_TRUE(!ec);

        binary_path_ = GetServerBinaryPath();
        ASSERT_FALSE(binary_path_.empty()) << "TEST_SRCDIR/TEST_WORKSPACE not set";
        ASSERT_EQ(0, access(binary_path_.c_str(), X_OK)) << "server binary not runnable: " << binary_path_;

        rpc_port_ = GetFreePort();
        http_port_ = GetFreePort();
        admin_rpc_port_ = GetFreePort();
        admin_http_port_ = GetFreePort();
        ASSERT_GT(rpc_port_, 0);
        ASSERT_GT(http_port_, 0);
        ASSERT_GT(admin_rpc_port_, 0);
        ASSERT_GT(admin_http_port_, 0);
    }

    void TearDown() override {
        if (child_pid_ > 0 && IsProcessAlive(child_pid_)) {
            kill(child_pid_, SIGKILL);
            int status = 0;
            waitpid(child_pid_, &status, 0);
        }
        std::error_code ec;
        std::filesystem::remove_all(work_dir_, ec);
    }

    // fork 出子进程 exec 真实 server 二进制，返回子进程 pid；失败返回 -1。
    pid_t StartServerChild() {
        std::vector<std::string> args = {"kv_cache_manager_bin",
                                         "-e",
                                         "kvcm.coordination.uri=memory://",
                                         "-e",
                                         "kvcm.service.rpc_port=" + std::to_string(rpc_port_),
                                         "-e",
                                         "kvcm.service.http_port=" + std::to_string(http_port_),
                                         "-e",
                                         "kvcm.service.admin_rpc_port=" + std::to_string(admin_rpc_port_),
                                         "-e",
                                         "kvcm.service.admin_http_port=" + std::to_string(admin_http_port_),
                                         "-e",
                                         "kvcm.service.enable_debug_service=true"};
        std::vector<char *> argv;
        argv.reserve(args.size() + 1);
        for (const auto &arg : args) {
            argv.push_back(strdup(arg.c_str()));
        }
        argv.push_back(nullptr);

        pid_t pid = fork();
        if (pid == 0) {
            // 信号屏蔽字会跨 exec 继承：清空它，否则被测进程可能收不到父进程发来的 SIGTERM。
            sigset_t no_block;
            sigemptyset(&no_block);
            sigprocmask(SIG_SETMASK, &no_block, nullptr);
            // 切换工作目录，让默认 alog 配置把日志写到本用例独占的目录里。
            if (chdir(work_dir_.c_str()) != 0) {
                _exit(127);
            }
            execv(binary_path_.c_str(), argv.data());
            _exit(127);
        }
        for (char *p : argv) {
            free(p);
        }
        if (pid < 0) {
            return -1;
        }
        child_pid_ = pid;
        return pid;
    }

    // Server::Start() 在 rpc 端口就绪后就返回，http 服务是在独立线程里异步起来的。
    // 这里等到显式配置的端口都在监听，让关闭路径覆盖到 http 服务的 Stop()。
    bool WaitForServerReady() {
        const std::vector<int> ports = {rpc_port_, http_port_, admin_http_port_};
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kServerStartTimeoutMs);
        while (std::chrono::steady_clock::now() < deadline) {
            if (!IsProcessAlive(child_pid_)) {
                ADD_FAILURE() << "server child exited before becoming ready, log:\n" << ServiceLog();
                return false;
            }
            bool all_listening = true;
            for (int port : ports) {
                if (!IsPortListening(port)) {
                    all_listening = false;
                    break;
                }
            }
            if (all_listening) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        ADD_FAILURE() << "server child did not listen on all ports in time, log:\n" << ServiceLog();
        return false;
    }

    // 等到子进程确实进入关闭流程（DoStop 的第一条日志落盘）。关闭一旦开始就只剩十几毫秒，
    // 所以这里用 1ms 粒度轮询，作为「第二个信号落在关闭进行中」的同步点。
    bool WaitForShutdownStarted() {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kShutdownTimeoutMs);
        while (std::chrono::steady_clock::now() < deadline) {
            if (ServiceLog().find("server stopping...") != std::string::npos) {
                return true;
            }
            if (!IsProcessAlive(child_pid_)) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return false;
    }

    std::string ServiceLog() { return ReadFile(work_dir_ + "/logs/kv_cache_manager.log"); }

protected:
    std::string work_dir_;
    std::string binary_path_;
    int rpc_port_ = 0;
    int http_port_ = 0;
    int admin_rpc_port_ = 0;
    int admin_http_port_ = 0;
    pid_t child_pid_ = -1;
};

// AC-6: 优雅退出返回 0，并在销毁 service/manager 之前完成完整的关闭日志序列。
TEST_F(CommandLineShutdownTest, SingleSigtermShutsDownGracefullyWithExitCodeZero) {
    ASSERT_GE(StartServerChild(), 0);
    ASSERT_TRUE(WaitForServerReady());

    ASSERT_EQ(0, kill(child_pid_, SIGTERM));

    int status = 0;
    ASSERT_TRUE(WaitChildWithTimeout(child_pid_, status, kShutdownTimeoutMs)) << "graceful shutdown hung";
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(0, WEXITSTATUS(status));

    std::string log = ServiceLog();
    EXPECT_NE(log.find("server stopping..."), std::string::npos) << log;
    EXPECT_NE(log.find("rpc server stopped."), std::string::npos) << log;
    EXPECT_NE(log.find("kvcm server stopped, goodbye!"), std::string::npos) << log;
    // 关闭日志序列只应出现一份（Server::Stop() 幂等 + 单一退出路径）。
    EXPECT_EQ(log.find("kvcm server stopped, goodbye!"), log.rfind("kvcm server stopped, goodbye!")) << log;
    child_pid_ = -1;
}

// AC-3: 优雅关闭进行中再次收到 SIGTERM → 立即以 128+SIGTERM 结束。
TEST_F(CommandLineShutdownTest, RepeatedSigtermForcesExitWithSignalCode) {
    ASSERT_GE(StartServerChild(), 0);
    ASSERT_TRUE(WaitForServerReady());

    ASSERT_EQ(0, kill(child_pid_, SIGTERM));
    // 标准信号不排队：必须等控制线程取走第一个信号（表现为关闭日志落盘）再补第二个，
    // 否则两个 SIGTERM 会合并成一个 pending 信号，只触发一次优雅关闭。
    ASSERT_TRUE(WaitForShutdownStarted()) << "server never entered shutdown, log:\n" << ServiceLog();
    // 子进程还没被回收，kill() 对僵尸进程同样返回 0，所以用 WNOHANG waitpid 判断是否仍在关闭中。
    int status = 0;
    ASSERT_EQ(0, waitpid(child_pid_, &status, WNOHANG)) << "graceful shutdown finished too fast, log:\n"
                                                        << ServiceLog();
    ASSERT_EQ(0, kill(child_pid_, SIGTERM));

    ASSERT_TRUE(WaitChildWithTimeout(child_pid_, status, kShutdownTimeoutMs)) << "force exit hung";
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(128 + SIGTERM, WEXITSTATUS(status));

    std::string log = ServiceLog();
    EXPECT_NE(log.find("force exit"), std::string::npos) << log;
    EXPECT_EQ(log.find("kvcm server stopped, goodbye!"), std::string::npos) << log;
    child_pid_ = -1;
}
