#include <unistd.h>

#include <chrono>
#include <csignal>
#include <thread>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/service/command_line.h"

using namespace kv_cache_manager;

class CommandLineTest : public TESTBASE {
public:
    void SetUp() override {}

    void TearDown() override {}

public:
    static std::vector<char *> BuildArgv(const std::vector<std::string> &arg_vec) {
        std::vector<char *> argv;
        for (const auto &s : arg_vec) {
            argv.push_back(strdup(s.c_str()));
        }
        return argv;
    }

    static void FreeArgv(std::vector<char *> &argv) {
        std::for_each(argv.begin(), argv.end(), [](char *p) { free(p); });
    }
};

TEST_F(CommandLineTest, TestParseCmdLine) {
    CommandArgs command_args;
    std::vector<std::string> arg_vec = {"kv_cache_manager",
                                        "-d",
                                        "-c",
                                        "etc/kv_cache_manager.conf",
                                        "-l",
                                        "etc/alog.conf",
                                        "-e",
                                        "key_a=value1",
                                        "-e",
                                        "key_b=value2",
                                        "-e",
                                        "kvcm.service.rpc_port=6381",
                                        "-e",
                                        "kvcm.service.http_port=6382"};
    auto argv = BuildArgv(arg_vec);
    command_args.ParseCmdLine(argv.size(), argv.data());
    FreeArgv(argv);
    ASSERT_TRUE(command_args.IsDaemon());
    ASSERT_EQ("etc/kv_cache_manager.conf", command_args.GetConfigFile());
    ASSERT_EQ("etc/alog.conf", command_args.GetLogConfigFile());
    auto environ = command_args.GetEnviron();
    ASSERT_EQ(4, environ.size());
    ASSERT_EQ("value1", environ["key_a"]);
    ASSERT_EQ("value2", environ["key_b"]);
    ASSERT_EQ("6381", environ["kvcm.service.rpc_port"]);
    ASSERT_EQ("6382", environ["kvcm.service.http_port"]);
}

TEST_F(CommandLineTest, TerminationSignalOnWorkerOnlyNotifiesMainThread) {
    // InstallSignalPolicy installs crash handlers and alternate signal-stack state.
    // Keep all of those process-wide side effects in the child.
    ASSERT_EXIT(
        {
            CommandLine command;
            // Same order as CommandLine::Run: mask first, then start the controller.
            if (!ShutdownController::BlockExitSignals()) {
                _exit(1);
            }
            if (!command.InstallSignalPolicy()) {
                _exit(1);
            }
            // raise() only marks the signal pending on the raising thread. Exit signals
            // are blocked in every thread, so the worker runs no shutdown logic and the
            // controller thread is not woken up.
            std::thread worker([] { std::raise(SIGTERM); });
            worker.join();
            if (command.controller_.IsShutdownRequested()) {
                _exit(1);
            }
            // Only a process-directed signal is picked up by the controller thread,
            // which then wakes the main thread to run the blocking shutdown.
            kill(getpid(), SIGTERM);
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!command.controller_.IsShutdownRequested() && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            _exit(command.controller_.ReceivedSignal() == SIGTERM ? 0 : 1);
        },
        ::testing::ExitedWithCode(0),
        "");
}
