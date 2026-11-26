#include <getopt.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#include "kv_cache_manager/service/command_line.h"
#include "kv_cache_manager/service/process_extension.h"

namespace kv_cache_manager {
namespace {
enum class Failure {
    START,
    EXCEPTION,
    SERVER_INIT,
    SERVER_START,
    TERMINATE_DURING_START
};

class RecordingExtension final : public ProcessExtension {
public:
    RecordingExtension(std::vector<std::string> &events, Failure failure) : events_(events), failure_(failure) {}
    ~RecordingExtension() override { events_.push_back("destroy"); }
    bool Start(const ProcessExtensionContext &context) override {
        events_.push_back("start");
        if (!context.config_file.empty() || context.overrides.at("test.option") != "test-value") {
            throw std::runtime_error("context lost");
        }
        events_.push_back("context");
        if (failure_ == Failure::TERMINATE_DURING_START) {
            raise(SIGTERM);
        }
        if (failure_ == Failure::EXCEPTION) {
            throw std::runtime_error("partial initialization");
        }
        return failure_ != Failure::START;
    }
    void Stop() noexcept override { events_.push_back("stop"); }

private:
    std::vector<std::string> &events_;
    Failure failure_;
};

void RunFailure(Failure failure) {
    std::vector<std::string> events;
    int result = 0;
    {
        CommandLine command(std::make_unique<RecordingExtension>(events, failure));
        // Invalid URI forces Server::Init failure, after extension startup.
        const char *uri = failure == Failure::SERVER_INIT ? "kvcm.coordination.uri=unsupported://"
                                                          : "kvcm.coordination.uri=memory://";
        int reservation = -1;
        sockaddr_in address{};
        if (failure == Failure::SERVER_START) {
            reservation = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
            address.sin_family = AF_INET;
            address.sin_addr.s_addr = htonl(INADDR_ANY);
            if (reservation < 0 || bind(reservation, reinterpret_cast<sockaddr *>(&address), sizeof(address)) ||
                listen(reservation, 1)) {
                _exit(2);
            }
            socklen_t size = sizeof(address);
            if (getsockname(reservation, reinterpret_cast<sockaddr *>(&address), &size)) {
                _exit(2);
            }
        }
        const std::string port = "kvcm.service.rpc_port=" + std::to_string(ntohs(address.sin_port));
        const char *argv[] = {"test",
                              "-e",
                              "test.option=test-value",
                              "-e",
                              uri,
                              "-e",
                              "kvcm.service.io_thread_num=1",
                              "-e",
                              port.c_str()};
        optind = 0;
        result = command.Run(9, argv);
        if (reservation >= 0) {
            close(reservation);
        }
    }
    const std::vector<std::string> expected{"start", "context", "stop", "destroy"};
    const int expected_result = failure == Failure::TERMINATE_DURING_START ? 0 : -1;
    _exit(result == expected_result && events == expected ? 0 : 1);
}

TEST(ProcessExtensionTest, FailedStartReleasesPartialResources) {
    EXPECT_EXIT(RunFailure(Failure::START), testing::ExitedWithCode(0), "");
}

TEST(ProcessExtensionTest, DefaultFactorySupportsAnUnconfiguredProcess) {
    auto extension = CreateProcessExtension();
    ASSERT_NE(extension, nullptr);
    ASSERT_TRUE(extension->Start({}));
    extension->Stop();
    extension->Stop();
}

TEST(ProcessExtensionTest, ThrowingStartReleasesPartialResources) {
    EXPECT_EXIT(RunFailure(Failure::EXCEPTION), testing::ExitedWithCode(0), "");
}

TEST(ProcessExtensionTest, ServerInitFailureStopsExtension) {
    EXPECT_EXIT(RunFailure(Failure::SERVER_INIT), testing::ExitedWithCode(0), "");
}

TEST(ProcessExtensionTest, ServerStartFailureStopsExtension) {
    EXPECT_EXIT(RunFailure(Failure::SERVER_START), testing::ExitedWithCode(0), "");
}

TEST(ProcessExtensionTest, TerminationDuringExtensionStartDrainsServerBeforeReturning) {
    EXPECT_EXIT(RunFailure(Failure::TERMINATE_DURING_START), testing::ExitedWithCode(0), "");
}
} // namespace
} // namespace kv_cache_manager
