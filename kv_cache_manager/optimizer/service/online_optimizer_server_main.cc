#include <csignal>
#include <getopt.h>
#include <stdio.h>
#include <string>
#include <unordered_map>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/service/online_optimizer_server.h"

namespace {
kv_cache_manager::OnlineOptimizerServer *g_server = nullptr;

void SignalHandler(int sig) {
    // Avoid non-async-signal-safe operations (logging, locking, thread joins).
    // Only set the shutdown flag and unblock WaitForShutdown(); full cleanup
    // happens in the main thread after grpc_server_->Wait() returns.
    if (g_server) {
        g_server->RequestShutdown();
    }
}

void PrintUsage(const char *prog_name) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "    -c, --config       config file path (JSON)\n"
            "    -e, --env          set config override, e.g., -e kvcm_optimizer.rpc_port=50053\n"
            "    -h, --help         display this help and exit\n"
            "\n",
            prog_name);
}
} // namespace

int main(int argc, char **argv) {
    std::string config_file;
    std::unordered_map<std::string, std::string> environ;

    const char *opt_string = "hc:e:";
    struct option long_opts[] = {
        {"help", 0, nullptr, 'h'}, {"config", 1, nullptr, 'c'}, {"env", 1, nullptr, 'e'}, {0, 0, 0, 0}};
    int opt;
    while ((opt = getopt_long(argc, argv, opt_string, long_opts, nullptr)) != -1) {
        switch (opt) {
        case 'h':
            PrintUsage(argv[0]);
            return 0;
        case 'c':
            config_file = optarg;
            break;
        case 'e': {
            std::string kv(optarg);
            auto pos = kv.find('=');
            if (pos == std::string::npos) {
                fprintf(stderr, "invalid env: %s (expected key=value)\n", optarg);
                return 1;
            }
            environ[kv.substr(0, pos)] = kv.substr(pos + 1);
            break;
        }
        default:
            PrintUsage(argv[0]);
            return 1;
        }
    }

    kv_cache_manager::LoggerBroker::InitLogger("");

    if (config_file.empty()) {
        KVCM_LOG_ERROR("--config / -c is required");
        return 1;
    }

    kv_cache_manager::OnlineOptimizerServer server;
    g_server = &server;

    std::signal(SIGINT, SignalHandler);
    std::signal(SIGTERM, SignalHandler);

    if (!server.Init(config_file, environ)) {
        KVCM_LOG_ERROR("Server init failed");
        return 1;
    }
    if (!server.Start()) {
        KVCM_LOG_ERROR("Server start failed");
        return 1;
    }

    server.WaitForShutdown();
    return 0;
}
