#pragma once

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <grpcpp/grpcpp.h>
#include <netinet/in.h>
#include <optional>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <signal.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "service/proto/admin_service.grpc.pb.h"

class ServiceProcessController {
    using PortsArray = std::array<int, 4>;

public:
    ~ServiceProcessController() { StopService(); }

    bool StartServiceInSubProcess(const std::filesystem::path &workspace_path) {
        if (!GenFreePorts()) {
            return false;
        }
        const auto local_path = workspace_path / "integration_test/install_root/usr/local";
        const auto *test_tmpdir = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
        if (!test_tmpdir) {
            test_tmpdir = std::getenv("TEST_TMPDIR");
        }
        auto path_template =
            ((test_tmpdir ? std::filesystem::path(test_tmpdir) : std::filesystem::temp_directory_path()) /
             "client-service-XXXXXX")
                .string();
        const auto *workdir = mkdtemp(path_template.data());
        if (!workdir) {
            return false;
        }
        workdir_ = workdir;
        std::filesystem::create_directories(workdir_ / "logs");
        std::filesystem::create_directories(workdir_ / "nfs");

        const auto config_path = workspace_path / "package/etc/default_startup_config.json";
        std::ifstream config_file(config_path);
        std::stringstream config_text;
        config_text << config_file.rdbuf();
        rapidjson::Document config;
        config.Parse(config_text.str().c_str());
        if (config.HasParseError() || !config.IsObject() || !config.HasMember("storage_config")) {
            fprintf(stderr, "cannot read startup configuration: %s\n", config_path.c_str());
            return false;
        }
        const auto nfs_path = (workdir_ / "nfs").string() + "/";
        config["storage_config"]["storage_spec"]["root_path"].SetString(nfs_path.c_str(), config.GetAllocator());
        rapidjson::StringBuffer serialized;
        rapidjson::Writer<rapidjson::StringBuffer> writer(serialized);
        config.Accept(writer);
        std::ofstream startup_config(workdir_ / "startup.json");
        startup_config << serialized.GetString();
        startup_config.close();
        if (!startup_config) {
            return false;
        }

        std::vector<std::string> args = {
            (local_path / "bin/kv_cache_manager_bin").string(),
            "--log_config_file",
            (workspace_path / "integration_test/client_test/client_test_service_logger.conf").string(),
            "--env",
            "kvcm.service.rpc_port=" + std::to_string(rpc_port()),
            "--env",
            "kvcm.service.http_port=" + std::to_string(http_port()),
            "--env",
            "kvcm.service.admin_rpc_port=" + std::to_string(admin_rpc_port()),
            "--env",
            "kvcm.service.admin_http_port=" + std::to_string(admin_http_port()),
            "--env",
            "kvcm.service.enable_debug_service=true",
            "--env",
            "kvcm.startup_config=" + (workdir_ / "startup.json").string(),
        };
        std::vector<char *> argv;
        for (auto &arg : args) {
            argv.push_back(arg.data());
        }
        argv.push_back(nullptr);
        const auto *old_library_path = std::getenv("LD_LIBRARY_PATH");
        const auto library_path = (local_path / "lib64").string() + ":" + (local_path / "lib").string() + ":" +
                                  (old_library_path ? old_library_path : "");
        fprintf(stdout, "client test service workdir [%s], rpc port [%d]\n", workdir_.c_str(), rpc_port());
        fflush(nullptr);
        service_pid_ = fork();
        if (service_pid_ == 0) {
            if (chdir(workdir_.c_str()) != 0 || !freopen("stdout.log", "w", stdout) ||
                !freopen("stderr.log", "w", stderr) || setenv("LD_LIBRARY_PATH", library_path.c_str(), 1) != 0) {
                _exit(127);
            }
            execv(argv[0], argv.data());
            _exit(127);
        }
        if (service_pid_ < 0) {
            return false;
        }

        // A successful leader-only request proves startup configuration and recovery completed.
        auto stub = kv_cache_manager::proto::admin::AdminService::NewStub(
            grpc::CreateChannel("127.0.0.1:" + std::to_string(admin_rpc_port()), grpc::InsecureChannelCredentials()));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (std::chrono::steady_clock::now() < deadline && !ReapExited()) {
            grpc::ClientContext context;
            context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(1));
            kv_cache_manager::proto::admin::GetInstanceGroupRequest request;
            request.set_name("default");
            kv_cache_manager::proto::admin::GetInstanceGroupResponse response;
            if (stub->GetInstanceGroup(&context, request, &response).ok() &&
                response.header().status().code() == kv_cache_manager::proto::admin::OK) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        fprintf(stderr, "service did not become ready; logs: %s\n", workdir_.c_str());
        StopService();
        return false;
    }

    bool StopService() {
        if (service_pid_ <= 0) {
            return true;
        }
        if (!ReapExited()) {
            kill(service_pid_, SIGTERM);
            if (!WaitForExit(std::chrono::seconds(10))) {
                fprintf(stderr, "service [%d] did not stop; logs: %s\n", service_pid_, workdir_.c_str());
                kill(service_pid_, SIGKILL);
                WaitForExit(std::chrono::seconds(5));
                return false;
            }
        }
        return WIFEXITED(service_exit_status_) && WEXITSTATUS(service_exit_status_) == 0;
    }

    int rpc_port() const { return service_ports_[0]; }
    int http_port() const { return service_ports_[1]; }
    int admin_rpc_port() const { return service_ports_[2]; }
    int admin_http_port() const { return service_ports_[3]; }

private:
    bool ReapExited() {
        const auto result = waitpid(service_pid_, &service_exit_status_, WNOHANG);
        if (result == service_pid_ || (result == -1 && errno == ECHILD)) {
            service_pid_ = -1;
            return true;
        }
        return false;
    }

    bool WaitForExit(std::chrono::seconds timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        do {
            if (ReapExited()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        } while (std::chrono::steady_clock::now() < deadline);
        return false;
    }

    bool GenFreePorts() {
        // Hold all four reservations together so the OS cannot select the same port twice.
        std::array<int, 4> sockets;
        sockets.fill(-1);
        bool success = true;
        for (size_t i = 0; i < service_ports_.size(); ++i) {
            auto port = GetFreePort(sockets[i]);
            if (!port.has_value()) {
                success = false;
                break;
            }
            service_ports_[i] = *port;
        }
        for (auto fd : sockets) {
            if (fd >= 0)
                close(fd);
        }
        return success;
    }

    std::optional<int> GetFreePort(int &sockfd) {
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd == -1) {
            return std::nullopt;
        }
        struct sockaddr_in addr = {};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        if (bind(sockfd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) == -1) {
            return std::nullopt;
        }
        socklen_t len = sizeof(addr);
        if (getsockname(sockfd, reinterpret_cast<struct sockaddr *>(&addr), &len) == -1) {
            return std::nullopt;
        }
        return ntohs(addr.sin_port);
    }

    pid_t service_pid_ = -1;
    int service_exit_status_ = 0;
    PortsArray service_ports_{};
    std::filesystem::path workdir_;
};
