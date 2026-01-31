#pragma once

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <netinet/in.h>
#include <optional>
#include <signal.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

class ServiceProcessController {
    using PortsArray = std::array<int, 4>;

public:
    bool StartServiceInSubProcess(const std::filesystem::path &workspace_path) {
        if (!GenFreePorts()) {
            return false;
        }
        auto logger_conf_path = workspace_path / "integration_test/client_test/client_test_service_logger.conf";
        install_root_ = workspace_path / "integration_test/install_root";
        local_path_ = install_root_ / "usr/local";
        bin_path_ = local_path_ / "bin";
        fprintf(stdout,
                "install root [%s]\nlocal path [%s]\nbin path [%s]\n",
                install_root_.c_str(),
                local_path_.c_str(),
                bin_path_.c_str());
        std::stringstream ss;
        ss << "LD_LIBRARY_PATH=" << local_path_ << "/lib64:" << local_path_ << "/lib:$LD_LIBRARY_PATH"
           << " PATH=" << bin_path_ << ":$PATH"
           << " kv_cache_manager_bin"
           << " --log_config_file " << logger_conf_path << " --env kvcm.service.rpc_port=" << rpc_port()
           << " --env kvcm.service.http_port=" << http_port()
           << " --env kvcm.service.admin_rpc_port=" << admin_rpc_port()
           << " --env kvcm.service.admin_http_port=" << admin_http_port()
           << " --env kvcm.service.enable_debug_service=true"
           << " -d";
        auto cmd = ss.str();
        fprintf(stdout, "start service cmd [%s]\n", cmd.c_str());
        fflush(stderr);
        fflush(stdout);
        int return_code = std::system(cmd.c_str());
        if (return_code < 0) {
            fprintf(stderr, "start service failed code [%d]\n", return_code);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::seconds(2));
        auto service_pid_option = GetServiceProcessPid(service_ports_);
        if (!service_pid_option.has_value() || !service_pid_option.value().second) {
            return false;
        }
        service_pid_ = service_pid_option.value().first;
        return true;
    }

    void StopService() {
        fprintf(stdout, "stop service\n");
        fflush(stdout);
        kill(service_pid_, SIGTERM);
        while (true) {
            auto result = GetServiceProcessPid(service_ports_);
            if (result.has_value() && !result.value().second) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    inline int rpc_port() const { return service_ports_[0]; }
    inline int http_port() const { return service_ports_[1]; }
    inline int admin_rpc_port() const { return service_ports_[2]; }
    inline int admin_http_port() const { return service_ports_[3]; }

private:
    std::optional<std::string> ExecuteCommandGetOutput(const std::string &cmd) {
        fprintf(stdout, "execute cmd [%s]\n", cmd.c_str());
        auto defer = [](FILE *fp) { pclose(fp); };
        std::unique_ptr<FILE, decltype(defer)> fp(popen(cmd.c_str(), "r"), defer);
        if (fp == nullptr) {
            fprintf(stderr, "execute cmd [%s] failed\n", cmd.c_str());
            return std::nullopt;
        }
        std::array<char, 128> buffer;
        std::stringstream ss;
        while (fgets(buffer.data(), buffer.size(), fp.get()) != nullptr) {
            ss << buffer.data();
        }
        std::string cmd_result = ss.str();
        fprintf(stdout, "execute cmd result [%s]\n", cmd_result.c_str());
        return cmd_result;
    }

    std::optional<std::pair<pid_t, bool>> GetServiceProcessPid(const PortsArray &ports) {
        std::stringstream ss;
        ss << "pgrep -f 'kv_cache_manager_bin";
        for (int port : ports) {
            ss << ".*" << port;
        }
        ss << "'";
        std::string cmd = ss.str();
        auto result = ExecuteCommandGetOutput(cmd);
        if (!result.has_value()) {
            return std::nullopt;
        }
        std::string result_str = result.value();
        if (result_str.empty()) {
            fprintf(stdout, "not find service\n");
            return std::pair<pid_t, bool>{-1, false};
        }
        std::stringstream result_ss(result_str);
        int pid = -1;
        result_ss >> pid;
        return std::pair<pid_t, bool>{pid, true};
    }

    bool GenFreePorts() {
        for (int i = 0; i < service_ports_.size(); ++i) {
            auto port = GetFreePort();
            if (!port.has_value()) {
                return false;
            }
            service_ports_[i] = port.value();
        }
        return true;
    }

    std::optional<int> GetFreePort() {
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd == -1) {
            fprintf(stderr, "create socket fail\n");
            return std::nullopt;
        }

        struct sockaddr_in addr;
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;

        if (bind(sockfd, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
            fprintf(stderr, "bind socket fail\n");
            close(sockfd);
            return std::nullopt;
        }

        socklen_t len = sizeof(addr);
        if (getsockname(sockfd, (struct sockaddr *)&addr, &len) == -1) {
            fprintf(stderr, "getsockname fail\n");
            close(sockfd);
            return std::nullopt;
        }

        int port = ntohs(addr.sin_port);

        close(sockfd);
        return port;
    }

    pid_t service_pid_ = -1;
    PortsArray service_ports_;
    std::filesystem::path install_root_;
    std::filesystem::path local_path_;
    std::filesystem::path bin_path_;
};