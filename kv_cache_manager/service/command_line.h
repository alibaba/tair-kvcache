#pragma once

#include <csignal>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace kv_cache_manager {

static constexpr const char *KVCM_ART = R"D(
__/\\\________/\\\__/\\\________/\\\________/\\\\\\\\\__/\\\\____________/\\\\
 _\/\\\_____/\\\//__\/\\\_______\/\\\_____/\\\////////__\/\\\\\\________/\\\\\\
  _\/\\\__/\\\//_____\//\\\______/\\\____/\\\/___________\/\\\//\\\____/\\\//\\\
   _\/\\\\\\//\\\______\//\\\____/\\\____/\\\_____________\/\\\\///\\\/\\\/_\/\\\
    _\/\\\//_\//\\\______\//\\\__/\\\____\/\\\_____________\/\\\__\///\\\/___\/\\\
     _\/\\\____\//\\\______\//\\\/\\\_____\//\\\____________\/\\\____\///_____\/\\\
      _\/\\\_____\//\\\______\//\\\\\_______\///\\\__________\/\\\_____________\/\\\
       _\/\\\______\//\\\______\//\\\__________\////\\\\\\\\\_\/\\\_____________\/\\\
        _\///________\///________\///______________\/////////__\///______________\///_
)D";

// TODO 生成
static constexpr const char *SYS_GLB_BUILD_TIME = "2020-08-08 08:08:08";
static constexpr const char *SYS_GLB_GIT_INFO = "...todo...git_info..";
static constexpr const char *SYS_GLB_VERSION = "1.0.0";

class Server;
class CommandLine;

class CommandArgs {
public:
    void ParseCmdLine(int argc, char *const argv[]);

    const std::string &GetConfigFile() { return config_file_; }

    const std::string &GetLogConfigFile() { return log_config_file_; }

    bool IsDaemon() { return daemon_; }

    const std::unordered_map<std::string, std::string> &GetEnviron() { return environ_; }

private:
    void PrintUsage(char *prog_name);

private:
    bool daemon_ = false;
    std::string log_config_file_;
    std::string config_file_;
    std::unordered_map<std::string, std::string> environ_;
};

class CommandLine {
public:
    int Run(int argc, const char *argv[]);
    CommandLine *GetNext() { return next_; };
    void Terminate();

private:
    void InitLogger(const std::string &log_config_file);
    void UpdateLogLevel(uint32_t log_level);
    void RegisterSignal();
    bool StartDaemon();

private:
    std::shared_ptr<Server> server_;
    CommandLine *next_ = nullptr;
};

} // namespace kv_cache_manager
