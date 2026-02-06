#include "kv_cache_manager/service/command_line.h"

#include <csignal>
#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <unistd.h>

#include "kv_cache_manager/service/server.h"
#include "kv_cache_manager/service/server_config.h"
namespace kv_cache_manager {

static std::mutex G_CMD_LOCK;
static CommandLine *G_CMD_LIST = nullptr;

void CommandArgs::ParseCmdLine(int argc, char *const argv[]) {
    int opt;
    const char *opt_string = "hvdl:c:e:";
    struct option long_opts[] = {{"help", 0, NULL, 'h'},
                                 {"version", 0, NULL, 'v'},
                                 {"daemon", 0, NULL, 'd'},
                                 {"log_config_file", 1, NULL, 'l'},
                                 {"config_file", 1, NULL, 'c'},
                                 {"env", 1, NULL, 'e'},
                                 {0, 0, 0, 0}};
    while ((opt = getopt_long(argc, argv, opt_string, long_opts, NULL)) != -1) {
        switch (opt) {
        case 'h':
            PrintUsage(argv[0]);
            exit(1);
        case 'v':
            fprintf(stderr, "BUILD_TIME: %s\nGIT: %s\n", SYS_GLB_BUILD_TIME, SYS_GLB_GIT_INFO);
            fprintf(stderr, "RPM VERSION: %s\n", SYS_GLB_VERSION);
            exit(1);
        case 'd':
            daemon_ = true;
            break;
        case 'l':
            if (optarg == nullptr) {
                fprintf(stderr, "ERROR: -l option requires an argument\n");
                exit(1);
            }
            log_config_file_ = std::string(optarg);
            break;
        case 'c':
            if (optarg == nullptr) {
                fprintf(stderr, "ERROR: -c option requires an argument\n");
                exit(1);
            }
            config_file_ = std::string(optarg);
            break;
        case 'e': {
            if (optarg == nullptr) {
                fprintf(stderr, "ERROR: -e option requires an argument\n");
                exit(1);
            }
            auto env_kv_pair = std::string(optarg);
            auto env_p = env_kv_pair.find('=');
            if (env_p == std::string::npos) {
                fprintf(stderr, "invalid env: %s\n", env_kv_pair.c_str());
                exit(1);
            }
            std::string env_key = env_kv_pair.substr(0, env_p);
            std::string env_value = env_kv_pair.substr(env_p + 1);
            environ_.insert({env_key, env_value});
            break;
        }
        default:
            fprintf(stderr, "invalid option: %c\n", opt);
        }
    }
}

void CommandArgs::PrintUsage(char *prog_name) {
    fprintf(stderr,
            "%s\n"
            "    -c, --config_file      config file path\n"
            "    -l, --log_config_file  config file path for logger\n"
            "    -e, --env              set env for kv_cache_manager, e.g., -e a=1 -e b=2\n"
            "    -d, --daemon           run worker as daemon\n"
            "    -h, --help             display this help and exit\n"
            "    -v, --version          version and build time\n"
            "\n",
            prog_name);
}

int CommandLine::Run(int argc, const char *argv[]) {
    // TODO
    // 1. parse args
    // 2. set env
    // 3. init Config
    // 4. daemon
    // 5. singal
    // 6. init logger
    // 7. make server by factory
    // 8. pidfile
    // 9. wait
    // 10. clear resource

    fprintf(stdout, "Start run command %p.\n", this);

    {
        std::unique_lock<std::mutex> lock(G_CMD_LOCK);
        next_ = G_CMD_LIST;
        G_CMD_LIST = this;
    }

    CommandArgs args;
    args.ParseCmdLine(argc, const_cast<char **>(argv));

    ServerConfig config;
    bool ret = config.Parse(args.GetConfigFile(), args.GetEnviron());
    if (!ret) {
        fprintf(stderr, "Parse config failed.\n");
        return -1;
    }

    if (args.IsDaemon() && !StartDaemon()) {
        fprintf(stderr, "Daemon failed.\n");
        return -1;
    }

    InitLogger(args.GetLogConfigFile());
    UpdateLogLevel(config.GetLogLevel());
    LoggerBroker::InitLogLevelFromEnv();

    RegisterSignal();

    server_.reset(new Server);
    server_->Init(config);
    server_->Start();
    server_->Wait();
    server_.reset();
    LoggerBroker::DestroyLogger();
    return 0;
}

void CommandLine::Terminate() {
    // TODO
    server_->Stop();
}

bool CommandLine::StartDaemon() {
    if (daemon(1, 1) == -1) {
        return false;
    }
    // 禁用标准输入
    int fd = open("/dev/null", 0);
    if (fd != -1) {
        dup2(fd, 0);
        close(fd);
    }
    return true;
}

void __sign_handler__(int sig) {
    std::unique_lock<std::mutex> lock(G_CMD_LOCK);
    switch (sig) {
    case SIGUSR1:
    case SIGUSR2:
    case SIGTERM:
    case SIGINT: {
        fprintf(stdout, "received singnal %d, try stop server.", sig);
        for (CommandLine *p = G_CMD_LIST; p; p = p->GetNext()) {
            p->Terminate();
        }
        break;
    }
    }
}

void CommandLine::InitLogger(const std::string &log_config_file) { LoggerBroker::InitLogger(log_config_file, false); }

void CommandLine::UpdateLogLevel(uint32_t log_level) {
    if (log_level == 0) {
        log_level = Logger::LEVEL_INFO;
    }
    LoggerBroker::SetLogLevel(log_level);
}

void CommandLine::RegisterSignal() {
    signal(SIGPIPE, SIG_IGN);
    signal(SIGUSR1, __sign_handler__);
    signal(SIGUSR2, __sign_handler__);
    signal(SIGTERM, __sign_handler__);
    signal(SIGINT, __sign_handler__);
}

} // namespace kv_cache_manager
