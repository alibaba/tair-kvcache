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