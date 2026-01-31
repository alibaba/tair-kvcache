#include "kv_cache_manager/service/command_line.h"

int main(int argc, const char *argv[]) {
    kv_cache_manager::CommandLine cmd;
    return cmd.Run(argc, argv);
}
