#include "client_test_base.h"

ServiceProcessController CLIENTTESTBASE::controller_ = ServiceProcessController();
std::filesystem::path CLIENTTESTBASE::workspace_path_ = "";