#include <string>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/service/server.h"

using namespace kv_cache_manager;

class ServerLifecycleTest : public TESTBASE {
protected:
    void TearDown() override {
        if (server_initialized_) {
            server_.Stop();
        }
    }

    bool StartRpcServer(bool enable_kv_meta) {
        ServerConfig config;
        if (!config.Parse("",
                          {{"kvcm.service.rpc_port", "0"},
                           {"kvcm.service.admin_rpc_port", "0"},
                           {"kvcm.kv_meta.enabled", enable_kv_meta ? "true" : "false"}}) ||
            !server_.Init(config)) {
            return false;
        }
        server_initialized_ = true;
        return server_.StartRpcServer();
    }

    Server server_;
    bool server_initialized_ = false;
};

TEST_F(ServerLifecycleTest, KvMetaAndFixedBlockMetaSharePrimaryRpcListener) {
    ASSERT_TRUE(StartRpcServer(true));

    EXPECT_NE(nullptr, server_.rpc_server_);
    EXPECT_NE(nullptr, server_.meta_service_);
    EXPECT_NE(nullptr, server_.kv_meta_service_);
    EXPECT_EQ(nullptr, server_.admin_rpc_server_);
}

TEST_F(ServerLifecycleTest, DisabledKvMetaDoesNotAlterPrimaryMetaListener) {
    ASSERT_TRUE(StartRpcServer(false));

    EXPECT_NE(nullptr, server_.rpc_server_);
    EXPECT_NE(nullptr, server_.meta_service_);
    EXPECT_EQ(nullptr, server_.kv_meta_service_);
    EXPECT_EQ(nullptr, server_.kv_meta_manager_);
}
