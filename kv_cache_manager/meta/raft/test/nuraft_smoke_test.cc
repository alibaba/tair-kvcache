#include <libnuraft/nuraft.hxx>

#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {
namespace raft_test {

class NuRaftSmokeTest : public TESTBASE {};

// 构造 raft_params，并验证默认值可读、链式 setter 可调用，证明 NuRaft 头/库都能链接成功。
TEST_F(NuRaftSmokeTest, RaftParamsDefaults) {
    nuraft::raft_params params;
    EXPECT_EQ(500, params.election_timeout_upper_bound_);
    EXPECT_EQ(250, params.election_timeout_lower_bound_);
    EXPECT_EQ(125, params.heart_beat_interval_);

    params.with_election_timeout_upper(800).with_election_timeout_lower(400);
    EXPECT_EQ(800, params.election_timeout_upper_bound_);
    EXPECT_EQ(400, params.election_timeout_lower_bound_);
}

TEST_F(NuRaftSmokeTest, BufferRoundTrip) {
    auto buf = nuraft::buffer::alloc(64);
    ASSERT_TRUE(buf);
    buf->put(static_cast<int32_t>(42));
    buf->pos(0);
    EXPECT_EQ(42, buf->get_int());
}

TEST_F(NuRaftSmokeTest, SrvConfigBasics) {
    nuraft::srv_config cfg(7, "127.0.0.1:9001");
    EXPECT_EQ(7, cfg.get_id());
    EXPECT_EQ("127.0.0.1:9001", cfg.get_endpoint());
}

} // namespace raft_test
} // namespace kv_cache_manager
