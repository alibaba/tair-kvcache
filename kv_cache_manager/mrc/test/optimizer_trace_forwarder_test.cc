#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/event/spec_events/optimizer_event.h"
#include "kv_cache_manager/mrc/optimizer_trace_forwarder.h"

namespace kv_cache_manager {
namespace {

std::shared_ptr<CacheGetEvent> MakeCacheGetEvent(const std::string &instance_id,
                                                 const std::vector<int64_t> &keys) {
    auto event = std::make_shared<CacheGetEvent>(instance_id);
    event->SetAddtionalArgs("query", keys, {}, {}, 0, {});
    return event;
}

} // namespace

TEST(OptimizerTraceForwarderTest, FiltersBeforeBoundedQueue) {
    OptimizerTraceForwarderConfig config;
    config.instance_allowlist = {"model-a"};
    OptimizerTraceForwarder forwarder(config, nullptr, nullptr);
    forwarder.InitBasicQueue(4);
    forwarder.running_ = true;

    EXPECT_TRUE(forwarder.Publish(MakeCacheGetEvent("model-b", {1, 2, 3})));
    EXPECT_EQ(0u, forwarder.BasicQueueSize());
    EXPECT_EQ(1u, forwarder.filtered_spans_.load());
    EXPECT_EQ(3u, forwarder.filtered_keys_.load());

    EXPECT_TRUE(forwarder.Publish(MakeCacheGetEvent("model-a", {4, 5})));
    EXPECT_EQ(1u, forwarder.BasicQueueSize());
    EXPECT_EQ(1u, forwarder.filtered_spans_.load());

    forwarder.running_ = false;
    forwarder.ClearBasicQueue();
}

TEST(OptimizerTraceForwarderTest, EmptyAllowlistPreservesAllInstances) {
    OptimizerTraceForwarderConfig config;
    OptimizerTraceForwarder forwarder(config, nullptr, nullptr);
    forwarder.InitBasicQueue(4);
    forwarder.running_ = true;

    EXPECT_TRUE(forwarder.Publish(MakeCacheGetEvent("model-b", {1})));
    EXPECT_EQ(1u, forwarder.BasicQueueSize());
    EXPECT_EQ(0u, forwarder.filtered_spans_.load());

    forwarder.running_ = false;
    forwarder.ClearBasicQueue();
}

} // namespace kv_cache_manager
