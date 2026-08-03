#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/redis/redis_executor.h"
#include "kv_cache_manager/common/standard_uri.h"

namespace sw {
namespace redis {
class RedisCluster;
}
} // namespace sw

namespace kv_cache_manager {

struct RedisClusterOptions;

class RedisClusterExecutor : public IRedisExecutor {
public:
    static std::shared_ptr<RedisClusterExecutor> GetOrCreate(const StandardUri &uri, std::string &out_error);

    explicit RedisClusterExecutor(std::shared_ptr<const RedisClusterOptions> options);
    ~RedisClusterExecutor() override;

    RedisClusterExecutor(const RedisClusterExecutor &) = delete;
    RedisClusterExecutor &operator=(const RedisClusterExecutor &) = delete;

    bool Open() override;
    bool IsReady() const noexcept override;
    std::vector<RedisReplyUPtr> ExecuteBatch(const std::vector<RedisCmdArgs> &cmds) override;

private:
    struct ClusterScanNodeState {
        uint64_t cursor = 0;
        bool done = false;
    };

    struct PipelineCommand {
        size_t original_index;
        const RedisCmdArgs *cmd;
        std::string route_key;
    };

    static bool ParseUInt64(const std::string &value, uint64_t &out);
    static std::string Uppercase(const std::string &value);
    static RedisReplyUPtr BuildScanReply(const std::string &cursor, const std::vector<std::string> &keys);
    static bool ReplyString(const redisReply *reply, std::string &out);
    static std::string ReplyError(const redisReply *reply);
    static bool ParseClusterScanCursor(const std::string &cursor,
                                       std::map<std::string, ClusterScanNodeState> &out,
                                       std::string &out_error);
    static std::string EncodeClusterScanCursor(const std::map<std::string, ClusterScanNodeState> &states);
    static bool IsKeyedCommand(const std::string &command);
    static bool ExtractRouteKey(const RedisCmdArgs &cmd, std::string &out_key, std::string &out_error);
    static bool IsRedirectReply(const redisReply *reply);

    RedisReplyUPtr ExecuteRaw(const RedisCmdArgs &cmd, const std::string &route_key);
    RedisReplyUPtr ExecuteScan(const RedisCmdArgs &cmd);
    RedisReplyUPtr ExecuteBroadcast(const RedisCmdArgs &cmd);
    void ExecutePipeline(const std::vector<PipelineCommand> &commands, std::vector<RedisReplyUPtr> &out_replies);
    void ExecuteOne(const PipelineCommand &command, std::vector<RedisReplyUPtr> &out_replies);
    std::string GetNodeIdentity(const std::string &route_key);
    bool StartPipelineWorkers();
    void StopPipelineWorkers();
    bool SubmitPipelineTask(std::packaged_task<void()> &&task);
    void PipelineWorkerRoutine();

    std::shared_ptr<const RedisClusterOptions> options_;
    std::unique_ptr<sw::redis::RedisCluster> cluster_;
    std::mutex open_mutex_;
    std::mutex global_command_mutex_;
    std::vector<std::thread> pipeline_workers_;
    std::deque<std::packaged_task<void()>> pipeline_tasks_;
    std::mutex pipeline_mutex_;
    std::condition_variable pipeline_condition_;
    bool pipeline_stop_ = true;
    std::atomic<uint64_t> random_sequence_{0};
    std::atomic<bool> ready_{false};
};

} // namespace kv_cache_manager
