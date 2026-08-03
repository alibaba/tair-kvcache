#include "kv_cache_manager/common/redis/redis_cluster_executor.h"

#include <cassert>
#include <cctype>
#include <charconv>
#include <chrono>
#include <climits>
#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <sstream>
#include <sw/redis++/redis++.h>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

struct RedisClusterOptions {
    std::string host;
    int port = 6379;
    std::string username = "default";
    std::string password;
    bool tls_enabled = false;
    std::string ca_file;
    std::string ca_dir;
    std::string client_cert_file;
    std::string client_key_file;
    std::string sni;
    int64_t connect_timeout_ms = 1000;
    int64_t socket_timeout_ms = 2000;
    int64_t topology_refresh_ms = 10000;
    int64_t pool_size_per_node = 32;
    int64_t pipeline_worker_count = 8;

    static bool ParseInt64(const std::string &value, int64_t &out);
    static bool ParseIntegerParam(const StandardUri &uri,
                                  const std::string &name,
                                  int64_t min_value,
                                  int64_t max_value,
                                  int64_t &in_out_value,
                                  std::string &out_error);
    static bool
    ParseBoolParam(const StandardUri &uri, const std::string &name, bool &in_out_value, std::string &out_error);
    static bool FromUri(const StandardUri &uri, RedisClusterOptions &out, std::string &out_error);
    std::string Identity() const;
    std::string RedactedDescription() const;
};

using ReplyUPtr = RedisReplyUPtr;
using CmdArgs = RedisCmdArgs;

bool RedisClusterOptions::ParseInt64(const std::string &value, int64_t &out) {
    if (value.empty()) {
        return false;
    }
    std::from_chars_result result = std::from_chars(value.data(), value.data() + value.size(), out);
    return result.ec == std::errc() && result.ptr == value.data() + value.size();
}

bool RedisClusterOptions::ParseIntegerParam(const StandardUri &uri,
                                            const std::string &name,
                                            int64_t min_value,
                                            int64_t max_value,
                                            int64_t &in_out_value,
                                            std::string &out_error) {
    if (!uri.HasParam(name)) {
        return true;
    }
    int64_t value = 0;
    if (!ParseInt64(uri.GetParam(name), value) || value < min_value || value > max_value) {
        out_error = "invalid redis cluster parameter: " + name;
        return false;
    }
    in_out_value = value;
    return true;
}

bool RedisClusterOptions::ParseBoolParam(const StandardUri &uri,
                                         const std::string &name,
                                         bool &in_out_value,
                                         std::string &out_error) {
    if (!uri.HasParam(name)) {
        return true;
    }
    const std::string value = uri.GetParam(name);
    if (value == "true" || value == "1") {
        in_out_value = true;
        return true;
    }
    if (value == "false" || value == "0") {
        in_out_value = false;
        return true;
    }
    out_error = "invalid redis cluster parameter: " + name;
    return false;
}

bool RedisClusterOptions::FromUri(const StandardUri &uri, RedisClusterOptions &out, std::string &out_error) {
    if (uri.GetProtocol() != "redis_cluster") {
        out_error = "redis cluster uri must use redis_cluster://";
        return false;
    }
    if (uri.GetHostName().empty()) {
        out_error = "redis cluster host is empty";
        return false;
    }
    out.host = uri.GetHostName();
    if (uri.GetPort() < 0 || uri.GetPort() > 65535) {
        out_error = "invalid redis cluster port";
        return false;
    }
    if (uri.GetPort() > 0) {
        out.port = static_cast<int>(uri.GetPort());
    }

    int64_t db = 0;
    if (!ParseIntegerParam(uri, "db", 0, 0, db, out_error)) {
        return false;
    }
    if (!ParseIntegerParam(uri, "connect_timeout_ms", 1, INT32_MAX, out.connect_timeout_ms, out_error) ||
        !ParseIntegerParam(uri, "socket_timeout_ms", 1, INT32_MAX, out.socket_timeout_ms, out_error) ||
        !ParseIntegerParam(uri, "cluster_topology_refresh_ms", 1, INT32_MAX, out.topology_refresh_ms, out_error) ||
        !ParseIntegerParam(uri, "cluster_pool_size_per_node", 1, INT32_MAX, out.pool_size_per_node, out_error) ||
        !ParseIntegerParam(uri, "cluster_pipeline_worker_count", 1, INT32_MAX, out.pipeline_worker_count, out_error) ||
        !ParseBoolParam(uri, "tls", out.tls_enabled, out_error)) {
        return false;
    }

    const std::string user_info = uri.GetUserInfo();
    if (!user_info.empty()) {
        const size_t separator = user_info.find(':');
        if (separator == std::string::npos || separator == 0 || separator + 1 == user_info.size()) {
            out_error = "redis cluster user info must be USERNAME:TOKEN";
            return false;
        }
        out.username = user_info.substr(0, separator);
        out.password = user_info.substr(separator + 1);
    }
    out.ca_file = uri.GetParam("ca_file");
    out.ca_dir = uri.GetParam("ca_dir");
    out.client_cert_file = uri.GetParam("client_cert_file");
    out.client_key_file = uri.GetParam("client_key_file");
    out.sni = uri.GetParam("sni");
    if (out.tls_enabled && out.ca_file.empty() && out.ca_dir.empty()) {
        out_error = "redis cluster TLS requires ca_file or ca_dir";
        return false;
    }
    return true;
}

std::string RedisClusterOptions::Identity() const {
    const size_t credential_hash = std::hash<std::string>()(username + "\n" + password);
    std::ostringstream stream;
    stream << host << ':' << port << '|' << std::hex << credential_hash << std::dec << '|' << tls_enabled << '|'
           << ca_file << '|' << ca_dir << '|' << client_cert_file << '|' << client_key_file << '|' << sni << '|'
           << connect_timeout_ms << '|' << socket_timeout_ms << '|' << topology_refresh_ms << '|' << pool_size_per_node
           << '|' << pipeline_worker_count;
    return stream.str();
}

std::string RedisClusterOptions::RedactedDescription() const {
    std::ostringstream stream;
    if (!password.empty()) {
        stream << "redis_cluster://***@";
    } else {
        stream << "redis_cluster://";
    }
    stream << host << ':' << port;
    return stream.str();
}

bool RedisClusterExecutor::ParseUInt64(const std::string &value, uint64_t &out) {
    if (value.empty()) {
        return false;
    }
    std::from_chars_result result = std::from_chars(value.data(), value.data() + value.size(), out);
    return result.ec == std::errc() && result.ptr == value.data() + value.size();
}

std::string RedisClusterExecutor::Uppercase(const std::string &value) {
    std::string result = value;
    for (char &ch : result) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return result;
}

ReplyUPtr RedisClusterExecutor::BuildScanReply(const std::string &cursor, const std::vector<std::string> &keys) {
    const auto append_bulk_string = [](const std::string &value, std::string &out) {
        out.append("$");
        out.append(std::to_string(value.size()));
        out.append("\r\n");
        out.append(value);
        out.append("\r\n");
    };

    std::string resp = "*2\r\n";
    append_bulk_string(cursor, resp);
    resp.append("*");
    resp.append(std::to_string(keys.size()));
    resp.append("\r\n");
    for (const std::string &key : keys) {
        append_bulk_string(key, resp);
    }

    redisReader *reader = redisReaderCreate();
    if (!reader) {
        KVCM_LOG_ERROR("create redis reader for cluster SCAN reply failed");
        return ReplyUPtr();
    }
    if (redisReaderFeed(reader, resp.data(), resp.size()) != REDIS_OK) {
        KVCM_LOG_ERROR("feed cluster SCAN reply to redis reader failed");
        redisReaderFree(reader);
        return ReplyUPtr();
    }
    void *raw_reply = nullptr;
    if (redisReaderGetReply(reader, &raw_reply) != REDIS_OK) {
        KVCM_LOG_ERROR("parse cluster SCAN reply failed");
        redisReaderFree(reader);
        freeReplyObject(raw_reply);
        return ReplyUPtr();
    }
    redisReaderFree(reader);
    return ReplyUPtr(static_cast<redisReply *>(raw_reply));
}

bool RedisClusterExecutor::ReplyString(const redisReply *reply, std::string &out) {
    if (!reply || (reply->type != REDIS_REPLY_STRING && reply->type != REDIS_REPLY_STATUS) || !reply->str) {
        return false;
    }
    out.assign(reply->str, reply->len);
    return true;
}

std::string RedisClusterExecutor::ReplyError(const redisReply *reply) {
    if (!reply) {
        return "empty redis reply";
    }
    if (reply->type == REDIS_REPLY_ERROR && reply->str) {
        return std::string(reply->str, reply->len);
    }
    return "unexpected redis reply";
}

bool RedisClusterExecutor::ParseClusterScanCursor(const std::string &cursor,
                                                  std::map<std::string, ClusterScanNodeState> &out,
                                                  std::string &out_error) {
    out.clear();
    if (cursor == "0") {
        return true;
    }
    static const std::string prefix = "rc1:";
    if (cursor.compare(0, prefix.size(), prefix) != 0 || cursor.size() == prefix.size()) {
        out_error = "invalid redis cluster scan cursor";
        return false;
    }
    size_t begin = prefix.size();
    while (begin < cursor.size()) {
        size_t end = cursor.find(',', begin);
        if (end == std::string::npos) {
            end = cursor.size();
        }
        const size_t equal = cursor.find('=', begin);
        if (equal == std::string::npos || equal >= end || equal == begin) {
            out_error = "invalid redis cluster scan cursor";
            return false;
        }
        const std::string node_id = cursor.substr(begin, equal - begin);
        const std::string value = cursor.substr(equal + 1, end - equal - 1);
        ClusterScanNodeState state;
        if (value == "d") {
            state.done = true;
        } else if (!ParseUInt64(value, state.cursor)) {
            out_error = "invalid redis cluster scan cursor";
            return false;
        }
        if (!out.emplace(node_id, state).second) {
            out_error = "duplicate node in redis cluster scan cursor";
            return false;
        }
        begin = end + 1;
    }
    return true;
}

std::string RedisClusterExecutor::EncodeClusterScanCursor(const std::map<std::string, ClusterScanNodeState> &states) {
    std::string cursor = "rc1:";
    bool first = true;
    for (const std::pair<const std::string, ClusterScanNodeState> &entry : states) {
        if (!first) {
            cursor.push_back(',');
        }
        cursor.append(entry.first);
        cursor.push_back('=');
        cursor.append(entry.second.done ? "d" : std::to_string(entry.second.cursor));
        first = false;
    }
    return cursor;
}

bool RedisClusterExecutor::IsKeyedCommand(const std::string &command) {
    static const std::unordered_set<std::string> commands = {
        "DEL", "HSET", "HDEL", "HMGET", "HGETALL", "EXISTS", "HSCAN", "GET", "SET", "PTTL", "PEXPIRE"};
    return commands.find(command) != commands.end();
}

bool RedisClusterExecutor::ExtractRouteKey(const CmdArgs &cmd, std::string &out_key, std::string &out_error) {
    if (cmd.empty()) {
        out_error = "empty redis cluster command";
        return false;
    }
    const std::string command = Uppercase(cmd[0]);
    if (IsKeyedCommand(command)) {
        if (cmd.size() < 2) {
            out_error = command + " requires a key";
            return false;
        }
        out_key = cmd[1];
        return true;
    }
    if (command == "EVAL" || command == "EVALSHA") {
        if (cmd.size() < 4) {
            out_error = command + " requires at least one key";
            return false;
        }
        int64_t key_count = 0;
        if (!RedisClusterOptions::ParseInt64(cmd[2], key_count) || key_count <= 0 ||
            static_cast<size_t>(key_count) + 3 > cmd.size()) {
            out_error = command + " has invalid key count";
            return false;
        }
        out_key = cmd[3];
        return true;
    }
    out_error = "unsupported redis cluster command: " + command;
    return false;
}

ReplyUPtr RedisClusterExecutor::ExecuteRaw(const CmdArgs &cmd, const std::string &route_key) {
    const auto sender = [&cmd](sw::redis::Connection &connection, const sw::redis::StringView &) {
        sw::redis::CmdArgs args;
        for (const std::string &arg : cmd) {
            args.append(sw::redis::StringView(arg.data(), arg.size()));
        }
        connection.send(args);
    };
    return cluster_->command(sender, sw::redis::StringView(route_key.data(), route_key.size()));
}

ReplyUPtr RedisClusterExecutor::ExecuteScan(const CmdArgs &cmd) {
    if (cmd.size() != 6 || Uppercase(cmd[2]) != "MATCH" || Uppercase(cmd[4]) != "COUNT") {
        KVCM_LOG_ERROR("invalid SCAN command for redis cluster");
        return ReplyUPtr();
    }
    int64_t limit = 0;
    if (!RedisClusterOptions::ParseInt64(cmd[5], limit) || limit <= 0) {
        KVCM_LOG_ERROR("invalid SCAN count for redis cluster");
        return ReplyUPtr();
    }

    std::map<std::string, ClusterScanNodeState> previous_states;
    std::string error;
    if (!ParseClusterScanCursor(cmd[1], previous_states, error)) {
        KVCM_LOG_ERROR("%s", error.c_str());
        return ReplyUPtr();
    }

    std::map<std::string, ClusterScanNodeState> next_states;
    std::vector<std::string> keys;
    cluster_->for_each([&](sw::redis::Redis &node) {
        if (!error.empty()) {
            return;
        }
        const CmdArgs myid_cmd = {"CLUSTER", "MYID"};
        sw::redis::ReplyUPtr myid_reply = node.command(myid_cmd.begin(), myid_cmd.end());
        std::string node_id;
        if (!ReplyString(myid_reply.get(), node_id)) {
            error = "cannot get redis cluster node id: " + ReplyError(myid_reply.get());
            return;
        }

        ClusterScanNodeState state;
        std::map<std::string, ClusterScanNodeState>::const_iterator previous = previous_states.find(node_id);
        if (previous != previous_states.end()) {
            state = previous->second;
        }
        if (!state.done && keys.size() < static_cast<size_t>(limit)) {
            const CmdArgs scan_cmd = {
                "SCAN", std::to_string(state.cursor), "MATCH", cmd[3], "COUNT", std::to_string(limit)};
            sw::redis::ReplyUPtr scan_reply = node.command(scan_cmd.begin(), scan_cmd.end());
            const redisReply *reply = scan_reply.get();
            if (!reply || reply->type == REDIS_REPLY_ERROR || reply->type != REDIS_REPLY_ARRAY ||
                reply->elements != 2 || !reply->element[0] || !reply->element[1] ||
                reply->element[1]->type != REDIS_REPLY_ARRAY) {
                error = "redis cluster node SCAN failed: " + ReplyError(reply);
                return;
            }
            std::string next_cursor;
            if (!ReplyString(reply->element[0], next_cursor) || !ParseUInt64(next_cursor, state.cursor)) {
                error = "redis cluster node returned invalid SCAN cursor";
                return;
            }
            state.done = state.cursor == 0;
            for (size_t i = 0; i < reply->element[1]->elements; ++i) {
                std::string key;
                if (!ReplyString(reply->element[1]->element[i], key)) {
                    error = "redis cluster node returned invalid SCAN key";
                    return;
                }
                keys.emplace_back(std::move(key));
            }
        }
        next_states[node_id] = state;
    });

    if (!error.empty()) {
        KVCM_LOG_ERROR("%s", error.c_str());
        return ReplyUPtr();
    }
    if (next_states.empty()) {
        KVCM_LOG_ERROR("redis cluster has no master nodes");
        return ReplyUPtr();
    }
    bool all_done = true;
    for (const std::pair<const std::string, ClusterScanNodeState> &entry : next_states) {
        if (!entry.second.done) {
            all_done = false;
            break;
        }
    }
    return BuildScanReply(all_done ? "0" : EncodeClusterScanCursor(next_states), keys);
}

ReplyUPtr RedisClusterExecutor::ExecuteBroadcast(const CmdArgs &cmd) {
    std::vector<ReplyUPtr> replies;
    std::string error;
    cluster_->for_each([&](sw::redis::Redis &node) {
        if (!error.empty()) {
            return;
        }
        sw::redis::ReplyUPtr node_reply = node.command(cmd.begin(), cmd.end());
        if (!node_reply || node_reply->type == REDIS_REPLY_ERROR) {
            error = ReplyError(node_reply.get());
            return;
        }
        replies.emplace_back(std::move(node_reply));
    });
    if (!error.empty()) {
        KVCM_LOG_ERROR("redis cluster broadcast failed: %s", error.c_str());
        return ReplyUPtr();
    }
    if (replies.empty()) {
        KVCM_LOG_ERROR("redis cluster has no master nodes");
        return ReplyUPtr();
    }

    const std::string command = Uppercase(cmd[0]);
    if (command == "SCRIPT" && cmd.size() >= 2 && Uppercase(cmd[1]) == "LOAD") {
        std::string sha;
        if (!ReplyString(replies[0].get(), sha)) {
            KVCM_LOG_ERROR("invalid SCRIPT LOAD reply");
            return ReplyUPtr();
        }
        for (size_t i = 1; i < replies.size(); ++i) {
            std::string node_sha;
            if (!ReplyString(replies[i].get(), node_sha) || node_sha != sha) {
                KVCM_LOG_ERROR("SCRIPT LOAD returned inconsistent SHA values");
                return ReplyUPtr();
            }
        }
    } else if (command == "SCRIPT" && cmd.size() >= 2 && Uppercase(cmd[1]) == "EXISTS") {
        for (size_t i = 0; i < replies.size(); ++i) {
            const redisReply *reply = replies[i].get();
            if (!reply || reply->type != REDIS_REPLY_ARRAY || reply->elements != 1 || !reply->element[0] ||
                reply->element[0]->type != REDIS_REPLY_INTEGER) {
                KVCM_LOG_ERROR("invalid SCRIPT EXISTS reply");
                return ReplyUPtr();
            }
            if (reply->element[0]->integer == 0) {
                return std::move(replies[i]);
            }
        }
    }
    return std::move(replies[0]);
}

bool RedisClusterExecutor::IsRedirectReply(const redisReply *reply) {
    if (!reply || reply->type != REDIS_REPLY_ERROR || !reply->str) {
        return false;
    }
    const std::string error(reply->str, reply->len);
    return error.compare(0, 6, "MOVED ") == 0 || error.compare(0, 4, "ASK ") == 0;
}

std::string RedisClusterExecutor::GetNodeIdentity(const std::string &route_key) {
    const sw::redis::StringView key(route_key.data(), route_key.size());
    const sw::redis::ConnectionOptions connection_options = cluster_->connection_options(key);
    return connection_options.host + ":" + std::to_string(connection_options.port);
}

void RedisClusterExecutor::ExecuteOne(const PipelineCommand &command, std::vector<ReplyUPtr> &out_replies) {
    try {
        out_replies[command.original_index] = ExecuteRaw(*command.cmd, command.route_key);
    } catch (const sw::redis::Error &error) {
        KVCM_LOG_ERROR("execute redis cluster command failed: %s", error.what());
    }
}

void RedisClusterExecutor::ExecutePipeline(const std::vector<PipelineCommand> &commands,
                                           std::vector<ReplyUPtr> &out_replies) {
    if (commands.empty()) {
        return;
    }

    try {
        const std::string &route_key = commands[0].route_key;
        sw::redis::Pipeline pipeline =
            cluster_->pipeline(sw::redis::StringView(route_key.data(), route_key.size()), false);
        for (const PipelineCommand &command : commands) {
            sw::redis::Pipeline &queued_pipeline = pipeline.command(command.cmd->begin(), command.cmd->end());
            assert(&queued_pipeline == &pipeline);
            (void)queued_pipeline;
        }

        sw::redis::QueuedReplies queued_replies = pipeline.exec();
        std::vector<ReplyUPtr> pipeline_replies = queued_replies.release();
        if (pipeline_replies.size() != commands.size()) {
            KVCM_LOG_ERROR("redis cluster pipeline reply count mismatch, command_count[%zu], reply_count[%zu]",
                           commands.size(),
                           pipeline_replies.size());
            return;
        }

        for (size_t i = 0; i < commands.size(); ++i) {
            if (IsRedirectReply(pipeline_replies[i].get())) {
                ExecuteOne(commands[i], out_replies);
            } else {
                out_replies[commands[i].original_index] = std::move(pipeline_replies[i]);
            }
        }
    } catch (const sw::redis::Error &) {
        for (const PipelineCommand &command : commands) {
            ExecuteOne(command, out_replies);
        }
    }
}

bool RedisClusterExecutor::StartPipelineWorkers() {
    {
        std::lock_guard<std::mutex> lock(pipeline_mutex_);
        if (!pipeline_workers_.empty()) {
            return true;
        }
        pipeline_stop_ = false;
    }

    try {
        pipeline_workers_.reserve(static_cast<size_t>(options_->pipeline_worker_count));
        for (int64_t i = 0; i < options_->pipeline_worker_count; ++i) {
            pipeline_workers_.emplace_back([this]() { PipelineWorkerRoutine(); });
        }
        return true;
    } catch (const std::system_error &error) {
        KVCM_LOG_ERROR("start redis cluster pipeline workers failed: %s", error.what());
        StopPipelineWorkers();
        return false;
    }
}

void RedisClusterExecutor::StopPipelineWorkers() {
    {
        std::lock_guard<std::mutex> lock(pipeline_mutex_);
        pipeline_stop_ = true;
    }
    pipeline_condition_.notify_all();
    for (std::thread &worker : pipeline_workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    pipeline_workers_.clear();
}

bool RedisClusterExecutor::SubmitPipelineTask(std::packaged_task<void()> &&task) {
    {
        std::lock_guard<std::mutex> lock(pipeline_mutex_);
        if (pipeline_stop_) {
            return false;
        }
        pipeline_tasks_.emplace_back(std::move(task));
    }
    pipeline_condition_.notify_one();
    return true;
}

void RedisClusterExecutor::PipelineWorkerRoutine() {
    while (true) {
        std::packaged_task<void()> task;
        {
            std::unique_lock<std::mutex> lock(pipeline_mutex_);
            pipeline_condition_.wait(lock, [this]() { return pipeline_stop_ || !pipeline_tasks_.empty(); });
            if (pipeline_stop_ && pipeline_tasks_.empty()) {
                return;
            }
            task = std::move(pipeline_tasks_.front());
            pipeline_tasks_.pop_front();
        }
        task();
    }
}

std::shared_ptr<RedisClusterExecutor> RedisClusterExecutor::GetOrCreate(const StandardUri &uri,
                                                                        std::string &out_error) {
    RedisClusterOptions parsed_options;
    if (!RedisClusterOptions::FromUri(uri, parsed_options, out_error)) {
        return nullptr;
    }
    std::shared_ptr<const RedisClusterOptions> options = std::make_shared<RedisClusterOptions>(parsed_options);
    const std::string identity = options->Identity();

    static std::mutex registry_mutex;
    static std::unordered_map<std::string, std::weak_ptr<RedisClusterExecutor>> registry;
    std::lock_guard<std::mutex> lock(registry_mutex);
    std::unordered_map<std::string, std::weak_ptr<RedisClusterExecutor>>::iterator existing = registry.find(identity);
    if (existing != registry.end()) {
        std::shared_ptr<RedisClusterExecutor> executor = existing->second.lock();
        if (executor) {
            return executor;
        }
    }
    std::shared_ptr<RedisClusterExecutor> executor = std::make_shared<RedisClusterExecutor>(std::move(options));
    registry[identity] = executor;
    return executor;
}

RedisClusterExecutor::RedisClusterExecutor(std::shared_ptr<const RedisClusterOptions> options)
    : options_(std::move(options)) {}

RedisClusterExecutor::~RedisClusterExecutor() {
    ready_.store(false, std::memory_order_release);
    StopPipelineWorkers();
}

bool RedisClusterExecutor::Open() {
    if (ready_.load(std::memory_order_acquire)) {
        return true;
    }
    std::lock_guard<std::mutex> lock(open_mutex_);
    if (ready_.load(std::memory_order_relaxed)) {
        return true;
    }

    try {
        sw::redis::ConnectionOptions connection_options;
        connection_options.host = options_->host;
        connection_options.port = options_->port;
        connection_options.user = options_->username;
        connection_options.password = options_->password;
        connection_options.db = 0;
        connection_options.connect_timeout = std::chrono::milliseconds(options_->connect_timeout_ms);
        connection_options.socket_timeout = std::chrono::milliseconds(options_->socket_timeout_ms);
        connection_options.tls.enabled = options_->tls_enabled;
        connection_options.tls.cacert = options_->ca_file;
        connection_options.tls.cacertdir = options_->ca_dir;
        connection_options.tls.cert = options_->client_cert_file;
        connection_options.tls.key = options_->client_key_file;
        connection_options.tls.sni = options_->sni;

        sw::redis::ConnectionPoolOptions pool_options;
        pool_options.size = static_cast<size_t>(options_->pool_size_per_node);
        pool_options.wait_timeout = std::chrono::milliseconds(options_->socket_timeout_ms);

        sw::redis::ClusterOptions cluster_options;
        cluster_options.slot_map_refresh_interval = std::chrono::milliseconds(options_->topology_refresh_ms);

        cluster_ = std::make_unique<sw::redis::RedisCluster>(
            connection_options, pool_options, sw::redis::Role::MASTER, cluster_options);
        if (!StartPipelineWorkers()) {
            cluster_.reset();
            return false;
        }
        ready_.store(true, std::memory_order_release);
        KVCM_LOG_INFO("redis cluster executor opened, endpoint[%s], pool_size_per_node[%ld], pipeline_workers[%ld]",
                      options_->RedactedDescription().c_str(),
                      options_->pool_size_per_node,
                      options_->pipeline_worker_count);
        return true;
    } catch (const sw::redis::Error &error) {
        cluster_.reset();
        KVCM_LOG_ERROR("open redis cluster executor failed, endpoint[%s], error[%s]",
                       options_->RedactedDescription().c_str(),
                       error.what());
        return false;
    } catch (const std::system_error &error) {
        cluster_.reset();
        KVCM_LOG_ERROR("open redis cluster executor failed, endpoint[%s], error[%s]",
                       options_->RedactedDescription().c_str(),
                       error.what());
        return false;
    }
}

bool RedisClusterExecutor::IsReady() const noexcept { return ready_.load(std::memory_order_acquire); }

std::vector<RedisReplyUPtr> RedisClusterExecutor::ExecuteBatch(const std::vector<RedisCmdArgs> &cmds) {
    std::vector<ReplyUPtr> replies;
    if (!IsReady() || !cluster_) {
        return replies;
    }
    replies.resize(cmds.size());

    std::unordered_map<std::string, std::vector<PipelineCommand>> node_commands;
    for (size_t i = 0; i < cmds.size(); ++i) {
        const CmdArgs &cmd = cmds[i];
        try {
            if (cmd.empty()) {
                KVCM_LOG_ERROR("empty redis cluster command, index[%zu]", i);
                continue;
            }
            const std::string command = Uppercase(cmd[0]);
            if (command == "SCAN") {
                std::lock_guard<std::mutex> lock(global_command_mutex_);
                replies[i] = ExecuteScan(cmd);
                continue;
            }
            if (command == "FLUSHALL" || command == "SCRIPT") {
                std::lock_guard<std::mutex> lock(global_command_mutex_);
                replies[i] = ExecuteBroadcast(cmd);
                continue;
            }

            std::string route_key;
            if (command == "RANDOMKEY") {
                const uint64_t sequence = random_sequence_.fetch_add(1, std::memory_order_relaxed);
                route_key = "kvcm:random:" + std::to_string(sequence);
            } else {
                std::string error;
                if (!ExtractRouteKey(cmd, route_key, error)) {
                    KVCM_LOG_ERROR("%s", error.c_str());
                    continue;
                }
            }

            const std::string node_identity = GetNodeIdentity(route_key);
            node_commands[node_identity].push_back(PipelineCommand{i, &cmd, std::move(route_key)});
        } catch (const sw::redis::Error &error) {
            KVCM_LOG_ERROR("prepare redis cluster command failed, index[%zu], error[%s]", i, error.what());
        }
    }

    std::vector<std::future<void>> futures;
    futures.reserve(node_commands.size());
    for (const std::pair<const std::string, std::vector<PipelineCommand>> &entry : node_commands) {
        const std::reference_wrapper<const std::vector<PipelineCommand>> commands = std::cref(entry.second);
        std::packaged_task<void()> task([this, commands, &replies]() { ExecutePipeline(commands.get(), replies); });
        std::future<void> future = task.get_future();
        if (!SubmitPipelineTask(std::move(task))) {
            ExecutePipeline(entry.second, replies);
            continue;
        }
        futures.emplace_back(std::move(future));
    }

    std::exception_ptr first_exception;
    for (std::future<void> &future : futures) {
        try {
            future.get();
        } catch (...) {
            if (!first_exception) {
                first_exception = std::current_exception();
            }
        }
    }
    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
    return replies;
}

} // namespace kv_cache_manager
