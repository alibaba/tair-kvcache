#include "kv_cache_manager/client/src/internal/async_rpc/http_client.h"

#include <algorithm>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/client/src/internal/async_rpc/async_slot.h"
#include "kv_cache_manager/client/src/internal/async_rpc/json_codec.h"
#include "kv_cache_manager/client/src/internal/async_rpc/rpc_util.h"
#include "ylt/coro_http/coro_http_client.hpp"
#include "ylt/coro_io/io_context_pool.hpp"

namespace kv_cache_manager::async_rpc {
namespace {

struct HttpCallOutcome {
    TransportError error = TransportError::kNone;
    int http_status = 0;
    std::string raw_error;
    std::string body;
    bool connection_closed = false;
    bool connected_now = false;
    double connect_ms = 0.0;
};

} // namespace

struct HttpClientRuntime::Impl {
    explicit Impl(uint32_t threads) : pool(threads == 0 ? 1u : threads) {
        runner = std::thread([this]() { pool.run(); });
    }

    ~Impl() { Stop(); }

    void Stop() {
        if (stopped) {
            return;
        }
        stopped = true;
        pool.stop();
        if (runner.joinable()) {
            runner.join();
        }
    }

    coro_io::io_context_pool pool;
    std::thread runner;
    bool stopped = false;
};

HttpClientRuntime::HttpClientRuntime(uint32_t reactor_threads) : impl_(std::make_unique<Impl>(reactor_threads)) {}

HttpClientRuntime::~HttpClientRuntime() = default;

uint32_t HttpClientRuntime::thread_count() const { return static_cast<uint32_t>(impl_->pool.pool_size()); }

void HttpClientRuntime::Shutdown() { impl_->Stop(); }

namespace {

// One HTTP/1.1 connection. cinatra clients are single-request; the pool
// guarantees a connection is used by at most one request at a time.
struct HttpConnection {
    std::shared_ptr<coro_http::coro_http_client> client;
    coro_io::ExecutorWrapper<> *executor = nullptr;
};

class HttpClient : public AsyncRpcClient {
public:
    HttpClient(std::shared_ptr<HttpClientRuntime> runtime,
               const ContinuationScheduler *scheduler,
               EndpointSet endpoints,
               ClientLimits limits)
        : runtime_(std::move(runtime))
        , scheduler_(scheduler)
        , endpoints_(std::move(endpoints))
        , limits_(limits)
        , meta_endpoint_(endpoints_.meta_http) {}

    ~HttpClient() override { Shutdown(); }

    Task<RpcResult> Call(Api api,
                         const google::protobuf::Message &request,
                         google::protobuf::Message *response,
                         CallOptions options) override;

    void SetMetaEndpoint(const std::string &endpoint) override {
        std::lock_guard<std::mutex> lock(mutex_);
        meta_endpoint_ = endpoint;
    }

    std::string MetaEndpoint() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return meta_endpoint_;
    }

    ClientStats Stats() const override {
        ClientStats stats;
        stats.kind = TransportKind::kHttp;
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto &entry : pools_) {
            EndpointStats endpoint_stats;
            endpoint_stats.endpoint = entry.first;
            endpoint_stats.role = entry.second->role;
            endpoint_stats.channels = entry.second->lanes[0].conns.size() + entry.second->lanes[1].conns.size();
            endpoint_stats.connections_current = entry.second->connections_current;
            endpoint_stats.connections_peak = entry.second->connections_peak;
            endpoint_stats.connections_created = entry.second->connections_created;
            endpoint_stats.connections_reused = entry.second->connections_reused;
            endpoint_stats.in_flight_current = entry.second->in_flight_current;
            endpoint_stats.in_flight_peak = entry.second->in_flight_peak;
            endpoint_stats.establish_latency_ms = entry.second->establish_latency_ms;
            stats.endpoints.push_back(std::move(endpoint_stats));
        }
        return stats;
    }

    void Shutdown() override {
        std::map<std::string, std::unique_ptr<EndpointPool>> pools;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            pools.swap(pools_);
        }
        for (auto &entry : pools) {
            for (auto &lane : entry.second->lanes) {
                for (auto &conn : lane.conns) {
                    if (conn->client) {
                        conn->client->close();
                    }
                }
            }
        }
    }

private:
    struct LanePool {
        uint32_t capacity = 1;
        std::vector<std::unique_ptr<HttpConnection>> conns;
        std::deque<HttpConnection *> free;
        std::deque<std::shared_ptr<AsyncSlot<HttpConnection *>>> waiters;
    };

    struct EndpointPool {
        std::string url;
        std::string role;
        LanePool lanes[2]; // [0] business, [1] control
        uint64_t connections_current = 0;
        uint64_t connections_peak = 0;
        uint64_t connections_created = 0;
        uint64_t connections_reused = 0;
        uint64_t in_flight_current = 0;
        uint64_t in_flight_peak = 0;
        std::vector<double> establish_latency_ms;
    };

    static size_t LaneIndex(RpcLane lane) { return lane == RpcLane::kBusiness ? 0 : 1; }

    EndpointPool *PoolFor(const std::string &url, const char *role) {
        auto it = pools_.find(url);
        if (it != pools_.end()) {
            return it->second.get();
        }
        auto pool = std::make_unique<EndpointPool>();
        pool->url = url;
        pool->role = role;
        pool->lanes[0].capacity = std::max<uint32_t>(1, limits_.http_connections_per_endpoint);
        pool->lanes[1].capacity = std::max<uint32_t>(1, limits_.http_control_connections_per_endpoint);
        EndpointPool *raw = pool.get();
        pools_.emplace(url, std::move(pool));
        return raw;
    }

    Task<HttpConnection *>
    Checkout(const std::string &url,
             const char *role,
             RpcLane lane,
             TimePoint deadline,
             CancellationToken cancel);
    void Checkin(const std::string &url, RpcLane lane, HttpConnection *conn, bool drop);

    std::string ResolveEndpoint(ServiceEndpoint endpoint, const char **role) const {
        if (endpoint == ServiceEndpoint::kAdmin) {
            *role = "admin";
            return endpoints_.admin_http;
        }
        *role = "meta";
        std::lock_guard<std::mutex> lock(mutex_);
        return meta_endpoint_;
    }

    std::shared_ptr<HttpClientRuntime> runtime_;
    const ContinuationScheduler *scheduler_;
    EndpointSet endpoints_;
    ClientLimits limits_;

    mutable std::mutex mutex_;
    std::string meta_endpoint_;
    std::map<std::string, std::unique_ptr<EndpointPool>> pools_;
};

Task<HttpConnection *> HttpClient::Checkout(
    const std::string &url, const char *role, RpcLane lane, TimePoint deadline, CancellationToken cancel) {
    std::shared_ptr<AsyncSlot<HttpConnection *>> slot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        EndpointPool *pool = PoolFor(url, role);
        LanePool &lane_pool = pool->lanes[LaneIndex(lane)];
        if (!lane_pool.free.empty()) {
            HttpConnection *conn = lane_pool.free.front();
            lane_pool.free.pop_front();
            ++pool->connections_reused;
            co_return conn;
        }
        if (lane_pool.conns.size() < lane_pool.capacity) {
            auto conn = std::make_unique<HttpConnection>();
            conn->executor = runtime_->impl().pool.get_executor();
            HttpConnection *raw = conn.get();
            lane_pool.conns.push_back(std::move(conn));
            co_return raw;
        }
        slot = std::make_shared<AsyncSlot<HttpConnection *>>(*scheduler_);
        lane_pool.waiters.push_back(slot);
    }
    scheduler_->schedule_at(deadline, [slot]() { slot->Complete(nullptr); });
    CancellationCallbackGuard guard(cancel, [slot]() { slot->Complete(nullptr); });
    HttpConnection *conn = co_await *slot;
    if (conn == nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pools_.find(url);
        if (it != pools_.end()) {
            LanePool &lane_pool = it->second->lanes[LaneIndex(lane)];
            for (auto wit = lane_pool.waiters.begin(); wit != lane_pool.waiters.end(); ++wit) {
                if (*wit == slot) {
                    lane_pool.waiters.erase(wit);
                    break;
                }
            }
        }
    }
    co_return conn;
}

void HttpClient::Checkin(const std::string &url, RpcLane lane, HttpConnection *conn, bool drop) {
    while (true) {
        std::shared_ptr<AsyncSlot<HttpConnection *>> next;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = pools_.find(url);
            if (it == pools_.end()) {
                return;
            }
            EndpointPool *pool = it->second.get();
            LanePool &lane_pool = pool->lanes[LaneIndex(lane)];
            if (drop && conn->client) {
                conn->client->close();
                conn->client.reset();
                if (pool->connections_current > 0) {
                    --pool->connections_current;
                }
            }
            if (lane_pool.waiters.empty()) {
                lane_pool.free.push_back(conn);
                return;
            }
            next = lane_pool.waiters.front();
            lane_pool.waiters.pop_front();
        }
        if (next->Complete(conn)) {
            return;
        }
    }
}

Task<RpcResult> HttpClient::Call(Api api,
                                 const google::protobuf::Message &request,
                                 google::protobuf::Message *response,
                                 CallOptions options) {
    const ApiInfo &info = GetApiInfo(api);
    TimePoint deadline = options.deadline;
    if (deadline.time_since_epoch().count() == 0) {
        deadline = Now() + limits_.default_rpc_timeout;
    }

    RpcResult result;
    std::string body;
    if (!MessageToJson(request, &body)) {
        result.transport_error = TransportError::kEncode;
        result.raw_error = "request could not be serialised to JSON";
        co_return result;
    }

    const char *role = "meta";
    const std::string base = ResolveEndpoint(info.endpoint, &role);
    HttpConnection *conn = co_await Checkout(base, role, options.lane, deadline, options.cancel);
    if (conn == nullptr) {
        result.transport_error =
            options.cancel.StopRequested() ? TransportError::kCancelled : TransportError::kTimeout;
        result.raw_error = "no HTTP connection slot before deadline";
        co_return result;
    }

    const TimePoint submitted_at = Now();
    Duration remaining = deadline - submitted_at;
    if (remaining <= Duration::zero()) {
        remaining = std::chrono::milliseconds(1);
    }

    bool created_connection = false;
    if (!conn->client) {
        conn->client = std::make_shared<coro_http::coro_http_client>(conn->executor);
        created_connection = true;
    } else if (conn->client->has_closed()) {
        created_connection = true;
    }
    conn->client->set_conn_timeout(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::min<Duration>(limits_.connect_timeout, remaining)));
    conn->client->set_req_timeout(std::chrono::duration_cast<std::chrono::milliseconds>(remaining));

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = pools_.find(base);
        if (it != pools_.end()) {
            EndpointPool *pool = it->second.get();
            ++pool->in_flight_current;
            pool->in_flight_peak = std::max(pool->in_flight_peak, pool->in_flight_current);
        }
    }

    auto slot = std::make_shared<AsyncSlot<HttpCallOutcome>>(*scheduler_);
    auto client = conn->client;
    const std::string url = base + std::string(info.http_path);
    const TimePoint connect_start = submitted_at;

    // The cinatra coroutine runs entirely on its own single-threaded
    // io_context; only the finished outcome is handed back to the Executor.
    std::move(client->async_post(url, std::move(body), coro_http::req_content_type::json))
        .via(conn->executor)
        .start([slot, client, created_connection, connect_start](auto &&maybe_result) {
            HttpCallOutcome outcome;
            outcome.connected_now = created_connection;
            outcome.connect_ms = created_connection ? ToMillis(Now() - connect_start) : 0.0;
            if (maybe_result.hasError()) {
                outcome.error = TransportError::kOther;
                outcome.raw_error = "http client raised an exception";
                outcome.connection_closed = true;
                slot->Complete(std::move(outcome));
                return;
            }
            const auto &resp = maybe_result.value();
            outcome.http_status = resp.status;
            if (resp.net_err) {
                outcome.raw_error = resp.net_err.message();
                outcome.connection_closed = true;
                if (resp.net_err == std::errc::timed_out || outcome.raw_error.find("timeout") != std::string::npos ||
                    outcome.raw_error.find("timed out") != std::string::npos) {
                    outcome.error = TransportError::kTimeout;
                } else if (created_connection && resp.status == 0) {
                    outcome.error = TransportError::kConnect;
                } else {
                    outcome.error = TransportError::kDisconnect;
                }
                slot->Complete(std::move(outcome));
                return;
            }
            outcome.body.assign(resp.resp_body.data(), resp.resp_body.size());
            if (resp.status < 200 || resp.status >= 300) {
                outcome.error = TransportError::kOther;
                outcome.raw_error = "http status " + std::to_string(resp.status);
            }
            slot->Complete(std::move(outcome));
        });

    // Cancellation closes the socket so the pending read fails immediately.
    {
        CancellationCallbackGuard guard(options.cancel, [client]() { client->close(); });
        HttpCallOutcome outcome = co_await *slot;
        result.rpc_latency = Now() - submitted_at;
        result.raw_status = outcome.http_status;
        result.transport_error = outcome.error;
        result.raw_error = outcome.raw_error;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = pools_.find(base);
            if (it != pools_.end()) {
                EndpointPool *pool = it->second.get();
                if (outcome.connected_now && outcome.error != TransportError::kConnect) {
                    ++pool->connections_created;
                    ++pool->connections_current;
                    pool->connections_peak = std::max(pool->connections_peak, pool->connections_current);
                    pool->establish_latency_ms.push_back(outcome.connect_ms);
                }
                if (pool->in_flight_current > 0) {
                    --pool->in_flight_current;
                }
            }
        }

        if (outcome.error == TransportError::kNone) {
            std::string parse_error;
            if (!JsonToMessage(outcome.body, response, &parse_error)) {
                result.transport_error = TransportError::kDecode;
                result.raw_error = "response JSON parse failed: " + parse_error;
            }
        }
        Checkin(base, options.lane, conn, outcome.connection_closed);
    }

    if (result.transport_error == TransportError::kNone) {
        ApplyServiceStatus(*response, &result);
    } else if (options.cancel.StopRequested() && result.transport_error == TransportError::kDisconnect) {
        result.transport_error = TransportError::kCancelled;
    }
    co_return result;
}

} // namespace

std::unique_ptr<AsyncRpcClient> MakeHttpClient(std::shared_ptr<HttpClientRuntime> runtime,
                                               const ContinuationScheduler *scheduler,
                                               EndpointSet endpoints,
                                               ClientLimits limits) {
    return std::make_unique<HttpClient>(std::move(runtime), scheduler, std::move(endpoints), limits);
}

} // namespace kv_cache_manager::async_rpc
