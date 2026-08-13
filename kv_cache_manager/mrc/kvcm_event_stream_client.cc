#include "kv_cache_manager/mrc/kvcm_event_stream_client.h"

#include <algorithm>
#include <array>
#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <limits>
#include <map>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/service_discovery_factory.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/mrc/online_mrc_fact_registry.h"

namespace kv_cache_manager {
namespace {

constexpr uint8_t kMessageHello = 1;
constexpr uint8_t kMessageEventBatch = 2;
constexpr uint32_t kProtocolVersion = 1;
constexpr int kMaxPollWaitMs = 1000;

constexpr char kMetricDiscoveredEndpoints[] = "online_mrc.discovered_endpoints";
constexpr char kMetricActiveConnections[] = "online_mrc.active_connections";
constexpr char kMetricReconnects[] = "online_mrc.reconnects";
constexpr char kMetricDecodeErrors[] = "online_mrc.decode_errors";
constexpr char kMetricReceivedBatches[] = "online_mrc.received_batches";
constexpr char kMetricReceiverQueueSize[] = "online_mrc.receiver_queue_size";
constexpr char kMetricReceiverQueueBytes[] = "online_mrc.receiver_queue_bytes";
constexpr char kMetricReceiverQueueCapacityBatches[] = "online_mrc.receiver_queue_capacity_batches";
constexpr char kMetricReceiverDroppedBatches[] = "online_mrc.receiver_dropped_batches";
constexpr char kMetricReceiverRejectedBatches[] = "online_mrc.receiver_rejected_batches";

std::string EndpointKey(const ServiceEndpoint &endpoint) {
    return endpoint.host.empty() ? endpoint.ip + ":" + std::to_string(endpoint.port) : endpoint.host;
}

bool SameEndpoint(const ServiceEndpoint &lhs, const ServiceEndpoint &rhs) {
    return lhs.ip == rhs.ip && lhs.port == rhs.port;
}

bool ConfigureNonBlockingFd(int fd) {
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
        return false;
    }
    const int fd_flags = fcntl(fd, F_GETFD, 0);
    return fd_flags >= 0 && fcntl(fd, F_SETFD, fd_flags | FD_CLOEXEC) == 0;
}

bool BuildHelloFrame(const std::string &optimizer_id, std::vector<uint8_t> &frame) {
    proto::optimizer::OptimizerHello hello;
    hello.set_protocol_version(kProtocolVersion);
    hello.set_optimizer_id(optimizer_id);
    std::string payload;
    if (!hello.SerializeToString(&payload)) {
        return false;
    }
    const uint32_t frame_length = static_cast<uint32_t>(payload.size() + 1);
    const uint32_t network_length = htonl(frame_length);
    frame.resize(sizeof(network_length) + frame_length);
    std::memcpy(frame.data(), &network_length, sizeof(network_length));
    frame[sizeof(network_length)] = kMessageHello;
    std::memcpy(frame.data() + sizeof(network_length) + 1, payload.data(), payload.size());
    return true;
}

} // namespace

class KvcmEventStreamClient::Connection {
public:
    enum class State { Disconnected, Connecting, SendingHello, Reading };

    explicit Connection(ServiceEndpoint value)
        : endpoint(std::move(value)), next_retry(std::chrono::steady_clock::now()) {}

    ~Connection() {
        if (fd >= 0) {
            close(fd);
        }
    }

    ServiceEndpoint endpoint;
    int fd = -1;
    State state = State::Disconnected;
    std::chrono::steady_clock::time_point next_retry;
    std::chrono::steady_clock::time_point connect_deadline;
    std::vector<uint8_t> write_buffer;
    size_t write_offset = 0;
    std::vector<uint8_t> read_buffer;
    size_t read_offset = 0;
    uint64_t attempt_count = 0;
    bool active_counted = false;
};

KvcmEventStreamClient::KvcmEventStreamClient(const OnlineMrcConfig &config,
                                             std::shared_ptr<OnlineMrcFactRegistry> fact_registry,
                                             std::shared_ptr<MetricsRegistry> metrics_registry)
    : config_(config)
    , fact_registry_(std::move(fact_registry))
    , metrics_registry_(std::move(metrics_registry))
    , optimizer_id_(StringUtil::GenerateRandomString(32)) {}

KvcmEventStreamClient::~KvcmEventStreamClient() {
    Stop();
    if (wake_read_fd_ >= 0) {
        close(wake_read_fd_);
        wake_read_fd_ = -1;
    }
    if (wake_write_fd_ >= 0) {
        close(wake_write_fd_);
        wake_write_fd_ = -1;
    }
}

bool KvcmEventStreamClient::Init() {
    discovery_ = ServiceDiscoveryFactory::CreateServiceDiscovery(config_.kvcm_service_discovery_url);
    if (!discovery_) {
        KVCM_LOG_ERROR("online mrc stream: failed to initialize KVCM service discovery[%s]",
                       config_.kvcm_service_discovery_url.c_str());
        return false;
    }
    if (!fact_registry_) {
        return false;
    }
    if (wake_read_fd_ >= 0 && wake_write_fd_ >= 0) {
        return true;
    }
    int wake_fds[2] = {-1, -1};
    if (pipe(wake_fds) != 0 || !ConfigureNonBlockingFd(wake_fds[0]) || !ConfigureNonBlockingFd(wake_fds[1])) {
        if (wake_fds[0] >= 0) {
            close(wake_fds[0]);
        }
        if (wake_fds[1] >= 0) {
            close(wake_fds[1]);
        }
        KVCM_LOG_ERROR("online mrc stream: failed to create event-loop wake pipe, errno=%d", errno);
        return false;
    }
    wake_read_fd_ = wake_fds[0];
    wake_write_fd_ = wake_fds[1];
    return true;
}

bool KvcmEventStreamClient::Start() {
    if (!discovery_ || !fact_registry_ || wake_read_fd_ < 0 || wake_write_fd_ < 0) {
        return false;
    }
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) {
        return true;
    }
    applied_endpoints_generation_ = std::numeric_limits<uint64_t>::max();
    ingress_stopped_.store(false, std::memory_order_release);
    consumer_thread_ = std::thread(&KvcmEventStreamClient::ConsumerLoop, this);
    event_loop_thread_ = std::thread(&KvcmEventStreamClient::EventLoop, this);
    discovery_thread_ = std::thread(&KvcmEventStreamClient::DiscoveryLoop, this);
    return true;
}

void KvcmEventStreamClient::Stop() {
    if (!running_.exchange(false)) {
        return;
    }
    wait_cv_.notify_all();
    WakeEventLoop();
    if (discovery_thread_.joinable()) {
        discovery_thread_.join();
    }
    if (event_loop_thread_.joinable()) {
        event_loop_thread_.join();
    }
    ingress_stopped_.store(true, std::memory_order_release);
    queue_cv_.notify_all();
    if (consumer_thread_.joinable()) {
        consumer_thread_.join();
    }
}

void KvcmEventStreamClient::WakeEventLoop() {
    if (wake_write_fd_ < 0) {
        return;
    }
    const uint8_t byte = 1;
    ssize_t rc;
    do {
        rc = write(wake_write_fd_, &byte, sizeof(byte));
    } while (rc < 0 && errno == EINTR);
}

void KvcmEventStreamClient::DrainWakePipe() {
    std::array<uint8_t, 64> buffer{};
    while (true) {
        const ssize_t rc = read(wake_read_fd_, buffer.data(), buffer.size());
        if (rc > 0) {
            continue;
        }
        if (rc < 0 && errno == EINTR) {
            continue;
        }
        return;
    }
}

bool KvcmEventStreamClient::Enqueue(proto::optimizer::CacheEventBatch batch, uint64_t wire_bytes) {
    std::lock_guard<std::mutex> guard(queue_mutex_);
    const size_t limit = static_cast<size_t>(std::max<int64_t>(config_.receiver_queue_max_batches, 1));
    if (queue_.size() >= limit) {
        dropped_batches_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    queue_.push_back(QueuedBatch{std::move(batch), wire_bytes});
    queue_bytes_ += wire_bytes;
    queue_cv_.notify_one();
    return true;
}

void KvcmEventStreamClient::ConsumerLoop() {
    while (true) {
        QueuedBatch queued;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this]() { return ingress_stopped_.load(std::memory_order_acquire) || !queue_.empty(); });
            if (queue_.empty()) {
                if (ingress_stopped_.load(std::memory_order_acquire)) {
                    break;
                }
                continue;
            }
            queued = std::move(queue_.front());
            queue_.pop_front();
            queue_bytes_ -= queued.wire_bytes;
        }
        if (!fact_registry_->Observe(queued.batch)) {
            rejected_batches_.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

void KvcmEventStreamClient::DiscoveryLoop() {
    while (running_) {
        std::vector<ServiceEndpoint> endpoints;
        if (discovery_->GetAllEndpoints(endpoints)) {
            UpdateDesiredEndpoints(endpoints);
        } else {
            KVCM_LOG_WARN("online mrc stream: failed to discover KVCM endpoints");
        }
        std::unique_lock<std::mutex> lock(wait_mutex_);
        wait_cv_.wait_for(lock,
                          std::chrono::milliseconds(std::max<int64_t>(config_.discovery_refresh_interval_ms, 1)),
                          [this]() { return !running_; });
        if (running_) {
            discovery_->Refresh();
        }
    }
}

void KvcmEventStreamClient::UpdateDesiredEndpoints(const std::vector<ServiceEndpoint> &endpoints) {
    std::unordered_map<std::string, ServiceEndpoint> desired;
    for (const auto &endpoint : endpoints) {
        if (endpoint.healthy && !endpoint.ip.empty() && endpoint.port > 0) {
            desired[EndpointKey(endpoint)] = endpoint;
        }
    }
    discovered_endpoints_.store(desired.size(), std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> guard(desired_endpoints_mutex_);
        desired_endpoints_ = std::move(desired);
        ++desired_endpoints_generation_;
    }
    WakeEventLoop();
}

void KvcmEventStreamClient::ApplyDesiredEndpoints() {
    std::unordered_map<std::string, ServiceEndpoint> desired;
    uint64_t generation = 0;
    {
        std::lock_guard<std::mutex> guard(desired_endpoints_mutex_);
        if (applied_endpoints_generation_ == desired_endpoints_generation_) {
            return;
        }
        desired = desired_endpoints_;
        generation = desired_endpoints_generation_;
    }

    for (auto it = connections_.begin(); it != connections_.end();) {
        const auto desired_it = desired.find(it->first);
        if (desired_it == desired.end()) {
            Disconnect(*it->second, false);
            it = connections_.erase(it);
            continue;
        }
        if (!SameEndpoint(it->second->endpoint, desired_it->second)) {
            Disconnect(*it->second, false);
            it->second->endpoint = desired_it->second;
            it->second->attempt_count = 0;
            it->second->next_retry = std::chrono::steady_clock::now();
        }
        ++it;
    }
    for (auto &[key, endpoint] : desired) {
        if (connections_.count(key) == 0) {
            connections_.emplace(key, std::make_unique<Connection>(std::move(endpoint)));
        }
    }
    managed_connections_.store(connections_.size(), std::memory_order_relaxed);
    applied_endpoints_generation_ = generation;
}

void KvcmEventStreamClient::BeginConnect(Connection &connection) {
    if (connection.state != Connection::State::Disconnected || connection.fd >= 0) {
        return;
    }
    if (connection.attempt_count > 0) {
        reconnects_.fetch_add(1, std::memory_order_relaxed);
    }
    ++connection.attempt_count;

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_NUMERICHOST;
    addrinfo *result = nullptr;
    const std::string port = std::to_string(connection.endpoint.port);
    if (getaddrinfo(connection.endpoint.ip.c_str(), port.c_str(), &hints, &result) != 0) {
        Disconnect(connection, true);
        return;
    }

    for (addrinfo *address = result; address && running_; address = address->ai_next) {
        const int fd = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (!ConfigureNonBlockingFd(fd)) {
            close(fd);
            continue;
        }
#ifdef SO_NOSIGPIPE
        const int no_sigpipe = 1;
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe));
#endif
        const int rc = connect(fd, address->ai_addr, address->ai_addrlen);
        if (rc == 0 || (rc < 0 && errno == EINPROGRESS)) {
            connection.fd = fd;
            if (rc == 0) {
                connection.state = Connection::State::SendingHello;
                if (!BuildHelloFrame(optimizer_id_, connection.write_buffer)) {
                    Disconnect(connection, true);
                }
            } else {
                connection.state = Connection::State::Connecting;
                connection.connect_deadline =
                    std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(std::max<int64_t>(config_.connect_timeout_ms, 1));
            }
            freeaddrinfo(result);
            return;
        }
        close(fd);
    }
    freeaddrinfo(result);
    Disconnect(connection, true);
}

bool KvcmEventStreamClient::FinishConnect(Connection &connection) {
    int socket_error = 0;
    socklen_t error_length = sizeof(socket_error);
    if (getsockopt(connection.fd, SOL_SOCKET, SO_ERROR, &socket_error, &error_length) != 0 || socket_error != 0) {
        return false;
    }
    connection.state = Connection::State::SendingHello;
    connection.write_offset = 0;
    return BuildHelloFrame(optimizer_id_, connection.write_buffer);
}

bool KvcmEventStreamClient::FlushHello(Connection &connection) {
    while (connection.write_offset < connection.write_buffer.size()) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags = MSG_NOSIGNAL;
#endif
        const ssize_t rc = send(connection.fd,
                                connection.write_buffer.data() + connection.write_offset,
                                connection.write_buffer.size() - connection.write_offset,
                                flags);
        if (rc > 0) {
            connection.write_offset += static_cast<size_t>(rc);
            continue;
        }
        if (rc < 0 && errno == EINTR) {
            continue;
        }
        if (rc < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return true;
        }
        return false;
    }
    connection.write_buffer.clear();
    connection.write_offset = 0;
    connection.state = Connection::State::Reading;
    connection.active_counted = true;
    active_connections_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool KvcmEventStreamClient::DecodeBufferedFrames(Connection &connection) {
    while (connection.read_buffer.size() - connection.read_offset >= sizeof(uint32_t)) {
        uint32_t network_length = 0;
        std::memcpy(&network_length, connection.read_buffer.data() + connection.read_offset, sizeof(network_length));
        const uint32_t frame_length = ntohl(network_length);
        if (frame_length < 1 || frame_length > static_cast<uint64_t>(std::max<int64_t>(config_.max_frame_bytes, 1))) {
            decode_errors_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        const size_t wire_bytes = sizeof(network_length) + static_cast<size_t>(frame_length);
        if (connection.read_buffer.size() - connection.read_offset < wire_bytes) {
            break;
        }
        const uint8_t *frame = connection.read_buffer.data() + connection.read_offset + sizeof(network_length);
        if (frame[0] != kMessageEventBatch) {
            decode_errors_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        proto::optimizer::CacheEventBatch batch;
        if (!batch.ParseFromArray(frame + 1, static_cast<int>(frame_length - 1))) {
            decode_errors_.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        received_batches_.fetch_add(1, std::memory_order_relaxed);
        Enqueue(std::move(batch), wire_bytes);
        connection.read_offset += wire_bytes;
    }

    if (connection.read_offset == connection.read_buffer.size()) {
        connection.read_buffer.clear();
        connection.read_offset = 0;
    } else if (connection.read_offset >= 64 * 1024 || connection.read_offset * 2 >= connection.read_buffer.size()) {
        connection.read_buffer.erase(connection.read_buffer.begin(),
                                     connection.read_buffer.begin() + static_cast<std::ptrdiff_t>(connection.read_offset));
        connection.read_offset = 0;
    }
    return true;
}

bool KvcmEventStreamClient::ReadAvailableFrames(Connection &connection) {
    std::array<uint8_t, 64 * 1024> buffer{};
    while (true) {
        const ssize_t rc = recv(connection.fd, buffer.data(), buffer.size(), 0);
        if (rc > 0) {
            connection.read_buffer.insert(connection.read_buffer.end(), buffer.begin(), buffer.begin() + rc);
            if (!DecodeBufferedFrames(connection)) {
                return false;
            }
            continue;
        }
        if (rc == 0) {
            return false;
        }
        if (errno == EINTR) {
            continue;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return true;
        }
        return false;
    }
}

void KvcmEventStreamClient::Disconnect(Connection &connection, bool schedule_retry) {
    if (connection.active_counted) {
        active_connections_.fetch_sub(1, std::memory_order_relaxed);
        connection.active_counted = false;
    }
    if (connection.fd >= 0) {
        close(connection.fd);
        connection.fd = -1;
    }
    connection.state = Connection::State::Disconnected;
    connection.write_buffer.clear();
    connection.write_offset = 0;
    connection.read_buffer.clear();
    connection.read_offset = 0;
    if (schedule_retry && running_) {
        connection.next_retry =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(std::max<int64_t>(config_.reconnect_interval_ms, 1));
    } else {
        connection.next_retry = std::chrono::steady_clock::time_point::max();
    }
}

void KvcmEventStreamClient::HandleConnectionEvent(Connection &connection, short revents) {
    if (revents & POLLNVAL) {
        Disconnect(connection, true);
        return;
    }
    if (connection.state == Connection::State::Connecting) {
        if (revents & (POLLOUT | POLLERR | POLLHUP)) {
            if (!FinishConnect(connection)) {
                Disconnect(connection, true);
            }
        }
        return;
    }
    if (connection.state == Connection::State::SendingHello) {
        if (revents & (POLLERR | POLLHUP)) {
            Disconnect(connection, true);
            return;
        }
        if ((revents & POLLOUT) && !FlushHello(connection)) {
            Disconnect(connection, true);
        }
        return;
    }
    if (connection.state == Connection::State::Reading) {
        if ((revents & POLLIN) && !ReadAvailableFrames(connection)) {
            Disconnect(connection, true);
            return;
        }
        if (revents & (POLLERR | POLLHUP)) {
            Disconnect(connection, true);
        }
    }
}

int KvcmEventStreamClient::ComputePollTimeoutMs() const {
    const auto now = std::chrono::steady_clock::now();
    auto deadline = now + std::chrono::milliseconds(kMaxPollWaitMs);
    for (const auto &[_, connection] : connections_) {
        if (connection->state == Connection::State::Disconnected) {
            deadline = std::min(deadline, connection->next_retry);
        } else if (connection->state == Connection::State::Connecting) {
            deadline = std::min(deadline, connection->connect_deadline);
        }
    }
    if (deadline <= now) {
        return 0;
    }
    const auto timeout = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
    return static_cast<int>(std::min<int64_t>(timeout, kMaxPollWaitMs));
}

void KvcmEventStreamClient::EventLoop() {
    while (running_) {
        ApplyDesiredEndpoints();
        const auto now = std::chrono::steady_clock::now();
        for (auto &[_, connection] : connections_) {
            if (connection->state == Connection::State::Disconnected && connection->next_retry <= now) {
                BeginConnect(*connection);
            }
        }

        std::vector<pollfd> poll_fds;
        std::vector<Connection *> polled_connections;
        poll_fds.push_back(pollfd{wake_read_fd_, POLLIN, 0});
        for (auto &[_, connection] : connections_) {
            if (connection->fd < 0) {
                continue;
            }
            short events = 0;
            if (connection->state == Connection::State::Connecting ||
                connection->state == Connection::State::SendingHello) {
                events = POLLOUT;
            } else if (connection->state == Connection::State::Reading) {
                events = POLLIN;
            }
            poll_fds.push_back(pollfd{connection->fd, events, 0});
            polled_connections.push_back(connection.get());
        }

        const int rc = poll(poll_fds.data(), poll_fds.size(), ComputePollTimeoutMs());
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            KVCM_LOG_WARN("online mrc stream: event-loop poll failed, errno=%d", errno);
            continue;
        }
        if (poll_fds[0].revents & POLLIN) {
            DrainWakePipe();
        }
        if (!running_) {
            break;
        }
        for (size_t i = 0; i < polled_connections.size(); ++i) {
            if (poll_fds[i + 1].revents != 0) {
                HandleConnectionEvent(*polled_connections[i], poll_fds[i + 1].revents);
            }
        }

        const auto after_poll = std::chrono::steady_clock::now();
        for (auto &[_, connection] : connections_) {
            if (connection->state == Connection::State::Connecting && connection->connect_deadline <= after_poll) {
                Disconnect(*connection, true);
            }
        }
    }

    for (auto &[_, connection] : connections_) {
        Disconnect(*connection, false);
    }
    connections_.clear();
    managed_connections_.store(0, std::memory_order_relaxed);
}

void KvcmEventStreamClient::ReportMetrics() {
    if (!metrics_registry_) {
        return;
    }
    size_t queue_size = 0;
    uint64_t queue_bytes = 0;
    {
        std::lock_guard<std::mutex> guard(queue_mutex_);
        queue_size = queue_.size();
        queue_bytes = queue_bytes_;
    }
    const MetricsTags tags;
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricDiscoveredEndpoints, tags, discovered_endpoints_.load());
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricActiveConnections, tags, active_connections_.load());
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricReconnects, tags, reconnects_.load());
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricDecodeErrors, tags, decode_errors_.load());
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricReceivedBatches, tags, received_batches_.load());
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricReceiverQueueSize, tags, queue_size);
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricReceiverQueueBytes, tags, queue_bytes);
    REPORT_DYNAMIC_GAUGE_(metrics_registry_,
                          kMetricReceiverQueueCapacityBatches,
                          tags,
                          std::max<int64_t>(config_.receiver_queue_max_batches, 1));
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricReceiverDroppedBatches, tags, dropped_batches_.load());
    REPORT_DYNAMIC_GAUGE_(metrics_registry_, kMetricReceiverRejectedBatches, tags, rejected_batches_.load());
}

size_t KvcmEventStreamClient::ConnectionCount() const {
    return managed_connections_.load(std::memory_order_relaxed);
}

} // namespace kv_cache_manager
