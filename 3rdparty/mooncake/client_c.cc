#include "client_c.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "client_service.h"
#include "utils.h"
#include <ylt/coro_http/coro_http_client.hpp>

#if __has_include("ha/ha_backend_factory.h") && \
    __has_include("ha/ha_types.h") &&            \
    __has_include("ha/leader_coordinator.h")
#define MOONCAKE_CLIENT_C_HAS_NEW_HA 1
#include "ha/ha_backend_factory.h"
#include "ha/ha_types.h"
#include "ha/leader_coordinator.h"
#elif __has_include("ha_helper.h")
#define MOONCAKE_CLIENT_C_HAS_LEGACY_HA 1
#include "ha_helper.h"
#endif

using namespace mooncake;

namespace {

constexpr uint16_t kDefaultMasterMetricsPort = 9003;

using ResolveMetricsHostFn = ErrorCode_t (*)(void *, std::string *);
using DestroyMetricsResolverStateFn = void (*)(void *);

struct MetricsResolverVTable {
    ResolveMetricsHostFn resolve_host = nullptr;
    DestroyMetricsResolverStateFn destroy_state = nullptr;
};

struct MetricsResolver {
    void *state = nullptr;
    MetricsResolverVTable vtable{};
};

struct ClientSession {
    ClientSession(std::shared_ptr<Client> native_client, uint16_t metrics_port,
                  MetricsResolver resolver)
        : client(std::move(native_client)),
          metrics_port(metrics_port),
          resolver_state(resolver.state),
          resolver_vtable(resolver.vtable) {}

    std::shared_ptr<Client> client;
    coro_http::coro_http_client http_client;
    uint16_t metrics_port = kDefaultMasterMetricsPort;
    void *resolver_state = nullptr;
    MetricsResolverVTable resolver_vtable{};
};

struct HttpResponse {
    int status = 0;
    std::string body;
};

ClientSession *GetSession(client_t handle) {
    return reinterpret_cast<ClientSession *>(handle);
}

Client *GetNativeClient(client_t handle) {
    ClientSession *session = GetSession(handle);
    if (session == nullptr) {
        return nullptr;
    }
    return session->client.get();
}

ErrorCode_t ToCErrorCode(ErrorCode error) {
    return static_cast<ErrorCode_t>(error);
}

uint16_t ResolveMetricsPortFromEnv() {
    const char *raw = std::getenv("MASTER_METRICS");
    if (raw == nullptr || raw[0] == '\0') {
        return kDefaultMasterMetricsPort;
    }

    errno = 0;
    char *end = nullptr;
    unsigned long parsed = std::strtoul(raw, &end, 10);
    if (raw == end || errno != 0 || end == nullptr || *end != '\0' ||
        parsed == 0 || parsed > std::numeric_limits<uint16_t>::max()) {
        LOG(WARNING) << "Invalid MASTER_METRICS=" << raw
                     << ", fallback to " << kDefaultMasterMetricsPort;
        return kDefaultMasterMetricsPort;
    }
    return static_cast<uint16_t>(parsed);
}

std::string ExtractHostFromAddress(std::string_view address) {
    if (address.empty()) {
        return {};
    }

    if (address.front() == '[') {
        const size_t bracket_pos = address.find(']');
        if (bracket_pos == std::string_view::npos || bracket_pos <= 1) {
            return {};
        }
        if (bracket_pos + 1 < address.size() &&
            address[bracket_pos + 1] != ':') {
            return {};
        }
        return std::string(address.substr(1, bracket_pos - 1));
    }

    const size_t first_colon = address.find(':');
    if (first_colon == std::string_view::npos) {
        return std::string(address);
    }

    const size_t last_colon = address.rfind(':');
    if (first_colon != last_colon) {
        return std::string(address);
    }
    if (first_colon == 0) {
        return {};
    }
    return std::string(address.substr(0, first_colon));
}

struct DirectMetricsResolverState {
    std::string host;
};

ErrorCode_t ResolveDirectMetricsHost(void *opaque, std::string *host) {
    if (opaque == nullptr || host == nullptr) {
        return MOONCAKE_ERROR_INVALID_PARAMS;
    }
    auto *state = reinterpret_cast<DirectMetricsResolverState *>(opaque);
    if (state->host.empty()) {
        return MOONCAKE_ERROR_INVALID_PARAMS;
    }
    *host = state->host;
    return MOONCAKE_ERROR_OK;
}

void DestroyDirectMetricsResolverState(void *opaque) {
    delete reinterpret_cast<DirectMetricsResolverState *>(opaque);
}

std::optional<HttpResponse> FetchHttpResponse(ClientSession &session,
                                              const std::string &url) {
    auto response = session.http_client.get(url);
    if (response.status <= 0) {
        return std::nullopt;
    }
    return HttpResponse{response.status, std::string(response.resp_body)};
}

bool ParseUint64Metric(const std::string &payload,
                       const std::string &metric_name, uint64_t &value) {
    std::istringstream input(payload);
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line.size() <= metric_name.size() ||
            line.compare(0, metric_name.size(), metric_name) != 0) {
            continue;
        }

        const char separator = line[metric_name.size()];
        if (separator != ' ' && separator != '\t') {
            continue;
        }

        const char *begin = line.c_str() + metric_name.size();
        while (*begin == ' ' || *begin == '\t') {
            ++begin;
        }

        errno = 0;
        char *end = nullptr;
        unsigned long long parsed = std::strtoull(begin, &end, 10);
        if (begin == end || errno != 0) {
            return false;
        }
        value = static_cast<uint64_t>(parsed);
        return true;
    }
    return false;
}

#if defined(MOONCAKE_CLIENT_C_HAS_NEW_HA)

struct NewHaMetricsResolverState {
    std::optional<ha::HABackendSpec> ha_backend_spec;
    std::unique_ptr<ha::LeaderCoordinator> leader_coordinator;
};

tl::expected<std::optional<ha::HABackendSpec>, ErrorCode> ParseHABackendSpec(
    std::string_view master_server_entry) {
    const size_t delimiter_pos = master_server_entry.find("://");
    if (delimiter_pos == std::string_view::npos) {
        return std::optional<ha::HABackendSpec>{std::nullopt};
    }

    auto backend_type =
        ha::ParseHABackendType(master_server_entry.substr(0, delimiter_pos));
    if (!backend_type.has_value()) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    return std::optional<ha::HABackendSpec>{ha::HABackendSpec{
        .type = backend_type.value(),
        .connstring = std::string(master_server_entry.substr(delimiter_pos + 3)),
        .cluster_namespace = "",
    }};
}

ErrorCode_t ResolveNewHaMetricsHost(void *opaque, std::string *host) {
    if (opaque == nullptr || host == nullptr) {
        return MOONCAKE_ERROR_INVALID_PARAMS;
    }

    auto *state = reinterpret_cast<NewHaMetricsResolverState *>(opaque);
    if (!state->ha_backend_spec.has_value()) {
        return MOONCAKE_ERROR_INVALID_PARAMS;
    }

    if (state->leader_coordinator == nullptr) {
        auto coordinator =
            ha::CreateLeaderCoordinator(state->ha_backend_spec.value());
        if (!coordinator) {
            return ToCErrorCode(coordinator.error());
        }
        state->leader_coordinator = std::move(coordinator.value());
    }

    auto current_view = state->leader_coordinator->ReadCurrentView();
    if (!current_view) {
        return ToCErrorCode(current_view.error());
    }
    if (!current_view.value().has_value()) {
        return MOONCAKE_ERROR_UNAVAILABLE_IN_CURRENT_STATUS;
    }

    std::string resolved_host =
        ExtractHostFromAddress(current_view.value()->leader_address);
    if (resolved_host.empty()) {
        return MOONCAKE_ERROR_INVALID_PARAMS;
    }
    *host = std::move(resolved_host);
    return MOONCAKE_ERROR_OK;
}

void DestroyNewHaMetricsResolverState(void *opaque) {
    delete reinterpret_cast<NewHaMetricsResolverState *>(opaque);
}

#elif defined(MOONCAKE_CLIENT_C_HAS_LEGACY_HA)

struct LegacyEtcdMetricsResolverState {
    std::string etcd_entry;
    std::unique_ptr<MasterViewHelper> master_view_helper;
    bool connected = false;
};

ErrorCode_t ResolveLegacyEtcdMetricsHost(void *opaque, std::string *host) {
    if (opaque == nullptr || host == nullptr) {
        return MOONCAKE_ERROR_INVALID_PARAMS;
    }

    auto *state = reinterpret_cast<LegacyEtcdMetricsResolverState *>(opaque);
    if (state->etcd_entry.empty()) {
        return MOONCAKE_ERROR_INVALID_PARAMS;
    }

    if (state->master_view_helper == nullptr) {
        state->master_view_helper = std::make_unique<MasterViewHelper>();
    }
    if (!state->connected) {
        auto err = state->master_view_helper->ConnectToEtcd(state->etcd_entry);
        if (err != ErrorCode::OK) {
            return ToCErrorCode(err);
        }
        state->connected = true;
    }

    std::string master_address;
    ViewVersionId master_version = 0;
    auto err = state->master_view_helper->GetMasterView(master_address,
                                                        master_version);
    if (err != ErrorCode::OK) {
        return ToCErrorCode(err);
    }

    std::string resolved_host = ExtractHostFromAddress(master_address);
    if (resolved_host.empty()) {
        return MOONCAKE_ERROR_INVALID_PARAMS;
    }
    *host = std::move(resolved_host);
    return MOONCAKE_ERROR_OK;
}

void DestroyLegacyEtcdMetricsResolverState(void *opaque) {
    delete reinterpret_cast<LegacyEtcdMetricsResolverState *>(opaque);
}

#endif

tl::expected<MetricsResolver, ErrorCode> BuildMetricsResolver(
    std::string_view master_server_entry) {
#if defined(MOONCAKE_CLIENT_C_HAS_NEW_HA)
    auto ha_backend_spec = ParseHABackendSpec(master_server_entry);
    if (!ha_backend_spec) {
        return tl::make_unexpected(ha_backend_spec.error());
    }
    if (ha_backend_spec.value().has_value()) {
        auto *state = new NewHaMetricsResolverState{
            .ha_backend_spec = std::move(ha_backend_spec.value()),
            .leader_coordinator = nullptr,
        };
        return MetricsResolver{
            .state = state,
            .vtable =
                MetricsResolverVTable{
                    .resolve_host = ResolveNewHaMetricsHost,
                    .destroy_state = DestroyNewHaMetricsResolverState,
                },
        };
    }
#elif defined(MOONCAKE_CLIENT_C_HAS_LEGACY_HA)
    constexpr std::string_view kEtcdScheme = "etcd://";
    if (master_server_entry.substr(0, kEtcdScheme.size()) == kEtcdScheme) {
        auto *state = new LegacyEtcdMetricsResolverState{
            .etcd_entry = std::string(master_server_entry.substr(kEtcdScheme.size())),
            .master_view_helper = nullptr,
            .connected = false,
        };
        return MetricsResolver{
            .state = state,
            .vtable =
                MetricsResolverVTable{
                    .resolve_host = ResolveLegacyEtcdMetricsHost,
                    .destroy_state = DestroyLegacyEtcdMetricsResolverState,
                },
        };
    }
#endif

    std::string direct_master_host = ExtractHostFromAddress(master_server_entry);
    if (direct_master_host.empty()) {
        return tl::make_unexpected(ErrorCode::INVALID_PARAMS);
    }

    auto *state = new DirectMetricsResolverState{
        .host = std::move(direct_master_host),
    };
    return MetricsResolver{
        .state = state,
        .vtable =
            MetricsResolverVTable{
                .resolve_host = ResolveDirectMetricsHost,
                .destroy_state = DestroyDirectMetricsResolverState,
            },
    };
}

ErrorCode_t ResolveMetricsHost(ClientSession &session, std::string *host) {
    if (session.resolver_vtable.resolve_host == nullptr) {
        return MOONCAKE_ERROR_INVALID_PARAMS;
    }
    return session.resolver_vtable.resolve_host(session.resolver_state, host);
}

std::string FormatHttpHost(std::string_view host) {
    if (host.find(':') != std::string_view::npos &&
        !(host.front() == '[' && host.back() == ']')) {
        return "[" + std::string(host) + "]";
    }
    return std::string(host);
}

std::string BuildMetricsBaseUrl(std::string_view host, uint16_t port) {
    return "http://" + FormatHttpHost(host) + ":" + std::to_string(port);
}

} // namespace

// Manage client object lifetime through a session that also owns HTTP context.
void *create_obj(std::shared_ptr<Client> client, uint16_t metrics_port,
                 MetricsResolver resolver) {
    return new ClientSession(std::move(client), metrics_port,
                             std::move(resolver));
}

void destroy_obj(void *handle) {
    ClientSession *session = GetSession(handle);
    if (session == nullptr) {
        return;
    }
    if (session->resolver_vtable.destroy_state != nullptr) {
        session->resolver_vtable.destroy_state(session->resolver_state);
    }
    delete session;
}

client_t mooncake_client_create(const char *local_hostname,
                                const char *metadata_connstring,
                                const char *protocol,
                                const char *rdma_devices,
                                const char *master_server_entry) {
    const char *resolved_master_server_entry =
        (master_server_entry != nullptr && master_server_entry[0] != '\0')
            ? master_server_entry
            : kDefaultMasterAddress.c_str();

    std::optional<std::string> device_names =
        (protocol != nullptr && strcmp(protocol, "rdma") == 0)
            ? std::optional<std::string>(rdma_devices)
            : std::nullopt;
    std::optional<std::shared_ptr<Client>> native = Client::Create(
        local_hostname, metadata_connstring, protocol, device_names,
        resolved_master_server_entry);
    if (!native) {
        return nullptr;
    }

    auto resolver = BuildMetricsResolver(resolved_master_server_entry);
    if (!resolver) {
        LOG(ERROR) << "Failed to build Mooncake client metrics resolver: "
                   << toString(resolver.error());
        return nullptr;
    }

    return create_obj(native.value(), ResolveMetricsPortFromEnv(),
                      std::move(resolver.value()));
}

ErrorCode_t mooncake_client_register_local_memory(
    client_t client, void *addr, size_t length, const char *location,
    bool remote_accessible, bool update_metadata) {
    Client *native_client = GetNativeClient(client);
    if (native_client == nullptr) {
        return MOONCAKE_ERROR_INVALID_PARAMS;
    }
    if (length == 0 || addr == nullptr) {
        return MOONCAKE_ERROR_OK;
    }
    auto result = native_client->RegisterLocalMemory(
        addr, length, location, remote_accessible, update_metadata);
    if (result) {
        return MOONCAKE_ERROR_OK;
    }

    LOG(ERROR) << "Failed to register local memory: "
               << toString(result.error());
    return ToCErrorCode(result.error());
}

ErrorCode_t mooncake_client_unregister_local_memory(client_t client,
                                                    void *addr,
                                                    bool update_metadata) {
    Client *native_client = GetNativeClient(client);
    if (native_client == nullptr) {
        return MOONCAKE_ERROR_INVALID_PARAMS;
    }
    if (addr == nullptr) {
        return MOONCAKE_ERROR_OK;
    }
    auto result = native_client->unregisterLocalMemory(addr, update_metadata);
    if (result) {
        return MOONCAKE_ERROR_OK;
    }

    LOG(ERROR) << "Failed to unregister local memory: "
               << toString(result.error());
    return ToCErrorCode(result.error());
}

ErrorCode_t mooncake_client_mount_segment(client_t client, size_t size,
                                          const char *protocol) {
    Client *native_client = GetNativeClient(client);
    if (native_client == nullptr || protocol == nullptr) {
        return MOONCAKE_ERROR_INVALID_PARAMS;
    }
    if (size == 0) {
        return MOONCAKE_ERROR_OK;
    }

    void *segment_ptr = allocate_buffer_allocator_memory(size);
    if (segment_ptr == nullptr) {
        LOG(ERROR) << "Failed to allocate segment memory";
        return 1;
    }

    auto result = native_client->MountSegment(segment_ptr, size, protocol);
    if (result) {
        return MOONCAKE_ERROR_OK;
    }

    LOG(ERROR) << "Failed to mount segment: " << toString(result.error());
    return ToCErrorCode(result.error());
}

ErrorCode_t mooncake_client_get(client_t client, const char *key,
                                Slice_t *slices, size_t slices_count) {
    Client *native_client = GetNativeClient(client);
    std::vector<Slice> slices_vector;
    for (size_t i = 0; i < slices_count; i++) {
        Slice slice;
        slice.ptr = slices[i].ptr;
        slice.size = slices[i].size;
        slices_vector.push_back(slice);
    }
    auto result = native_client->Get(key, slices_vector);
    if (result) {
        return MOONCAKE_ERROR_OK;
    }

    LOG(ERROR) << "Failed to get: " << toString(result.error());
    return ToCErrorCode(result.error());
}

ErrorCode_t mooncake_client_put(client_t client, const char *key,
                                Slice_t *slices, size_t slices_count,
                                const ReplicateConfig_t config) {
    Client *native_client = GetNativeClient(client);
    std::vector<Slice> slices_vector;
    for (size_t i = 0; i < slices_count; i++) {
        Slice slice;
        slice.ptr = slices[i].ptr;
        slice.size = slices[i].size;
        slices_vector.push_back(slice);
    }
    ReplicateConfig cpp_config;
    cpp_config.replica_num = config.replica_num;
    auto result = native_client->Put(key, slices_vector, cpp_config);
    if (result) {
        return MOONCAKE_ERROR_OK;
    }

    LOG(ERROR) << "Failed to put: " << toString(result.error());
    return ToCErrorCode(result.error());
}

ErrorCode_t mooncake_client_query(client_t client, const char *key) {
    Client *native_client = GetNativeClient(client);
    auto result = native_client->IsExist(key);
    if (!result) {
        LOG(ERROR) << "Failed to query: " << toString(result.error());
        return ToCErrorCode(result.error());
    }
    return result.value() ? MOONCAKE_ERROR_OK : MOONCAKE_ERROR_OBJECT_NOT_FOUND;
}

ErrorCode_t mooncake_client_remove(client_t client, const char *key) {
    Client *native_client = GetNativeClient(client);
    auto result = native_client->Remove(key);
    if (result) {
        return MOONCAKE_ERROR_OK;
    }

    LOG(ERROR) << "Failed to remove: " << toString(result.error());
    return ToCErrorCode(result.error());
}

ErrorCode_t mooncake_client_remove_all(client_t client) {
    Client *native_client = GetNativeClient(client);
    auto result = native_client->RemoveAll();
    if (result) {
        return MOONCAKE_ERROR_OK;
    }

    LOG(ERROR) << "Failed to remove all: " << toString(result.error());
    return ToCErrorCode(result.error());
}

ErrorCode_t mooncake_client_get_store_status(client_t client,
                                             MooncakeStoreStatus_t *status) {
    if (status == nullptr) {
        return MOONCAKE_ERROR_INVALID_PARAMS;
    }

    ClientSession *session = GetSession(client);
    if (session == nullptr) {
        return MOONCAKE_ERROR_INVALID_PARAMS;
    }

    *status = {};

    std::string metrics_host;
    ErrorCode_t resolve_ec = ResolveMetricsHost(*session, &metrics_host);
    if (resolve_ec != MOONCAKE_ERROR_OK) {
        LOG(ERROR) << "Failed to resolve Mooncake store metrics host: "
                   << resolve_ec;
        return resolve_ec;
    }

    const std::string base_url =
        BuildMetricsBaseUrl(metrics_host, session->metrics_port);

    auto health_response = FetchHttpResponse(*session, base_url + "/health");
    if (!health_response.has_value()) {
        LOG(ERROR) << "Failed to fetch Mooncake store health from " << base_url;
        return MOONCAKE_ERROR_RPC_FAIL;
    }

    status->healthy = (health_response->status == 200);
    status->health_status_code = health_response->status;

    auto metrics_response = FetchHttpResponse(*session, base_url + "/metrics");
    if (!metrics_response.has_value() || metrics_response->status != 200) {
        LOG(ERROR) << "Failed to fetch Mooncake store metrics from "
                   << base_url << ", status="
                   << (metrics_response.has_value() ? metrics_response->status
                                                    : -1);
        return MOONCAKE_ERROR_RPC_FAIL;
    }

    if (!ParseUint64Metric(metrics_response->body, "master_allocated_bytes",
                           status->allocated_bytes) ||
        !ParseUint64Metric(metrics_response->body,
                           "master_total_capacity_bytes",
                           status->total_capacity_bytes)) {
        LOG(ERROR) << "Failed to parse Mooncake store metrics payload";
        return MOONCAKE_ERROR_INTERNAL_ERROR;
    }

    if (status->total_capacity_bytes > 0) {
        status->used_ratio =
            static_cast<double>(status->allocated_bytes) /
            static_cast<double>(status->total_capacity_bytes);
    }

    return MOONCAKE_ERROR_OK;
}

void mooncake_client_destroy(client_t client) { destroy_obj(client); }

uint64_t mooncake_max_slice_size() { return kMaxSliceSize; }
