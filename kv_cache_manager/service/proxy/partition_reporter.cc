#include "kv_cache_manager/service/proxy/partition_reporter.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <google/protobuf/util/json_util.h>
#include <grpcpp/grpcpp.h>
#include <iterator>
#include <unordered_map>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/service_discovery_factory.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/routing/routing_table.h"

namespace kv_cache_manager {
bool PartitionReporter::OwnsInstance(std::string_view instance) const {
    const auto bucket = InstanceBucketPolicy().GetBucket({{}, instance}, target_.bucket_owners_size());
    return bucket && target_.bucket_owners(*bucket) == target_.partition();
}

bool PartitionReporter::Init(const std::string &target_file, const std::string &discovery_uri) {
    std::ifstream stream(target_file);
    if (!stream) {
        KVCM_LOG_ERROR("cannot open partition target %s", target_file.c_str());
        return false;
    }
    const std::string json{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    auto status = google::protobuf::util::JsonStringToMessage(json, &target_);
    if (!status.ok() || target_.cluster().empty() || target_.version() == 0 ||
        std::find(target_.bucket_owners().begin(), target_.bucket_owners().end(), target_.partition()) ==
            target_.bucket_owners().end()) {
        KVCM_LOG_ERROR("invalid partition target %s", target_file.c_str());
        return false;
    }
    discovery_ = ServiceDiscoveryFactory::CreateServiceDiscovery(discovery_uri);
    return discovery_ != nullptr;
}

void PartitionReporter::Start(std::string endpoint, std::function<bool()> serving) {
    thread_ = std::thread([this, endpoint = std::move(endpoint), serving = std::move(serving)] {
        using Stub = proto::proxy::ProxyControlService::Stub;
        std::unordered_map<std::string, std::shared_ptr<Stub>> stubs;
        proto::proxy::ReplicaReport report;
        *report.mutable_target() = target_;
        report.set_process_id(StringUtil::GenerateRandomString(32));
        report.set_grpc_endpoint(endpoint);
        for (;;) {
            std::vector<ServiceEndpoint> proxies;
            if (discovery_->GetAllEndpoints(proxies)) {
                report.set_sequence(report.sequence() + 1);
                report.set_serving_leader(serving());
                // Parallel calls share one deadline. A slow proxy cannot delay registration on its peers.
                grpc::CompletionQueue queue;
                struct Call {
                    grpc::ClientContext context;
                    proto::proxy::ReportAck response;
                    grpc::Status status;
                    std::unique_ptr<grpc::ClientAsyncResponseReader<proto::proxy::ReportAck>> reader;
                };
                std::vector<std::unique_ptr<Call>> calls;
                std::unordered_map<std::string, std::shared_ptr<Stub>> current;
                const auto deadline = std::chrono::system_clock::now() + std::chrono::seconds(1);
                for (const auto &proxy : proxies) {
                    if (!current.emplace(proxy.host, stubs[proxy.host]).second) {
                        continue;
                    }
                    auto &stub = current.at(proxy.host);
                    if (!stub) {
                        stub = proto::proxy::ProxyControlService::NewStub(
                            grpc::CreateChannel(proxy.host, grpc::InsecureChannelCredentials()));
                    }
                    auto call = std::make_unique<Call>();
                    call->context.set_deadline(deadline);
                    call->reader = stub->AsyncReportReplica(&call->context, report, &queue);
                    call->reader->Finish(&call->response, &call->status, call.get());
                    calls.push_back(std::move(call));
                }
                stubs.swap(current);
                queue.Shutdown();
                void *tag;
                bool ok;
                while (queue.Next(&tag, &ok)) {
                    auto *call = static_cast<Call *>(tag);
                    if (!ok || !call->status.ok()) {
                        KVCM_LOG_WARN("proxy registration failed: %s", call->status.error_message().c_str());
                    }
                }
            }
            std::unique_lock lock(mutex_);
            if (wake_.wait_for(lock, std::chrono::seconds(2), [this] { return stopping_; })) {
                return;
            }
        }
    });
}

void PartitionReporter::Stop() {
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
}
} // namespace kv_cache_manager
