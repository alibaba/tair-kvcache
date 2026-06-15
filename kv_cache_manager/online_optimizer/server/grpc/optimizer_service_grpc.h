#pragma once

#include <grpcpp/grpcpp.h>
#include <memory>

#include "kv_cache_manager/protocol/protobuf/optimizer_service.grpc.pb.h"

namespace kv_cache_manager {

class OptimizerServiceImpl;
class MetricsRegistry;

class OptimizerServiceGRpc final : public proto::optimizer::OptimizerService::Service {
public:
    OptimizerServiceGRpc(std::shared_ptr<OptimizerServiceImpl> service_impl,
                         std::shared_ptr<MetricsRegistry> metrics_registry);

    // InstanceGroup CRUD
    grpc::Status CreateInstanceGroup(grpc::ServerContext *context,
                                     const proto::optimizer::CreateInstanceGroupRequest *request,
                                     proto::optimizer::CommonResponse *response) override;

    grpc::Status UpdateInstanceGroup(grpc::ServerContext *context,
                                     const proto::optimizer::UpdateInstanceGroupRequest *request,
                                     proto::optimizer::CommonResponse *response) override;

    grpc::Status RemoveInstanceGroup(grpc::ServerContext *context,
                                     const proto::optimizer::RemoveInstanceGroupRequest *request,
                                     proto::optimizer::CommonResponse *response) override;

    grpc::Status GetInstanceGroup(grpc::ServerContext *context,
                                  const proto::optimizer::GetInstanceGroupRequest *request,
                                  proto::optimizer::GetInstanceGroupResponse *response) override;

    grpc::Status ListInstanceGroups(grpc::ServerContext *context,
                                    const proto::optimizer::ListInstanceGroupsRequest *request,
                                    proto::optimizer::ListInstanceGroupsResponse *response) override;

    // Instance management
    grpc::Status RegisterInstance(grpc::ServerContext *context,
                                  const proto::optimizer::OptimizerRegisterInstanceRequest *request,
                                  proto::optimizer::OptimizerRegisterInstanceResponse *response) override;

    grpc::Status RemoveInstance(grpc::ServerContext *context,
                                const proto::optimizer::OptimizerRemoveInstanceRequest *request,
                                proto::optimizer::OptimizerRemoveInstanceResponse *response) override;

    grpc::Status GetInstance(grpc::ServerContext *context,
                             const proto::optimizer::OptimizerGetInstanceRequest *request,
                             proto::optimizer::OptimizerGetInstanceResponse *response) override;

    // TraceQuery
    grpc::Status TraceQuery(grpc::ServerContext *context,
                            const proto::optimizer::TraceQueryRequest *request,
                            proto::optimizer::TraceQueryResponse *response) override;

    grpc::Status ListInstances(grpc::ServerContext *context,
                               const proto::optimizer::OptimizerListInstancesRequest *request,
                               proto::optimizer::OptimizerListInstancesResponse *response) override;

    grpc::Status ResetStats(grpc::ServerContext *context,
                            const proto::optimizer::OptimizerResetStatsRequest *request,
                            proto::optimizer::OptimizerResetStatsResponse *response) override;

private:
    std::shared_ptr<OptimizerServiceImpl> service_impl_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
};

} // namespace kv_cache_manager
