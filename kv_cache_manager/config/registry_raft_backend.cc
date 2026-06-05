#include "kv_cache_manager/config/registry_raft_backend.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/meta/raft/raft_coordinator.h"
#include "kv_cache_manager/meta/raft/raft_log_codec.h"

namespace kv_cache_manager {

raft_meta::RaftCoordinator *RegistryRaftBackend::Coordinator() const {
    if (coordinator_override_) {
        return coordinator_override_;
    }
    auto *coord = raft_meta::RaftCoordinator::GetInstance();
    if (!coord) {
        KVCM_LOG_ERROR("RegistryRaftBackend: no RaftCoordinator installed");
    }
    return coord;
}

ErrorCode RegistryRaftBackend::Init(const StandardUri & /*standard_uri*/) noexcept {
    if (!Coordinator()) {
        KVCM_LOG_ERROR("RegistryRaftBackend: Init failed, RaftCoordinator not available");
        return EC_ERROR;
    }
    return EC_OK;
}

ErrorCode RegistryRaftBackend::Load(const std::string &key,
                                    std::map<std::string, std::string> &out_value) noexcept {
    auto *coord = Coordinator();
    if (!coord) {
        return EC_ERROR;
    }
    return coord->RegistryLoad(key, out_value);
}

ErrorCode RegistryRaftBackend::Save(const std::string &key,
                                    const std::map<std::string, std::string> &value) noexcept {
    auto *coord = Coordinator();
    if (!coord) {
        return EC_ERROR;
    }
    raft_meta::LogOp op;
    op.type = raft_meta::OpType::kRegistrySave;
    op.registry_key = key;
    op.registry_fields = value;
    return coord->AppendAndWait(raft_meta::Encode(op));
}

ErrorCode RegistryRaftBackend::Delete(const std::string &key) noexcept {
    auto *coord = Coordinator();
    if (!coord) {
        return EC_ERROR;
    }
    raft_meta::LogOp op;
    op.type = raft_meta::OpType::kRegistryDelete;
    op.registry_key = key;
    return coord->AppendAndWait(raft_meta::Encode(op));
}

} // namespace kv_cache_manager
