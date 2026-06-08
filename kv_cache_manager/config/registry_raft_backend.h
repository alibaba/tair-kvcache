#pragma once

#include <map>
#include <string>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/config/registry_storage_backend.h"

namespace kv_cache_manager {

namespace raft_meta {
class RaftCoordinator;
} // namespace raft_meta

class RegistryRaftBackend : public RegistryStorageBackend {
public:
    RegistryRaftBackend() = default;
    ~RegistryRaftBackend() override = default;

    ErrorCode Init(const StandardUri &standard_uri) noexcept override;
    ErrorCode Load(const std::string &key, std::map<std::string, std::string> &out_value) noexcept override;
    ErrorCode Save(const std::string &key, const std::map<std::string, std::string> &value) noexcept override;
    ErrorCode Delete(const std::string &key) noexcept override;
    ErrorCode SaveField(const std::string &key, const std::string &field_id,
                        const std::string &value) noexcept override;
    ErrorCode DeleteField(const std::string &key, const std::string &field_id) noexcept override;

    void SetCoordinatorForTest(raft_meta::RaftCoordinator *coord) { coordinator_override_ = coord; }

private:
    raft_meta::RaftCoordinator *Coordinator() const;
    raft_meta::RaftCoordinator *coordinator_override_ = nullptr;
};

} // namespace kv_cache_manager
