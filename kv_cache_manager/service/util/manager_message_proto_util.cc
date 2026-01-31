#include "manager_message_proto_util.h"

#include <type_traits>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {
void ProtoConvert::StorageConfigToProto(const StorageConfig &storage_config,
                                        proto::admin::StorageConfig *proto_storage_config) {
    // 设置global_unique_name
    proto_storage_config->set_global_unique_name(storage_config.global_unique_name());
    proto_storage_config->set_check_storage_available_when_open(storage_config.check_storage_available_when_open());
    auto type = storage_config.type();
    // 根据实际存储类型设置global_unique_name
    if (type == DataStorageType::DATA_STORAGE_TYPE_HF3FS) {
        const auto &three_fs_storage = *std::dynamic_pointer_cast<ThreeFSStorageSpec>(storage_config.storage_spec());
        auto *threefs = proto_storage_config->mutable_threefs();
        threefs->set_cluster_name(three_fs_storage.cluster_name());
        threefs->set_mountpoint(three_fs_storage.mountpoint());
        threefs->set_root_dir(three_fs_storage.root_dir());
        threefs->set_key_count_per_file(three_fs_storage.key_count_per_file());
        threefs->set_touch_file_when_create(three_fs_storage.touch_file_when_create());
    } else if (type == DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS) {
        const auto &vcns_threefs_storage =
            *std::dynamic_pointer_cast<VcnsThreeFSStorageSpec>(storage_config.storage_spec());
        auto *vcns_threefs = proto_storage_config->mutable_vcnsthreefs();
        vcns_threefs->set_cluster_name(vcns_threefs_storage.cluster_name());
        vcns_threefs->set_mountpoint(vcns_threefs_storage.mountpoint());
        vcns_threefs->set_root_dir(vcns_threefs_storage.root_dir());
        vcns_threefs->set_key_count_per_file(vcns_threefs_storage.key_count_per_file());
        vcns_threefs->set_touch_file_when_create(vcns_threefs_storage.touch_file_when_create());
        vcns_threefs->set_remote_host(vcns_threefs_storage.remote_host());
        vcns_threefs->set_remote_port(vcns_threefs_storage.remote_port());
        vcns_threefs->set_meta_storage_uri(vcns_threefs_storage.meta_storage_uri());
    } else if (type == DataStorageType::DATA_STORAGE_TYPE_MOONCAKE) {
        const auto &mooncake_storage = *std::dynamic_pointer_cast<MooncakeStorageSpec>(storage_config.storage_spec());
        // proto_storage_config->set_global_unique_name(mooncake_storage.get_global_unique_name());
        auto *mooncake = proto_storage_config->mutable_mooncake();
        mooncake->set_local_hostname(mooncake_storage.local_hostname());
        mooncake->set_metadata_connstring(mooncake_storage.metadata_connstring());
        mooncake->set_protocol(mooncake_storage.protocol());
        mooncake->set_rdma_device(mooncake_storage.rdma_device());
        mooncake->set_master_service_entry(mooncake_storage.master_server_entry());
    } else if (type == DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL) {
        const auto &tair_mem_pool_storage =
            *std::dynamic_pointer_cast<TairMemPoolStorageSpec>(storage_config.storage_spec());
        // proto_storage_config->set_global_unique_name(tair_mem_pool_storage.get_global_unique_name());
        auto *tair_mem_pool = proto_storage_config->mutable_tair_mem_pool();
        tair_mem_pool->set_domain(tair_mem_pool_storage.domain());
        tair_mem_pool->set_timeout(tair_mem_pool_storage.timeout());
        tair_mem_pool->set_enable_vipserver(tair_mem_pool_storage.enable_vipserver());
    } else if (type == DataStorageType::DATA_STORAGE_TYPE_NFS) {
        const auto &nfs_storage = *std::dynamic_pointer_cast<NfsStorageSpec>(storage_config.storage_spec());
        auto *nfs = proto_storage_config->mutable_nfs();
        nfs->set_root_path(nfs_storage.root_path());
        nfs->set_key_count_per_file(nfs_storage.key_count_per_file());
    }
}

void ProtoConvert::StorageFromProto(const proto::admin::StorageConfig *proto_storage_config,
                                    StorageConfig &storage_config) {
    // 设置global_unique_name
    storage_config.set_global_unique_name(proto_storage_config->global_unique_name());
    storage_config.set_check_storage_available_when_open(proto_storage_config->check_storage_available_when_open());
    // 根据proto中的storage_type_case设置对应的存储类型
    switch (proto_storage_config->storage_spec_case()) {
    case proto::admin::StorageConfig::kThreefs: {
        ThreeFSStorageSpec spec;
        spec.set_cluster_name(proto_storage_config->threefs().cluster_name());
        spec.set_mountpoint(proto_storage_config->threefs().mountpoint());
        spec.set_root_dir(proto_storage_config->threefs().root_dir());
        spec.set_key_count_per_file(proto_storage_config->threefs().key_count_per_file());
        spec.set_touch_file_when_create(proto_storage_config->threefs().touch_file_when_create());
        storage_config.set_storage_spec(std::make_shared<ThreeFSStorageSpec>(spec));
        storage_config.set_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
        break;
    }
    case proto::admin::StorageConfig::kVcnsthreefs: {
        VcnsThreeFSStorageSpec spec;
        spec.set_cluster_name(proto_storage_config->vcnsthreefs().cluster_name());
        spec.set_mountpoint(proto_storage_config->vcnsthreefs().mountpoint());
        spec.set_root_dir(proto_storage_config->vcnsthreefs().root_dir());
        spec.set_key_count_per_file(proto_storage_config->vcnsthreefs().key_count_per_file());
        spec.set_touch_file_when_create(proto_storage_config->vcnsthreefs().touch_file_when_create());
        spec.set_remote_host(proto_storage_config->vcnsthreefs().remote_host());
        spec.set_remote_port(proto_storage_config->vcnsthreefs().remote_port());
        spec.set_meta_storage_uri(proto_storage_config->vcnsthreefs().meta_storage_uri());
        storage_config.set_storage_spec(std::make_shared<VcnsThreeFSStorageSpec>(spec));
        storage_config.set_type(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS);
        break;
    }
    case proto::admin::StorageConfig::kMooncake: {
        MooncakeStorageSpec spec;
        spec.set_local_hostname(proto_storage_config->mooncake().local_hostname());
        spec.set_metadata_connstring(proto_storage_config->mooncake().metadata_connstring());
        spec.set_protocol(proto_storage_config->mooncake().protocol());
        spec.set_rdma_device(proto_storage_config->mooncake().rdma_device());
        spec.set_master_server_entry(proto_storage_config->mooncake().master_service_entry());
        storage_config.set_storage_spec(std::make_shared<MooncakeStorageSpec>(spec));
        storage_config.set_type(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE);
        break;
    }
    case proto::admin::StorageConfig::kTairMemPool: {
        TairMemPoolStorageSpec spec;
        spec.set_domain(proto_storage_config->tair_mem_pool().domain());
        spec.set_timeout(proto_storage_config->tair_mem_pool().timeout());
        spec.set_enable_vipserver(proto_storage_config->tair_mem_pool().enable_vipserver());
        storage_config.set_storage_spec(std::make_shared<TairMemPoolStorageSpec>(spec));
        storage_config.set_type(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL);
        break;
    }
    case proto::admin::StorageConfig::kNfs: {
        NfsStorageSpec spec;
        spec.set_root_path(proto_storage_config->nfs().root_path());
        spec.set_key_count_per_file(proto_storage_config->nfs().key_count_per_file());
        storage_config.set_storage_spec(std::make_shared<NfsStorageSpec>(spec));
        storage_config.set_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
        break;
    }
    default:
        storage_config.set_type(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN);
        KVCM_LOG_WARN("Unknown storage type in request proto: storage_type should be : threefs, mooncake, "
                      "tair_mem_pool or nfs");
        break;
    }
}

void ProtoConvert::CacheConfigToProto(const CacheConfig &cache_config_info,
                                      proto::admin::CacheConfig *proto_cache_config) {
    // 转换reclaim_strategy
    auto *reclaim_strategy = proto_cache_config->mutable_reclaim_strategy();
    reclaim_strategy->set_storage_unique_name(cache_config_info.reclaim_strategy()->storage_unique_name());
    reclaim_strategy->set_reclaim_policy(
        static_cast<proto::admin::ReclaimPolicy>(cache_config_info.reclaim_strategy()->reclaim_policy()));

    // 转换trigger_strategy
    auto *trigger_strategy = reclaim_strategy->mutable_trigger_strategy();
    trigger_strategy->set_used_size(cache_config_info.reclaim_strategy()->trigger_strategy().used_size());
    trigger_strategy->set_used_percentage(cache_config_info.reclaim_strategy()->trigger_strategy().used_percentage());

    reclaim_strategy->set_trigger_period_seconds(cache_config_info.reclaim_strategy()->trigger_period_seconds());
    reclaim_strategy->set_reclaim_step_size(cache_config_info.reclaim_strategy()->reclaim_step_size());
    reclaim_strategy->set_reclaim_step_percentage(cache_config_info.reclaim_strategy()->reclaim_step_percentage());
    reclaim_strategy->set_delay_before_delete_ms(cache_config_info.reclaim_strategy()->delay_before_delete_ms());

    // 转换data_storage_strategy (cache_prefer_strategy)
    proto_cache_config->set_data_storage_strategy(
        static_cast<proto::admin::CachePreferStrategy>(cache_config_info.cache_prefer_strategy()));

    // 转换meta_indexer_config
    auto origin_meta_indexer_config = cache_config_info.meta_indexer_config();
    if (cache_config_info.meta_indexer_config()) {
        auto *meta_indexer_config = proto_cache_config->mutable_meta_indexer_config();
        meta_indexer_config->set_max_key_count(origin_meta_indexer_config->GetMaxKeyCount());
        meta_indexer_config->set_mutex_shard_num(origin_meta_indexer_config->GetMutexShardNum());
        meta_indexer_config->set_batch_key_size(origin_meta_indexer_config->GetBatchKeySize());

        // 转换meta_storage_backend_config
        auto origin_meta_storage_backend_config = origin_meta_indexer_config->GetMetaStorageBackendConfig();
        if (cache_config_info.meta_indexer_config()->GetMetaStorageBackendConfig()) {
            auto *meta_storage_backend_config = meta_indexer_config->mutable_meta_storage_backend_config();
            meta_storage_backend_config->set_storage_type(origin_meta_storage_backend_config->GetStorageType());
            meta_storage_backend_config->set_storage_uri(origin_meta_storage_backend_config->GetStorageUri());
        }

        // 转换meta_cache_policy_config
        auto origin_meta_cache_policy_config = origin_meta_indexer_config->GetMetaCachePolicyConfig();
        if (cache_config_info.meta_indexer_config()->GetMetaCachePolicyConfig()) {
            auto *meta_cache_policy_config = meta_indexer_config->mutable_meta_cache_policy_config();
            meta_cache_policy_config->set_capacity(origin_meta_cache_policy_config->GetCapacity());
            meta_cache_policy_config->set_type(origin_meta_cache_policy_config->GetCachePolicyType());
            meta_cache_policy_config->set_cache_shard_bits(origin_meta_cache_policy_config->GetCacheShardBits());
            meta_cache_policy_config->set_high_pri_pool_ratio(origin_meta_cache_policy_config->GetHighPriPoolRatio());
        }
    }
}

void ProtoConvert::CacheConfigFromProto(const proto::admin::CacheConfig *proto_cache_config,
                                        CacheConfig &cache_config_info) {
    // 转换reclaim_strategy
    std::shared_ptr<CacheReclaimStrategy> reclaim_strategy = std::make_shared<CacheReclaimStrategy>();
    reclaim_strategy->set_storage_unique_name(proto_cache_config->reclaim_strategy().storage_unique_name());
    reclaim_strategy->set_reclaim_policy(
        static_cast<ReclaimPolicy>(proto_cache_config->reclaim_strategy().reclaim_policy()));

    // 转换trigger_strategy
    TriggerStrategy trigger_strategy;
    trigger_strategy.set_used_size(proto_cache_config->reclaim_strategy().trigger_strategy().used_size());
    trigger_strategy.set_used_percentage(proto_cache_config->reclaim_strategy().trigger_strategy().used_percentage());
    reclaim_strategy->set_trigger_strategy(trigger_strategy);

    reclaim_strategy->set_trigger_period_seconds(proto_cache_config->reclaim_strategy().trigger_period_seconds());
    reclaim_strategy->set_reclaim_step_size(proto_cache_config->reclaim_strategy().reclaim_step_size());
    reclaim_strategy->set_reclaim_step_percentage(proto_cache_config->reclaim_strategy().reclaim_step_percentage());
    reclaim_strategy->set_delay_before_delete_ms(proto_cache_config->reclaim_strategy().delay_before_delete_ms());

    cache_config_info.set_reclaim_strategy(reclaim_strategy);

    // 转换data_storage_strategy (cache_prefer_strategy)
    cache_config_info.set_cache_prefer_strategy(
        static_cast<CachePreferStrategy>(proto_cache_config->data_storage_strategy()));

    // 转换meta_indexer_config
    auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
    meta_indexer_config->SetMaxKeyCount(proto_cache_config->meta_indexer_config().max_key_count());
    meta_indexer_config->SetMutexShardNum(proto_cache_config->meta_indexer_config().mutex_shard_num());
    meta_indexer_config->SetBatchKeySize(proto_cache_config->meta_indexer_config().batch_key_size());

    // 转换meta_storage_backend_config
    auto meta_storage_backend_config = std::make_shared<MetaStorageBackendConfig>();
    meta_storage_backend_config->SetStorageType(
        proto_cache_config->meta_indexer_config().meta_storage_backend_config().storage_type());
    meta_storage_backend_config->SetStorageUri(
        proto_cache_config->meta_indexer_config().meta_storage_backend_config().storage_uri());
    meta_indexer_config->SetMetaStorageBackendConfig(meta_storage_backend_config);

    // 转换meta_cache_policy_config
    auto meta_cache_policy_config = std::make_shared<MetaCachePolicyConfig>();
    meta_cache_policy_config->SetCapacity(
        proto_cache_config->meta_indexer_config().meta_cache_policy_config().capacity());
    meta_cache_policy_config->SetType(proto_cache_config->meta_indexer_config().meta_cache_policy_config().type());
    meta_cache_policy_config->SetCacheShardBits(
        proto_cache_config->meta_indexer_config().meta_cache_policy_config().cache_shard_bits());
    meta_cache_policy_config->SetHighPriPoolRatio(
        proto_cache_config->meta_indexer_config().meta_cache_policy_config().high_pri_pool_ratio());
    meta_indexer_config->SetMetaCachePolicyConfig(meta_cache_policy_config);

    cache_config_info.set_meta_indexer_config(meta_indexer_config);
}

void ProtoConvert::InstanceGroupToProto(const InstanceGroup &instance_group_info,
                                        proto::admin::InstanceGroup *proto_instance_group) {
    proto_instance_group->set_name(instance_group_info.name());
    for (const auto &candidate : instance_group_info.storage_candidates()) {
        proto_instance_group->add_storage_candidates(candidate);
    }
    proto_instance_group->set_global_quota_group_name(instance_group_info.global_quota_group_name());
    proto_instance_group->set_max_instance_count(instance_group_info.max_instance_count());

    auto *quota = proto_instance_group->mutable_quota();
    quota->set_capacity(instance_group_info.quota().capacity());

    // 处理quota_config - proto中是repeated字段
    const auto &quota_configs = instance_group_info.quota().quota_config();
    for (const auto &quota_config : quota_configs) {
        auto *proto_quota_config = quota->add_quota_config();
        proto_quota_config->set_storage_type(static_cast<proto::admin::StorageType>(quota_config.storage_spec()));
        proto_quota_config->set_capacity(quota_config.capacity());
    }

    // 处理cache_config
    if (instance_group_info.cache_config()) {
        auto *cache_config = proto_instance_group->mutable_cache_config();
        CacheConfigToProto(*instance_group_info.cache_config(), cache_config);
    }

    proto_instance_group->set_user_data(instance_group_info.user_data());
    proto_instance_group->set_version(instance_group_info.version());
}
void ProtoConvert::InstanceGroupFromProto(const proto::admin::InstanceGroup *proto_instance_group,
                                          InstanceGroup &instance_group_info) {
    instance_group_info.set_name(proto_instance_group->name());
    std::vector<std::string> storage_candidates(proto_instance_group->storage_candidates().begin(),
                                                proto_instance_group->storage_candidates().end());
    instance_group_info.set_storage_candidates(storage_candidates);
    instance_group_info.set_global_quota_group_name(proto_instance_group->global_quota_group_name());
    instance_group_info.set_max_instance_count(proto_instance_group->max_instance_count());

    InstanceGroupQuota quota;
    quota.set_capacity(proto_instance_group->quota().capacity());

    // 处理quota_config - proto中是repeated字段
    std::vector<QuotaConfig> quota_configs;
    for (const auto &proto_quota_config : proto_instance_group->quota().quota_config()) {
        QuotaConfig quota_config;
        DataStorageType data_storage_type;
        auto storage_type = proto_quota_config.storage_type();
        DataStorageTypeFromProto(storage_type, data_storage_type);
        quota_config.set_storage_type(data_storage_type);
        quota_config.set_capacity(proto_quota_config.capacity());
        quota_configs.push_back(quota_config);
    }
    quota.set_quota_config(quota_configs);

    instance_group_info.set_quota(quota);

    // 处理cache_config
    if (proto_instance_group->has_cache_config()) {
        auto cache_config = std::make_shared<CacheConfig>();
        CacheConfigFromProto(&proto_instance_group->cache_config(), *cache_config);
        instance_group_info.set_cache_config(cache_config);
    }

    instance_group_info.set_user_data(proto_instance_group->user_data());
    instance_group_info.set_version(proto_instance_group->version());
}

void ProtoConvert::AccountFromProto(const proto::admin::Account *proto_account, Account &account_info) {
    account_info.set_user_name(proto_account->user_name());
    switch (proto_account->role()) {
    case proto::admin::ROLE_USER:
        account_info.set_role(AccountRole::ROLE_USER);
        break;
    case proto::admin::ROLE_ADMIN:
        account_info.set_role(AccountRole::ROLE_ADMIN);
        break;
    default:
        account_info.set_role(AccountRole::ROLE_USER); // 默认值
        break;
    }
}
void ProtoConvert::AccountToProto(const Account &account_info, proto::admin::Account *proto_account) {
    proto_account->set_user_name(account_info.user_name());
    switch (account_info.role()) {
    case AccountRole::ROLE_USER:
        proto_account->set_role(proto::admin::ROLE_USER);
        break;
    case AccountRole::ROLE_ADMIN:
        proto_account->set_role(proto::admin::ROLE_ADMIN);
        break;
    default:
        proto_account->set_role(proto::admin::ROLE_USER); // 默认值
        break;
    }
}
} // namespace kv_cache_manager
