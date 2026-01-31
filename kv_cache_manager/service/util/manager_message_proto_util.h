#pragma once

#include <type_traits>

#include "kv_cache_manager/config/account.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/cache_reclaim_strategy.h"
#include "kv_cache_manager/config/data_storage_strategy.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_group_quota.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/meta_cache_policy_config.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/config/model_deployment.h"
#include "kv_cache_manager/config/trigger_strategy.h"
#include "kv_cache_manager/data_storage/common_define.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/cache_location.h"
#include "kv_cache_manager/manager/cache_location_view.h"
#include "kv_cache_manager/protocol/protobuf/admin_service.pb.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
namespace kv_cache_manager {

class ProtoConvert {
public:
    // 将ModelDeployment转换为protobuf的ModelDeployment
    template <typename T>
    static void ModelDeploymentToProto(const ModelDeployment &model_deployment_info, T *proto_model_deployment);

    template <typename T>
    static void ModelDeploymentFromProto(const T *proto_model_deployment, ModelDeployment &model_deployment_info);

    template <typename T>
    static void CacheLocationViewToProto(const CacheLocationView &cache_location_info, T *proto_cache_location);

    template <typename T>
    static void BlockMaskToProto(const BlockMask &block_mask_info, T *proto_block_mask);

    template <typename T>
    static void BlockMaskFromProto(const T *proto_block_mask, BlockMask &block_mask_info);

    template <typename T>
    static void DataStorageTypeToProto(const DataStorageType &data_storage_type_info, T *proto_data_storage_type);
    template <typename T>
    static void DataStorageTypeFromProto(const T proto_data_storage_type, DataStorageType &data_storage_type_info);

    static void StorageConfigToProto(const StorageConfig &storage_config,
                                     proto::admin::StorageConfig *proto_storage_config);

    static void StorageFromProto(const proto::admin::StorageConfig *proto_storage_config,
                                 StorageConfig &storage_config);

    static void InstanceGroupToProto(const InstanceGroup &instance_group_info,
                                     proto::admin::InstanceGroup *proto_instance_group);

    static void InstanceGroupFromProto(const proto::admin::InstanceGroup *proto_instance_group,
                                       InstanceGroup &instance_group_info);

    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::InstanceInfo> ||
                            std::is_same_v<T, proto::admin::InstanceInfo>>
    InstanceInfoToProto(const InstanceInfo &instance_info, T *proto_instance_info);

    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::InstanceInfo> ||
                            std::is_same_v<T, proto::admin::InstanceInfo>>
    InstanceInfoFromProto(const T *proto_instance_info, InstanceInfo &instance_info);
    static void CacheConfigToProto(const CacheConfig &cache_config_info, proto::admin::CacheConfig *proto_cache_config);
    static void CacheConfigFromProto(const proto::admin::CacheConfig *proto_cache_config,
                                     CacheConfig &cache_config_info);
    static void AccountToProto(const Account &account_info, proto::admin::Account *proto_account);
    static void AccountFromProto(const proto::admin::Account *proto_account, Account &account_info);

    // LocationSpecInfo转换函数
    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpecInfo> ||
                            std::is_same_v<T, proto::admin::LocationSpecInfo>>
    LocationSpecInfoToProto(const LocationSpecInfo &location_spec_info, T *proto_location_spec_info);

    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpecInfo> ||
                            std::is_same_v<T, proto::admin::LocationSpecInfo>>
    LocationSpecInfoFromProto(const T *proto_location_spec_info, LocationSpecInfo &location_spec_info);

    // LocationSpecInfos转换函数
    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpecInfo> ||
                            std::is_same_v<T, proto::admin::LocationSpecInfo>>
    LocationSpecInfosToProto(const std::vector<LocationSpecInfo> &location_spec_infos,
                             google::protobuf::RepeatedPtrField<T> *proto_location_spec_infos);

    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpecInfo> ||
                            std::is_same_v<T, proto::admin::LocationSpecInfo>>
    LocationSpecInfosFromProto(const google::protobuf::RepeatedPtrField<T> &proto_location_spec_infos,
                               std::vector<LocationSpecInfo> &location_spec_infos);

    // LocationSpecGroup转换函数
    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpecGroup> ||
                            std::is_same_v<T, proto::admin::LocationSpecGroup>>
    LocationSpecGroupToProto(const LocationSpecGroup &location_spec_group, T *proto_location_spec_group);

    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpecGroup> ||
                            std::is_same_v<T, proto::admin::LocationSpecGroup>>
    LocationSpecGroupFromProto(const T *proto_location_spec_group, LocationSpecGroup &location_spec_group);

    // LocationSpecGroups转换函数
    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpecGroup> ||
                            std::is_same_v<T, proto::admin::LocationSpecGroup>>
    LocationSpecGroupsToProto(const std::vector<LocationSpecGroup> &location_spec_groups,
                              google::protobuf::RepeatedPtrField<T> *proto_location_spec_groups);

    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpecGroup> ||
                            std::is_same_v<T, proto::admin::LocationSpecGroup>>
    LocationSpecGroupsFromProto(const google::protobuf::RepeatedPtrField<T> &proto_location_spec_groups,
                                std::vector<LocationSpecGroup> &location_spec_groups);
    // LocationSpec转换函数
    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpec> ||
                            std::is_same_v<T, proto::admin::LocationSpec>>
    LocationSpecToProto(const LocationSpec &location_spec_info, T *proto_location_spec);
    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpec> ||
                            std::is_same_v<T, proto::admin::LocationSpec>>
    LocationSpecFromProto(const T *proto_location_spec, LocationSpec &location_spec_info);
    // LocationSpecs转换函数
    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpec> ||
                            std::is_same_v<T, proto::admin::LocationSpec>>
    LocationSpecsToProto(const std::vector<LocationSpec> &location_spec_infos,
                         google::protobuf::RepeatedPtrField<T> *proto_location_specs);
    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpec> ||
                            std::is_same_v<T, proto::admin::LocationSpec>>
    LocationSpecsFromProto(const google::protobuf::RepeatedPtrField<T> &proto_location_specs,
                           std::vector<LocationSpec> &location_spec_infos);
    // CacheLocation转换函数
    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::CacheLocation> ||
                            std::is_same_v<T, proto::admin::CacheLocation>>
    CacheLocationToProto(const CacheLocation &cache_location_info, T *proto_cache_location);
    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::CacheLocation> ||
                            std::is_same_v<T, proto::admin::CacheLocation>>
    CacheLocationFromProto(const T *proto_cache_location, CacheLocation &cache_location_info);
    // CacheLocations转换函数
    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::CacheLocation> ||
                            std::is_same_v<T, proto::admin::CacheLocation>>
    CacheLocationsToProto(const std::vector<CacheLocation> &cache_location_infos,
                          google::protobuf::RepeatedPtrField<T> *proto_cache_locations);
    template <typename T>
    static std::enable_if_t<std::is_same_v<T, proto::meta::CacheLocation> ||
                            std::is_same_v<T, proto::admin::CacheLocation>>
    CacheLocationsFromProto(const google::protobuf::RepeatedPtrField<T> &proto_cache_locations,
                            std::vector<CacheLocation> &cache_location_infos);
};
// DONE
template <typename T>
void ProtoConvert::ModelDeploymentToProto(const ModelDeployment &model_deployment_info, T *proto_model_deployment) {

    static_assert(std::is_same_v<T, proto::meta::ModelDeployment> || std::is_same_v<T, proto::admin::ModelDeployment>,
                  "T must be either proto::meta::ModelDeployment or proto::admin::ModelDeployment");

    // 填充基础字段
    proto_model_deployment->set_model_name(model_deployment_info.model_name());
    proto_model_deployment->set_dtype(model_deployment_info.dtype());
    proto_model_deployment->set_use_mla(model_deployment_info.use_mla());
    proto_model_deployment->set_tp_size(model_deployment_info.tp_size());
    proto_model_deployment->set_dp_size(model_deployment_info.dp_size());
    proto_model_deployment->set_lora_name(model_deployment_info.lora_name());
    proto_model_deployment->set_pp_size(model_deployment_info.pp_size());
    proto_model_deployment->set_extra(model_deployment_info.extra());
    proto_model_deployment->set_user_data(model_deployment_info.user_data());
}
// DONE
template <typename T>
void ProtoConvert::ModelDeploymentFromProto(const T *proto_model_deployment, ModelDeployment &model_deployment_info) {
    static_assert(std::is_same_v<T, proto::meta::ModelDeployment> || std::is_same_v<T, proto::admin::ModelDeployment>,
                  "T must be either proto::meta::ModelDeployment or proto::admin::ModelDeployment");
    // 填充基础字段
    model_deployment_info.set_model_name(proto_model_deployment->model_name());
    model_deployment_info.set_dtype(proto_model_deployment->dtype());
    model_deployment_info.set_use_mla(proto_model_deployment->use_mla());
    model_deployment_info.set_tp_size(proto_model_deployment->tp_size());
    model_deployment_info.set_dp_size(proto_model_deployment->dp_size());
    model_deployment_info.set_lora_name(proto_model_deployment->lora_name());
    model_deployment_info.set_pp_size(proto_model_deployment->pp_size());
    model_deployment_info.set_extra(proto_model_deployment->extra());
    model_deployment_info.set_user_data(proto_model_deployment->user_data());
}
// DONE
template <typename T>
void ProtoConvert::CacheLocationViewToProto(const CacheLocationView &cache_location_info, T *proto_cache_location) {
    static_assert(std::is_same_v<T, proto::meta::CacheLocation> || std::is_same_v<T, proto::admin::CacheLocation>,
                  "T must be either proto::meta::CacheLocation or proto::admin::CacheLocation");
    proto_cache_location->set_spec_size(cache_location_info.spec_size());
    if constexpr (std::is_same_v<T, proto::meta::CacheLocation>) {
        proto::meta::StorageType type;
        DataStorageTypeToProto(cache_location_info.type(), &type);
        proto_cache_location->set_type(type);
    } else if constexpr (std::is_same_v<T, proto::admin::CacheLocation>) {
        proto::admin::StorageType type;
        DataStorageTypeToProto(cache_location_info.type(), &type);
        proto_cache_location->set_type(type);
    }

    for (const auto &location_spec : cache_location_info.location_specs()) {
        auto *proto_spec = proto_cache_location->add_location_specs();
        proto_spec->set_name(location_spec.name());
        proto_spec->set_uri(location_spec.uri());
    }
}
// DONE
template <typename T>
void ProtoConvert::BlockMaskToProto(const BlockMask &block_mask_info, T *proto_block_mask) {
    static_assert(std::is_same_v<T, proto::meta::BlockMask>, "T must be proto::meta::BlockMask");
    proto_block_mask->clear_info();
    if (std::holds_alternative<BlockMaskVector>(block_mask_info)) {
        // 处理BlockMaskVector类型
        const auto &mask_vector = std::get<BlockMaskVector>(block_mask_info);
        auto bool_masks = proto_block_mask->mutable_bool_masks();
        bool_masks->clear_values();
        for (const auto &value : mask_vector) {
            bool_masks->add_values(value);
        }
    } else if (std::holds_alternative<BlockMaskOffset>(block_mask_info)) {
        // 处理BlockMaskOffset类型
        const auto &mask_offset = std::get<BlockMaskOffset>(block_mask_info);
        proto_block_mask->set_offset(static_cast<int64_t>(mask_offset));
    }
}
// DONE
template <typename T>
void ProtoConvert::BlockMaskFromProto(const T *proto_block_mask, BlockMask &block_mask_info) {
    static_assert(std::is_same_v<T, proto::meta::BlockMask> || std::is_same_v<T, proto::admin::BlockMask>,
                  "T must be proto::meta::BlockMask or proto::admin::BlockMask");
    switch (proto_block_mask->info_case()) {
    case T::InfoCase::kOffset: {
        // 处理offset类型
        block_mask_info = BlockMaskOffset(static_cast<size_t>(proto_block_mask->offset()));
        break;
    }
    case T::InfoCase::kBoolMasks: {
        // 处理bool_masks类型
        BlockMaskVector mask_vector;
        const auto &bool_masks = proto_block_mask->bool_masks();
        mask_vector.reserve(bool_masks.values_size());
        for (const auto &value : bool_masks.values()) {
            mask_vector.push_back(value);
        }
        block_mask_info = std::move(mask_vector);
        break;
    }
    default: {
        // 默认情况，创建空的BlockMaskVector
        block_mask_info = BlockMaskVector();
        break;
    }
    }
}
// DONE
template <typename T>
void ProtoConvert::DataStorageTypeToProto(const DataStorageType &data_storage_type_info, T *proto_data_storage_type) {
    static_assert(std::is_same_v<T, proto::meta::StorageType> || std::is_same_v<T, proto::admin::StorageType>,
                  "T must be either proto::meta::DataStorage or proto::admin::DataStorage");
    switch (data_storage_type_info) {
    case DataStorageType::DATA_STORAGE_TYPE_HF3FS: {
        *proto_data_storage_type = T::ST_3FS;
        break;
    }
    case DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS: {
        *proto_data_storage_type = T::ST_VCNS_3FS;
        break;
    }
    case DataStorageType::DATA_STORAGE_TYPE_MOONCAKE: {
        *proto_data_storage_type = T::ST_MOONCAKE;
        break;
    }
    case DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL: {
        *proto_data_storage_type = T::ST_TAIRMEMPOOL;
        break;
    }
    case DataStorageType::DATA_STORAGE_TYPE_UNKNOWN: {
        *proto_data_storage_type = T::ST_UNSPECIFIED;
        break;
    }
    case DataStorageType::DATA_STORAGE_TYPE_NFS: {
        *proto_data_storage_type = T::ST_NFS;
    }
    default: {
        // Handle unknown storage type case if necessary
        break;
    }
    }
}
// DONE
template <typename T>
void ProtoConvert::DataStorageTypeFromProto(const T proto_data_storage_type, DataStorageType &data_storage_type_info) {
    static_assert(std::is_same_v<T, proto::meta::StorageType> || std::is_same_v<T, proto::admin::StorageType>,
                  "T must be either proto::meta::DataStorage or proto::admin::DataStorage");
    switch (proto_data_storage_type) {
    case T::ST_UNSPECIFIED: {
        data_storage_type_info = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
        break;
    }
    case T::ST_3FS: {
        data_storage_type_info = DataStorageType::DATA_STORAGE_TYPE_HF3FS;
        break;
    }
    case T::ST_VCNS_3FS: {
        data_storage_type_info = DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS;
        break;
    }
    case T::ST_MOONCAKE: {
        data_storage_type_info = DataStorageType::DATA_STORAGE_TYPE_MOONCAKE;
        break;
    }
    case T::ST_TAIRMEMPOOL: {
        data_storage_type_info = DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL;
        break;
    }
    case T::ST_NFS: {
        data_storage_type_info = DataStorageType::DATA_STORAGE_TYPE_NFS;
        break;
    }
    default: {
        break;
    }
    }
}

template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::InstanceInfo> || std::is_same_v<T, proto::admin::InstanceInfo>>
ProtoConvert::InstanceInfoToProto(const InstanceInfo &instance_info, T *proto_instance_info) {
    proto_instance_info->set_quota_group_name(instance_info.quota_group_name());
    proto_instance_info->set_instance_group_name(instance_info.instance_group_name());
    proto_instance_info->set_instance_id(instance_info.instance_id());
    proto_instance_info->set_block_size(instance_info.block_size());

    // 添加location_spec_infos字段
    LocationSpecInfosToProto(instance_info.location_spec_infos(), proto_instance_info->mutable_location_spec_infos());

    ProtoConvert::ModelDeploymentToProto(instance_info.model_deployment(),
                                         proto_instance_info->mutable_model_deployment());

    // 添加location_spec_groups字段
    LocationSpecGroupsToProto(instance_info.location_spec_groups(),
                              proto_instance_info->mutable_location_spec_groups());
}

template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::InstanceInfo> || std::is_same_v<T, proto::admin::InstanceInfo>>
ProtoConvert::InstanceInfoFromProto(const T *proto_instance_info, InstanceInfo &instance_info) {
    instance_info.set_quota_group_name(proto_instance_info->quota_group_name());
    instance_info.set_instance_group_name(proto_instance_info->instance_group_name());
    instance_info.set_instance_id(proto_instance_info->instance_id());
    instance_info.set_block_size(proto_instance_info->block_size());

    // 处理location_spec_infos字段
    std::vector<LocationSpecInfo> location_spec_infos;
    LocationSpecInfosFromProto(proto_instance_info->location_spec_infos(), location_spec_infos);
    instance_info.set_location_spec_infos(location_spec_infos);

    ModelDeployment model_deployment;
    ProtoConvert::ModelDeploymentFromProto(&proto_instance_info->model_deployment(), model_deployment);
    instance_info.set_model_deployment(model_deployment);

    // 处理location_spec_groups字段
    std::vector<LocationSpecGroup> location_spec_groups;
    LocationSpecGroupsFromProto(proto_instance_info->location_spec_groups(), location_spec_groups);
    instance_info.set_location_spec_groups(location_spec_groups);
}

template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpecInfo> || std::is_same_v<T, proto::admin::LocationSpecInfo>>
ProtoConvert::LocationSpecInfoToProto(const LocationSpecInfo &location_spec_info, T *proto_location_spec_info) {
    proto_location_spec_info->set_name(location_spec_info.name());
    proto_location_spec_info->set_size(location_spec_info.size());
}

template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpecInfo> || std::is_same_v<T, proto::admin::LocationSpecInfo>>
ProtoConvert::LocationSpecInfoFromProto(const T *proto_location_spec_info, LocationSpecInfo &location_spec_info) {
    location_spec_info.set_name(proto_location_spec_info->name());
    location_spec_info.set_size(proto_location_spec_info->size());
}

template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpecInfo> || std::is_same_v<T, proto::admin::LocationSpecInfo>>
ProtoConvert::LocationSpecInfosToProto(const std::vector<LocationSpecInfo> &location_spec_infos,
                                       google::protobuf::RepeatedPtrField<T> *proto_location_spec_infos) {
    for (const auto &location_spec_info : location_spec_infos) {
        auto *proto_location_spec_info = proto_location_spec_infos->Add();
        LocationSpecInfoToProto(location_spec_info, proto_location_spec_info);
    }
}

template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpecInfo> || std::is_same_v<T, proto::admin::LocationSpecInfo>>
ProtoConvert::LocationSpecInfosFromProto(const google::protobuf::RepeatedPtrField<T> &proto_location_spec_infos,
                                         std::vector<LocationSpecInfo> &location_spec_infos) {
    location_spec_infos.reserve(proto_location_spec_infos.size());
    for (const auto &proto_location_spec_info : proto_location_spec_infos) {
        LocationSpecInfo location_spec_info;
        LocationSpecInfoFromProto(&proto_location_spec_info, location_spec_info);
        location_spec_infos.push_back(std::move(location_spec_info));
    }
}

template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpecGroup> ||
                 std::is_same_v<T, proto::admin::LocationSpecGroup>>
ProtoConvert::LocationSpecGroupToProto(const LocationSpecGroup &location_spec_group, T *proto_location_spec_group) {
    proto_location_spec_group->set_name(location_spec_group.name());
    for (const auto &spec_name : location_spec_group.spec_names()) {
        proto_location_spec_group->add_spec_names(spec_name);
    }
}

template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpecGroup> ||
                 std::is_same_v<T, proto::admin::LocationSpecGroup>>
ProtoConvert::LocationSpecGroupFromProto(const T *proto_location_spec_group, LocationSpecGroup &location_spec_group) {
    location_spec_group.set_name(proto_location_spec_group->name());
    std::vector<std::string> spec_names;
    for (const auto &spec_name : proto_location_spec_group->spec_names()) {
        spec_names.push_back(spec_name);
    }
    location_spec_group.set_spec_names(spec_names);
}

template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpecGroup> ||
                 std::is_same_v<T, proto::admin::LocationSpecGroup>>
ProtoConvert::LocationSpecGroupsToProto(const std::vector<LocationSpecGroup> &location_spec_groups,
                                        google::protobuf::RepeatedPtrField<T> *proto_location_spec_groups) {
    for (const auto &location_spec_group : location_spec_groups) {
        auto *proto_location_spec_group = proto_location_spec_groups->Add();
        LocationSpecGroupToProto(location_spec_group, proto_location_spec_group);
    }
}

template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpecGroup> ||
                 std::is_same_v<T, proto::admin::LocationSpecGroup>>
ProtoConvert::LocationSpecGroupsFromProto(const google::protobuf::RepeatedPtrField<T> &proto_location_spec_groups,
                                          std::vector<LocationSpecGroup> &location_spec_groups) {
    location_spec_groups.reserve(proto_location_spec_groups.size());
    for (const auto &proto_location_spec_group : proto_location_spec_groups) {
        LocationSpecGroup location_spec_group;
        LocationSpecGroupFromProto(&proto_location_spec_group, location_spec_group);
        location_spec_groups.push_back(std::move(location_spec_group));
    }
}
template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpec> || std::is_same_v<T, proto::admin::LocationSpec>>
ProtoConvert::LocationSpecToProto(const LocationSpec &location_spec_info, T *proto_location_spec) {
    proto_location_spec->set_name(location_spec_info.name());
    proto_location_spec->set_uri(location_spec_info.uri());
}
template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpec> || std::is_same_v<T, proto::admin::LocationSpec>>
ProtoConvert::LocationSpecFromProto(const T *proto_location_spec, LocationSpec &location_spec_info) {
    location_spec_info.set_name(proto_location_spec->name());
    location_spec_info.set_uri(proto_location_spec->uri());
}
template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpec> || std::is_same_v<T, proto::admin::LocationSpec>>
ProtoConvert::LocationSpecsToProto(const std::vector<LocationSpec> &location_spec_infos,
                                   google::protobuf::RepeatedPtrField<T> *proto_location_specs) {
    for (const auto &location_spec_info : location_spec_infos) {
        auto *proto_location_spec = proto_location_specs->Add();
        LocationSpecToProto(location_spec_info, proto_location_spec);
    }
}
template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::LocationSpec> || std::is_same_v<T, proto::admin::LocationSpec>>
ProtoConvert::LocationSpecsFromProto(const google::protobuf::RepeatedPtrField<T> &proto_location_specs,
                                     std::vector<LocationSpec> &location_spec_infos) {
    location_spec_infos.reserve(proto_location_specs.size());
    for (const auto &proto_location_spec : proto_location_specs) {
        LocationSpec location_spec_info;
        LocationSpecFromProto(&proto_location_spec, location_spec_info);
        location_spec_infos.push_back(std::move(location_spec_info));
    }
}
template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::CacheLocation> || std::is_same_v<T, proto::admin::CacheLocation>>
ProtoConvert::CacheLocationToProto(const CacheLocation &cache_location_info, T *proto_cache_location) {
    proto_cache_location->set_spec_size(cache_location_info.spec_size());
    if constexpr (std::is_same_v<T, proto::meta::CacheLocation>) {
        proto::meta::StorageType type;
        DataStorageTypeToProto(cache_location_info.type(), &type);
        proto_cache_location->set_type(type);
    } else if constexpr (std::is_same_v<T, proto::admin::CacheLocation>) {
        proto::admin::StorageType type;
        DataStorageTypeToProto(cache_location_info.type(), &type);
        proto_cache_location->set_type(type);
    }
    LocationSpecsToProto(cache_location_info.location_specs(), proto_cache_location->mutable_location_specs());
}
template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::CacheLocation> || std::is_same_v<T, proto::admin::CacheLocation>>
ProtoConvert::CacheLocationFromProto(const T *proto_cache_location, CacheLocation &cache_location_info) {
    cache_location_info.set_spec_size(proto_cache_location->spec_size());
    if constexpr (std::is_same_v<T, proto::meta::CacheLocation>) {
        DataStorageType data_storage_type;
        DataStorageTypeFromProto(proto_cache_location->type(), data_storage_type);
        cache_location_info.set_type(data_storage_type);
    } else if constexpr (std::is_same_v<T, proto::admin::CacheLocation>) {
        DataStorageType data_storage_type;
        DataStorageTypeFromProto(proto_cache_location->type(), data_storage_type);
        cache_location_info.set_type(data_storage_type);
    }
    std::vector<LocationSpec> location_spec_infos;
    LocationSpecsFromProto(proto_cache_location->location_specs(), location_spec_infos);
    cache_location_info.set_location_specs(std::move(location_spec_infos));
}
template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::CacheLocation> || std::is_same_v<T, proto::admin::CacheLocation>>
ProtoConvert::CacheLocationsToProto(const std::vector<CacheLocation> &cache_location_infos,
                                    google::protobuf::RepeatedPtrField<T> *proto_cache_locations) {
    for (const auto &cache_location_info : cache_location_infos) {
        auto *proto_cache_location = proto_cache_locations->Add();
        CacheLocationToProto(cache_location_info, proto_cache_location);
    }
}
template <typename T>
std::enable_if_t<std::is_same_v<T, proto::meta::CacheLocation> || std::is_same_v<T, proto::admin::CacheLocation>>
ProtoConvert::CacheLocationsFromProto(const google::protobuf::RepeatedPtrField<T> &proto_cache_locations,
                                      std::vector<CacheLocation> &cache_location_infos) {
    for (const auto &proto_cache_location : proto_cache_locations) {
        CacheLocation cache_location_info;
        CacheLocationFromProto(&proto_cache_location, cache_location_info);
        cache_location_infos.push_back(std::move(cache_location_info));
    }
}
} // namespace kv_cache_manager