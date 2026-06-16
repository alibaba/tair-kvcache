#include <pybind11/functional.h>
#include <pybind11/gil.h>
#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/optimizer/config/insight_simulator_types.h"
#include "kv_cache_manager/optimizer/config/instance_config.h"
#include "kv_cache_manager/optimizer/config/instance_group_config.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/config/optimizer_config_loader.h"
#include "kv_cache_manager/optimizer/manager/optimizer_loader.h"
#include "kv_cache_manager/optimizer/manager/optimizer_manager.h"
#include "kv_cache_manager/optimizer/config/hierarchical_replay_config_loader.h"
#include "kv_cache_manager/optimizer/manager/hierarchical_replay_manager.h"

namespace py = pybind11;
namespace kvcm = kv_cache_manager;

namespace {
struct OSException {
    int errcode;
};
} // namespace

PYBIND11_MODULE(kvcm_py_optimizer, module) {
    pybind11::register_local_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (const OSException &e) {
            errno = e.errcode;
            PyErr_SetFromErrno(PyExc_OSError);
        }
    });

    module.doc() = "kvcm_py_optimizer pybind11 extension";

    // 绑定枚举
    py::native_enum<kvcm::EvictionPolicyType>(module, "EvictionPolicyType", "enum.Enum")
        .value("UNSPECIFIED", kvcm::EvictionPolicyType::POLICY_UNSPECIFIED)
        .value("LRU", kvcm::EvictionPolicyType::POLICY_LRU)
        .value("RANDOM_LRU", kvcm::EvictionPolicyType::POLICY_RANDOM_LRU)
        .value("LEAF_AWARE_LRU", kvcm::EvictionPolicyType::POLICY_LEAF_AWARE_LRU)
        .value("TTL", kvcm::EvictionPolicyType::POLICY_TTL)
        .finalize();

    py::native_enum<kvcm::HitRatePerspective>(module, "HitRatePerspective", "enum.Enum")
        .value("KVCM_L3", kvcm::HitRatePerspective::KVCM_L3)
        .value("ENGINE_LOCAL", kvcm::HitRatePerspective::ENGINE_LOCAL)
        .finalize();

    py::native_enum<kvcm::DataStorageType>(module, "DataStorageType", "enum.Enum")
        .value("DATA_STORAGE_TYPE_UNKNOWN", kvcm::DataStorageType::DATA_STORAGE_TYPE_UNKNOWN)
        .value("DATA_STORAGE_TYPE_HF3FS", kvcm::DataStorageType::DATA_STORAGE_TYPE_HF3FS)
        .value("DATA_STORAGE_TYPE_MOONCAKE", kvcm::DataStorageType::DATA_STORAGE_TYPE_MOONCAKE)
        .value("DATA_STORAGE_TYPE_TAIR_MEMPOOL", kvcm::DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL)
        .value("DATA_STORAGE_TYPE_NFS", kvcm::DataStorageType::DATA_STORAGE_TYPE_NFS)
        .value("DATA_STORAGE_TYPE_VCNS_HF3FS", kvcm::DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS)
        .finalize();
    // 绑定insight_simulator_types.h中的结构体
    py::class_<kvcm::GetCacheLocationRes>(module, "GetCacheLocationRes")
        .def_readonly("trace_id", &kvcm::GetCacheLocationRes::trace_id)
        .def_readonly("kvcm_hit_length", &kvcm::GetCacheLocationRes::kvcm_hit_length)
        .def_readonly("hit_indices", &kvcm::GetCacheLocationRes::hit_indices);

    py::class_<kvcm::WriteCacheRes>(module, "WriteCacheRes")
        .def_readonly("trace_id", &kvcm::WriteCacheRes::trace_id)
        .def_readonly("kvcm_write_length", &kvcm::WriteCacheRes::kvcm_write_length)
        .def_readonly("kvcm_write_hit_length", &kvcm::WriteCacheRes::kvcm_write_hit_length);

    // 绑定 RadixTreeExport 结构体
    py::class_<kvcm::RadixTreeIndex::RadixTreeExportNode>(module, "RadixTreeExportNode")
        .def(py::init<>())
        .def_readonly("node_id", &kvcm::RadixTreeIndex::RadixTreeExportNode::node_id)
        .def_readonly("access_count", &kvcm::RadixTreeIndex::RadixTreeExportNode::access_count)
        .def_readonly("last_access_time", &kvcm::RadixTreeIndex::RadixTreeExportNode::last_access_time)
        .def_readonly("total_blocks", &kvcm::RadixTreeIndex::RadixTreeExportNode::total_blocks)
        .def_readonly("cached_blocks", &kvcm::RadixTreeIndex::RadixTreeExportNode::cached_blocks)
        .def_readonly("is_leaf", &kvcm::RadixTreeIndex::RadixTreeExportNode::is_leaf)
        .def_readonly("parent_id", &kvcm::RadixTreeIndex::RadixTreeExportNode::parent_id);

    py::class_<kvcm::RadixTreeIndex::RadixTreeExport>(module, "RadixTreeExport")
        .def(py::init<>())
        .def_readonly("instance_id", &kvcm::RadixTreeIndex::RadixTreeExport::instance_id)
        .def_readonly("nodes", &kvcm::RadixTreeIndex::RadixTreeExport::nodes)
        .def_readonly("edges", &kvcm::RadixTreeIndex::RadixTreeExport::edges);

    // 绑定optimizer_config.h中的OptimizerConfig类
    py::class_<kvcm::OptimizerConfig>(module, "OptimizerConfig")
        .def(py::init<>())
        .def("set_trace_file_path", &kvcm::OptimizerConfig::set_trace_file_path)
        .def("trace_file_path", &kvcm::OptimizerConfig::trace_file_path)
        .def("set_output_result_path", &kvcm::OptimizerConfig::set_output_result_path)
        .def("output_result_path", &kvcm::OptimizerConfig::output_result_path);

    py::class_<kvcm::OptInstanceGroupConfig>(module, "OptInstanceGroupConfig")
        .def(py::init<>())
        .def("quota_capacity", &kvcm::OptInstanceGroupConfig::quota_capacity)
        .def("set_quota_capacity", &kvcm::OptInstanceGroupConfig::set_quota_capacity)
        .def("used_percentage", &kvcm::OptInstanceGroupConfig::used_percentage)
        .def("set_used_percentage", &kvcm::OptInstanceGroupConfig::set_used_percentage);

    // 绑定 vector 类型，确保元素保持引用关系
    py::bind_vector<std::vector<kvcm::OptInstanceGroupConfig>>(module, "VectorOptInstanceGroupConfig");

    // 绑定OptimizerConfigLoader
    py::class_<kvcm::OptimizerConfigLoader>(module, "OptimizerConfigLoader")
        .def(py::init<>())
        .def("load", &kvcm::OptimizerConfigLoader::Load)
        .def("config", &kvcm::OptimizerConfigLoader::get_config);

    py::class_<kvcm::LoggerBroker>(module, "LoggerBroker")
        .def_static("InitLogger", &kvcm::LoggerBroker::InitLogger)
        .def_static("SetLogLevel", &kvcm::LoggerBroker::SetLogLevel)
        .def_static("DestroyLogger", &kvcm::LoggerBroker::DestroyLogger);

    // 绑定OptimizerManager类
    py::class_<kvcm::OptimizerManager>(module, "OptimizerManager")
        .def(py::init<const kvcm::OptimizerConfig &, bool, bool, kvcm::HitRatePerspective>(),
             py::arg("config"),
             py::arg("enable_lifecycle_tracking") = false,
             py::arg("enable_template_analysis") = false,
             py::arg("hit_rate_perspective") = kvcm::HitRatePerspective::KVCM_L3,
             "Initialize OptimizerManager. Set enable_lifecycle_tracking=True to track block lifecycle (uses ~10GB "
             "more memory). Set enable_template_analysis=True to enable template prefix analysis (slower replay)")
        .def("Init", &kvcm::OptimizerManager::Init, py::call_guard<py::gil_scoped_release>())
        .def("DirectRun", &kvcm::OptimizerManager::DirectRun, py::call_guard<py::gil_scoped_release>())
        .def("AnalyzeResults",
             &kvcm::OptimizerManager::AnalyzeResults,
             py::call_guard<py::gil_scoped_release>(),
             "Finalize, export and reset all registered trackers")
        .def("ExportRadixTrees", &kvcm::OptimizerManager::ExportRadixTrees, py::call_guard<py::gil_scoped_release>())
        .def("WriteCache",
             &kvcm::OptimizerManager::WriteCache,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("instance_id"),
             py::arg("trace_id"),
             py::arg("timestamp"),
             py::arg("block_ids"),
             py::arg("ttl_seconds") = int64_t(0))
        .def(
            "GetCacheLocation",
            [](kvcm::OptimizerManager &self,
               const std::string &instance_id,
               const std::string &trace_id,
               const int64_t timestamp,
               const std::vector<int64_t> &block_ids,
               const py::object &block_mask_obj,
               const int64_t input_len,
               const std::string &query_type) {
                kvcm::BlockMask block_mask;
                if (py::isinstance<std::vector<bool>>(block_mask_obj)) {
                    block_mask = block_mask_obj.cast<std::vector<bool>>();
                } else if (py::isinstance<py::int_>(block_mask_obj)) {
                    block_mask = block_mask_obj.cast<size_t>();
                } else {
                    throw std::invalid_argument("block_mask must be either a list of bools or an integer bitmask.");
                }
                return self.GetCacheLocation(
                    instance_id, trace_id, timestamp, block_ids, block_mask, input_len, true, true, query_type);
            },
            py::call_guard<py::gil_scoped_release>(),
            py::arg("instance_id"),
            py::arg("trace_id"),
            py::arg("timestamp"),
            py::arg("block_ids"),
            py::arg("block_mask"),
            py::arg("input_len"),
            py::arg("query_type") = "prefix_match")
        .def("ClearCache",
             &kvcm::OptimizerManager::ClearCache,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("instance_id"),
             "Clear cache for a specific instance without resetting statistics")
        .def("ClearAllCaches",
             &kvcm::OptimizerManager::ClearAllCaches,
             py::call_guard<py::gil_scoped_release>(),
             "Clear caches for all instances without resetting statistics")
        .def("ClearCacheAndResetStats",
             &kvcm::OptimizerManager::ClearCacheAndResetStats,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("instance_id"),
             "Clear cache and reset statistics for a specific instance")
        .def("ClearAllCachesAndResetStats",
             &kvcm::OptimizerManager::ClearAllCachesAndResetStats,
             py::call_guard<py::gil_scoped_release>(),
             "Clear caches and reset statistics for all instances");


    // =========================================================================
    // Hierarchical Replay Manager bindings
    // =========================================================================

    // HierarchicalGetCacheLocationRes
    py::class_<kvcm::HierarchicalGetCacheLocationRes>(module, "HierarchicalGetCacheLocationRes")
        .def_readonly("trace_id", &kvcm::HierarchicalGetCacheLocationRes::trace_id)
        .def_readonly("engine_hit_length", &kvcm::HierarchicalGetCacheLocationRes::engine_hit_length)
        .def_readonly("peer_hit_length", &kvcm::HierarchicalGetCacheLocationRes::peer_hit_length)
        .def_readonly("storage_pool_hit_length", &kvcm::HierarchicalGetCacheLocationRes::storage_pool_hit_length)
        .def_readonly("total_hit_length", &kvcm::HierarchicalGetCacheLocationRes::total_hit_length);

    // HierarchicalReplayConfigLoader
    py::class_<kvcm::HierarchicalReplayConfigLoader>(module, "HierarchicalReplayConfigLoader")
        .def(py::init<>())
        .def("load", &kvcm::HierarchicalReplayConfigLoader::Load)
        .def("config", &kvcm::HierarchicalReplayConfigLoader::get_config);

    // HierarchicalReplayConfig
    py::class_<kvcm::HierarchicalReplayConfig>(module, "HierarchicalReplayConfig")
        .def(py::init<>())
        .def("trace_file_path", &kvcm::HierarchicalReplayConfig::trace_file_path)
        .def("set_trace_file_path", &kvcm::HierarchicalReplayConfig::set_trace_file_path)
        .def("output_result_path", &kvcm::HierarchicalReplayConfig::output_result_path)
        .def("set_output_result_path", &kvcm::HierarchicalReplayConfig::set_output_result_path)
        .def("infer_scheduling_strategy", &kvcm::HierarchicalReplayConfig::infer_scheduling_strategy)
        .def("set_infer_scheduling_strategy", &kvcm::HierarchicalReplayConfig::set_infer_scheduling_strategy)
        .def("infer_active_windows_from_trace", &kvcm::HierarchicalReplayConfig::infer_active_windows_from_trace)
        .def("set_infer_active_windows_from_trace", &kvcm::HierarchicalReplayConfig::set_infer_active_windows_from_trace);


    // ChooseBestEngineRes
    py::class_<kvcm::ChooseBestEngineRes>(module, "ChooseBestEngineRes")
        .def_readonly("engine_instance_id", &kvcm::ChooseBestEngineRes::engine_instance_id)
        .def_readonly("hit_count", &kvcm::ChooseBestEngineRes::hit_count);

    // HierarchicalReplayManager
    py::class_<kvcm::HierarchicalReplayManager>(module, "HierarchicalReplayManager")
        .def(py::init<const kvcm::HierarchicalReplayConfig &>())
        .def("Init", &kvcm::HierarchicalReplayManager::Init, py::call_guard<py::gil_scoped_release>())
        .def("DirectRun", &kvcm::HierarchicalReplayManager::DirectRun, py::call_guard<py::gil_scoped_release>(),
             "Run full trace replay from the configured trace file")
        .def("AnalyzeResults", &kvcm::HierarchicalReplayManager::AnalyzeResults,
             py::call_guard<py::gil_scoped_release>(),
             "Finalize and export hierarchical hit rate analysis")
        .def("GetCacheLocation", &kvcm::HierarchicalReplayManager::GetCacheLocation,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("engine_instance_id"),
             py::arg("trace_id"),
             py::arg("timestamp"),
             py::arg("block_ids"),
             py::arg("input_len"),
             py::arg("query_type") = "prefix_match",
             "Query cache with hierarchical read: engine local -> P2P peer -> storage pool")
        .def("ChooseBestEngine", &kvcm::HierarchicalReplayManager::ChooseBestEngine,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("block_ids"),
             py::arg("timestamp"),
             "Find the engine with the longest prefix match for given block_ids (single C++ call)")
        .def("WriteCache", &kvcm::HierarchicalReplayManager::WriteCache,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("engine_instance_id"),
             py::arg("trace_id"),
             py::arg("timestamp"),
             py::arg("block_ids"),
             py::arg("ttl_seconds") = int64_t(0),
             "Write blocks to hierarchical cache with tier flow propagation")
        .def("WriteCacheWithTtlUs", &kvcm::HierarchicalReplayManager::WriteCacheWithTtlUs,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("engine_instance_id"),
             py::arg("trace_id"),
             py::arg("timestamp"),
             py::arg("block_ids"),
             py::arg("ttl_us"),
             "Write blocks with microsecond TTL granularity");

} // namespace kv_cache_manager
