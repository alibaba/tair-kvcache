#include <pybind11/gil.h>
#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kv_cache_manager/client/include/kv_meta_object_client.h"
#include "kv_cache_manager/client/include/transfer_client.h"

namespace py = pybind11;
namespace kvcm = kv_cache_manager;

namespace {
struct OSException {
    int errcode;
};

struct PyRegistSpan {
    kvcm::RegistSpan span;
    int fd{-1};
    py::object owner{py::none()};

    void set_base_as_uint64(uint64_t base_ptr) { span.set_base_as_uint64(base_ptr); }
    uint64_t base_as_uint64() const { return span.base_as_uint64(); }
    void set_size(size_t size) { span.size = size; }
    size_t size() const { return span.size; }
};

struct PyInitParams {
    kvcm::RoleType role_type{kvcm::RoleType::UNKNOWN};
    std::shared_ptr<PyRegistSpan> regist_span;
    std::string self_location_spec_name;
    std::string storage_configs;

    kvcm::InitParams ToCpp() const {
        kvcm::InitParams params;
        params.role_type = role_type;
        params.regist_span = regist_span == nullptr ? nullptr : &regist_span->span;
        params.self_location_spec_name = self_location_spec_name;
        params.storage_configs = storage_configs;
        return params;
    }
};

struct PyKvMetaObjectClientConfig {
    kvcm::KvMetaClientConfig metadata;
    std::string instance_group;
    std::string user_data;
    std::string transfer_client_config;
    PyInitParams transfer_init_params;
    std::uint64_t max_object_bytes{1024ULL * 1024ULL * 1024ULL};
    std::int32_t write_timeout_seconds{30};

    kvcm::KvMetaObjectClientConfig ToCpp() const {
        kvcm::KvMetaObjectClientConfig config;
        config.metadata = metadata;
        config.instance_group = instance_group;
        config.user_data = user_data;
        config.transfer_client_config = transfer_client_config;
        config.transfer_init_params = transfer_init_params.ToCpp();
        config.max_object_bytes = max_object_bytes;
        config.write_timeout_seconds = write_timeout_seconds;
        return config;
    }
};
} // namespace

PYBIND11_MODULE(kvcm_py_client, module) {
    pybind11::register_local_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (const OSException &e) {
            errno = e.errcode;
            PyErr_SetFromErrno(PyExc_OSError);
        }
    });

    module.doc() = "kvcm_py_client pybind11 extension";

    // 绑定枚举
    py::native_enum<kvcm::ClientErrorCode>(module, "ClientErrorCode", "enum.Enum")
        .value("ER_OK", kvcm::ClientErrorCode::ER_OK)
        .value("ER_INVALID_STUB", kvcm::ClientErrorCode::ER_INVALID_STUB)
        .value("ER_INVALID_GRPCSTATUS", kvcm::ClientErrorCode::ER_INVALID_GRPCSTATUS)
        .value("ER_INVALID_PARAMS", kvcm::ClientErrorCode::ER_INVALID_PARAMS)
        .value("ER_INVALID_ROLETYPE", kvcm::ClientErrorCode::ER_INVALID_ROLETYPE)
        .value("ER_INVALID_CLIENT_CONFIG", kvcm::ClientErrorCode::ER_INVALID_CLIENT_CONFIG)
        .value("ER_INVALID_STORAGE_CONFIG", kvcm::ClientErrorCode::ER_INVALID_STORAGE_CONFIG)
        .value("ER_INVALID_SDKWRAPPER_CONFIG", kvcm::ClientErrorCode::ER_INVALID_SDKWRAPPER_CONFIG)
        .value("ER_INVALID_SDKBACKEND_CONFIG", kvcm::ClientErrorCode::ER_INVALID_SDKBACKEND_CONFIG)
        .value("ER_CONNECT_FAIL", kvcm::ClientErrorCode::ER_CONNECT_FAIL)
        .value("ER_THREADPOOL_ERROR", kvcm::ClientErrorCode::ER_THREADPOOL_ERROR)
        .value("ER_SKIPINIT", kvcm::ClientErrorCode::ER_SKIPINIT)
        .value("ER_METACLIENT_INIT_ERROR", kvcm::ClientErrorCode::ER_METACLIENT_INIT_ERROR)
        .value("ER_TRANSFERCLIENT_INIT_ERROR", kvcm::ClientErrorCode::ER_TRANSFERCLIENT_INIT_ERROR)
        .value("ER_MANAGERCLIENT_INIT_ERROR", kvcm::ClientErrorCode::ER_MANAGERCLIENT_INIT_ERROR)
        .value("ER_CLIENT_NOT_EXISTS", kvcm::ClientErrorCode::ER_CLIENT_NOT_EXISTS)
        .value("ER_SERVICE_NO_STATUS", kvcm::ClientErrorCode::ER_SERVICE_NO_STATUS)
        .value("ER_SERVICE_INTERNAL_ERROR", kvcm::ClientErrorCode::ER_SERVICE_INTERNAL_ERROR)
        .value("ER_SERVICE_UNSUPPORTED", kvcm::ClientErrorCode::ER_SERVICE_UNSUPPORTED)
        .value("ER_SERVICE_INVALID_ARGUMENT", kvcm::ClientErrorCode::ER_SERVICE_INVALID_ARGUMENT)
        .value("ER_SERVICE_DUPLICATE_ENTITY", kvcm::ClientErrorCode::ER_SERVICE_DUPLICATE_ENTITY)
        .value("ER_SERVICE_INSTANCE_NOT_EXIST", kvcm::ClientErrorCode::ER_SERVICE_INSTANCE_NOT_EXIST)
        .value("ER_SERVICE_NOT_LEADER", kvcm::ClientErrorCode::ER_SERVICE_NOT_LEADER)
        .value("ER_SERVICE_NOT_READY", kvcm::ClientErrorCode::ER_SERVICE_NOT_READY)
        .value("ER_SERVICE_REACH_MAX_ENTITY_CAPACITY",
               kvcm::ClientErrorCode::ER_SERVICE_REACH_MAX_ENTITY_CAPACITY)
        .value("ER_SERVICE_RESOURCE_EXHAUSTED", kvcm::ClientErrorCode::ER_SERVICE_RESOURCE_EXHAUSTED)
        .value("ER_SERVICE_NOT_FOUND", kvcm::ClientErrorCode::ER_SERVICE_NOT_FOUND)
        .value("ER_SERVICE_WRITE_IN_PROGRESS", kvcm::ClientErrorCode::ER_SERVICE_WRITE_IN_PROGRESS)
        .value("ER_SERVICE_SESSION_NOT_FOUND", kvcm::ClientErrorCode::ER_SERVICE_SESSION_NOT_FOUND)
        .value("ER_SERVICE_SIZE_MISMATCH", kvcm::ClientErrorCode::ER_SERVICE_SIZE_MISMATCH)
        .value("ER_SERVICE_IO_ERROR", kvcm::ClientErrorCode::ER_SERVICE_IO_ERROR)
        .value("ER_SDK_TIMEOUT", kvcm::ClientErrorCode::ER_SDK_TIMEOUT)
        .value("ER_GETSDK_ERROR", kvcm::ClientErrorCode::ER_GETSDK_ERROR)
        .value("ER_CREATESDK_ERROR", kvcm::ClientErrorCode::ER_CREATESDK_ERROR)
        .value("ER_SDKINIT_ERROR", kvcm::ClientErrorCode::ER_SDKINIT_ERROR)
        .value("ER_SDKREAD_ERROR", kvcm::ClientErrorCode::ER_SDKREAD_ERROR)
        .value("ER_SDKWRITE_ERROR", kvcm::ClientErrorCode::ER_SDKWRITE_ERROR)
        .value("ER_SDKALLOC_ERROR", kvcm::ClientErrorCode::ER_SDKALLOC_ERROR)
        .value("ER_INVALID_ADDRESS", kvcm::ClientErrorCode::ER_INVALID_ADDRESS)
        .value("ER_INVALID_LOCAL_BUFFERS", kvcm::ClientErrorCode::ER_INVALID_LOCAL_BUFFERS)
        .value("ER_UNSUPPORTED_MEMORY_TYPE", kvcm::ClientErrorCode::ER_UNSUPPORTED_MEMORY_TYPE)
        .value("ER_UNCONSISTENT_MEMORY_TYPE", kvcm::ClientErrorCode::ER_UNCONSISTENT_MEMORY_TYPE)
        .value("ER_FILE_IO_ERROR", kvcm::ClientErrorCode::ER_FILE_IO_ERROR)
        .value("ER_CUDAMEMCPY_ERROR", kvcm::ClientErrorCode::ER_CUDAMEMCPY_ERROR)
        .value("ER_EXTRACT_SLICES_ERROR", kvcm::ClientErrorCode::ER_EXTRACT_SLICES_ERROR)
        .value("ER_CUDA_STREAM_CREATE_ERROR", kvcm::ClientErrorCode::ER_CUDA_STREAM_CREATE_ERROR)
        .value("ER_CUDA_STREAM_SYNCHRONIZE_ERROR", kvcm::ClientErrorCode::ER_CUDA_STREAM_SYNCHRONIZE_ERROR)
        .value("ER_CUDA_STREAM_DESTROY_ERROR", kvcm::ClientErrorCode::ER_CUDA_STREAM_DESTROY_ERROR)
        .value("ER_CUDA_HOST_REGISTER_ERROR", kvcm::ClientErrorCode::ER_CUDA_HOST_REGISTER_ERROR)
        .finalize();

    py::native_enum<kvcm::MemoryType>(module, "MemoryType", "enum.Enum")
        .value("CPU", kvcm::MemoryType::CPU)
        .value("GPU", kvcm::MemoryType::GPU)
        .finalize();

    py::native_enum<kvcm::RoleType>(module, "RoleType", "enum.Enum")
        .value("UNKNOWN", kvcm::RoleType::UNKNOWN)
        .value("WORKER", kvcm::RoleType::WORKER)
        .value("SCHEDULER", kvcm::RoleType::SCHEDULER)
        .value("HYBRID", kvcm::RoleType::HYBRID)
        .finalize();

    // 绑定结构体
    py::class_<kvcm::Iov, py::smart_holder>(module, "Iov")
        .def(py::init<>())
        .def_readwrite("type", &kvcm::Iov::type)
        .def_property("base", &kvcm::Iov::base_as_uint64, &kvcm::Iov::set_base_as_uint64)
        .def_readwrite("size", &kvcm::Iov::size)
        .def_readwrite("ignore", &kvcm::Iov::ignore);

    py::class_<kvcm::BlockBuffer, py::smart_holder>(module, "BlockBuffer")
        .def(py::init<>())
        .def_readwrite("iovs", &kvcm::BlockBuffer::iovs);

    py::class_<PyRegistSpan, py::smart_holder>(module, "RegistSpan")
        .def(py::init<>())
        .def_property("base", &PyRegistSpan::base_as_uint64, &PyRegistSpan::set_base_as_uint64)
        .def_property("size", &PyRegistSpan::size, &PyRegistSpan::set_size)
        .def_readwrite("fd", &PyRegistSpan::fd)
        .def_readwrite("owner", &PyRegistSpan::owner);

    py::class_<PyInitParams, py::smart_holder>(module, "InitParams")
        .def(py::init<>())
        .def_readwrite("role_type", &PyInitParams::role_type)
        .def_readwrite("regist_span", &PyInitParams::regist_span)
        .def_readwrite("self_location_spec_name", &PyInitParams::self_location_spec_name)
        .def_readwrite("storage_configs", &PyInitParams::storage_configs);

    py::class_<kvcm::KvMetaClientConfig, py::smart_holder>(module, "KvMetaClientConfig")
        .def(py::init<>())
        .def_readwrite("addresses", &kvcm::KvMetaClientConfig::addresses)
        .def_readwrite("instance_id", &kvcm::KvMetaClientConfig::instance_id)
        .def_readwrite("call_timeout_ms", &kvcm::KvMetaClientConfig::call_timeout_ms);

    py::class_<PyKvMetaObjectClientConfig, py::smart_holder>(module, "KvMetaObjectClientConfig")
        .def(py::init<>())
        .def_readwrite("metadata", &PyKvMetaObjectClientConfig::metadata)
        .def_readwrite("instance_group", &PyKvMetaObjectClientConfig::instance_group)
        .def_readwrite("user_data", &PyKvMetaObjectClientConfig::user_data)
        .def_readwrite("transfer_client_config", &PyKvMetaObjectClientConfig::transfer_client_config)
        .def_readwrite("transfer_init_params", &PyKvMetaObjectClientConfig::transfer_init_params)
        .def_readwrite("max_object_bytes", &PyKvMetaObjectClientConfig::max_object_bytes)
        .def_readwrite("write_timeout_seconds", &PyKvMetaObjectClientConfig::write_timeout_seconds);

    py::class_<kvcm::ForwardContext, py::smart_holder>(module, "ForwardContext")
        .def(py::init<>())
        .def_readwrite("metas", &kvcm::ForwardContext::metas)
        .def_readwrite("sw_size", &kvcm::ForwardContext::sw_size);

    py::class_<kvcm::TransferTraceInfo, py::smart_holder>(module, "TransferTraceInfo")
        .def(py::init<>())
        .def_readwrite("need_print", &kvcm::TransferTraceInfo::need_print)
        .def_readwrite("block_ids", &kvcm::TransferTraceInfo::block_ids);

    // 由于pybind11会自动处理std::vector和std::map到Python list/dict的转换，
    // 我们不需要显式绑定这些标准容器类型，直接使用Python的list和dict即可
    // 保留这些类型定义以支持C++接口，但使用Python原生类型进行交互

    // 绑定TransferClient类
    py::class_<kvcm::TransferClient, py::smart_holder>(module, "TransferClient", py::dynamic_attr())
        .def_static(
            "Create",
            [](const std::string &client_config, const PyInitParams &py_init_params) -> py::object {
                auto init_params = py_init_params.ToCpp();
                auto captured_span = py_init_params.regist_span;
                py::object captured_owner = captured_span == nullptr ? py::none() : captured_span->owner;
                std::unique_ptr<kvcm::TransferClient> client;
                {
                    py::gil_scoped_release release;
                    if (captured_span != nullptr && captured_span->fd != -1) {
                        kvcm::SharedMemoryRegistration registration;
                        registration.base = captured_span->span.base;
                        registration.size = captured_span->span.size;
                        registration.fd = captured_span->fd;
                        client = kvcm::TransferClient::Create(client_config, init_params, registration);
                    } else {
                        client = kvcm::TransferClient::Create(client_config, init_params);
                    }
                }
                if (client == nullptr) {
                    return py::none();
                }
                auto py_client = py::cast(std::move(client));
                py_client.attr("_regist_span") = py::cast(captured_span);
                py_client.attr("_regist_span_owner") = captured_owner;
                return py_client;
            },
            py::arg("client_config"),
            py::arg("init_params"))
        .def("LoadKvCaches",
             &kvcm::TransferClient::LoadKvCaches,
             py::arg("uri_str_vec"),
             py::arg("block_buffers"),
             py::arg("trace_info") = nullptr,
             py::call_guard<py::gil_scoped_release>())
        .def("SaveKvCaches",
             &kvcm::TransferClient::SaveKvCaches,
             py::arg("uri_str_vec"),
             py::arg("block_buffers"),
             py::arg("trace_info") = nullptr,
             py::call_guard<py::gil_scoped_release>());

    py::class_<kvcm::KvMetaObjectClient, py::smart_holder>(module, "KvMetaObjectClient", py::dynamic_attr())
        .def_static(
            "Create",
            [](const std::string &trace_id, const PyKvMetaObjectClientConfig &py_config) -> py::tuple {
                auto config = py_config.ToCpp();
                auto captured_span = py_config.transfer_init_params.regist_span;
                py::object captured_owner = captured_span == nullptr ? py::none() : captured_span->owner;
                std::pair<kvcm::ClientErrorCode, std::unique_ptr<kvcm::KvMetaObjectClient>> result;
                {
                    py::gil_scoped_release release;
                    if (captured_span != nullptr && captured_span->fd != -1) {
                        kvcm::SharedMemoryRegistration registration;
                        registration.base = captured_span->span.base;
                        registration.size = captured_span->span.size;
                        registration.fd = captured_span->fd;
                        result = kvcm::KvMetaObjectClient::Create(trace_id, config, registration);
                    } else {
                        result = kvcm::KvMetaObjectClient::Create(trace_id, config);
                    }
                }
                if (result.second == nullptr) {
                    return py::make_tuple(result.first, py::none());
                }
                auto py_client = py::cast(std::move(result.second));
                py_client.attr("_regist_span") = py::cast(captured_span);
                py_client.attr("_regist_span_owner") = captured_owner;
                return py::make_tuple(result.first, py_client);
            },
            py::arg("trace_id"),
            py::arg("config"))
        .def("LoadObjects",
             &kvcm::KvMetaObjectClient::LoadObjects,
             py::arg("trace_id"),
             py::arg("keys"),
             py::arg("expected_value_sizes"),
             py::arg("object_buffers"),
             py::call_guard<py::gil_scoped_release>())
        .def("SaveObjects",
             &kvcm::KvMetaObjectClient::SaveObjects,
             py::arg("trace_id"),
             py::arg("keys"),
             py::arg("value_sizes"),
             py::arg("object_buffers"),
             py::call_guard<py::gil_scoped_release>())
        .def("Remove",
             &kvcm::KvMetaObjectClient::Remove,
             py::arg("trace_id"),
             py::arg("keys"),
             py::call_guard<py::gil_scoped_release>());

} // namespace kv_cache_manager
