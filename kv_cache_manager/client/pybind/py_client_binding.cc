#include <optional>

#include <pybind11/gil.h>
#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kv_cache_manager/client/include/transfer_client.h"
#include "kv_cache_manager/client/pybind/py_span_selection.h"

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
    std::vector<std::shared_ptr<PyRegistSpan>> regist_spans;
    std::string self_location_spec_name;
    std::string storage_configs;

    // regist_spans wins for the host-memory channel; the singular regist_span is
    // what legacy callers (and mooncake) get.
    std::vector<std::shared_ptr<PyRegistSpan>> HostRegistSpans() const {
        if (!regist_spans.empty()) {
            return regist_spans;
        }
        if (regist_span != nullptr) {
            return {regist_span};
        }
        return {};
    }

    // regist_span is filled in Create from a value snapshot, see there.
    kvcm::InitParams ToCpp() const {
        kvcm::InitParams params;
        params.role_type = role_type;
        params.self_location_spec_name = self_location_spec_name;
        params.storage_configs = storage_configs;
        return params;
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
        .value("ER_TASK_CANCELLED", kvcm::ClientErrorCode::ER_TASK_CANCELLED)
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
        .def_readwrite("regist_spans", &PyInitParams::regist_spans)
        .def_readwrite("self_location_spec_name", &PyInitParams::self_location_spec_name)
        .def_readwrite("storage_configs", &PyInitParams::storage_configs);

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
                // Snapshot both registration entries first: the explicit singular is
                // separate from the host-span list and only one of them has to be set.
                auto singular = py_init_params.regist_span;
                auto span_list = py_init_params.regist_spans;
                auto spans = py_init_params.HostRegistSpans();
                auto init_params = py_init_params.ToCpp();
                std::vector<const kvcm::RegistSpan *> host_span_pointers;
                for (const auto &span : spans) {
                    host_span_pointers.push_back(span == nullptr ? nullptr : &span->span);
                }
                // The backend holds raw addresses into these objects for the client's
                // lifetime, so the objects and their owners live on the client. The
                // singular gets its own reference: when both entries are set it is not
                // part of the host-span list.
                py::list owners;
                for (const auto &span : spans) {
                    owners.append(span == nullptr ? py::none() : span->owner);
                }
                py::object singular_owner = singular == nullptr ? py::none() : singular->owner;
                auto registration_of = [](const std::shared_ptr<PyRegistSpan> &span) {
                    if (span == nullptr || span->fd == -1) {
                        return std::optional<kvcm::SharedMemoryRegistration>{};
                    }
                    return std::optional<kvcm::SharedMemoryRegistration>{
                        kvcm::SharedMemoryRegistration{span->span.base, span->span.size, span->fd}};
                };
                // A host-span entry is either a complete registration or nothing at
                // all: dropping a half-filled one would leave the backend with fewer
                // segments than the caller handed over.
                auto append_host_registration = [](const std::shared_ptr<PyRegistSpan> &span,
                                                   size_t index,
                                                   std::vector<kvcm::SharedMemoryRegistration> &registrations) {
                    if (span == nullptr || (span->span.base == nullptr && span->span.size == 0 && span->fd == -1)) {
                        return;
                    }
                    std::string missing_fields;
                    if (span->span.base == nullptr) {
                        missing_fields += " base";
                    }
                    if (span->span.size == 0) {
                        missing_fields += " size";
                    }
                    if (span->fd < 0) {
                        missing_fields += " fd";
                    }
                    if (!missing_fields.empty()) {
                        throw std::invalid_argument("regist_spans[" + std::to_string(index) +
                                                    "] is incomplete, missing:" + missing_fields);
                    }
                    registrations.push_back(
                        kvcm::SharedMemoryRegistration{span->span.base, span->span.size, span->fd});
                };
                // Everything read after the release must come from these snapshots:
                // another thread may rebind regist_spans/regist_span meanwhile.
                const bool list_form = !span_list.empty();
                // Only the list channel is strict; the singular channel keeps its
                // legacy lenient semantics (see registration_of above).
                std::vector<kvcm::SharedMemoryRegistration> host_registrations;
                for (size_t i = 0; i < span_list.size(); ++i) {
                    append_host_registration(span_list[i], i, host_registrations);
                }
                const auto single_registration = registration_of(singular);
                // Single-region backends read this copy without the GIL; it lives for the
                // whole Create call, which is as long as anything reads it.
                auto singular_span = kvcm::SelectSingularRegistSpan(
                    singular == nullptr ? nullptr : &singular->span, host_span_pointers);
                init_params.regist_span = singular_span ? &*singular_span : nullptr;
                std::unique_ptr<kvcm::TransferClient> client;
                {
                    py::gil_scoped_release release;
                    if (list_form) {
                        client = host_registrations.empty()
                                     ? kvcm::TransferClient::Create(client_config, init_params)
                                     : kvcm::TransferClient::Create(client_config, init_params, host_registrations);
                    } else if (single_registration) {
                        client = kvcm::TransferClient::Create(client_config, init_params, *single_registration);
                    } else {
                        client = kvcm::TransferClient::Create(client_config, init_params);
                    }
                }
                if (client == nullptr) {
                    return py::none();
                }
                auto py_client = py::cast(std::move(client));
                py_client.attr("_regist_spans") = py::cast(spans);
                py_client.attr("_regist_spans_owners") = owners;
                py_client.attr("_regist_span") = py::cast(singular);
                py_client.attr("_regist_span_owner") = singular_owner;
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

} // namespace kv_cache_manager
