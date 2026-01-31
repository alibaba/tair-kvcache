#include <pybind11/gil.h>
#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kv_cache_manager/client/include/transfer_client.h"

namespace py = pybind11;
namespace kvcm = kv_cache_manager;

namespace {
struct OSException {
    int errcode;
};
} // namespace

PYBIND11_MODULE(kvcm_py_client, module) {
    pybind11::register_exception_translator([](std::exception_ptr p) {
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

    py::class_<kvcm::RegistSpan, py::smart_holder>(module, "RegistSpan")
        .def(py::init<>())
        .def_property("base", &kvcm::RegistSpan::base_as_uint64, &kvcm::RegistSpan::set_base_as_uint64)
        .def_readwrite("size", &kvcm::RegistSpan::size);

    py::class_<kvcm::InitParams, py::smart_holder>(module, "InitParams")
        .def(py::init<>())
        .def_readwrite("role_type", &kvcm::InitParams::role_type)
        .def_readwrite("regist_span", &kvcm::InitParams::regist_span)
        .def_readwrite("self_location_spec_name", &kvcm::InitParams::self_location_spec_name)
        .def_readwrite("storage_configs", &kvcm::InitParams::storage_configs);

    py::class_<kvcm::ForwardContext, py::smart_holder>(module, "ForwardContext")
        .def(py::init<>())
        .def_readwrite("metas", &kvcm::ForwardContext::metas)
        .def_readwrite("sw_size", &kvcm::ForwardContext::sw_size);

    // 由于pybind11会自动处理std::vector和std::map到Python list/dict的转换，
    // 我们不需要显式绑定这些标准容器类型，直接使用Python的list和dict即可
    // 保留这些类型定义以支持C++接口，但使用Python原生类型进行交互

    // 绑定TransferClient类
    py::class_<kvcm::TransferClient, py::smart_holder>(module, "TransferClient")
        .def_static("Create", &kvcm::TransferClient::Create, py::call_guard<py::gil_scoped_release>())
        .def("LoadKvCaches", &kvcm::TransferClient::LoadKvCaches, py::call_guard<py::gil_scoped_release>())
        .def("SaveKvCaches", &kvcm::TransferClient::SaveKvCaches, py::call_guard<py::gil_scoped_release>());

} // namespace kv_cache_manager
