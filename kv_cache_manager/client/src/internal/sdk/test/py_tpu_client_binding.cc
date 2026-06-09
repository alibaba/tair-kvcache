// pybind11 binding for TpuClient — for Python-level testing with JAX
// Build: bazel build //kv_cache_manager/client/src/internal/sdk/test:py_tpu_client --define=using_tpu=true

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kv_cache_manager/client/src/internal/sdk/tpu_client.h"

namespace py = pybind11;
namespace kvcm = kv_cache_manager;

PYBIND11_MODULE(py_tpu_client, m) {
    m.doc() = "TpuClient pybind11 binding for Python-level TPU testing";

    // 绑定 ClientErrorCode（只绑定常用值）
    py::enum_<kvcm::ClientErrorCode>(m, "ClientErrorCode")
        .value("ER_OK", kvcm::ER_OK)
        .value("ER_INVALID_PARAMS", kvcm::ER_INVALID_PARAMS)
        .value("ER_TPU_PJRT_INIT_ERROR", kvcm::ER_TPU_PJRT_INIT_ERROR)
        .value("ER_TPU_DMA_MAP_ERROR", kvcm::ER_TPU_DMA_MAP_ERROR)
        .value("ER_TPU_BUFFER_TRANSFER_ERROR", kvcm::ER_TPU_BUFFER_TRANSFER_ERROR)
        .value("ER_TPU_RAWBUFFER_ERROR", kvcm::ER_TPU_RAWBUFFER_ERROR)
        .value("ER_TPU_EVENT_ERROR", kvcm::ER_TPU_EVENT_ERROR)
        .export_values();

    // PJRT_Buffer / PJRT_Event / PJRT_RawBuffer 是不完整类型（opaque），
    // Python 端统一用 uintptr_t (int) 传递指针地址，无需 py::class_ 绑定。

    // 绑定 TpuClient
    py::class_<kvcm::TpuClient>(m, "TpuClient")
        .def(py::init<>())

        // Init / Destroy
        .def("init", &kvcm::TpuClient::Init)
        .def("destroy", &kvcm::TpuClient::Destroy)

        // DMA map/unmap
        .def("dma_map", [](kvcm::TpuClient& self, uintptr_t addr, size_t size) {
            return self.DmaMap(reinterpret_cast<void*>(addr), size);
        }, py::arg("addr"), py::arg("size"))
        .def("dma_unmap", [](kvcm::TpuClient& self, uintptr_t addr) {
            return self.DmaUnmap(reinterpret_cast<void*>(addr));
        }, py::arg("addr"))

        // Synchronous BufferFromHost: int_addr → PJRT_Buffer*
        .def("buffer_from_host", [](kvcm::TpuClient& self, uintptr_t addr, size_t size)
                -> py::tuple {
            PJRT_Buffer* buf = nullptr;
            auto ec = self.BufferFromHost(reinterpret_cast<const void*>(addr), size, buf);
            return py::make_tuple(ec, reinterpret_cast<uintptr_t>(buf));
        }, py::arg("addr"), py::arg("size"))

        // Synchronous BufferToHost: PJRT_Buffer* → int_addr
        .def("buffer_to_host", [](kvcm::TpuClient& self, uintptr_t buf_addr,
                                 uintptr_t dst_addr, size_t size) {
            return self.BufferToHost(reinterpret_cast<PJRT_Buffer*>(buf_addr),
                                     reinterpret_cast<void*>(dst_addr), size);
        }, py::arg("buffer"), py::arg("dst_addr"), py::arg("size"))

        .def("destroy_buffer", [](kvcm::TpuClient& self, uintptr_t buf_addr) {
            self.DestroyBuffer(reinterpret_cast<PJRT_Buffer*>(buf_addr));
        }, py::arg("buffer"))

        // Async BufferFromHost
        .def("buffer_from_host_async", [](kvcm::TpuClient& self, uintptr_t addr, size_t size)
                -> py::tuple {
            PJRT_Buffer* buf = nullptr;
            PJRT_Event* ev = nullptr;
            auto ec = self.BufferFromHostAsync(reinterpret_cast<const void*>(addr), size, buf, ev);
            return py::make_tuple(ec, reinterpret_cast<uintptr_t>(buf),
                                  reinterpret_cast<uintptr_t>(ev));
        }, py::arg("addr"), py::arg("size"))

        // Async BufferToHost
        .def("buffer_to_host_async", [](kvcm::TpuClient& self, uintptr_t buf_addr,
                                        uintptr_t dst_addr, size_t size) -> py::tuple {
            PJRT_Event* ev = nullptr;
            auto ec = self.BufferToHostAsync(reinterpret_cast<PJRT_Buffer*>(buf_addr),
                                             reinterpret_cast<void*>(dst_addr), size, ev);
            return py::make_tuple(ec, reinterpret_cast<uintptr_t>(ev));
        }, py::arg("buffer"), py::arg("dst_addr"), py::arg("size"))

        // Event management
        .def("wait_event", [](kvcm::TpuClient& self, uintptr_t ev_addr) {
            return self.WaitEvent(reinterpret_cast<PJRT_Event*>(ev_addr));
        }, py::arg("event"))
        .def("wait_events", [](kvcm::TpuClient& self, std::vector<uintptr_t>& ev_addrs) {
            std::vector<PJRT_Event*> events;
            for (auto a : ev_addrs)
                events.push_back(reinterpret_cast<PJRT_Event*>(a));
            return self.WaitEvents(events);
        }, py::arg("events"))
        .def("destroy_event", [](kvcm::TpuClient& self, uintptr_t ev_addr) {
            self.DestroyEvent(reinterpret_cast<PJRT_Event*>(ev_addr));
        }, py::arg("event"))

        // RawBuffer
        .def("has_raw_buffer_extension", &kvcm::TpuClient::HasRawBufferExtension)
        .def("create_raw_alias", [](kvcm::TpuClient& self, uintptr_t buf_addr) -> py::tuple {
            PJRT_RawBuffer* raw = nullptr;
            auto ec = self.CreateRawAlias(reinterpret_cast<PJRT_Buffer*>(buf_addr), raw);
            return py::make_tuple(ec, reinterpret_cast<uintptr_t>(raw));
        }, py::arg("buffer"))
        .def("raw_buffer_from_host",
             [](kvcm::TpuClient& self, uintptr_t raw_addr, uintptr_t src_addr,
                int64_t offset, int64_t size) -> py::tuple {
            PJRT_Event* ev = nullptr;
            auto ec = self.RawBufferFromHost(reinterpret_cast<PJRT_RawBuffer*>(raw_addr),
                                             reinterpret_cast<const void*>(src_addr),
                                             offset, size, ev);
            return py::make_tuple(ec, reinterpret_cast<uintptr_t>(ev));
        }, py::arg("raw"), py::arg("src_addr"), py::arg("offset"), py::arg("size"))
        .def("raw_buffer_to_host",
             [](kvcm::TpuClient& self, uintptr_t raw_addr, uintptr_t dst_addr,
                int64_t offset, int64_t size) -> py::tuple {
            PJRT_Event* ev = nullptr;
            auto ec = self.RawBufferToHost(reinterpret_cast<PJRT_RawBuffer*>(raw_addr),
                                           reinterpret_cast<void*>(dst_addr),
                                           offset, size, ev);
            return py::make_tuple(ec, reinterpret_cast<uintptr_t>(ev));
        }, py::arg("raw"), py::arg("dst_addr"), py::arg("offset"), py::arg("size"))
        .def("raw_buffer_get_device_size", [](kvcm::TpuClient& self, uintptr_t raw_addr)
                -> py::tuple {
            size_t sz = 0;
            auto ec = self.RawBufferGetDeviceSize(reinterpret_cast<PJRT_RawBuffer*>(raw_addr), sz);
            return py::make_tuple(ec, sz);
        }, py::arg("raw"))
        .def("destroy_raw_buffer", [](kvcm::TpuClient& self, uintptr_t raw_addr) {
            self.DestroyRawBuffer(reinterpret_cast<PJRT_RawBuffer*>(raw_addr));
        }, py::arg("raw"));
}
