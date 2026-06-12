#pragma once

// jax_utils.h — Extract PJRT_Buffer* from jax.Array Python objects.
//
// Adapted from tpu-raiden: frameworks/jax/jax_utils.h
// Uses pybind11 (not nanobind) to match the tair-kvcache binding layer.
//
// Required deps: jaxlib/py_array.h, xla/python/ifrt, xla/python/pjrt_ifrt,
//                xla/pjrt, xla/pjrt/c_api_client

#ifdef USING_TPU

#include <Python.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include "jaxlib/py_array.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/pjrt_ifrt/pjrt_array.h"

namespace kv_cache_manager {
namespace tpu {

// =====================================================================
// JAX internal PyArrayObject layout (matches jaxlib/py_array.h)
// =====================================================================

struct PyArrayObject {
    PyObject_HEAD;
#if PY_VERSION_HEX < 0x030C0000
    PyObject* weakrefs;
    PyObject* dict;
#endif  // PY_VERSION_HEX < 0x030C0000
    bool initialized;
    alignas(jax::PyArray::Storage) char array_storage[sizeof(jax::PyArray::Storage)];
};

// =====================================================================
// Core extraction helpers
// =====================================================================

/// Access PyArray::Storage from a raw PyObject* (JAX internal struct).
inline jax::PyArray::Storage* GetPyArrayStorageFromObject(
    PyArrayObject* py_array_object) {
    return std::launder(
        reinterpret_cast<jax::PyArray::Storage*>(py_array_object->array_storage));
}

/// Downcast ifrt::Array to PjRtCompatibleArray (only valid for pjrt_ifrt runtime).
inline xla::ifrt::PjRtCompatibleArray* CastToPjRtCompatibleArray(
    xla::ifrt::Array* ifrt_array) {
    if (ifrt_array == nullptr) return nullptr;
    if (ifrt_array->client()->runtime_type() == "pjrt_ifrt") {
        return static_cast<xla::ifrt::PjRtCompatibleArray*>(ifrt_array);
    }
    return nullptr;
}

/// Extract the first PjRtBuffer* from a jax.Array PyObject*.
inline xla::PjRtBuffer* GetPjrtBufferFromPyObject(PyObject* obj) {
    auto* py_array_obj = reinterpret_cast<PyArrayObject*>(obj);
    if (!py_array_obj->initialized) {
        throw std::runtime_error("PyArrayObject not initialized");
    }
    auto* storage = GetPyArrayStorageFromObject(py_array_obj);
    xla::ifrt::Array* ifrt_array = storage->ifrt_array.get();

    auto* arr = CastToPjRtCompatibleArray(ifrt_array);
    if (arr == nullptr) {
        throw std::runtime_error("Not a PjRt compatible array");
    }
    return arr->pjrt_buffers().front().get();
}

/// Extract ifrt::Array* from a jax.Array PyObject*.
inline xla::ifrt::Array* GetIfrtArrayFromPyObject(PyObject* obj) {
    auto* py_array_obj = reinterpret_cast<PyArrayObject*>(obj);
    if (!py_array_obj->initialized) {
        throw std::runtime_error("PyArrayObject not initialized");
    }
    auto* storage = GetPyArrayStorageFromObject(py_array_obj);
    return storage->ifrt_array.get();
}

// =====================================================================
// PjRtBuffer → PJRT_Buffer* (C API handle) conversion
// =====================================================================

/// Get the underlying C API PJRT_Buffer* handle from a PjRtBuffer*.
/// Returns nullptr if the buffer is not a C API wrapper (e.g. CommonPjRtBuffer).
inline PJRT_Buffer* GetCBufferHandle(xla::PjRtBuffer* buf) {
    auto* capi_buf = dynamic_cast<xla::PjRtCApiBuffer*>(buf);
    return capi_buf ? capi_buf->c_buffer() : nullptr;
}

/// Get the PJRT_Api* function pointer table from a PjRtBuffer*.
inline const PJRT_Api* GetPjrtCApi(xla::PjRtBuffer* buf) {
    auto* capi_buf = dynamic_cast<xla::PjRtCApiBuffer*>(buf);
    return capi_buf ? capi_buf->pjrt_c_api() : nullptr;
}

// =====================================================================
// Multi-shard extraction (using pybind11 object)
// =====================================================================

/// Extract PjRtBuffer* for each shard of a multi-shard jax.Array.
/// Uses pybind11::object (not nanobind) to match the project binding layer.
template <typename PyObjectType>
inline std::vector<xla::PjRtBuffer*> ExtractPjRtBuffersFromPyArray(
    const PyObjectType& jax_array) {
    std::vector<xla::PjRtBuffer*> result;

    // jax_array.addressable_shards → [Shard, Shard, ...]
    auto addressable_shards = jax_array.attr("addressable_shards");
    size_t num_shards = pybind11::len(addressable_shards);
    result.reserve(num_shards);

    for (size_t i = 0; i < num_shards; ++i) {
        auto shard = addressable_shards[i];
        auto shard_data = shard.attr("data");
        result.push_back(GetPjrtBufferFromPyObject(
            reinterpret_cast<PyObject*>(shard_data.ptr())));
    }
    return result;
}

}  // namespace tpu
}  // namespace kv_cache_manager

#endif  // USING_TPU
