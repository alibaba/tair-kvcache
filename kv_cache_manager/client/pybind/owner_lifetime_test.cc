#include <memory>
#include <pybind11/pybind11.h>

#include "kv_cache_manager/client/pybind/python_client_owner.h"

namespace py = pybind11;

namespace {
// This test exercises native holder/GC ordering, without registering memory or
// pretending to validate a data transfer backend.
class TestClient {
public:
    TestClient(py::object owner, py::list events) : owner_(py::weakref(owner)), events_(std::move(events)) {}

    void Close() noexcept {
        py::gil_scoped_acquire acquire;
        if (!closed_) {
            events_.attr("append")(py::make_tuple("close", !owner_().is_none()));
            closed_ = true;
        }
    }

    ~TestClient() {
        py::gil_scoped_acquire acquire;
        events_.attr("append")(py::make_tuple("destroy", !owner_().is_none()));
        owner_ = py::object();
        events_ = py::object();
    }

private:
    py::object owner_;
    py::object events_;
    bool closed_{false};
};
} // namespace

PYBIND11_MODULE(kvcm_owner_lifetime_test, module) {
    py::class_<TestClient, py::smart_holder>(module, "Client", py::dynamic_attr()).def("close", &TestClient::Close);
    module.def("create", [](py::object owner, py::list events) {
        auto client = std::make_unique<TestClient>(owner, events);
        return kv_cache_manager::MakePythonOwnedClient(std::move(client),
                                                       [owner = std::move(owner)]() mutable { owner = py::object(); });
    });
}
