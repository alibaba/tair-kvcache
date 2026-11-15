#pragma once

#include <memory>
#include <pybind11/gil.h>
#include <utility>

namespace kv_cache_manager {

// Python GC may clear dynamic attributes before disposing the native holder.
// Keep registration owners in its deleter through Close and SDK destruction.
// release_owners must clear Python references while the GIL is still held.
template <typename Client, typename ReleaseOwners>
std::shared_ptr<Client> MakePythonOwnedClient(std::unique_ptr<Client> client, ReleaseOwners release_owners) {
    return std::shared_ptr<Client>(client.release(),
                                   [release_owners = std::move(release_owners)](Client *owned) mutable {
                                       pybind11::gil_scoped_acquire acquire;
                                       {
                                           pybind11::gil_scoped_release release;
                                           owned->Close();
                                           delete owned;
                                       }
                                       release_owners();
                                   });
}

} // namespace kv_cache_manager
