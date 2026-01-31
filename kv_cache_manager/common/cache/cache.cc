//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "kv_cache_manager/common/cache/cache.h"

#include "kv_cache_manager/common/cache/lru_cache.h"

namespace kv_cache_manager {

const bool kDefaultToAdaptiveMutex = false;
const Cache::CacheItemHelper kNoopCacheItemHelper{};

// namespace {

// static void NoopDelete(Cache::ObjectPtr /*obj*/,
//                        MemoryAllocator* /*allocator*/) {
//   assert(false);
// }

// static size_t SliceSize(Cache::ObjectPtr obj) {
//   return static_cast<Slice*>(obj)->size();
// }

// static Status SliceSaveTo(Cache::ObjectPtr from_obj, size_t from_offset,
//                           size_t length, char* out) {
//   const Slice& slice = *static_cast<Slice*>(from_obj);
//   std::memcpy(out, slice.data() + from_offset, length);
//   return Status::OK();
// }

// static Status NoopCreate(const Slice& /*data*/, CompressionType /*type*/,
//                          CacheTier /*source*/, Cache::CreateContext* /*ctx*/,
//                          MemoryAllocator* /*allocator*/,
//                          Cache::ObjectPtr* /*out_obj*/,
//                          size_t* /*out_charge*/) {
//   assert(false);
//   return Status::NotSupported();
// }

// static Cache::CacheItemHelper kBasicCacheItemHelper(CacheEntryRole::kMisc,
//                                                     &NoopDelete);
// }  // namespace

// const Cache::CacheItemHelper kSliceCacheItemHelper{
//     CacheEntryRole::kMisc, &NoopDelete, &SliceSize,
//     &SliceSaveTo,          &NoopCreate, &kBasicCacheItemHelper,
// };

// Status SecondaryCache::CreateFromString(
//     const ConfigOptions& config_options, const std::string& value,
//     std::shared_ptr<SecondaryCache>* result) {
//   if (value.find("compressed_secondary_cache://") == 0) {
//     std::string args = value;
//     args.erase(0, std::strlen("compressed_secondary_cache://"));
//     Status status;
//     std::shared_ptr<SecondaryCache> sec_cache;

//     CompressedSecondaryCacheOptions sec_cache_opts;
//     status = OptionTypeInfo::ParseStruct(config_options, "",
//                                          &comp_sec_cache_options_type_info, "",
//                                          args, &sec_cache_opts);
//     if (status.ok()) {
//       sec_cache = NewCompressedSecondaryCache(sec_cache_opts);
//     }

//     if (status.ok()) {
//       result->swap(sec_cache);
//     }
//     return status;
//   } else {
//     return LoadSharedObject<SecondaryCache>(config_options, value, result);
//   }
// }

ErrorCode
Cache::CreateFromString(const ConfigOptions &config_options, const std::string &value, std::shared_ptr<Cache> *result) {
    return EC_OK;
}

bool Cache::AsyncLookupHandle::IsReady() {
    // return pending_handle == nullptr || pending_handle->IsReady();
    return false;
}

bool Cache::AsyncLookupHandle::IsPending() { return pending_handle != nullptr; }

Cache::Handle *Cache::AsyncLookupHandle::Result() {
    assert(!IsPending());
    return result_handle;
}

void Cache::StartAsyncLookup(AsyncLookupHandle &async_handle) {
    async_handle.found_dummy_entry = false; // in case re-used
    assert(!async_handle.IsPending());
    async_handle.result_handle = Lookup(
        async_handle.key, async_handle.helper, async_handle.create_context, async_handle.priority, async_handle.stats);
}

Cache::Handle *Cache::Wait(AsyncLookupHandle &async_handle) {
    WaitAll(&async_handle, 1);
    return async_handle.Result();
}

void Cache::WaitAll(AsyncLookupHandle *async_handles, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (async_handles[i].IsPending()) {
            // If a pending handle gets here, it should be marked at "to be handled
            // by a caller" by that caller erasing the pending_cache on it.
            assert(async_handles[i].pending_cache == nullptr);
        }
    }
}

void Cache::SetEvictionCallback(EvictionCallback &&fn) {
    // Overwriting non-empty with non-empty could indicate a bug
    assert(!eviction_callback_ || !fn);
    eviction_callback_ = std::move(fn);
}

} // namespace kv_cache_manager
