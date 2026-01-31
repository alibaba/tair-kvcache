//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// Common hash functions with convenient interfaces. If hashing a
// statically-sized input in a performance-critical context, consider
// calling a specific hash implementation directly, such as
// XXH3_64bits from xxhash.h.
//
// Since this is a very common header, implementation details are kept
// out-of-line. Out-of-lining also aids in tracking the time spent in
// hashing functions. Inlining is of limited benefit for runtime-sized
// hash inputs.

#pragma once

#include <cstddef>
#include <cstdint>

#include "kv_cache_manager/common/hash/xxph3.h"

namespace kv_cache_manager {

inline uint32_t Lower32of64(uint64_t v) { return static_cast<uint32_t>(v); }

// We are standardizing on a preview release of XXH3, because that's
// the best available at time of standardizing.
//
// In testing (mostly Intel Skylake), this hash function is much more
// thorough than Hash32 and is almost universally faster. Hash() only
// seems faster when passing runtime-sized keys of the same small size
// (less than about 24 bytes) thousands of times in a row; this seems
// to allow the branch predictor to work some magic. XXH3's speed is
// much less dependent on branch prediction.
//
// Hashing with a prefix extractor is potentially a common case of
// hashing objects of small, predictable size. We could consider
// bundling hash functions specialized for particular lengths with
// the prefix extractors.
inline uint64_t Hash64(const char *data, size_t n, uint64_t seed) { return XXPH3_64bits_withSeed(data, n, seed); }

} // namespace kv_cache_manager
