//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// See port_example.h for documentation for the following types/functions.

#pragma once

#include <thread>

// #include "rocksdb/port_defs.h"

// size_t printf formatting named in the manner of C99 standard formatting
// strings such as PRIu64
// in fact, we could use that one
#define ROCKSDB_PRIszt "zu"

#define __declspec(S)

#undef PLATFORM_IS_LITTLE_ENDIAN
#if defined(OS_MACOSX)
#include <machine/endian.h>
#if defined(__DARWIN_LITTLE_ENDIAN) && defined(__DARWIN_BYTE_ORDER)
#define PLATFORM_IS_LITTLE_ENDIAN (__DARWIN_BYTE_ORDER == __DARWIN_LITTLE_ENDIAN)
#endif
#elif defined(OS_SOLARIS)
#include <sys/isa_defs.h>
#ifdef _LITTLE_ENDIAN
#define PLATFORM_IS_LITTLE_ENDIAN true
#else
#define PLATFORM_IS_LITTLE_ENDIAN false
#endif
#include <alloca.h>
#elif defined(OS_AIX)
#include <arpa/nameser_compat.h>
#include <sys/types.h>
#define PLATFORM_IS_LITTLE_ENDIAN (BYTE_ORDER == LITTLE_ENDIAN)
#include <alloca.h>
#elif defined(OS_FREEBSD) || defined(OS_OPENBSD) || defined(OS_NETBSD) || defined(OS_DRAGONFLYBSD) ||                  \
    defined(OS_ANDROID)
#include <sys/endian.h>
#include <sys/types.h>
#define PLATFORM_IS_LITTLE_ENDIAN (_BYTE_ORDER == _LITTLE_ENDIAN)
#else
#include <endian.h>
#endif
#include <limits>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <string>

#ifndef PLATFORM_IS_LITTLE_ENDIAN
#define PLATFORM_IS_LITTLE_ENDIAN (__BYTE_ORDER == __LITTLE_ENDIAN)
#endif

#if defined(OS_MACOSX) || defined(OS_SOLARIS) || defined(OS_FREEBSD) || defined(OS_NETBSD) || defined(OS_OPENBSD) ||   \
    defined(OS_DRAGONFLYBSD) || defined(OS_ANDROID) || defined(CYGWIN) || defined(OS_AIX)
// Use fread/fwrite/fflush on platforms without _unlocked variants
#define fread_unlocked fread
#define fwrite_unlocked fwrite
#define fflush_unlocked fflush
#endif

#if defined(OS_MACOSX) || defined(OS_FREEBSD) || defined(OS_OPENBSD) || defined(OS_DRAGONFLYBSD)
// Use fsync() on platforms without fdatasync()
#define fdatasync fsync
#endif

#if defined(OS_ANDROID) && __ANDROID_API__ < 9
// fdatasync() was only introduced in API level 9 on Android. Use fsync()
// when targeting older platforms.
#define fdatasync fsync
#endif

namespace kv_cache_manager {

extern const bool kDefaultToAdaptiveMutex;

namespace port {
constexpr bool kLittleEndian = PLATFORM_IS_LITTLE_ENDIAN;
#undef PLATFORM_IS_LITTLE_ENDIAN

using Thread = std::thread;

static inline void AsmVolatilePause() {
#if defined(__i386__) || defined(__x86_64__)
    asm volatile("pause");
#elif defined(__aarch64__)
    asm volatile("isb");
#elif defined(__powerpc64__)
    asm volatile("or 27,27,27");
#elif defined(__loongarch64)
    asm volatile("dbar 0");
#endif
    // it's okay for other platforms to be no-ops
}

using OnceType = pthread_once_t;
#define LEVELDB_ONCE_INIT PTHREAD_ONCE_INIT

#ifndef CACHE_LINE_SIZE
// To test behavior with non-native cache line size, e.g. for
// Bloom filters, set TEST_CACHE_LINE_SIZE to the desired test size.
// This disables ALIGN_AS to keep it from failing compilation.
#ifdef TEST_CACHE_LINE_SIZE
#define CACHE_LINE_SIZE TEST_CACHE_LINE_SIZE
#define ALIGN_AS(n) /*empty*/
#else
#if defined(__s390__)
#if defined(__GNUC__) && __GNUC__ < 7
#define CACHE_LINE_SIZE 64U
#else
#define CACHE_LINE_SIZE 256U
#endif
#elif defined(__powerpc__) || defined(__aarch64__)
#define CACHE_LINE_SIZE 128U
#else
#define CACHE_LINE_SIZE 64U
#endif
#define ALIGN_AS(n) alignas(n)
#endif
#endif

static_assert((CACHE_LINE_SIZE & (CACHE_LINE_SIZE - 1)) == 0, "Cache line size must be a power of 2 number of bytes");

inline void *cacheline_aligned_alloc(size_t size) {
#if __GNUC__ < 5 && defined(__SANITIZE_ADDRESS__)
    return malloc(size);
#elif (_POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600 || defined(__APPLE__))
    void *m;
    errno = posix_memalign(&m, CACHE_LINE_SIZE, size);
    return errno ? nullptr : m;
#else
    return malloc(size);
#endif
}

inline void cacheline_aligned_free(void *memblock) { free(memblock); }

#if defined(__aarch64__)
//  __builtin_prefetch(..., 1) turns into a prefetch into prfm pldl3keep. On
// arm64 we want this as close to the core as possible to turn it into a
// L1 prefetech unless locality == 0 in which case it will be turned into a
// non-temporal prefetch
#define PREFETCH(addr, rw, locality) __builtin_prefetch(addr, rw, locality >= 1 ? 3 : locality)
#else
#define PREFETCH(addr, rw, locality) __builtin_prefetch(addr, rw, locality)
#endif

} // namespace port
} // namespace kv_cache_manager
