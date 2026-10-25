#include <cerrno>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

#include "kv_cache_manager/common/lock_free_thread_pool.h"
#include "kv_cache_manager/common/unittest.h"

namespace {

// This executable alone intercepts thread creation. Only the thread opening
// the pool injects failures; workers and unrelated background threads do not.
thread_local int fail_after = -1;
thread_local int creation_attempts = 0;
thread_local int joined_threads = 0;

class ScopedThreadCreationFailure {
public:
    explicit ScopedThreadCreationFailure(int successful_threads) {
        fail_after = successful_threads;
        creation_attempts = 0;
        joined_threads = 0;
    }
    ~ScopedThreadCreationFailure() { fail_after = -1; }
};

} // namespace

extern "C" int
pthread_create(pthread_t *thread, const pthread_attr_t *attributes, void *(*routine)(void *), void *argument) noexcept {
    using Create = int (*)(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
    static const auto real_create = reinterpret_cast<Create>(dlsym(RTLD_NEXT, "pthread_create"));
    if (fail_after >= 0 && creation_attempts++ >= fail_after) {
        return EAGAIN;
    }
    return real_create(thread, attributes, routine, argument);
}

extern "C" int pthread_join(pthread_t thread, void **result) {
    using Join = int (*)(pthread_t, void **);
    static const auto real_join = reinterpret_cast<Join>(dlsym(RTLD_NEXT, "pthread_join"));
    const int error = real_join(thread, result);
    if (fail_after >= 0 && error == 0) {
        ++joined_threads;
    }
    return error;
}

namespace {

int CheckFailedStartup(int successful_threads) {
    using kv_cache_manager::LockFreeThreadPool;
    {
        ScopedThreadCreationFailure failure(successful_threads);
        LockFreeThreadPool pool(4, 8, "start_failure");
        if (pool.start() || pool.start()) {
            return 1;
        }
        if (creation_attempts != successful_threads + 1 || joined_threads != successful_threads) {
            return 2;
        }
        if (pool.enqueue([] {})) {
            return 3;
        }
        pool.stop();
    }
    // A failed pool must neither leak workers nor prevent another pool starting.
    LockFreeThreadPool healthy(2, 8, "after_failure");
    if (!healthy.start() || healthy.async([] { return 42; }).get() != 42) {
        return 4;
    }
    healthy.stop();
    return 0;
}

[[noreturn]] void CheckInChild(int successful_threads) {
    // A regression must fail promptly rather than hang the entire test runner.
    alarm(5);
    const int result = CheckFailedStartup(successful_threads);
    alarm(0);
    _exit(result);
}

} // namespace

class LockFreeThreadPoolStartFailureDeathTest : public TESTBASE {
public:
    void SetUp() override { ::testing::GTEST_FLAG(death_test_style) = "threadsafe"; }
};

TEST_F(LockFreeThreadPoolStartFailureDeathTest, FirstThreadFailureReturnsFalse) {
    ASSERT_EXIT(CheckInChild(0), ::testing::ExitedWithCode(0), "");
}

TEST_F(LockFreeThreadPoolStartFailureDeathTest, PartialStartupJoinsCreatedThread) {
    ASSERT_EXIT(CheckInChild(1), ::testing::ExitedWithCode(0), "");
}

TEST_F(LockFreeThreadPoolStartFailureDeathTest, LastThreadFailureJoinsAllCreatedThreads) {
    ASSERT_EXIT(CheckInChild(3), ::testing::ExitedWithCode(0), "");
}
