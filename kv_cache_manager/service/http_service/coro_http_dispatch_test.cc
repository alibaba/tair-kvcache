#include <chrono>
#include <future>
#include <memory>
#include <shared_mutex>
#include <stdexcept>
#include <thread>

#include "google/protobuf/wrappers.pb.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/metrics/metrics_lifecycle.h"
#include "kv_cache_manager/service/http_service/coro_http_service.h"
#include "kv_cache_manager/service/http_service/meta_service_http.h"

namespace kv_cache_manager {
namespace {

using namespace std::chrono_literals;
using StringMessage = google::protobuf::StringValue;

class Gate {
public:
    Gate() : ready_(open_.get_future().share()) {}
    void Open() { open_.set_value(); }
    void Wait() const {
        if (ready_.wait_for(5s) != std::future_status::ready) {
            throw std::runtime_error("test gate timed out");
        }
    }

private:
    std::promise<void> open_;
    std::shared_future<void> ready_;
};

struct HttpExchange {
    explicit HttpExchange(std::string json) : request(parser, nullptr), response(nullptr), body(std::move(json)) {
        request.set_body(body);
    }
    cinatra::http_parser parser;
    coro_http::coro_http_request request;
    coro_http::coro_http_response response;
    std::string body;
};

struct Reply {
    coro_http::status_type status;
    std::string body;
    std::thread::id thread;
};

class IoLoop {
public:
    IoLoop() : executor_(io_.get_executor()), work_(asio::make_work_guard(io_)), thread_([this] { io_.run(); }) {}
    ~IoLoop() {
        work_.reset();
        thread_.join();
    }

    std::future<Reply> Start(CoroHttpService::HandlerType handler, std::string body) {
        auto exchange = std::make_shared<HttpExchange>(std::move(body));
        auto done = std::make_shared<std::promise<Reply>>();
        auto future = done->get_future();
        // The registered handler owns its coroutine-lambda capture state until
        // completion, just as the router does in the actual HTTP server.
        auto owner = std::make_shared<CoroHttpService::HandlerType>(std::move(handler));
        (*owner)(exchange->request, exchange->response)
            .via(&executor_)
            .start([exchange, done, owner](async_simple::Try<void> result) {
                if (result.hasError()) {
                    done->set_exception(result.getException());
                } else {
                    done->set_value({exchange->response.status(),
                                     std::string(exchange->response.content()),
                                     std::this_thread::get_id()});
                }
            });
        return future;
    }

    void Post(std::function<void()> callback) { asio::post(io_, std::move(callback)); }
    std::thread::id thread_id() const { return thread_.get_id(); }

private:
    asio::io_context io_{1};
    coro_io::ExecutorWrapper<> executor_;
    asio::executor_work_guard<asio::io_context::executor_type> work_;
    std::thread thread_;
};

class DispatchService : public CoroHttpService {
public:
    using Work = std::function<CachedJsonResponse(StringMessage *, StringMessage *)>;

    DispatchService(ServiceRequestExecutor &executor, Work work) : executor_(executor), work_(std::move(work)) {}
    void Init() override {}
    void RegisterHandler() override {}

    HandlerType Handler(bool arena = false) {
        return arena ? GetArenaHandler<DispatchService, StringMessage, StringMessage>(&DispatchService::Call)
                     : GetHandler<DispatchService, StringMessage, StringMessage>(&DispatchService::Call);
    }
    HandlerType SynchronousHandler() {
        return GetHandler<DispatchService, StringMessage, StringMessage>(&DispatchService::SynchronousCall);
    }

    async_simple::coro::Lazy<HttpDispatchResult>
    Call(coro_http::coro_http_connection *, StringMessage *request, StringMessage *response) {
        return DispatchHttpRequest(
            executor_,
            [work = work_, request, response] { return work(request, response); },
            [response](const char *message) { response->set_value(message); });
    }
    void SynchronousCall(coro_http::coro_http_connection *, StringMessage *request, StringMessage *response) {
        response->set_value(request->value());
    }

private:
    ServiceRequestExecutor &executor_;
    Work work_;
};

TEST(CoroHttpDispatchTest, SlowBusinessAllowsAnotherHandlerOnSameIoAndReturnsToIo) {
    ServiceRequestExecutor executor(2);
    IoLoop io;
    auto gate = std::make_shared<Gate>();
    std::promise<std::thread::id> started;
    DispatchService service(executor, [&](StringMessage *request, StringMessage *response) {
        if (request->value() == "slow") {
            started.set_value(std::this_thread::get_id());
            gate->Wait();
        }
        response->set_value(request->value());
        return CoroHttpService::CachedJsonResponse{};
    });
    auto started_future = started.get_future();
    auto slow = io.Start(service.Handler(), R"("slow")");
    ASSERT_EQ(started_future.wait_for(5s), std::future_status::ready);
    EXPECT_NE(started_future.get(), io.thread_id());
    auto fast = io.Start(service.Handler(), R"("fast")");
    ASSERT_EQ(fast.wait_for(5s), std::future_status::ready);
    auto fast_reply = fast.get();
    EXPECT_EQ(fast_reply.body, R"("fast")");
    EXPECT_EQ(fast_reply.thread, io.thread_id());
    EXPECT_EQ(slow.wait_for(0ms), std::future_status::timeout);
    gate->Open();
    ASSERT_EQ(slow.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(slow.get().thread, io.thread_id());
    executor.CloseAdmission(ServiceRequestExecutor::ShutdownMode::kDrain);
    executor.Drain();
}

TEST(CoroHttpDispatchTest, ArenaAndCachedJsonSurviveDispatch) {
    ServiceRequestExecutor executor(1);
    IoLoop io;
    DispatchService service(executor, [](StringMessage *request, StringMessage *response) {
        EXPECT_NE(request->GetArena(), nullptr);
        EXPECT_EQ(request->GetArena(), response->GetArena());
        EXPECT_EQ(request->value(), "arena");
        response->set_value("unused fallback");
        return CoroHttpService::CachedJsonResponse{R"("cached")"};
    });
    auto future = io.Start(service.Handler(true), R"("arena")");
    ASSERT_EQ(future.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(future.get().body, R"("cached")");
    executor.CloseAdmission(ServiceRequestExecutor::ShutdownMode::kDrain);
    executor.Drain();
}

TEST(CoroHttpDispatchTest, DrainWaitsForIoContinuationIncludingWorkerException) {
    for (bool fail : {false, true}) {
        ServiceRequestExecutor executor(1);
        IoLoop io;
        auto business_gate = std::make_shared<Gate>();
        auto io_gate = std::make_shared<Gate>();
        std::promise<void> entered;
        DispatchService service(executor, [&](StringMessage *, StringMessage *response) {
            entered.set_value();
            business_gate->Wait();
            if (fail) {
                throw std::runtime_error("business failed");
            }
            response->set_value("done");
            return CoroHttpService::CachedJsonResponse{};
        });
        auto entered_future = entered.get_future();
        auto response = io.Start(service.Handler(), R"("request")");
        ASSERT_EQ(entered_future.wait_for(5s), std::future_status::ready);
        std::promise<void> io_blocked;
        auto io_blocked_future = io_blocked.get_future();
        io.Post([&io_blocked, io_gate] {
            io_blocked.set_value();
            io_gate->Wait();
        });
        ASSERT_EQ(io_blocked_future.wait_for(5s), std::future_status::ready);
        std::promise<void> worker_finished;
        auto worker_finished_future = worker_finished.get_future();
        ServiceRequestExecutor::Request marker;
        marker.run = [&worker_finished](ServiceRequestExecutor::CompletionToken) { worker_finished.set_value(); };
        marker.on_discard = [](ServiceRequestExecutor::CompletionToken) {};
        ASSERT_EQ(executor.TrySubmit(std::move(marker)), ServiceRequestExecutor::SubmitResult::kAccepted);
        business_gate->Open();
        ASSERT_EQ(worker_finished_future.wait_for(5s), std::future_status::ready);
        executor.CloseAdmission(ServiceRequestExecutor::ShutdownMode::kDrain);
        auto drain = std::async(std::launch::async, [&] { executor.Drain(); });
        EXPECT_EQ(drain.wait_for(50ms), std::future_status::timeout);
        io_gate->Open();
        ASSERT_EQ(response.wait_for(5s), std::future_status::ready);
        if (fail) {
            EXPECT_THROW(response.get(), std::runtime_error);
        } else {
            EXPECT_EQ(response.get().thread, io.thread_id());
        }
        EXPECT_EQ(drain.wait_for(5s), std::future_status::ready);
    }
}

TEST(CoroHttpDispatchTest, FullDiscardAndClosedCompleteWithoutExecutingBusiness) {
    ServiceRequestExecutor executor(1, 1);
    IoLoop io;
    auto gate = std::make_shared<Gate>();
    std::promise<void> entered;
    auto entered_future = entered.get_future();
    ServiceRequestExecutor::Request blocker;
    blocker.run = [&entered, gate](ServiceRequestExecutor::CompletionToken) {
        entered.set_value();
        gate->Wait();
    };
    blocker.on_discard = [](ServiceRequestExecutor::CompletionToken) {};
    ASSERT_EQ(executor.TrySubmit(std::move(blocker)), ServiceRequestExecutor::SubmitResult::kAccepted);
    ASSERT_EQ(entered_future.wait_for(5s), std::future_status::ready);
    DispatchService service(executor, [](StringMessage *, StringMessage *) {
        ADD_FAILURE() << "rejected or discarded task must not execute";
        return CoroHttpService::CachedJsonResponse{};
    });
    auto pending = io.Start(service.Handler(), R"("pending")");
    // A later operation on this single IO thread runs after the first handler
    // has submitted and suspended, so the one waiting queue slot is occupied.
    std::promise<void> submitted;
    auto submitted_future = submitted.get_future();
    io.Post([&submitted] { submitted.set_value(); });
    ASSERT_EQ(submitted_future.wait_for(5s), std::future_status::ready);
    auto full = io.Start(service.Handler(), R"("full")");
    ASSERT_EQ(full.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(full.get().body, R"("service request queue is full")");
    executor.CloseAdmission(ServiceRequestExecutor::ShutdownMode::kDiscardPending);
    ASSERT_EQ(pending.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(pending.get().body, R"("service is stopping")");
    auto closed = io.Start(service.Handler(), R"("closed")");
    ASSERT_EQ(closed.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(closed.get().body, R"("service is stopping")");
    gate->Open();
    executor.Drain();
}

class MetaDispatchService : public MetaServiceHttp {
public:
    using MetaServiceHttp::MetaServiceHttp;
    HandlerType MissingInstanceHandler(bool arena = false) {
        if (arena) {
            return GetArenaHandler<MetaServiceHttp,
                                   proto::meta::GetCacheMetaRequest,
                                   proto::meta::GetCacheMetaResponse>(&MetaServiceHttp::GetCacheMeta);
        }
        return GetHandler<MetaServiceHttp, proto::meta::GetCacheMetaRequest, proto::meta::GetCacheMetaResponse>(
            &MetaServiceHttp::GetCacheMeta);
    }
};

TEST(CoroHttpDispatchTest, OverloadedMetaRejectsBeforeParsingJson) {
    auto executor = std::make_shared<ServiceRequestExecutor>(1, 1);
    auto gate = std::make_shared<Gate>();
    std::promise<void> entered;
    auto ready = entered.get_future();
    executor->TrySubmit({[&entered, gate](ServiceRequestExecutor::CompletionToken) {
                             entered.set_value();
                             gate->Wait();
                         },
                         [](ServiceRequestExecutor::CompletionToken) {}});
    ASSERT_EQ(std::future_status::ready, ready.wait_for(5s));
    executor->TrySubmit(
        {[](ServiceRequestExecutor::CompletionToken) {}, [](ServiceRequestExecutor::CompletionToken) {}});
    auto metrics = std::make_shared<MetricsRegistry>();
    MetaDispatchService service(metrics, nullptr, std::make_shared<RegistryManager>("", metrics), executor);
    service.Init();
    IoLoop io;
    for (bool arena : {false, true}) {
        const auto reply = io.Start(service.MissingInstanceHandler(arena), "not json").get();
        EXPECT_EQ(coro_http::status_type::ok, reply.status);
        EXPECT_NE(std::string::npos, reply.body.find("SERVICE_NOT_READY"));
    }
    EXPECT_EQ(2u, executor->GetSnapshot().rejected);
    EXPECT_EQ(1u, executor->GetSnapshot().queued);
    executor->CloseAdmission();
    const auto closed = io.Start(service.MissingInstanceHandler(), "not json").get();
    EXPECT_NE(std::string::npos, closed.body.find("SERVICE_NOT_READY"));
    EXPECT_NE(std::string::npos, closed.body.find("stopping"));
    gate->Open();
    executor->Drain();
}

TEST(CoroHttpDispatchTest, MetaCollectorLookupWaitsOnWorkerWithoutBlockingSingleIoThread) {
    auto executor = std::make_shared<ServiceRequestExecutor>(1);
    auto metrics = std::make_shared<MetricsRegistry>();
    auto lifecycle = std::make_shared<MetricsLifecycle>();
    auto registry = std::make_shared<RegistryManager>("", metrics);
    // The unknown-instance preparation returns before dereferencing impl.
    MetaDispatchService service(metrics, nullptr, registry, executor, lifecycle);
    service.Init();
    IoLoop io;
    std::unique_lock<std::shared_mutex> hold_lifecycle(lifecycle->mut_);
    auto request = io.Start(service.MissingInstanceHandler(), R"({"trace_id":"http-cold","instance_id":"missing"})");
    std::promise<void> io_progress;
    auto progress = io_progress.get_future();
    io.Post([&] { io_progress.set_value(); });
    // Always unlock before fatal assertions/destruction, so a regression
    // blocks this check for at most two seconds and cannot hang the test.
    const auto io_status = progress.wait_for(2s);
    const auto request_status = request.wait_for(0ms);
    if (io_status == std::future_status::ready) {
        const auto deadline = std::chrono::steady_clock::now() + 2s;
        while (executor->GetSnapshot().running == 0 && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(1ms);
        }
    }
    const auto snapshot = executor->GetSnapshot();
    hold_lifecycle.unlock();
    EXPECT_EQ(std::future_status::ready, io_status);
    EXPECT_EQ(std::future_status::timeout, request_status);
    EXPECT_EQ(1u, snapshot.running);
    ASSERT_EQ(std::future_status::ready, request.wait_for(5s));
    const auto reply = request.get();
    EXPECT_EQ(coro_http::status_type::ok, reply.status);
    EXPECT_EQ(io.thread_id(), reply.thread);
    EXPECT_NE(std::string::npos, reply.body.find("INSTANCE_NOT_EXIST"));
    executor->CloseAdmission(ServiceRequestExecutor::ShutdownMode::kDrain);
    executor->Drain();
}

TEST(CoroHttpDispatchTest, InvalidJsonAndSynchronousHandlersKeepOriginalBehavior) {
    ServiceRequestExecutor executor(1);
    IoLoop io;
    DispatchService service(executor, [](StringMessage *, StringMessage *) {
        ADD_FAILURE() << "invalid request must not reach business";
        return CoroHttpService::CachedJsonResponse{};
    });
    auto invalid = io.Start(service.Handler(), "not json");
    ASSERT_EQ(invalid.wait_for(5s), std::future_status::ready);
    auto reply = invalid.get();
    EXPECT_EQ(reply.status, coro_http::status_type::bad_request);
    EXPECT_EQ(reply.body, "{}");
    executor.CloseAdmission(ServiceRequestExecutor::ShutdownMode::kDiscardPending);
    auto synchronous = io.Start(service.SynchronousHandler(), R"("admin still works")");
    ASSERT_EQ(synchronous.wait_for(5s), std::future_status::ready);
    EXPECT_EQ(synchronous.get().body, R"("admin still works")");
    executor.Drain();
}

} // namespace
} // namespace kv_cache_manager
