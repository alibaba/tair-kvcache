#pragma once

namespace kv_cache_manager {

namespace proto::optimizer {
class TraceQueryRequest;
}

// Where the optimizer stream publisher hands finished replay events to the
// subscription queue consumed by the gRPC service.
//
// Send() is fire-and-forget by contract. The return value only says whether
// the event entered a subscriber queue, never whether the peer processed it.
class EventSink {
public:
    virtual ~EventSink() = default;

    // Never blocks and never applies back pressure: a sink that cannot ship
    // right now must drop and return false. Events are analysis samples, and
    // losing samples is always preferable to stalling the caller.
    virtual bool Send(const proto::optimizer::TraceQueryRequest &event) = 0;

    // Releases queue resources. Must tolerate being called more than once
    // and without a preceding Send().
    virtual void Stop() = 0;
};

} // namespace kv_cache_manager
