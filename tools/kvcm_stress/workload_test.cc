#include <map>

#include "kv_cache_manager/common/unittest.h"
#include "tools/kvcm_stress/workload.h"

using namespace kv_cache_manager;
using namespace kv_cache_manager::stress;
namespace {
void Check(bool ok) {
    if (!ok)
        throw std::runtime_error("test check failed");
}

class MemoryTransfer : public TransferClient {
public:
    std::map<std::string, std::vector<char>> data;
    bool fail_put = false, change_uri = false, corrupt = false;
    size_t reads = 0;
    ClientErrorCode Init(const std::string &, const InitParams &) override { return ER_OK; }
    std::pair<ClientErrorCode, UriStrVec>
    SaveKvCaches(const UriStrVec &uris, const BlockBuffers &buffers, std::shared_ptr<TransferTraceInfo>) override {
        if (fail_put)
            return {ER_SDKWRITE_ERROR, {}};
        Check(uris.size() == buffers.size());
        for (size_t i = 0; i < uris.size(); ++i) {
            const auto &iov = buffers[i].iovs.at(0);
            Check(iov.type == MemoryType::CPU);
            const auto *ptr = static_cast<const char *>(iov.base);
            data[uris[i]] = std::vector<char>(ptr, ptr + iov.size);
        }
        auto actual = uris;
        if (change_uri)
            actual.at(0) += "changed";
        return {ER_OK, actual};
    }
    ClientErrorCode
    LoadKvCaches(const UriStrVec &uris, const BlockBuffers &buffers, std::shared_ptr<TransferTraceInfo>) override {
        ++reads;
        for (size_t i = 0; i < uris.size(); ++i) {
            const auto &payload = data.at(uris[i]);
            Check(payload.size() == buffers[i].iovs.at(0).size);
            std::memcpy(buffers[i].iovs[0].base, payload.data(), payload.size());
            if (corrupt)
                static_cast<char *>(buffers[i].iovs[0].base)[0] ^= 1;
        }
        return ER_OK;
    }
};
class MemoryMeta : public MetaClient {
public:
    std::map<int64_t, Location> published;
    std::vector<int64_t> pending;
    Locations pending_locations;
    BlockMask mask = BlockMaskOffset{0};
    bool fail_start = false, fail_finish = false, malformed = false, short_response = false, miss = false;
    size_t finishes = 0;
    BlockMaskVector last_success;
    ClientErrorCode Init(const std::string &, const InitParams &) override { return ER_OK; }
    void Shutdown() override {}
    std::pair<ClientErrorCode, WriteLocation> StartWrite(const std::string &,
                                                         const std::vector<int64_t> &keys,
                                                         const std::vector<int64_t> &,
                                                         const std::vector<std::string> &,
                                                         int64_t,
                                                         int32_t) override {
        if (fail_start)
            return {ER_CONNECT_FAIL, {}};
        pending.clear();
        pending_locations.clear();
        for (size_t i = 0; i < keys.size(); ++i) {
            bool skip = std::holds_alternative<BlockMaskOffset>(mask) ? i < std::get<BlockMaskOffset>(mask)
                                                                      : std::get<BlockMaskVector>(mask).at(i);
            if (!skip) {
                pending.push_back(keys[i]);
                pending_locations.push_back({{"tp0", "pace://" + std::to_string(keys[i])}});
            }
        }
        auto locations = pending_locations;
        if (malformed)
            locations.at(0).clear();
        if (short_response)
            locations.pop_back();
        return {ER_OK, {"session", mask, locations}};
    }
    ClientErrorCode FinishWrite(const std::string &,
                                const std::string &session,
                                const BlockMask &success,
                                const Locations &locations) override {
        Check(session == "session");
        ++finishes;
        if (const auto *values = std::get_if<BlockMaskVector>(&success)) {
            last_success = *values;
            Check(locations.size() == pending.size());
        } else {
            Check(std::get<BlockMaskOffset>(success) == 0);
            last_success.assign(pending.size(), false);
        }
        Check(last_success.size() == pending.size());
        if (fail_finish)
            return ER_CONNECT_FAIL;
        for (size_t i = 0; i < pending.size(); ++i)
            if (last_success[i])
                published[pending[i]] = locations[i];
        return ER_OK;
    }
    std::pair<ClientErrorCode, Locations> MatchLocation(const std::string &,
                                                        QueryType type,
                                                        const std::vector<int64_t> &keys,
                                                        const std::vector<int64_t> &,
                                                        const BlockMask &,
                                                        int32_t,
                                                        const std::vector<std::string> &specs) override {
        Check(type == QueryType::QT_BATCH_GET && specs == std::vector<std::string>{"tp0"});
        Locations out;
        for (auto key : keys) {
            Check(published.count(key));
            out.push_back(miss ? Location{} : published.at(key));
        }
        return {ER_OK, out};
    }
    std::pair<ClientErrorCode, Metas> MatchMeta(const std::string &,
                                                const std::vector<int64_t> &,
                                                const std::vector<int64_t> &,
                                                const BlockMask &,
                                                int32_t) override {
        return {ER_OK, {}};
    }
    std::pair<ClientErrorCode, int64_t> MatchLocationLen(
        const std::string &, QueryType, const std::vector<int64_t> &, const std::vector<int64_t> &, int32_t) override {
        return {ER_OK, 0};
    }
    ClientErrorCode RemoveCache(const std::string &,
                                const std::vector<int64_t> &,
                                const std::vector<int64_t> &,
                                const BlockMask &) override {
        return ER_OK;
    }
    const std::string &GetStorageConfig() const override {
        static const std::string empty;
        return empty;
    }
    std::pair<ClientErrorCode, BackendLocations> GetCacheLocationsByBackend(const std::string &,
                                                                            const std::vector<int64_t> &,
                                                                            const std::vector<int64_t> &,
                                                                            const BlockMask &,
                                                                            const std::vector<std::string> &,
                                                                            StorageType,
                                                                            BackendSelectStrategy) override {
        return {ER_OK, {}};
    }
    std::pair<ClientErrorCode, HostCacheState> GetHostCacheState(const std::string &,
                                                                 QueryType,
                                                                 const std::vector<int64_t> &,
                                                                 const std::vector<std::string> &,
                                                                 int32_t,
                                                                 const std::vector<StorageType> &) override {
        return {ER_OK, {}};
    }
};
struct Fixture {
    Options o;
    Stats stats;
    MemoryMeta meta;
    MemoryTransfer transfer;
    std::atomic<uint64_t> next{1000};
    Fixture() {
        o.batch = 4;
        o.bytes = 65;
        o.pool = 8;
        o.spec = "tp0";
        o.verify = true;
        o.query_percent = 100;
    }
    Worker Make() { return Worker(meta, transfer, o, stats, next, 42); }
};
void RoundTrip() {
    Fixture f;
    auto w = f.Make();
    for (int i = 0; i < 10; ++i)
        Check(w.Write()); // Wrap the bounded key ring.
    Check(f.stats.written == 40);
    for (int i = 0; i < 10; ++i)
        Check(w.Step());
    Check(f.stats.read == 40 && f.stats.corrupt == 0 && f.stats.phases[QUERY].calls == 10);
    f.transfer.corrupt = true;
    Check(!w.Step() && f.stats.corrupt > 0);
}
void Masks() {
    for (BlockMask mask : {BlockMask{BlockMaskOffset{2}},
                           BlockMask{BlockMaskVector{true, false, true, false}},
                           BlockMask{BlockMaskOffset{4}}}) {
        Fixture f;
        f.meta.mask = mask;
        auto w = f.Make();
        Check(w.Write());
        size_t count = std::holds_alternative<BlockMaskOffset>(mask) && std::get<BlockMaskOffset>(mask) == 4 ? 0 : 2;
        Check(f.stats.written == count && f.stats.skipped == 4 - count);
        Check(f.meta.last_success == BlockMaskVector(count, true));
        if (count)
            Check(w.Step() && f.stats.read == 4);
        else
            Check(f.stats.phases[PUT].calls == 0 && !w.HasKeys() && f.meta.finishes == 1);
    }
}
void Failures() {
    for (int failure = 0; failure < 6; ++failure) {
        Fixture f;
        f.meta.fail_start = failure == 0;
        f.transfer.fail_put = failure == 1;
        f.meta.fail_finish = failure == 2;
        f.transfer.change_uri = failure == 3;
        f.meta.malformed = failure == 4;
        f.meta.short_response = failure == 5;
        auto w = f.Make();
        Check(!w.Write());
        Check(!w.HasKeys() && f.stats.written == 0 && f.meta.published.empty());
        Check(f.meta.finishes == (failure == 0 ? 0 : 1));
        if (failure == 1 || failure == 3 || failure == 4 || failure == 5)
            Check(f.meta.last_success == BlockMaskVector(4, false));
    }
}
void QueryOnlyAndMiss() {
    Fixture f;
    auto w = f.Make();
    Check(w.Write());
    f.o.read_percent = 0;
    Check(w.Step());
    Check(f.stats.hits == 4 && f.transfer.reads == 0);
    f.o.read_percent = 100;
    f.meta.miss = true;
    Check(w.Step());
    Check(f.stats.queried == 8 && f.stats.hits == 4 && f.transfer.reads == 0);
}
} // namespace
class WorkloadTest : public TESTBASE {};
TEST_F(WorkloadTest, RoundTripAndCorruption) { RoundTrip(); }
TEST_F(WorkloadTest, CompactMasks) { Masks(); }
TEST_F(WorkloadTest, FailureCleanup) { Failures(); }
TEST_F(WorkloadTest, MetadataOnlyAndMiss) { QueryOnlyAndMiss(); }
