// C1a / C1b / C2 / C3 / C4 classifier tests, driven only through the same
// response shapes the real transport produces.
#include <gtest/gtest.h>

#include "tools/kvcm_swarm/clients/v6d/checks.h"

namespace kvcm_swarm {
namespace {

class ChecksTest : public ::testing::Test {
protected:
    void SetUp() override {
        expected_.SetReporterLive(Self());
        expected_.SetReporterLive(Peer());
    }

    static ReporterIdentity Self() { return ReporterIdentity{"inst", "10.0.0.1:1"}; }
    static ReporterIdentity Peer() { return ReporterIdentity{"inst", "10.0.0.2:1"}; }

    static HotLocationKey Key(int64_t block_key, const std::string &host) {
        HotLocationKey key;
        key.block_key = block_key;
        key.spec_name = "v6d_4096";
        key.reporter = ReporterIdentity{"inst", host};
        return key;
    }

    static LookupItem Item(int64_t block_key, bool masked) {
        LookupItem item;
        item.block_key = block_key;
        item.spec_name = "v6d_4096";
        item.object_key = "obj" + std::to_string(block_key);
        item.masked = masked;
        return item;
    }

    // Builds a response whose positional shape matches the request.
    static meta::GetCacheLocationsByBackendResponse
    MakeResponse(size_t keys, const std::map<size_t, std::pair<meta::StorageType, std::string>> &locations) {
        meta::GetCacheLocationsByBackendResponse response;
        for (size_t i = 0; i < keys; ++i) {
            auto *vector = response.add_key_locations();
            const auto it = locations.find(i);
            if (it == locations.end()) {
                continue;
            }
            auto *location = vector->add_locations();
            location->set_type(it->second.first);
            auto *spec = location->add_location_specs();
            spec->set_name("v6d_4096");
            spec->set_uri(it->second.second);
        }
        return response;
    }

    void ConfirmPeer(int64_t block_key) {
        expected_.HotPendingCreate(Key(block_key, "10.0.0.2:1"));
        expected_.HotConfirm(Key(block_key, "10.0.0.2:1"));
    }

    ExpectedLocations expected_;
    EvidenceSink evidence_;
    V6dChecks checks_{expected_, evidence_, "inst"};
};

TEST_F(ChecksTest, PrefixEligibilityLooksAtTheFirstUnmaskedKey) {
    ConfirmPeer(2);
    const std::vector<LookupItem> items = {Item(1, true), Item(2, false), Item(3, false)};
    const LookupExpectation expectation = checks_.BeforeLookup(LookupTier::kHot, items, FullSelector::kPrefix, Self());
    ASSERT_EQ(expectation.remote_eligible_indices.size(), 1u);
    EXPECT_EQ(expectation.remote_eligible_indices[0], 1u);
    EXPECT_EQ(expectation.unmasked_keys, 2u);

    // With the candidate on a later key only, PREFIX is not eligible.
    ExpectedLocations other;
    other.SetReporterLive(Self());
    other.SetReporterLive(Peer());
    V6dChecks other_checks(other, evidence_, "inst");
    other.HotPendingCreate(Key(3, "10.0.0.2:1"));
    other.HotConfirm(Key(3, "10.0.0.2:1"));
    const LookupExpectation not_eligible =
        other_checks.BeforeLookup(LookupTier::kHot, items, FullSelector::kPrefix, Self());
    EXPECT_TRUE(not_eligible.remote_eligible_indices.empty());
    // COVERAGE considers any unmasked key.
    const LookupExpectation coverage =
        other_checks.BeforeLookup(LookupTier::kHot, items, FullSelector::kCoverage, Self());
    ASSERT_EQ(coverage.remote_eligible_indices.size(), 1u);
    EXPECT_EQ(coverage.remote_eligible_indices[0], 2u);
}

TEST_F(ChecksTest, RequesterOwnPossibleLocationDisablesEligibility) {
    ConfirmPeer(2);
    expected_.HotPendingCreate(Key(2, "10.0.0.1:1"));
    const std::vector<LookupItem> items = {Item(2, false)};
    const LookupExpectation expectation =
        checks_.BeforeLookup(LookupTier::kHot, items, FullSelector::kCoverage, Self());
    EXPECT_TRUE(expectation.remote_eligible_indices.empty())
        << "a requester that may already hold the object cannot distinguish a legal empty answer";
}

TEST_F(ChecksTest, StableEligibleQueryWithAPeerHitPasses) {
    ConfirmPeer(2);
    const std::vector<LookupItem> items = {Item(2, false)};
    const LookupExpectation expectation =
        checks_.BeforeLookup(LookupTier::kHot, items, FullSelector::kCoverage, Self());
    ASSERT_FALSE(expectation.remote_eligible_indices.empty());
    const auto response = MakeResponse(1, {{0, {meta::ST_EVENT_REPORT_L2, "vineyard://10.0.0.2:1/mem?s_version=abc"}}});
    std::vector<std::string> hot;
    std::vector<std::string> cold;
    checks_.OnLookupResult(items, expectation, response, Self(), true, &hot, &cold);
    const V6dCheckCounters counters = checks_.Counters();
    EXPECT_EQ(counters.stable_eligible_queries, 1u);
    EXPECT_EQ(counters.stable_eligible_returned, 1u);
    EXPECT_EQ(counters.c1b_violations, 0u);
    EXPECT_EQ(counters.c1a_violations, 0u);
    EXPECT_EQ(hot[0], "10.0.0.2:1");
}

TEST_F(ChecksTest, PrefixHitOnAnotherKeyDoesNotSatisfyAvailability) {
    ConfirmPeer(1);
    ConfirmPeer(2);
    const std::vector<LookupItem> items = {Item(1, false), Item(2, false)};
    const LookupExpectation expectation = checks_.BeforeLookup(LookupTier::kHot, items, FullSelector::kPrefix, Self());
    ASSERT_EQ(expectation.remote_eligible_indices, std::vector<size_t>({0}));

    const auto response = MakeResponse(2, {{1, {meta::ST_EVENT_REPORT_L2, "vineyard://10.0.0.2:1/mem"}}});
    std::vector<std::string> hot;
    std::vector<std::string> cold;
    checks_.OnLookupResult(items, expectation, response, Self(), true, &hot, &cold);
    EXPECT_EQ(checks_.Counters().c1b_violations, 1u);
}

TEST_F(ChecksTest, CoverageMayUseAnyEligibleKey) {
    ConfirmPeer(1);
    ConfirmPeer(2);
    const std::vector<LookupItem> items = {Item(1, false), Item(2, false)};
    const LookupExpectation expectation =
        checks_.BeforeLookup(LookupTier::kHot, items, FullSelector::kCoverage, Self());
    ASSERT_EQ(expectation.remote_eligible_indices, std::vector<size_t>({0, 1}));

    const auto response = MakeResponse(2, {{1, {meta::ST_EVENT_REPORT_L2, "vineyard://10.0.0.2:1/mem"}}});
    std::vector<std::string> hot;
    std::vector<std::string> cold;
    checks_.OnLookupResult(items, expectation, response, Self(), true, &hot, &cold);
    EXPECT_EQ(checks_.Counters().stable_eligible_returned, 1u);
    EXPECT_EQ(checks_.Counters().c1b_violations, 0u);
}

TEST_F(ChecksTest, StableEligibleQueryWithNoRemoteLocationIsAC1bViolation) {
    ConfirmPeer(2);
    const std::vector<LookupItem> items = {Item(2, false)};
    const LookupExpectation expectation =
        checks_.BeforeLookup(LookupTier::kHot, items, FullSelector::kCoverage, Self());
    ASSERT_FALSE(expectation.remote_eligible_indices.empty());
    const auto response = MakeResponse(1, {});
    std::vector<std::string> hot;
    std::vector<std::string> cold;
    checks_.OnLookupResult(items, expectation, response, Self(), true, &hot, &cold);
    EXPECT_EQ(checks_.Counters().c1b_violations, 1u);
    EXPECT_EQ(evidence_.violations().Count("C1b_remote_availability"), 1u);
    const auto observations = checks_.Snapshot("v6d_deployment");
    const auto c1b = std::find_if(observations.begin(), observations.end(), [](const InvariantObservation &o) {
        return o.check_name == "C1b_remote_availability";
    });
    ASSERT_NE(c1b, observations.end());
    EXPECT_EQ(c1b->status, CheckStatus::kFail);
}

TEST_F(ChecksTest, CandidateChangeDuringTheQueryInvalidatesTheSample) {
    ConfirmPeer(2);
    const std::vector<LookupItem> items = {Item(2, false)};
    const LookupExpectation expectation =
        checks_.BeforeLookup(LookupTier::kHot, items, FullSelector::kCoverage, Self());
    ASSERT_FALSE(expectation.remote_eligible_indices.empty());
    // The peer deletes its location while the query is in flight.
    expected_.HotPendingDelete(Key(2, "10.0.0.2:1"));
    expected_.HotRemove(Key(2, "10.0.0.2:1"));
    const auto response = MakeResponse(1, {});
    std::vector<std::string> hot;
    std::vector<std::string> cold;
    checks_.OnLookupResult(items, expectation, response, Self(), true, &hot, &cold);
    const V6dCheckCounters counters = checks_.Counters();
    EXPECT_EQ(counters.invalidated_queries, 1u);
    EXPECT_EQ(counters.stable_eligible_queries, 0u);
    EXPECT_EQ(counters.c1b_violations, 0u);
}

TEST_F(ChecksTest, ZeroStableEligibleSamplesReportsNotRun) {
    const std::vector<LookupItem> items = {Item(5, false)};
    const LookupExpectation expectation =
        checks_.BeforeLookup(LookupTier::kHot, items, FullSelector::kCoverage, Self());
    EXPECT_TRUE(expectation.remote_eligible_indices.empty());
    const auto response = MakeResponse(1, {});
    std::vector<std::string> hot;
    std::vector<std::string> cold;
    checks_.OnLookupResult(items, expectation, response, Self(), true, &hot, &cold);
    const auto observations = checks_.Snapshot("v6d_deployment");
    for (const auto &observation : observations) {
        if (observation.check_name == "C1b_remote_availability") {
            EXPECT_EQ(observation.status, CheckStatus::kNotRun);
            EXPECT_EQ(observation.checked, 0u);
        }
    }
}

TEST_F(ChecksTest, UnknownReporterLocationIsAC1aViolation) {
    const std::vector<LookupItem> items = {Item(2, false)};
    const LookupExpectation expectation =
        checks_.BeforeLookup(LookupTier::kHot, items, FullSelector::kCoverage, Self());
    const auto response = MakeResponse(1, {{0, {meta::ST_EVENT_REPORT_L2, "vineyard://10.9.9.9:9/mem"}}});
    std::vector<std::string> hot;
    std::vector<std::string> cold;
    checks_.OnLookupResult(items, expectation, response, Self(), true, &hot, &cold);
    EXPECT_EQ(checks_.Counters().c1a_violations, 1u);
    EXPECT_TRUE(hot[0].empty());
}

TEST_F(ChecksTest, SelfLocationIsLegalButNeverCountedAsRemoteReuse) {
    expected_.HotPendingCreate(Key(4, "10.0.0.1:1"));
    expected_.HotConfirm(Key(4, "10.0.0.1:1"));
    const std::vector<LookupItem> items = {Item(4, false)};
    const LookupExpectation expectation =
        checks_.BeforeLookup(LookupTier::kHot, items, FullSelector::kCoverage, Self());
    const auto response = MakeResponse(1, {{0, {meta::ST_EVENT_REPORT_L2, "vineyard://10.0.0.1:1/mem"}}});
    std::vector<std::string> hot;
    std::vector<std::string> cold;
    checks_.OnLookupResult(items, expectation, response, Self(), true, &hot, &cold);
    EXPECT_EQ(checks_.Counters().c1a_violations, 0u);
    EXPECT_TRUE(hot[0].empty()) << "a self location is excluded from remote reuse";
}

TEST_F(ChecksTest, ColdOnlyQueryMustNotReturnAnEventReportLocation) {
    const std::vector<LookupItem> items = {Item(6, false)};
    const LookupExpectation expectation =
        checks_.BeforeLookup(LookupTier::kCold, items, FullSelector::kCoverage, Self());
    EXPECT_FALSE(expectation.hot_backend_requested);
    const auto response = MakeResponse(1, {{0, {meta::ST_EVENT_REPORT_L2, "vineyard://10.0.0.2:1/mem"}}});
    std::vector<std::string> hot;
    std::vector<std::string> cold;
    checks_.OnLookupResult(items, expectation, response, Self(), true, &hot, &cold);
    EXPECT_EQ(checks_.Counters().c1a_violations, 1u);
}

TEST_F(ChecksTest, ColdLocationIsReportedAsAColdHit) {
    const std::vector<LookupItem> items = {Item(6, false)};
    const LookupExpectation expectation =
        checks_.BeforeLookup(LookupTier::kCold, items, FullSelector::kCoverage, Self());
    const auto response = MakeResponse(1, {{0, {meta::ST_NFS, "file://nfs_01/x?blkid=0&size=4096"}}});
    std::vector<std::string> hot;
    std::vector<std::string> cold;
    checks_.OnLookupResult(items, expectation, response, Self(), true, &hot, &cold);
    EXPECT_EQ(checks_.Counters().c1a_violations, 0u);
    EXPECT_EQ(cold[0], "file://nfs_01/x?blkid=0&size=4096");
}

TEST_F(ChecksTest, ResponseLengthMismatchIsAC2Violation) {
    const std::vector<LookupItem> items = {Item(1, false), Item(2, false)};
    const LookupExpectation expectation = checks_.BeforeLookup(LookupTier::kHot, items, FullSelector::kPrefix, Self());
    const auto response = MakeResponse(1, {});
    std::vector<std::string> hot;
    std::vector<std::string> cold;
    checks_.OnLookupResult(items, expectation, response, Self(), true, &hot, &cold);
    EXPECT_EQ(checks_.Counters().c2_violations, 1u);
}

TEST_F(ChecksTest, WrongSpecOrMaskedPositionIsAC2Violation) {
    const std::vector<LookupItem> items = {Item(1, true), Item(2, false)};
    const LookupExpectation expectation =
        checks_.BeforeLookup(LookupTier::kHot, items, FullSelector::kCoverage, Self());
    meta::GetCacheLocationsByBackendResponse response;
    auto *masked_vector = response.add_key_locations();
    auto *masked_location = masked_vector->add_locations();
    masked_location->set_type(meta::ST_EVENT_REPORT_L2);
    auto *masked_spec = masked_location->add_location_specs();
    masked_spec->set_name("v6d_4096");
    masked_spec->set_uri("vineyard://10.0.0.2:1/mem");
    auto *vector = response.add_key_locations();
    auto *location = vector->add_locations();
    location->set_type(meta::ST_EVENT_REPORT_L2);
    auto *spec = location->add_location_specs();
    spec->set_name("v6d_9999");
    spec->set_uri("vineyard://10.0.0.2:1/mem");
    std::vector<std::string> hot;
    std::vector<std::string> cold;
    checks_.OnLookupResult(items, expectation, response, Self(), true, &hot, &cold);
    EXPECT_EQ(checks_.Counters().c2_violations, 2u);
}

TEST_F(ChecksTest, StartWriteAndReportEventShapeChecks) {
    checks_.CheckStartWriteShape(4, 2, 2, "ok");
    EXPECT_EQ(checks_.Counters().c2_violations, 0u);
    checks_.CheckStartWriteShape(4, 2, 3, "mismatch");
    EXPECT_EQ(checks_.Counters().c2_violations, 1u);
    checks_.CheckReportEventShape(3, 0, "empty is allowed");
    checks_.CheckReportEventShape(3, 3, "one per event");
    EXPECT_EQ(checks_.Counters().c2_violations, 1u);
    checks_.CheckReportEventShape(3, 2, "partial");
    EXPECT_EQ(checks_.Counters().c2_violations, 2u);
}

TEST_F(ChecksTest, C3RequiresBothWritableAndMaskedCoverage) {
    auto observations = checks_.Snapshot("v6d_deployment");
    auto find = [&observations](const std::string &name) {
        return *std::find_if(observations.begin(), observations.end(), [&name](const InvariantObservation &o) {
            return o.check_name == name;
        });
    };
    EXPECT_EQ(find("C3_capacity_pressure_eviction").status, CheckStatus::kNotRun);

    checks_.RecordCompletedEviction(2, 1, 2, 3, 3);
    observations = checks_.Snapshot("v6d_deployment");
    const InvariantObservation c3 = find("C3_capacity_pressure_eviction");
    EXPECT_EQ(c3.status, CheckStatus::kPass);
    EXPECT_EQ(c3.counters.at("writable_completed"), 2);
    EXPECT_EQ(c3.counters.at("masked_completed"), 1);
    EXPECT_EQ(c3.counters.at("cold_allocations_confirmed"), 2);

    checks_.RecordCompletedEviction(2, 1, 1, 3, 2);
    observations = checks_.Snapshot("v6d_deployment");
    EXPECT_EQ(find("C3_capacity_pressure_eviction").status, CheckStatus::kFail);
}

TEST_F(ChecksTest, C4IsAlwaysNonGatingInconclusive) {
    const auto observations = checks_.Snapshot("v6d_deployment");
    const auto c4 = std::find_if(observations.begin(), observations.end(), [](const InvariantObservation &o) {
        return o.check_name == "C4_server_metric_cross_check";
    });
    ASSERT_NE(c4, observations.end());
    EXPECT_EQ(c4->status, CheckStatus::kInconclusive);
    EXPECT_NE(c4->reason.find("TODO"), std::string::npos);
}

TEST(VineyardUriTest, HostPortAndMediumParsing) {
    EXPECT_EQ(ParseVineyardHostPort("vineyard://10.0.0.1:40000/mem?s_version=x"), "10.0.0.1:40000");
    EXPECT_EQ(ParseVineyardMedium("vineyard://10.0.0.1:40000/mem?s_version=x"), "mem");
    EXPECT_EQ(ParseVineyardHostPort("file://nfs_01/x"), "");
    EXPECT_EQ(ParseVineyardHostPort("vineyard://nohost/mem"), "");
}

} // namespace
} // namespace kvcm_swarm
