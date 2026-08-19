// ExpectedLocations tests: hot and cold provenance, the full state machine,
// revisions and reporter retirement.
#include <gtest/gtest.h>

#include "tools/kvcm_swarm/clients/v6d/expected_locations.h"

namespace kvcm_swarm {
namespace {

HotLocationKey HotKey(int64_t block_key, const std::string &host) {
    HotLocationKey key;
    key.block_key = block_key;
    key.spec_name = "v6d_4096";
    key.reporter = ReporterIdentity{"inst", host};
    return key;
}

ColdLocationKey ColdKey(int64_t block_key, const std::string &uri) {
    ColdLocationKey key;
    key.block_key = block_key;
    key.spec_name = "v6d_4096";
    key.storage_uri = uri;
    return key;
}

TEST(ExpectedLocationsTest, HotLifecycleTransitions) {
    ExpectedLocations expected;
    expected.SetReporterLive(ReporterIdentity{"inst", "h1"});
    const HotLocationKey key = HotKey(1, "h1");
    expected.HotPendingCreate(key);
    EXPECT_EQ(expected.CheckHotAcceptable(key, Now()).state, LocationState::kPendingCreate);
    expected.HotConfirm(key);
    EXPECT_EQ(expected.CheckHotAcceptable(key, Now()).state, LocationState::kConfirmed);
    expected.HotPendingDelete(key);
    EXPECT_EQ(expected.CheckHotAcceptable(key, Now()).state, LocationState::kPendingDelete);
    // pending-delete that was explicitly not executed returns to confirmed.
    expected.HotNotExecuted(key, true);
    EXPECT_EQ(expected.CheckHotAcceptable(key, Now()).state, LocationState::kConfirmed);
    expected.HotPendingDelete(key);
    expected.HotRemove(key);
    const auto removed = expected.CheckHotAcceptable(key, Now() + std::chrono::milliseconds(10));
    EXPECT_EQ(removed.state, LocationState::kRemoved);
    EXPECT_FALSE(removed.state_allows) << "no client-side delete-visibility grace period";
    // A query issued before the removal may still legitimately observe it.
    const auto earlier = expected.CheckHotAcceptable(key, Now() - std::chrono::seconds(1));
    EXPECT_TRUE(earlier.removed_after_query);
    EXPECT_TRUE(earlier.state_allows);
}

TEST(ExpectedLocationsTest, UnknownStateKeepsDirectionAndStaysAcceptable) {
    ExpectedLocations expected;
    expected.SetReporterLive(ReporterIdentity{"inst", "h1"});
    const HotLocationKey key = HotKey(2, "h1");
    expected.HotPendingCreate(key);
    expected.HotUnknown(key, false);
    const auto acceptance = expected.CheckHotAcceptable(key, Now());
    EXPECT_EQ(acceptance.state, LocationState::kUnknown);
    EXPECT_TRUE(acceptance.state_allows) << "soundness must admit still-possible unknown states";
    EXPECT_EQ(expected.Stats().hot_unknown, 1u);
    EXPECT_FALSE(expected.UnresolvedSummary(4).empty());
}

TEST(ExpectedLocationsTest, RemoteCandidateSnapshotExcludesTheRequesterAndDeadReporters) {
    ExpectedLocations expected;
    const ReporterIdentity self{"inst", "self"};
    const ReporterIdentity peer{"inst", "peer"};
    const ReporterIdentity dead{"inst", "dead"};
    expected.SetReporterLive(self);
    expected.SetReporterLive(peer);
    expected.SetReporterLive(dead);

    expected.HotPendingCreate(HotKey(7, "self"));
    expected.HotConfirm(HotKey(7, "self"));
    auto snapshot = expected.SnapshotCandidates(7, "v6d_4096", self);
    EXPECT_TRUE(snapshot.requester_has_possible_local);
    EXPECT_FALSE(snapshot.has_confirmed_remote);

    expected.HotPendingCreate(HotKey(8, "peer"));
    expected.HotConfirm(HotKey(8, "peer"));
    snapshot = expected.SnapshotCandidates(8, "v6d_4096", self);
    EXPECT_TRUE(snapshot.has_confirmed_remote);
    EXPECT_FALSE(snapshot.requester_has_possible_local);
    expected.HotPendingCreate(HotKey(9, "dead"));
    expected.HotConfirm(HotKey(9, "dead"));
    // A reporter that stopped heartbeating is unavailable, so a normal query
    // must not be expected to return its location.
    expected.SetReporterUnavailable(dead, false);
    snapshot = expected.SnapshotCandidates(9, "v6d_4096", self);
    EXPECT_FALSE(snapshot.has_confirmed_remote);
}

TEST(ExpectedLocationsTest, HostDownRetiresEveryLocationOfTheReporter) {
    ExpectedLocations expected;
    const ReporterIdentity peer{"inst", "peer"};
    expected.SetReporterLive(peer);
    for (int64_t key = 0; key < 3; ++key) {
        expected.HotPendingCreate(HotKey(key, "peer"));
        expected.HotConfirm(HotKey(key, "peer"));
    }
    const uint64_t before = expected.liveness_revision();
    expected.RetireReporter(peer);
    EXPECT_GT(expected.liveness_revision(), before);
    EXPECT_EQ(expected.Stats().hot_confirmed, 0u);
    EXPECT_EQ(expected.Stats().hot_removed, 3u);
    EXPECT_TRUE(expected.CheckHotAcceptable(HotKey(0, "peer"), Now()).retired_reporter);
}

TEST(ExpectedLocationsTest, ColdProvenanceHasNoMachineOwner) {
    ExpectedLocations expected;
    const ColdLocationKey key = ColdKey(11, "file://nfs_01/x?size=4096");
    expected.ColdPendingCreate(key);
    EXPECT_EQ(expected.Stats().cold_pending_create, 1u);
    expected.ColdConfirm(key, 4096);
    EXPECT_EQ(expected.Stats().cold_confirmed, 1u);
    EXPECT_EQ(expected.Stats().cold_confirmed_bytes, 4096u);
    // Confirming twice must not double count.
    expected.ColdConfirm(key, 4096);
    EXPECT_EQ(expected.Stats().cold_confirmed_bytes, 4096u);
    // Retiring the reporter that issued the write leaves the allocation alone.
    expected.SetReporterLive(ReporterIdentity{"inst", "writer"});
    expected.RetireReporter(ReporterIdentity{"inst", "writer"});
    EXPECT_EQ(expected.Stats().cold_confirmed, 1u);

    const ColdLocationKey unknown_key = ColdKey(12, "file://nfs_01/y");
    expected.ColdPendingCreate(unknown_key);
    expected.ColdUnknown(unknown_key);
    EXPECT_EQ(expected.Stats().cold_unknown, 1u);
    const ColdLocationKey aborted_key = ColdKey(13, "file://nfs_01/z");
    expected.ColdPendingCreate(aborted_key);
    expected.ColdNotExecuted(aborted_key);
    EXPECT_EQ(expected.Stats().cold_removed, 1u);
}

TEST(ExpectedLocationsTest, CandidateRevisionAdvancesOnEveryChange) {
    ExpectedLocations expected;
    expected.SetReporterLive(ReporterIdentity{"inst", "h"});
    const uint64_t start = expected.candidate_revision();
    expected.HotPendingCreate(HotKey(1, "h"));
    const uint64_t after_create = expected.candidate_revision();
    EXPECT_GT(after_create, start);
    expected.HotConfirm(HotKey(1, "h"));
    EXPECT_GT(expected.candidate_revision(), after_create);

    const uint64_t after_hot_change = expected.candidate_revision();
    expected.ColdPendingCreate(ColdKey(2, "file://nfs_01/x"));
    expected.ColdConfirm(ColdKey(2, "file://nfs_01/x"), 4096);
    EXPECT_EQ(expected.candidate_revision(), after_hot_change)
        << "cold allocation changes cannot invalidate a hot remote-availability sample";
}

} // namespace
} // namespace kvcm_swarm
