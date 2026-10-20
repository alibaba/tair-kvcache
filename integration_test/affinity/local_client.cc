#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "kv_cache_manager/client/include/manager_client.h"

namespace kvcm = kv_cache_manager;

namespace {
void Require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void RequireOk(kvcm::ClientErrorCode ec, const std::string &operation) {
    Require(ec == kvcm::ER_OK, operation + " failed: " + std::to_string(ec));
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " CLIENT_CONFIG.json\n";
        return 2;
    }
    try {
        std::ifstream input(argv[1]);
        Require(input.is_open(), "Cannot open client config");
        const std::string config{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        kvcm::InitParams params;
        params.role_type = kvcm::RoleType::HYBRID;
        params.self_location_spec_name = "spec_0";
        auto client = kvcm::ManagerClient::Create(config, params);
        Require(client != nullptr, "ManagerClient initialization failed");
        const auto caller = client->GetCallerNode();
        Require(!caller.empty(), "Caller node is empty; configure the local caller provider");

        const int64_t key =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
        const std::string trace = "local_sdk_" + std::to_string(key);
        std::cout << "KEY=" << key << "\nCALLER=" << caller << std::endl;
        auto [start_ec, write] = client->StartWrite(trace + "_start", {key}, {}, {}, 60);
        RequireOk(start_ec, "StartWrite");
        Require(!write.write_session_id.empty(), "StartWrite returned an empty session");
        Require(write.locations.size() == 1 && write.locations.front().size() == 1,
                "StartWrite must allocate one block with one spec");
        const auto &allocated = write.locations.front().front();
        Require(allocated.spec_name == "spec_0" && !allocated.uri.empty(), "Unexpected allocated spec or URI");
        Require(allocated.node_id == caller, "Allocated node differs from caller");

        std::array<unsigned char, 4096> expected{}, actual{};
        for (size_t i = 0; i < expected.size(); ++i) {
            expected[i] = static_cast<unsigned char>((i * 17 + key % 251) % 251 + 1);
        }
        kvcm::UriStrVec saved;
        try {
            auto [save_ec, uris] = client->SaveKvCaches(
                {allocated.uri}, {{{kvcm::Iov{kvcm::MemoryType::CPU, expected.data(), expected.size(), false}}}});
            RequireOk(save_ec, "SaveKvCaches");
            Require(uris.size() == 1 && !uris.front().empty(), "SaveKvCaches returned invalid URIs");
            saved = std::move(uris);
        } catch (...) {
            const auto abort_ec =
                client->FinishWrite(trace + "_abort", write.write_session_id, kvcm::BlockMaskOffset{0}, {});
            std::cerr << "Save rollback FinishWrite(false): " << abort_ec << std::endl;
            throw;
        }
        write.locations.front().front().uri = saved.front();
        std::cout << "URI=" << saved.front() << std::endl;

        const auto match = [&](const std::string &phase) {
            std::vector<kvcm::ClientReplicationHint> hints;
            auto [ec, locations] = client->MatchLocation(trace + phase,
                                                         kvcm::QueryType::QT_PREFIX_MATCH,
                                                         {key},
                                                         {},
                                                         kvcm::BlockMaskOffset{0},
                                                         0,
                                                         {"spec_0"},
                                                         hints);
            RequireOk(ec, "MatchLocation" + phase);
            Require(hints.empty(), "Local read unexpectedly returned a replication hint");
            return locations;
        };
        Require(match("_before_finish").empty(), "Saved bytes became visible before FinishWrite");
        // An ambiguous publication response must not trigger a second, aborting FinishWrite.
        RequireOk(
            client->FinishWrite(trace + "_finish", write.write_session_id, kvcm::BlockMaskOffset{1}, write.locations),
            "FinishWrite");
        const auto matched = match("_after_finish");
        Require(matched == write.locations, "Published location differs from the saved location");
        Require(matched.front().front().node_id == client->GetCallerNode(), "Matched node differs from caller");
        RequireOk(client->LoadKvCaches({matched.front().front().uri},
                                       {{{kvcm::Iov{kvcm::MemoryType::CPU, actual.data(), actual.size(), false}}}}),
                  "LoadKvCaches");
        for (size_t i = 0; i < expected.size(); ++i) {
            Require(actual[i] == expected[i], "Readback mismatch at byte " + std::to_string(i));
        }
        std::cout << "LOCAL_SDK_OK" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "LOCAL_SDK_ERROR: " << error.what() << std::endl;
        return 1;
    }
}
