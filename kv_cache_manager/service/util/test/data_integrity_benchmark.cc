#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "service/util/fast_proto_json_codec.h"

namespace kv_cache_manager {
namespace {

constexpr size_t kBlockCount = 4096;
volatile size_t benchmark_sink = 0;

template <typename Operation>
double MeasureMicros(int iterations, Operation operation) {
    for (int i = 0; i < 3; ++i) {
        operation();
    }
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        operation();
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - begin).count() / iterations;
}

proto::meta::GetCacheLocationResponse CreateResponse(bool include_checksums) {
    proto::meta::GetCacheLocationResponse response;
    response.mutable_header()->mutable_status()->set_code(proto::meta::OK);
    response.mutable_header()->set_request_id("integrity-benchmark");
    for (size_t i = 0; i < kBlockCount; ++i) {
        auto *location = response.add_locations();
        location->set_type(proto::meta::ST_NFS);
        location->set_spec_size(2);
        auto *tp0 = location->add_location_specs();
        tp0->set_name("tp0");
        tp0->set_uri("file:///cache/model/tp0/block");
        auto *tp1 = location->add_location_specs();
        tp1->set_name("tp1");
        tp1->set_uri("file:///cache/model/tp1/block");
        if (include_checksums) {
            tp0->set_checksum(static_cast<int64_t>(0x123456789ABC0000ULL ^ i));
            tp0->set_checksum_present(true);
            tp1->set_checksum(static_cast<int64_t>(0xFEDCBA9876540000ULL ^ i));
            tp1->set_checksum_present(true);
        }
    }
    return response;
}

bool RunChecksumComparisonBenchmark() {
    std::vector<int64_t> expected(kBlockCount);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] = static_cast<int64_t>(0x123456789ABC0000ULL ^ i);
    }
    auto actual = expected;
    const double match_us = MeasureMicros(2000, [&] {
        const auto result = VerifyBatchChecksums(expected, actual, ChecksumValidationStage::CVS_META_ROUND_TRIP);
        benchmark_sink += result.faulty_indices.size() + (result.mismatch ? 1 : 0);
    });
    actual.back() ^= 1;
    const double mismatch_us = MeasureMicros(2000, [&] {
        const auto result = VerifyBatchChecksums(expected, actual, ChecksumValidationStage::CVS_READ_OUTPUT);
        benchmark_sink += result.faulty_indices.size();
    });
    const auto validation = VerifyBatchChecksums(expected, actual, ChecksumValidationStage::CVS_READ_OUTPUT);
    if (!validation.mismatch || validation.faulty_indices != std::vector<size_t>{kBlockCount - 1}) {
        std::cerr << "checksum comparison fixture validation failed\n";
        return false;
    }
    std::cout << "VerifyBatchChecksums (" << kBlockCount << " blocks)\n"
              << "  match=" << match_us << " us/batch (" << match_us * 1000 / kBlockCount << " ns/block)\n"
              << "  one mismatch=" << mismatch_us << " us/batch (" << mismatch_us * 1000 / kBlockCount
              << " ns/block)\n";
    return true;
}

bool RunWireBenchmark() {
    const auto without_checksums = CreateResponse(false);
    const auto with_checksums = CreateResponse(true);
    std::string binary_without;
    std::string binary_with;
    std::string json_without;
    std::string json_with;
    if (!without_checksums.SerializeToString(&binary_without) || !with_checksums.SerializeToString(&binary_with) ||
        !FastProtoJsonCodec::TryToJson(without_checksums, json_without) ||
        !FastProtoJsonCodec::TryToJson(with_checksums, json_with)) {
        std::cerr << "wire fixture serialization failed\n";
        return false;
    }

    const double binary_serialize_without = MeasureMicros(30, [&] {
        std::string output;
        if (!without_checksums.SerializeToString(&output)) {
            std::abort();
        }
        benchmark_sink += output.size();
    });
    const double binary_serialize_with = MeasureMicros(30, [&] {
        std::string output;
        if (!with_checksums.SerializeToString(&output)) {
            std::abort();
        }
        benchmark_sink += output.size();
    });
    const double binary_parse_without = MeasureMicros(30, [&] {
        proto::meta::GetCacheLocationResponse output;
        if (!output.ParseFromString(binary_without)) {
            std::abort();
        }
        benchmark_sink += output.locations_size();
    });
    const double binary_parse_with = MeasureMicros(30, [&] {
        proto::meta::GetCacheLocationResponse output;
        if (!output.ParseFromString(binary_with)) {
            std::abort();
        }
        benchmark_sink += output.locations_size();
    });
    const double json_serialize_without = MeasureMicros(20, [&] {
        std::string output;
        if (!FastProtoJsonCodec::TryToJson(without_checksums, output)) {
            std::abort();
        }
        benchmark_sink += output.size();
    });
    const double json_serialize_with = MeasureMicros(20, [&] {
        std::string output;
        if (!FastProtoJsonCodec::TryToJson(with_checksums, output)) {
            std::abort();
        }
        benchmark_sink += output.size();
    });

    std::cout << "GetCacheLocationResponse (" << kBlockCount << " locations, 2 checksummed specs/location)\n"
              << "  protobuf bytes: opt-out=" << binary_without.size() << " opt-in=" << binary_with.size()
              << " delta=" << binary_with.size() - binary_without.size() << " ("
              << static_cast<double>(binary_with.size() - binary_without.size()) / kBlockCount << " bytes/block)\n"
              << "  protobuf serialize: opt-out=" << binary_serialize_without << " us opt-in=" << binary_serialize_with
              << " us\n"
              << "  protobuf parse: opt-out=" << binary_parse_without << " us opt-in=" << binary_parse_with << " us\n"
              << "  HTTP JSON bytes: opt-out=" << json_without.size() << " opt-in=" << json_with.size()
              << " delta=" << json_with.size() - json_without.size() << " ("
              << static_cast<double>(json_with.size() - json_without.size()) / kBlockCount << " bytes/block)\n"
              << "  HTTP JSON serialize: opt-out=" << json_serialize_without << " us opt-in=" << json_serialize_with
              << " us\n";
    return true;
}

} // namespace
} // namespace kv_cache_manager

int main() {
    using namespace kv_cache_manager;
    std::cout << std::fixed << std::setprecision(2);
    if (!RunChecksumComparisonBenchmark() || !RunWireBenchmark()) {
        return 1;
    }
    return benchmark_sink == 0 ? 1 : 0;
}
