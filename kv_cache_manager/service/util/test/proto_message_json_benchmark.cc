#include <chrono>
#include <cstdlib>
#include <google/protobuf/stubs/stringpiece.h>
#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>
#include <iomanip>
#include <iostream>
#include <string>

#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "service/util/fast_proto_json_codec.h"

namespace kv_cache_manager {
namespace {

constexpr int kKeyCount = 4096;
constexpr int kIterations = 20;
volatile size_t benchmark_sink = 0;

::google::protobuf::util::JsonPrintOptions CreateJsonPrintOptions() {
    ::google::protobuf::util::JsonPrintOptions options;
    options.add_whitespace = false;
    options.always_print_primitive_fields = true;
    options.always_print_enums_as_ints = false;
    options.preserve_proto_field_names = true;
    return options;
}

::google::protobuf::util::JsonParseOptions CreateJsonParseOptions() {
    ::google::protobuf::util::JsonParseOptions options;
    options.ignore_unknown_fields = true;
    options.case_insensitive_enum_parsing = false;
    return options;
}

bool GenericToJson(const ::google::protobuf::Message &message, std::string *json) {
    static const auto options = CreateJsonPrintOptions();
    return ::google::protobuf::util::MessageToJsonString(message, json, options).ok();
}

bool GenericFromJson(const std::string &json, ::google::protobuf::Message *message) {
    static const auto options = CreateJsonParseOptions();
    const ::google::protobuf::StringPiece input(json.data(), static_cast<ptrdiff_t>(json.size()));
    return ::google::protobuf::util::JsonStringToMessage(input, message, options).ok();
}

proto::meta::GetCacheLocationsByBackendRequest CreateRequest() {
    proto::meta::GetCacheLocationsByBackendRequest request;
    request.set_trace_id("trace-access-log-benchmark");
    request.set_instance_id("instance-production-shaped");
    request.set_query_type(proto::meta::QT_BATCH_GET);
    request.mutable_block_mask()->set_offset(0);
    request.set_sw_size(128);
    request.add_location_spec_names("full_attention");
    request.add_location_spec_names("mamba_state");
    auto *threefs = request.add_backend_selectors();
    threefs->set_backend_type(proto::meta::ST_3FS);
    threefs->set_strategy(proto::meta::LSS_V6D_PREFIX);
    auto *mempool = request.add_backend_selectors();
    mempool->set_backend_type(proto::meta::ST_TAIRMEMPOOL);
    mempool->set_strategy(proto::meta::LSS_WEIGHTED_RANDOM);

    // The access-log hot case contains 4K 64-bit block keys and the matching
    // token ids. Large decimal block keys reproduce the observed ~100-KiB
    // request JSON rather than benchmarking a tiny synthetic message.
    constexpr int64_t kFirstBlockKey = 8000000000000000000LL;
    for (int i = 0; i < kKeyCount; ++i) {
        request.add_block_keys(kFirstBlockKey + i);
        request.add_token_ids(i);
    }
    return request;
}

proto::meta::GetCacheLocationsByBackendResponse CreateResponse() {
    proto::meta::GetCacheLocationsByBackendResponse response;
    response.mutable_header()->mutable_status()->set_code(proto::meta::OK);
    response.mutable_header()->mutable_status()->set_message("");
    response.mutable_header()->set_request_id("request-access-log-benchmark");
    response.mutable_header()->set_tracer_result("lookup=complete");

    for (int i = 0; i < kKeyCount; ++i) {
        auto *locations = response.add_key_locations();
        // A mostly-miss batch with sparse cache hits matches the response
        // shape that makes repeated nested reflection work visible.
        if (i % 16 != 0) {
            continue;
        }
        auto *location = locations->add_locations();
        location->set_type(i % 32 == 0 ? proto::meta::ST_3FS : proto::meta::ST_TAIRMEMPOOL);
        location->set_spec_size(2);
        auto *attention = location->add_location_specs();
        attention->set_name("full_attention");
        attention->set_uri("3fs://cache-cluster/model/layer?offset=1048576&size=2097152");
        auto *mamba = location->add_location_specs();
        mamba->set_name("mamba_state");
        mamba->set_uri("pace://cache-node/block?offset=0&size=262144&media_type=1");
    }
    return response;
}

template <typename Operation>
double MeasureMicros(Operation operation) {
    for (int i = 0; i < 2; ++i) {
        operation();
    }
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < kIterations; ++i) {
        operation();
    }
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - begin).count() / kIterations;
}

template <typename MessageType>
bool VerifyEquivalent(const MessageType &message, const char *fixture_name, std::string *json) {
    std::string generic_json;
    std::string fast_json;
    if (!GenericToJson(message, &generic_json) || !FastProtoJsonCodec::TryToJson(message, fast_json)) {
        std::cerr << fixture_name << ": serialization failed\n";
        return false;
    }
    if (generic_json != fast_json) {
        std::cerr << fixture_name << ": fast JSON differs from protobuf JSON\n";
        return false;
    }

    MessageType generic_message;
    MessageType fast_message;
    if (!GenericFromJson(generic_json, &generic_message) ||
        !FastProtoJsonCodec::TryFromJson(generic_json, &fast_message)) {
        std::cerr << fixture_name << ": parsing failed\n";
        return false;
    }
    if (!::google::protobuf::util::MessageDifferencer::Equals(message, generic_message) ||
        !::google::protobuf::util::MessageDifferencer::Equals(message, fast_message)) {
        std::cerr << fixture_name << ": parsed protobuf differs from source\n";
        return false;
    }
    *json = std::move(generic_json);
    return true;
}

template <typename MessageType>
void RunBenchmark(const MessageType &message, const std::string &json, const char *fixture_name) {
    const double generic_to_json = MeasureMicros([&] {
        std::string output;
        if (!GenericToJson(message, &output)) {
            std::abort();
        }
        benchmark_sink += output.size();
    });
    const double fast_to_json = MeasureMicros([&] {
        std::string output;
        if (!FastProtoJsonCodec::TryToJson(message, output)) {
            std::abort();
        }
        benchmark_sink += output.size();
    });
    const double generic_from_json = MeasureMicros([&] {
        MessageType output;
        if (!GenericFromJson(json, &output)) {
            std::abort();
        }
        benchmark_sink += output.GetDescriptor()->field_count();
    });
    const double fast_from_json = MeasureMicros([&] {
        MessageType output;
        if (!FastProtoJsonCodec::TryFromJson(json, &output)) {
            std::abort();
        }
        benchmark_sink += output.GetDescriptor()->field_count();
    });

    std::cout << fixture_name << " (" << json.size() << " bytes, " << kKeyCount << " keys)\n"
              << "  PB -> JSON  protobuf=" << generic_to_json << " us  fast=" << fast_to_json
              << " us  speedup=" << generic_to_json / fast_to_json << "x\n"
              << "  JSON -> PB  protobuf=" << generic_from_json << " us  fast=" << fast_from_json
              << " us  speedup=" << generic_from_json / fast_from_json << "x\n";
}

} // namespace
} // namespace kv_cache_manager

int main() {
    using namespace kv_cache_manager;
    const auto request = CreateRequest();
    const auto response = CreateResponse();
    std::string request_json;
    std::string response_json;
    if (!VerifyEquivalent(request, "GetCacheLocationsByBackendRequest", &request_json) ||
        !VerifyEquivalent(response, "GetCacheLocationsByBackendResponse", &response_json)) {
        return 1;
    }

    std::cout << std::fixed << std::setprecision(1);
    RunBenchmark(request, request_json, "GetCacheLocationsByBackendRequest");
    RunBenchmark(response, response_json, "GetCacheLocationsByBackendResponse");
    return benchmark_sink == 0 ? 1 : 0;
}
