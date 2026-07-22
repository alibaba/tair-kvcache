#include "kv_cache_manager/optimizer/liteHit/lite_hit.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kv_cache_manager {

namespace {

constexpr std::size_t kCompactionSlackPositions = 4096;

uint64_t SaturatingAdd(uint64_t lhs, uint64_t rhs) {
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        return std::numeric_limits<uint64_t>::max();
    }
    return lhs + rhs;
}

uint64_t SaturatingMultiply(uint64_t lhs, uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return std::numeric_limits<uint64_t>::max();
    }
    return lhs * rhs;
}

} // namespace

LiteHit::LiteHit(std::vector<int64_t> capacity_blocks, uint64_t block_size_tokens)
    : input_capacities_(std::move(capacity_blocks)), block_size_tokens_(block_size_tokens) {
    if (block_size_tokens_ == 0) {
        throw std::invalid_argument("LiteHit block_size_tokens must be positive");
    }

    finite_capacities_.reserve(input_capacities_.size());
    for (int64_t capacity : input_capacities_) {
        if (capacity < kInfiniteCapacity) {
            throw std::invalid_argument("LiteHit capacity must be non-negative or -1 for infinity");
        }
        if (capacity == kInfiniteCapacity) {
            has_infinite_capacity_ = true;
        } else {
            finite_capacities_.push_back(capacity);
            largest_finite_capacity_ = std::max(largest_finite_capacity_, static_cast<uint64_t>(capacity));
        }
    }

    std::sort(finite_capacities_.begin(), finite_capacities_.end());
    finite_capacities_.erase(std::unique(finite_capacities_.begin(), finite_capacities_.end()),
                             finite_capacities_.end());

    input_to_internal_capacity_.reserve(input_capacities_.size());
    for (int64_t capacity : input_capacities_) {
        if (capacity == kInfiniteCapacity) {
            input_to_internal_capacity_.push_back(finite_capacities_.size());
        } else {
            const auto it = std::lower_bound(finite_capacities_.begin(), finite_capacities_.end(), capacity);
            input_to_internal_capacity_.push_back(static_cast<std::size_t>(it - finite_capacities_.begin()));
        }
    }

    hit_deltas_.assign(finite_capacities_.size() + (has_infinite_capacity_ ? 1 : 0), 0);
}

LiteHit::RequestResult LiteHit::ProcessRequest(const std::vector<int64_t> &block_keys, uint64_t input_token_len) {
    ValidateRequest(block_keys, input_token_len);

    const std::vector<uint64_t> request_hit_deltas = EvaluateRequestPrefix(block_keys);
    for (std::size_t i = 0; i < hit_deltas_.size(); ++i) {
        hit_deltas_[i] = SaturatingAdd(hit_deltas_[i], request_hit_deltas[i]);
    }

    CommitRequest(block_keys);
    PruneToTrackedCapacity();
    MaybeCompactPositions();

    request_count_ = SaturatingAdd(request_count_, 1);
    total_input_tokens_ = SaturatingAdd(total_input_tokens_, input_token_len);

    RequestResult result;
    result.input_token_len = input_token_len;
    result.capacity_results = MakeCapacityResults(ResolveHitDeltas(request_hit_deltas), input_token_len);
    return result;
}

LiteHit::AnalysisResult LiteHit::Analyze(const std::vector<TraceRequest> &requests,
                                         const RequestResultCallback &on_request) {
    Reset();
    for (std::size_t request_index = 0; request_index < requests.size(); ++request_index) {
        const auto &request = requests[request_index];
        const RequestResult result = ProcessRequest(request.block_keys, request.input_token_len);
        if (on_request) {
            on_request(request_index, result);
        }
    }
    return GetResult();
}

void LiteHit::ValidateRequest(const std::vector<int64_t> &block_keys, uint64_t input_token_len) const {
    const uint64_t expected_complete_blocks = input_token_len / block_size_tokens_;
    if (expected_complete_blocks != static_cast<uint64_t>(block_keys.size())) {
        throw std::invalid_argument(
            "LiteHit block_keys must contain exactly floor(input_token_len / block_size_tokens) complete blocks");
    }
}

std::vector<uint64_t> LiteHit::EvaluateRequestPrefix(const std::vector<int64_t> &block_keys) const {
    std::vector<uint64_t> request_hit_deltas(hit_deltas_.size(), 0);
    if (block_keys.empty() || request_hit_deltas.empty()) {
        return request_hit_deltas;
    }

    // All ranks are read from one immutable request-start snapshot. Repeated
    // keys reuse the same snapshot entry. A cold key stops every capacity's
    // prefix, so its later occurrence cannot revive the prefix.
    std::unordered_map<int64_t, SnapshotEntry> snapshot_entries;
    snapshot_entries.reserve(block_keys.size());
    for (int64_t block_key : block_keys) {
        auto [entry_it, inserted] = snapshot_entries.emplace(block_key, SnapshotEntry{});
        if (!inserted) {
            continue;
        }
        const auto previous = last_positions_.find(block_key);
        if (previous != last_positions_.end()) {
            entry_it->second.is_resident = true;
            entry_it->second.required_capacity = ReuseDistance(previous->second) + 1;
        }
    }

    uint64_t prefix_required_capacity = 0;
    for (int64_t block_key : block_keys) {
        const SnapshotEntry &entry = snapshot_entries.at(block_key);
        if (!entry.is_resident) {
            break;
        }
        prefix_required_capacity = std::max(prefix_required_capacity, entry.required_capacity);
        const std::size_t capacity_index = FirstCapacityAtLeast(prefix_required_capacity);
        if (capacity_index < request_hit_deltas.size()) {
            request_hit_deltas[capacity_index]++;
        }
    }
    return request_hit_deltas;
}

void LiteHit::CommitRequest(const std::vector<int64_t> &block_keys) {
    if (block_keys.empty()) {
        return;
    }

    // State commits tail-to-head: sequentially touching the request in
    // reverse order produces a final LRU order determined only by each
    // distinct key's last touch, which is its first occurrence visited
    // back-to-front. The chain head therefore ends up most recent and the
    // eviction victim is always a chain leaf. Remove old markers once, then
    // append those touches in reverse request order.
    std::unordered_map<int64_t, std::size_t> first_occurrence;
    first_occurrence.reserve(block_keys.size());
    for (std::size_t i = block_keys.size(); i > 0; --i) {
        first_occurrence[block_keys[i - 1]] = i - 1;
    }

    for (const auto &[block_key, _] : first_occurrence) {
        const auto previous = last_positions_.find(block_key);
        if (previous != last_positions_.end()) {
            fenwick_.Add(previous->second, -1);
            last_positions_.erase(previous);
        }
    }

    for (std::size_t i = block_keys.size(); i > 0; --i) {
        const int64_t block_key = block_keys[i - 1];
        if (first_occurrence.at(block_key) != i - 1) {
            continue;
        }
        fenwick_.AppendZero();
        const std::size_t current_position = fenwick_.size();
        fenwick_.Add(current_position, 1);
        last_positions_[block_key] = current_position;
        position_order_.emplace_back(current_position, block_key);
    }
}

void LiteHit::PruneToTrackedCapacity() {
    if (has_infinite_capacity_) {
        return;
    }

    while (last_positions_.size() > largest_finite_capacity_) {
        while (!position_order_.empty()) {
            const auto [position, block_key] = position_order_.front();
            position_order_.pop_front();
            const auto active = last_positions_.find(block_key);
            if (active == last_positions_.end() || active->second != position) {
                continue;
            }
            fenwick_.Add(position, -1);
            last_positions_.erase(active);
            break;
        }
    }
}

void LiteHit::MaybeCompactPositions() {
    const std::size_t active_positions = last_positions_.size();
    if (fenwick_.size() <= kCompactionSlackPositions) {
        return;
    }
    const std::size_t positions_over_slack = fenwick_.size() - kCompactionSlackPositions;
    if (active_positions >= (positions_over_slack + 1) / 2) {
        return;
    }

    std::vector<std::pair<std::size_t, int64_t>> ordered_positions;
    ordered_positions.reserve(active_positions);
    for (const auto &[block_key, position] : last_positions_) {
        ordered_positions.emplace_back(position, block_key);
    }
    std::sort(ordered_positions.begin(), ordered_positions.end());

    DynamicFenwickTree compacted_fenwick;
    position_order_.clear();
    for (const auto &[_, block_key] : ordered_positions) {
        compacted_fenwick.AppendZero();
        const std::size_t compacted_position = compacted_fenwick.size();
        compacted_fenwick.Add(compacted_position, 1);
        last_positions_[block_key] = compacted_position;
        position_order_.emplace_back(compacted_position, block_key);
    }
    fenwick_ = std::move(compacted_fenwick);
}

std::size_t LiteHit::FirstCapacityAtLeast(uint64_t required_capacity) const {
    auto finite_it = finite_capacities_.end();
    if (required_capacity <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        finite_it = std::lower_bound(
            finite_capacities_.begin(), finite_capacities_.end(), static_cast<int64_t>(required_capacity));
    }
    if (finite_it != finite_capacities_.end()) {
        return static_cast<std::size_t>(finite_it - finite_capacities_.begin());
    }
    if (has_infinite_capacity_) {
        return finite_capacities_.size();
    }
    return hit_deltas_.size();
}

std::vector<uint64_t> LiteHit::ResolveHitDeltas(const std::vector<uint64_t> &hit_deltas) const {
    std::vector<uint64_t> internal_hits(hit_deltas.size(), 0);
    uint64_t running_hits = 0;
    for (std::size_t i = 0; i < hit_deltas.size(); ++i) {
        running_hits = SaturatingAdd(running_hits, hit_deltas[i]);
        internal_hits[i] = running_hits;
    }

    std::vector<uint64_t> result;
    result.reserve(input_to_internal_capacity_.size());
    for (std::size_t internal_index : input_to_internal_capacity_) {
        result.push_back(internal_hits[internal_index]);
    }
    return result;
}

uint64_t LiteHit::ReuseDistance(std::size_t previous_position) const {
    return fenwick_.PrefixSum(fenwick_.size()) - fenwick_.PrefixSum(previous_position);
}

std::vector<LiteHit::CapacityResult> LiteHit::MakeCapacityResults(const std::vector<uint64_t> &hit_counts,
                                                                  uint64_t input_tokens) const {
    std::vector<CapacityResult> results;
    results.reserve(input_capacities_.size());
    for (std::size_t i = 0; i < input_capacities_.size(); ++i) {
        CapacityResult result;
        result.capacity_blocks = input_capacities_[i];
        result.hit_count = hit_counts[i];
        result.hit_tokens = SaturatingMultiply(result.hit_count, block_size_tokens_);
        result.input_tokens = input_tokens;
        result.hit_rate =
            input_tokens == 0 ? 0.0 : static_cast<double>(result.hit_tokens) / static_cast<double>(input_tokens);
        results.push_back(result);
    }
    return results;
}

LiteHit::AnalysisResult LiteHit::GetResult() const {
    AnalysisResult result;
    result.request_count = request_count_;
    result.total_input_tokens = total_input_tokens_;
    result.capacity_results = MakeCapacityResults(ResolveHitDeltas(hit_deltas_), total_input_tokens_);
    return result;
}

void LiteHit::Reset() {
    std::fill(hit_deltas_.begin(), hit_deltas_.end(), 0);
    fenwick_.Clear();
    last_positions_.clear();
    position_order_.clear();
    request_count_ = 0;
    total_input_tokens_ = 0;
}

uint64_t LiteHit::memory_usage_bytes() const {
    uint64_t bytes = fenwick_.memory_usage_bytes();
    bytes =
        SaturatingAdd(bytes, SaturatingMultiply(static_cast<uint64_t>(last_positions_.bucket_count()), sizeof(void *)));
    constexpr uint64_t kEstimatedHashNodeOverhead = sizeof(void *) * 2;
    bytes =
        SaturatingAdd(bytes,
                      SaturatingMultiply(static_cast<uint64_t>(last_positions_.size()),
                                         sizeof(std::pair<const int64_t, std::size_t>) + kEstimatedHashNodeOverhead));
    bytes =
        SaturatingAdd(bytes, SaturatingMultiply(static_cast<uint64_t>(hit_deltas_.capacity()), sizeof(hit_deltas_[0])));
    bytes = SaturatingAdd(
        bytes, SaturatingMultiply(static_cast<uint64_t>(finite_capacities_.capacity()), sizeof(finite_capacities_[0])));
    bytes = SaturatingAdd(
        bytes,
        SaturatingMultiply(static_cast<uint64_t>(position_order_.size()), sizeof(std::pair<std::size_t, int64_t>)));
    return bytes;
}

} // namespace kv_cache_manager
