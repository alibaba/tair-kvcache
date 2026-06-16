#include "kv_cache_manager/common/string_util.h"

#include <cmath>
#include <sstream>

namespace kv_cache_manager {

std::vector<double> StringUtil::ParseBucketBoundaries(const std::string &buckets_str) {
    std::vector<double> boundaries;
    if (buckets_str.empty()) {
        return boundaries;
    }
    // Reject leading/trailing commas and empty tokens upfront
    if (buckets_str.front() == ',' || buckets_str.back() == ',') {
        return {};
    }
    std::istringstream iss(buckets_str);
    std::string token;

    while (std::getline(iss, token, ',')) {
        Trim(token);
        if (token.empty()) {
            return {};  // reject empty tokens (e.g. "1,,5")
        }
        try {
            size_t pos = 0;
            double val = std::stod(token, &pos);
            if (pos != token.size()) {
                return {};  // reject partial consumption (e.g. "1s")
            }
            if (val <= 0.0 || !std::isfinite(val)) {
                return {};
            }
            boundaries.push_back(val);
        } catch (const std::exception &) {
            return {};
        }
    }

    // Check strictly ascending order
    for (size_t i = 1; i < boundaries.size(); ++i) {
        if (boundaries[i] <= boundaries[i - 1]) {
            return {};
        }
    }

    return boundaries;
}

} // namespace kv_cache_manager
