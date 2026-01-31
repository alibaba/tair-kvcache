#pragma once

#include <sstream>

#include "common.h"

namespace kv_cache_manager {
class DebugStringUtil {
public:
    template <typename T>
    static inline std::string ToString(const std::vector<T> &vec) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            oss << vec[i];
            if (i + 1 < vec.size())
                oss << ", ";
        }
        oss << "]";
        return oss.str();
    }

    static inline std::string ToString(const BlockMask &mask) {
        std::ostringstream oss;
        std::visit(
            [&oss](const auto &mask) {
                using T = std::decay_t<decltype(mask)>;
                if constexpr (std::is_same_v<BlockMaskVector, T>) {
                    oss << "{block_mask_vec_size:" << mask.size() << ",block_mask_vec:[";
                    for (auto x : mask) {
                        oss << x << ',';
                    }
                    oss << ']';
                } else if constexpr (std::is_same_v<BlockMaskOffset, T>) {
                    oss << "{block_mask_offset:" << mask;
                }
                oss << "}";
            },
            mask);
        return oss.str();
    }

    static inline std::string ToString(const Location &location) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < location.size(); ++i) {
            if (i > 0)
                oss << ", ";
            oss << "{"
                << "spec:" << location[i].spec_name << ",uri:" << location[i].uri << "}";
        }
        oss << "]";
        return oss.str();
    }

    static inline std::string ToString(const Locations &locations) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < locations.size(); ++i) {
            if (i > 0)
                oss << ", ";
            oss << ToString(locations[i]);
        }

        oss << "]";
        return oss.str();
    }

    static inline std::string ToString(const BlockBuffers &block_buffers) {
        std::ostringstream oss;
        oss << "{count:" << block_buffers.size() << ", blocks:[";
        for (size_t i = 0; i < block_buffers.size(); ++i) {
            if (i > 0) {
                oss << ", ";
            }
            oss << "{idx:" << i << ", iov_count:" << block_buffers[i].iovs.size() << ", iovs:[";
            for (size_t j = 0; j < block_buffers[i].iovs.size(); ++j) {
                const auto &iov = block_buffers[i].iovs[j];
                if (j > 0) {
                    oss << ", ";
                }
                oss << "{idx:" << j << ",type:" << static_cast<uint32_t>(iov.type) << ",addr:" << iov.base
                    << ",size:" << iov.size << ",ignore:" << (iov.ignore ? "true" : "false") << "}";
            }
            oss << "]}";
        }
        oss << "]}";
        return oss.str();
    }
};
} // namespace kv_cache_manager