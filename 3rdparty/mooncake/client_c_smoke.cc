// Cross-node put/get smoke test for the mooncake C client.
//
// Run one "server" (mounts a store segment) and one "client" (puts values
// into the remote segment, reads them back and verifies) on different hosts,
// so the data path really goes over the configured transport (e.g. efa):
//
//   host A: mooncake_master
//   host A: client_c_smoke --role=server --hostname=<A ip> --master=<A>:50051
//                          --protocol=efa --device=rdmap...
//   host B: client_c_smoke --role=client --hostname=<B ip> --master=<A>:50051
//                          --protocol=efa --device=rdmap...

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gflags/gflags.h>
#include <string>
#include <thread>
#include <vector>

#include "client_c.h"

DEFINE_string(role, "client", "server: mount a segment and wait; client: put/get/verify");
DEFINE_string(hostname, "", "local ip (or ip:port) advertised to peers");
DEFINE_string(metadata, "P2PHANDSHAKE", "mooncake metadata connstring");
DEFINE_string(master, "localhost:50051", "mooncake master entry");
DEFINE_string(protocol, "efa", "transport protocol: efa, rdma or tcp");
DEFINE_string(device, "", "comma separated device names (required for efa/rdma)");
DEFINE_uint64(segment_mb, 4096, "server: segment size to mount");
DEFINE_uint64(value_kb, 4096, "client: bytes per key");
DEFINE_uint64(num_keys, 64, "client: number of keys");
DEFINE_uint64(rounds, 3, "client: put/get rounds");

namespace {

std::vector<Slice_t> MakeSlices(char *base, size_t len) {
    const size_t max_slice = mooncake_max_slice_size();
    std::vector<Slice_t> slices;
    for (size_t off = 0; off < len; off += max_slice) {
        slices.push_back({base + off, std::min(max_slice, len - off)});
    }
    return slices;
}

void Fill(char *buf, size_t len, uint64_t seed) {
    auto *words = reinterpret_cast<uint64_t *>(buf);
    for (size_t i = 0; i < len / sizeof(uint64_t); i++) {
        words[i] = seed * 0x9E3779B97F4A7C15ULL + i;
    }
}

double GiBps(size_t bytes, std::chrono::steady_clock::duration d) {
    return bytes / std::chrono::duration<double>(d).count() / (1ULL << 30);
}

int RunServer(client_t client) {
    ErrorCode_t ec = mooncake_client_mount_segment(client, FLAGS_segment_mb << 20, FLAGS_protocol.c_str());
    if (ec != MOONCAKE_ERROR_OK) {
        fprintf(stderr, "mount_segment failed: %d\n", ec);
        return 1;
    }
    printf("segment of %lu MiB mounted over %s, serving (Ctrl-C to stop)\n", FLAGS_segment_mb, FLAGS_protocol.c_str());
    fflush(stdout);
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(60));
    }
}

int RunClient(client_t client) {
    const size_t value_len = FLAGS_value_kb << 10;
    const size_t total = value_len * FLAGS_num_keys;
    void *mem = nullptr;
    if (posix_memalign(&mem, 1 << 21, total * 2) != 0) {
        fprintf(stderr, "alloc %zu bytes failed\n", total * 2);
        return 1;
    }
    char *src = static_cast<char *>(mem);
    char *dst = src + total;
    ErrorCode_t ec = mooncake_client_register_local_memory(client, mem, total * 2, "cpu:0", false, false);
    if (ec != MOONCAKE_ERROR_OK) {
        fprintf(stderr, "register_local_memory failed: %d\n", ec);
        return 1;
    }
    for (uint64_t k = 0; k < FLAGS_num_keys; k++) {
        Fill(src + k * value_len, value_len, k + 1);
    }

    // Unique per run so leftovers of an earlier run can never satisfy a get.
    const std::string run_id = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    int failures = 0;
    for (uint64_t r = 0; r < FLAGS_rounds; r++) {
        std::vector<std::string> keys;
        for (uint64_t k = 0; k < FLAGS_num_keys; k++) {
            keys.push_back("client_c_smoke_" + FLAGS_hostname + "_" + run_id + "_" + std::to_string(r) + "_" +
                           std::to_string(k));
        }

        auto t0 = std::chrono::steady_clock::now();
        for (uint64_t k = 0; k < FLAGS_num_keys; k++) {
            auto slices = MakeSlices(src + k * value_len, value_len);
            ReplicateConfig_t cfg;
            ec = mooncake_client_put(client, keys[k].c_str(), slices.data(), slices.size(), cfg);
            if (ec != MOONCAKE_ERROR_OK) {
                fprintf(stderr, "put %s failed: %d\n", keys[k].c_str(), ec);
                return 1;
            }
        }
        auto t1 = std::chrono::steady_clock::now();

        memset(dst, 0, total);
        for (uint64_t k = 0; k < FLAGS_num_keys; k++) {
            auto slices = MakeSlices(dst + k * value_len, value_len);
            ec = mooncake_client_get(client, keys[k].c_str(), slices.data(), slices.size());
            if (ec != MOONCAKE_ERROR_OK) {
                fprintf(stderr, "get %s failed: %d\n", keys[k].c_str(), ec);
                return 1;
            }
        }
        auto t2 = std::chrono::steady_clock::now();

        uint64_t bad_keys = 0;
        for (uint64_t k = 0; k < FLAGS_num_keys; k++) {
            const char *want = src + k * value_len;
            const char *got = dst + k * value_len;
            if (memcmp(want, got, value_len) != 0) {
                size_t first = 0, bad = 0, zero = 0;
                for (size_t b = 0; b < value_len; b++) {
                    if (want[b] != got[b]) {
                        if (bad++ == 0)
                            first = b;
                        zero += got[b] == 0;
                    }
                }
                if (bad_keys++ < 8) {
                    fprintf(stderr,
                            "data mismatch on %s: first_bad_offset=%zu bad_bytes=%zu (zero %zu)\n",
                            keys[k].c_str(),
                            first,
                            bad,
                            zero);
                }
            }
        }
        // Best effort: objects stay under a short lease right after get.
        for (uint64_t k = 0; k < FLAGS_num_keys; k++) {
            mooncake_client_remove(client, keys[k].c_str());
        }
        failures += bad_keys;
        printf("round %lu: %lu x %lu KiB  put %.2f GiB/s  get %.2f GiB/s  bad_keys=%lu\n",
               r,
               FLAGS_num_keys,
               FLAGS_value_kb,
               GiBps(total, t1 - t0),
               GiBps(total, t2 - t1),
               bad_keys);
    }
    mooncake_client_unregister_local_memory(client, mem, false);
    free(mem);
    return failures ? 1 : 0;
}

} // namespace

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_hostname.empty()) {
        fprintf(stderr, "--hostname is required\n");
        return 2;
    }
    client_t client = mooncake_client_create(FLAGS_hostname.c_str(),
                                             FLAGS_metadata.c_str(),
                                             FLAGS_protocol.c_str(),
                                             FLAGS_device.c_str(),
                                             FLAGS_master.c_str());
    if (client == nullptr) {
        fprintf(stderr, "mooncake_client_create failed\n");
        return 1;
    }
    int rc = FLAGS_role == "server" ? RunServer(client) : RunClient(client);
    mooncake_client_destroy(client);
    return rc;
}
