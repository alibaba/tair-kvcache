#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/liteHit/facts_query.h"

// Post-hoc capacity query over a LiteHit facts CSV.
//
// Usage:
//   lite_hit_facts_query_main <facts_csv> <output_log> <capacity_gb>...
// A negative capacity means infinite.
int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <facts_csv> <output_log> <capacity_gb>...\n", argv[0]);
        return 1;
    }

    const std::string facts_path = argv[1];
    const std::string output_path = argv[2];
    std::vector<double> capacity_gb;
    capacity_gb.reserve(argc - 3);
    for (int i = 3; i < argc; ++i) {
        char *end = nullptr;
        const double gb = std::strtod(argv[i], &end);
        if (end == argv[i] || *end != '\0') {
            fprintf(stderr, "invalid capacity_gb: %s\n", argv[i]);
            return 1;
        }
        capacity_gb.push_back(gb);
    }

    std::string error;
    if (!kv_cache_manager::RunLiteHitFactsQuery(facts_path, capacity_gb, output_path, error)) {
        fprintf(stderr, "facts query failed: %s\n", error.c_str());
        return 1;
    }
    printf("facts query done: %s\n", output_path.c_str());
    return 0;
}
