// kvcm_swarm: a standalone client-behavior runner for KVCM.
//
// It reads one JSON run configuration, connects to an existing KVCM
// deployment, runs the configured behaviors and writes a complete fact report.
// It needs no Python and never creates or destroys deployment resources.
#include <cstring>
#include <iostream>
#include <string>

#include "tools/kvcm_swarm/app/run_coordinator.h"
#include "tools/kvcm_swarm/clients/registry.h"
#include "tools/kvcm_swarm/scenario/loader.h"

namespace {

void PrintUsage() {
    std::cout << "usage: kvcm_swarm --config <run_config.json> [--validate-only]\n"
              << "\n"
              << "  --config PATH      JSON run configuration (the only configuration entry point)\n"
              << "  --validate-only    parse and validate locally, create no transport and send no RPC\n"
              << "\n"
              << "exit codes: 0 ok, 2 configuration invalid, 3 preflight failed,\n"
              << "            4 initialize failed, 5 report generation failed\n";
}

} // namespace

int main(int argc, char **argv) {
    std::string config_path;
    bool validate_only = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (std::strcmp(argv[i], "--validate-only") == 0) {
            validate_only = true;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            PrintUsage();
            return 0;
        } else {
            std::cerr << "kvcm_swarm: unrecognised argument '" << argv[i] << "'\n";
            PrintUsage();
            return static_cast<int>(kvcm_swarm::ExitCode::kConfigInvalid);
        }
    }
    if (config_path.empty()) {
        std::cerr << "kvcm_swarm: --config is required\n";
        PrintUsage();
        return static_cast<int>(kvcm_swarm::ExitCode::kConfigInvalid);
    }

    const kvcm_swarm::BehaviorRegistry registry = kvcm_swarm::MakeDefaultRegistry();
    // Local validation is pure: no transport is created and no RPC is sent.
    kvcm_swarm::LoadResult load = kvcm_swarm::LoadScenarioFromFile(config_path, registry);
    if (!load.ok) {
        std::cerr << "kvcm_swarm: configuration is invalid (" << load.errors.size() << " problem(s)):\n";
        for (const auto &error : load.errors) {
            std::cerr << "  - " << error << "\n";
        }
        return static_cast<int>(kvcm_swarm::ExitCode::kConfigInvalid);
    }
    if (validate_only) {
        std::cout << "kvcm_swarm: configuration '" << load.config.name << "' is valid\n";
        return 0;
    }

    kvcm_swarm::RunCoordinator coordinator(std::move(load.config), registry);
    return static_cast<int>(coordinator.Run());
}
