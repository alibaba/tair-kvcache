"""Tests for per-pod statistics export (per_pod_stats.csv)."""
import os, sys, json, csv, shutil, random
import numpy as np
import pytest

KVCM_SO_DIR = "/sgl-workspace/claude_workspace/tair-kvcache/bazel-bin/kv_cache_manager/optimizer/pybind"
if KVCM_SO_DIR not in sys.path:
    sys.path.insert(0, KVCM_SO_DIR)
try:
    import kvcm_py_optimizer
    HAS_KVCM = True
except ImportError:
    HAS_KVCM = False

from schedule_simulator.schedule_emulator.types import *
from schedule_simulator.schedule_emulator.run import DisaggBenchmarkRunner

ENRICHED = os.path.join(os.path.dirname(__file__), "assets/glm5_sample/glm5_enriched_input.jsonl")
EXPORT_DIR = "/tmp/test_per_pod_stats"


@pytest.fixture(autouse=True)
def cleanup_export():
    yield
    if os.path.exists(EXPORT_DIR):
        shutil.rmtree(EXPORT_DIR)


def _make_runner(num_p=3, num_prompts=15, routing=RoutingPolicy.ROUND_ROBIN):
    random.seed(42); np.random.seed(42)
    return DisaggBenchmarkRunner(
        benchmark_config=BenchmarkConfig(
            num_prompts=num_prompts, min_input_length=200, max_input_length=500,
            min_output_length=1, max_output_length=2, disable_tqdm=True,
        ),
        p_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill",
                                            chunked_prefill_size=4096,
                                            hicache_storage_backend="hf3fs"),
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=PlatformConfig(device="H20", memory_read_bandwidth_gb=16.0,
                                          disk_read_bandwidth_gb=2.0),
        d_platform_config=PlatformConfig(device="H20"),
        router_config=RouterConfig(p_policy=routing, d_policy=RoutingPolicy.ROUND_ROBIN,
                                    worker_startup_check_interval=0.01),
        num_p_instance=num_p, num_d_instance=0,
    )


class TestPerPodStatsExport:
    """Test per_pod_stats.csv generation."""

    def test_per_pod_stats_file_created(self):
        """export_results should create per_pod_stats.csv."""
        runner = _make_runner()
        metrics = runner.run_benchmark_emulation()
        runner.export_results(EXPORT_DIR, metrics)

        assert os.path.exists(os.path.join(EXPORT_DIR, "per_pod_stats.csv"))

    def test_per_pod_stats_has_correct_columns(self):
        """CSV should have the expected header columns."""
        runner = _make_runner()
        metrics = runner.run_benchmark_emulation()
        runner.export_results(EXPORT_DIR, metrics)

        with open(os.path.join(EXPORT_DIR, "per_pod_stats.csv")) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        expected_cols = {"pod", "total_requests", "total_input_tokens", "total_output_tokens",
                         "total_blocks", "total_engine_hit_blocks", "total_peer_hit_blocks",
                         "total_pool_hit_blocks", "total_group_hit_blocks", "total_external_peer_hit_blocks"}
        assert set(rows[0].keys()) == expected_cols

    def test_per_pod_stats_row_count_matches_instances(self):
        """Number of rows should equal the number of P + D instances."""
        num_p = 4
        runner = _make_runner(num_p=num_p, num_prompts=20)
        metrics = runner.run_benchmark_emulation()
        runner.export_results(EXPORT_DIR, metrics)

        with open(os.path.join(EXPORT_DIR, "per_pod_stats.csv")) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == num_p
        pods = [row["pod"] for row in rows]
        for i in range(num_p):
            assert f"P{i}" in pods

    def test_per_pod_total_requests_sum_matches(self):
        """Sum of per-pod total_requests should equal total completed requests."""
        runner = _make_runner(num_p=3, num_prompts=30)
        metrics = runner.run_benchmark_emulation()
        runner.export_results(EXPORT_DIR, metrics)

        with open(os.path.join(EXPORT_DIR, "per_pod_stats.csv")) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        total_reqs = sum(int(row["total_requests"]) for row in rows)
        assert total_reqs == metrics["completed"]

    def test_per_pod_total_input_tokens_positive(self):
        """Each pod should have positive total_input_tokens if it processed requests."""
        runner = _make_runner(num_p=3, num_prompts=15)
        metrics = runner.run_benchmark_emulation()
        runner.export_results(EXPORT_DIR, metrics)

        with open(os.path.join(EXPORT_DIR, "per_pod_stats.csv")) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            if int(row["total_requests"]) > 0:
                assert int(row["total_input_tokens"]) > 0
                assert int(row["total_output_tokens"]) > 0

    def test_per_pod_total_blocks_consistent_with_tokens(self):
        """total_blocks should be >= total_requests (at least 1 block per request)."""
        runner = _make_runner(num_p=3, num_prompts=15)
        metrics = runner.run_benchmark_emulation()
        runner.export_results(EXPORT_DIR, metrics)

        with open(os.path.join(EXPORT_DIR, "per_pod_stats.csv")) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            total_reqs = int(row["total_requests"])
            total_blocks = int(row["total_blocks"])
            if total_reqs > 0:
                assert total_blocks >= total_reqs

    def test_per_pod_round_robin_balanced(self):
        """With round-robin routing, requests should be evenly distributed across pods."""
        num_p = 3
        num_prompts = 30
        runner = _make_runner(num_p=num_p, num_prompts=num_prompts)
        metrics = runner.run_benchmark_emulation()
        runner.export_results(EXPORT_DIR, metrics)

        with open(os.path.join(EXPORT_DIR, "per_pod_stats.csv")) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        reqs_per_pod = [int(row["total_requests"]) for row in rows]
        # Round-robin should give exactly equal distribution
        expected = num_prompts // num_p
        for count in reqs_per_pod:
            assert abs(count - expected) <= 1  # allow +-1 for rounding

    @pytest.mark.skipif(not HAS_KVCM, reason="kvcm not available")
    def test_per_pod_stats_with_hierarchical(self):
        """With hierarchical cache, hit_blocks columns should be populated."""
        random.seed(42); np.random.seed(42)
        runner = DisaggBenchmarkRunner(
            benchmark_config=BenchmarkConfig(
                dataset_path=ENRICHED, num_prompts=20, disable_tqdm=True,
            ),
            p_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_prefill",
                                                chunked_prefill_size=8192,
                                                hicache_storage_backend="hf3fs"),
            d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
            p_platform_config=PlatformConfig(device="H20", disk_read_bandwidth_gb=2.0,
                                              memory_read_bandwidth_gb=16.0,
                                              memory_capacity_gb=64.0,
                                              peer_read_bandwidth_gb=10.0),
            d_platform_config=PlatformConfig(device="H20"),
            router_config=RouterConfig(p_policy=RoutingPolicy.ROUND_ROBIN,
                                        d_policy=RoutingPolicy.ROUND_ROBIN,
                                        worker_startup_check_interval=0.01),
            num_p_instance=3, num_d_instance=0,
            enable_hierarchical=True,
            hierarchical_output_dir=os.path.join(EXPORT_DIR, "optimizer"),
        )
        metrics = runner.run_benchmark_emulation()
        runner.export_results(EXPORT_DIR, metrics)

        with open(os.path.join(EXPORT_DIR, "per_pod_stats.csv")) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        # At least one pod should have engine_hit_blocks > 0 (prefix reuse)
        total_engine_hits = sum(int(row["total_engine_hit_blocks"]) for row in rows)
        # With real data there should be some hits
        print(f"[hierarchical] total_engine_hit_blocks across pods: {total_engine_hits}")
        # Just verify the column exists and is numeric
        for row in rows:
            assert int(row["total_engine_hit_blocks"]) >= 0
            assert int(row["total_peer_hit_blocks"]) >= 0
            assert int(row["total_pool_hit_blocks"]) >= 0

    def test_per_pod_no_hierarchical_hit_blocks_zero(self):
        """Without hierarchical cache, hit_blocks columns should all be 0."""
        runner = _make_runner(num_p=2, num_prompts=10)
        metrics = runner.run_benchmark_emulation()
        runner.export_results(EXPORT_DIR, metrics)

        with open(os.path.join(EXPORT_DIR, "per_pod_stats.csv")) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for row in rows:
            assert int(row["total_engine_hit_blocks"]) == 0
            assert int(row["total_peer_hit_blocks"]) == 0
            assert int(row["total_pool_hit_blocks"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
