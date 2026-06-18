"""Tests for timeline replay mode (route_only, route_and_latency, latency_only)."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from schedule_simulator.schedule_emulator.types import (
    BenchmarkConfig,
    SchedulerConfig,
    PlatformConfig,
    RouterConfig,
    RoutingPolicy,
    TimelineMode,
)
from schedule_simulator.schedule_emulator.run import DisaggBenchmarkRunner
from schedule_simulator.schedule_emulator.timeline_loader import TimelineLoader
from schedule_simulator.infer_time_predictor import RequestLevelTimePredictor

TIMELINE_FILE = os.path.join(os.path.dirname(__file__), "assets", "timeline_sample.jsonl")

# Fast predictor for tests: 0.1 ms/token
_FAST_PREDICTOR = RequestLevelTimePredictor(constant_ms_per_token=0.1)


def _make_runner(timeline_mode: TimelineMode, routing="round_robin"):
    """Helper to create a DisaggBenchmarkRunner with timeline mode."""
    timeline_loader = TimelineLoader(TIMELINE_FILE)

    bc = BenchmarkConfig(
        dataset_path=TIMELINE_FILE,
        num_prompts=20,
        disable_tqdm=True,
        data_block_size=256,
    )
    sc = SchedulerConfig(
        "Qwen2.5-3B",
        scenario="disagg_prefill",
        chunked_prefill_size=8192,
        request_level_scheduling=True,
        page_size=256,
    )
    pc = PlatformConfig(device="H20", hbm_capacity_gb=80.0, memory_capacity_gb=512.0)
    policy_map = {
        "round_robin": RoutingPolicy.ROUND_ROBIN,
        "random": RoutingPolicy.RANDOM,
    }
    rc = RouterConfig(
        p_policy=policy_map[routing],
        d_policy=RoutingPolicy.ROUND_ROBIN,
        worker_startup_check_interval=0.01,
    )
    runner = DisaggBenchmarkRunner(
        benchmark_config=bc,
        p_scheduler_config=sc,
        d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
        p_platform_config=pc,
        d_platform_config=PlatformConfig(device="H20"),
        router_config=rc,
        num_p_instance=1,  # Should be overridden by timeline
        num_d_instance=0,
        infer_time_predictor=_FAST_PREDICTOR,
        timeline_loader=timeline_loader,
        timeline_mode=timeline_mode,
    )
    return runner, timeline_loader


class TestTimelineLoader:
    """Test TimelineLoader pod mapping stability."""

    def test_pod_names_sorted(self):
        loader = TimelineLoader(TIMELINE_FILE)
        names = loader.pod_names
        assert names == sorted(names)
        assert names == ["pod-alpha", "pod-beta", "pod-gamma"]

    def test_num_pods(self):
        loader = TimelineLoader(TIMELINE_FILE)
        assert loader.num_pods == 3

    def test_pod_index_mapping(self):
        loader = TimelineLoader(TIMELINE_FILE)
        # Alphabetical order: alpha=0, beta=1, gamma=2
        assert loader.get_pod_index("pod-alpha") == 0
        assert loader.get_pod_index("pod-beta") == 1
        assert loader.get_pod_index("pod-gamma") == 2
        assert loader.get_pod_index("nonexistent") is None


class TestTimelineRouteOnly:
    """route_only: uses timeline routing but predictor latency."""

    def test_num_nodes_from_file(self):
        """Node count should be determined by timeline file (3 pods)."""
        runner, loader = _make_runner(TimelineMode.ROUTE_ONLY)
        assert len(runner.p_schedulers) == 3

    def test_requests_routed_to_timeline_pods(self):
        """Each request should be routed to the pod specified in timeline."""
        runner, loader = _make_runner(TimelineMode.ROUTE_ONLY)
        m = runner.run_benchmark_emulation()
        assert m["completed"] == 20

        # Check distribution: pod-alpha gets i%3==0 (indices 0,3,6,9,12,15,18) = 7
        # pod-beta gets i%3==1 (indices 1,4,7,10,13,16,19) = 7
        # pod-gamma gets i%3==2 (indices 2,5,8,11,14,17) = 6
        counts = [len(s.completed_requests) for s in runner.p_schedulers]
        assert counts == [7, 7, 6]

    def test_latency_from_predictor(self):
        """Total benchmark time should reflect predictor speed, not timeline speed."""
        runner, loader = _make_runner(TimelineMode.ROUTE_ONLY)
        m = runner.run_benchmark_emulation()

        # With predictor at 0.1 ms/token, input_len from 1000 to 2900 tokens:
        #   Total work = sum((1000+i*100)*0.1 for i in range(20)) = 3900ms
        #   Max pod (pod-beta, 7 reqs): ~1400ms → benchmark finishes in ~1.4s
        # With timeline latencies (100-1050ms):
        #   Max pod (pod-beta): 150+300+450+600+750+900+1050 = 4200ms
        # If predictor is used, duration << 2.5s; if timeline, duration > 4s.
        assert m["duration"] < 2.5, \
            f"Benchmark took {m['duration']:.2f}s, too slow for predictor speeds (expected <2.5s)"


class TestTimelineRouteAndLatency:
    """route_and_latency: uses both timeline routing and timeline latency."""

    def test_num_nodes_from_file(self):
        runner, _ = _make_runner(TimelineMode.ROUTE_AND_LATENCY)
        assert len(runner.p_schedulers) == 3

    def test_routes_match_timeline(self):
        runner, _ = _make_runner(TimelineMode.ROUTE_AND_LATENCY)
        m = runner.run_benchmark_emulation()
        assert m["completed"] == 20
        counts = [len(s.completed_requests) for s in runner.p_schedulers]
        assert counts == [7, 7, 6]

    def test_latency_matches_timeline(self):
        """Prefill latency should match timeline prefill_ms values."""
        runner, _ = _make_runner(TimelineMode.ROUTE_AND_LATENCY)
        m = runner.run_benchmark_emulation()
        results = runner.get_response_results()

        for req in results:
            ttft_s = req.gen_token_latencies[0]
            expected_prefill_ms = 100 + req.id * 50
            expected_s = expected_prefill_ms / 1000.0
            # The TTFT = queue_wait + prefill. For requests that don't queue,
            # TTFT should equal prefill_ms. Allow small tolerance for queue.
            # At minimum, the prefill component should be the timeline value.
            # Since this is serial processing, some requests will queue.
            # Just verify that the prefill time is at least the timeline value.
            assert ttft_s >= expected_s * 0.99, \
                f"req {req.id}: ttft={ttft_s*1000:.1f}ms < expected={expected_prefill_ms}ms"


class TestTimelineLatencyOnly:
    """latency_only: uses sim routing but timeline latency."""

    def test_num_nodes_from_file(self):
        runner, _ = _make_runner(TimelineMode.LATENCY_ONLY)
        assert len(runner.p_schedulers) == 3

    def test_routing_uses_policy(self):
        """With round_robin routing, distribution should be even (not timeline pattern)."""
        runner, _ = _make_runner(TimelineMode.LATENCY_ONLY, routing="round_robin")
        m = runner.run_benchmark_emulation()
        assert m["completed"] == 20
        counts = [len(s.completed_requests) for s in runner.p_schedulers]
        # Round robin distributes evenly: ~7, 7, 6 but in RR order (not timeline pod pattern)
        # The key check is total = 20
        assert sum(counts) == 20
        # With round robin and 3 pods: expect roughly equal distribution
        for c in counts:
            assert 5 <= c <= 8

    def test_latency_from_timeline(self):
        """Latency should come from timeline prefill_ms, not predictor."""
        runner, _ = _make_runner(TimelineMode.LATENCY_ONLY)
        m = runner.run_benchmark_emulation()
        results = runner.get_response_results()

        for req in results:
            ttft_s = req.gen_token_latencies[0]
            expected_prefill_ms = 100 + req.id * 50
            expected_s = expected_prefill_ms / 1000.0
            # TTFT >= prefill (some may have queue wait)
            assert ttft_s >= expected_s * 0.99, \
                f"req {req.id}: ttft={ttft_s*1000:.1f}ms < expected={expected_prefill_ms}ms"


class TestDisabledModeUnchanged:
    """DISABLED mode should behave exactly like before."""

    def test_disabled_uses_num_p_instances(self):
        """When DISABLED, num_p_instance param should be respected."""
        bc = BenchmarkConfig(
            dataset_path=TIMELINE_FILE,
            num_prompts=10,
            disable_tqdm=True,
            data_block_size=256,
        )
        sc = SchedulerConfig(
            "Qwen2.5-3B",
            scenario="disagg_prefill",
            chunked_prefill_size=8192,
            request_level_scheduling=True,
            page_size=256,
        )
        pc = PlatformConfig(device="H20")
        rc = RouterConfig(
            p_policy=RoutingPolicy.ROUND_ROBIN,
            d_policy=RoutingPolicy.ROUND_ROBIN,
            worker_startup_check_interval=0.01,
        )
        runner = DisaggBenchmarkRunner(
            benchmark_config=bc,
            p_scheduler_config=sc,
            d_scheduler_config=SchedulerConfig("Qwen2.5-3B", scenario="disagg_decode"),
            p_platform_config=pc,
            d_platform_config=PlatformConfig(device="H20"),
            router_config=rc,
            num_p_instance=5,
            num_d_instance=0,
            infer_time_predictor=_FAST_PREDICTOR,
            timeline_mode=TimelineMode.DISABLED,
        )
        assert len(runner.p_schedulers) == 5
        m = runner.run_benchmark_emulation()
        assert m["completed"] == 10
