"""Benchmark configuration from environment variables."""

import os


class BenchmarkConfig:
    """All benchmark parameters populated from BENCH_* environment variables."""

    def __init__(self):
        self.protocol = os.getenv("BENCH_PROTOCOL", "grpc").lower()
        self.optimizer_address = os.getenv("BENCH_OPTIMIZER_ADDRESS", "127.0.0.1:50052")

        self.instance_id = os.getenv("BENCH_INSTANCE_ID", "benchmark-instance-0")
        self.instance_group = os.getenv("BENCH_INSTANCE_GROUP", "benchmark-group")

        self.qps = int(os.getenv("BENCH_QPS", "100"))
        self.thread_count = int(os.getenv("BENCH_THREAD_COUNT", "4"))
        self.block_keys_count = int(os.getenv("BENCH_BLOCK_KEYS_COUNT", "10"))
        self.max_block_key = int(os.getenv("BENCH_MAX_BLOCK_KEY", "-1"))
        self.prefix_reuse_ratio = float(os.getenv("BENCH_PREFIX_REUSE_RATIO", "0.5"))
        self.prefix_reuse_length = int(os.getenv("BENCH_PREFIX_REUSE_LENGTH", "0"))

        self.duration_seconds = int(os.getenv("BENCH_DURATION_SECONDS", "60"))
        self.warmup_seconds = int(os.getenv("BENCH_WARMUP_SECONDS", "5"))
        self.report_interval = int(os.getenv("BENCH_REPORT_INTERVAL", "10"))

        self.block_size = int(os.getenv("BENCH_BLOCK_SIZE", "16"))
        self.capacity_gb_list = [
            float(x.strip()) for x in os.getenv("BENCH_CAPACITY_GB", "1.0").split(",") if x.strip()
        ] or [1.0]
        self.block_bytes = int(os.getenv("BENCH_BLOCK_BYTES", "0"))
        self.eviction_policy = os.getenv("BENCH_EVICTION_POLICY", "OPTIMIZER_EVICTION_POLICY_LRU")
        self.shared_group_quota = os.getenv("BENCH_SHARED_GROUP_QUOTA", "false").lower() in ("true", "1", "yes")
        self.ttl_seconds = int(os.getenv("BENCH_TTL_SECONDS", "0"))
        self.enable_theoretical_max_cache = os.getenv(
            "BENCH_ENABLE_THEORETICAL_MAX_CACHE", "true").lower() in ("true", "1", "yes")

        self.mode = os.getenv("BENCH_MODE", "setup_and_run")

        self.connection_timeout = float(os.getenv("BENCH_CONNECTION_TIMEOUT", "5.0"))
        self.request_timeout = float(os.getenv("BENCH_REQUEST_TIMEOUT", "10.0"))

        self.trace_data_dir = os.getenv("BENCH_TRACE_DATA_DIR", "")
        self.trace_speed_factor = float(os.getenv("BENCH_TRACE_SPEED_FACTOR", "1.0"))
        self.trace_loop = os.getenv("BENCH_TRACE_LOOP", "false").lower() in ("true", "1", "yes")
        self.trace_loop_count = int(os.getenv("BENCH_TRACE_LOOP_COUNT", "0"))
        self.trace_max_requests = int(os.getenv("BENCH_TRACE_MAX_REQUESTS", "0"))

        # A positive loop count means the benchmark must loop to reach that count.
        if self.trace_loop_count > 0:
            self.trace_loop = True

    def __str__(self) -> str:
        lines = [
            "=== Benchmark Configuration ===",
            f"  protocol:            {self.protocol}",
            f"  optimizer_address:   {self.optimizer_address}",
            f"  instance_id:         {self.instance_id}",
            f"  instance_group:      {self.instance_group}",
            f"  qps:                 {self.qps}",
            f"  thread_count:        {self.thread_count}",
            f"  block_keys_count:    {self.block_keys_count}",
            f"  max_block_key:       {self.max_block_key} (-1=unlimited)",
            f"  prefix_reuse_ratio:  {self.prefix_reuse_ratio}",
            f"  prefix_reuse_length: {self.prefix_reuse_length} (0=random)",
            f"  duration_seconds:    {self.duration_seconds}",
            f"  warmup_seconds:      {self.warmup_seconds}",
            f"  report_interval:     {self.report_interval}",
            f"  block_size:          {self.block_size}",
            f"  capacity_gb:         {self.capacity_gb_list}",
            f"  block_bytes:         {self.block_bytes} (0=placeholder/unlimited)",
            f"  eviction_policy:     {self.eviction_policy}",
            f"  shared_group_quota:  {self.shared_group_quota}",
            f"  ttl_seconds:         {self.ttl_seconds}",
            f"  enable_theoretical_max_cache: {self.enable_theoretical_max_cache}",
            f"  mode:                {self.mode}",
            f"  connection_timeout:  {self.connection_timeout}",
            f"  request_timeout:     {self.request_timeout}",
            f"  trace_data_dir:      {self.trace_data_dir}",
            f"  trace_speed_factor:  {self.trace_speed_factor}",
            f"  trace_loop:          {self.trace_loop}",
            f"  trace_loop_count:    {self.trace_loop_count} (0=unlimited)",
            f"  trace_max_requests:  {self.trace_max_requests} (0=unlimited)",
        ]
        return "\n".join(lines)
