"""Benchmark configuration from environment variables."""

import os


class BenchmarkConfig:
    """All benchmark parameters, populated from environment variables with BENCH_ prefix."""

    def __init__(self):
        # Optimizer server address (host:port)
        self.optimizer_address = os.getenv("BENCH_OPTIMIZER_ADDRESS", "127.0.0.1:8082")

        # Instance identification
        self.instance_id = os.getenv("BENCH_INSTANCE_ID", "benchmark-instance-0")
        self.instance_group = os.getenv("BENCH_INSTANCE_GROUP", "benchmark-group")

        # Load parameters
        self.qps = int(os.getenv("BENCH_QPS", "100"))
        self.thread_count = int(os.getenv("BENCH_THREAD_COUNT", "4"))
        self.block_keys_count = int(os.getenv("BENCH_BLOCK_KEYS_COUNT", "10"))
        self.max_block_key = int(os.getenv("BENCH_MAX_BLOCK_KEY", "-1"))
        self.prefix_reuse_ratio = float(os.getenv("BENCH_PREFIX_REUSE_RATIO", "0.5"))
        # Prefix reuse length: 0 = random length, >0 = fixed reuse length
        self.prefix_reuse_length = int(os.getenv("BENCH_PREFIX_REUSE_LENGTH", "0"))

        # Timing
        self.duration_seconds = int(os.getenv("BENCH_DURATION_SECONDS", "60"))
        self.warmup_seconds = int(os.getenv("BENCH_WARMUP_SECONDS", "5"))
        self.report_interval = int(os.getenv("BENCH_REPORT_INTERVAL", "10"))

        # Instance setup parameters
        self.block_size = int(os.getenv("BENCH_BLOCK_SIZE", "16"))
        self.capacity_gb = float(os.getenv("BENCH_CAPACITY_GB", "1.0"))
        self.indexer_type = os.getenv("BENCH_INDEXER_TYPE", "bst_lru")
        self.max_key_count = int(os.getenv("BENCH_MAX_KEY_COUNT", "0"))

        # Benchmark mode: "setup_and_run" | "run_only" | "setup_only"
        self.mode = os.getenv("BENCH_MODE", "setup_and_run")

        # HTTP connection pool
        self.connection_timeout = float(os.getenv("BENCH_CONNECTION_TIMEOUT", "5.0"))
        self.request_timeout = float(os.getenv("BENCH_REQUEST_TIMEOUT", "10.0"))

    @property
    def base_url(self) -> str:
        address = self.optimizer_address
        if not address.startswith("http"):
            address = f"http://{address}"
        return address

    def __str__(self) -> str:
        lines = [
            "=== Benchmark Configuration ===",
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
            f"  capacity_gb:         {self.capacity_gb}",
            f"  indexer_type:        {self.indexer_type}",
            f"  max_key_count:       {self.max_key_count}",
            f"  mode:                {self.mode}",
            f"  connection_timeout:  {self.connection_timeout}",
            f"  request_timeout:     {self.request_timeout}",
        ]
        return "\n".join(lines)
