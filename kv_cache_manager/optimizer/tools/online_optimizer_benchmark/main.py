"""Benchmark entry point for online_optimizer TraceQuery."""

import sys
import logging

from .core.config import BenchmarkConfig
from .core.benchmark_client import create_benchmark_client
from .core.instance import setup_instance
from .core.stats import StatsCollector
from .runners.live import BenchmarkRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    config = BenchmarkConfig()
    logger.info("\n%s", config)

    client = create_benchmark_client(config)

    try:
        if config.mode in ("setup_and_run", "setup_only", "trace_replay"):
            setup_instance(client, config)

        if config.mode == "setup_only":
            logger.info("Setup complete. Exiting (mode=setup_only).")
            return

        stats = StatsCollector(report_interval=config.report_interval)

        if config.mode == "trace_replay":
            if not config.trace_data_dir:
                logger.error("BENCH_TRACE_DATA_DIR is required for trace_replay mode.")
                sys.exit(1)
            from .runners.trace_replay import TraceReplayRunner
            runner = TraceReplayRunner(config, client, stats)
        else:
            runner = BenchmarkRunner(config, client, stats)

        runner.run()

        # NOTE: teardown removed to avoid accidental deletion of production instances.
        # Use the kvcm_ops CLI to remove instances manually if needed.

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception:
        logger.exception("Benchmark failed")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
