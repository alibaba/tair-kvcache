"""Benchmark entry point for online_optimizer TraceQuery."""

import sys
import logging

from .config import BenchmarkConfig
from .client import OptimizerClient
from .setup import setup_instance, teardown_instance
from .stats import StatsCollector
from .runner import BenchmarkRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    config = BenchmarkConfig()
    logger.info("\n%s", config)

    client = OptimizerClient(config)

    try:
        if config.mode in ("setup_and_run", "setup_only"):
            setup_instance(client, config)

        if config.mode == "setup_only":
            logger.info("Setup complete. Exiting (mode=setup_only).")
            return

        stats = StatsCollector(report_interval=config.report_interval)
        runner = BenchmarkRunner(config, client, stats)
        runner.run()

        if config.mode == "setup_and_run":
            teardown_instance(client, config)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception:
        logger.exception("Benchmark failed")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()
