from __future__ import annotations

import argparse
import asyncio

from subscriber import logger
from subscriber.config import SubscriberConfig
from subscriber.main import run


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subscriber",
        description="tair-kvcache subscriber",
    )
    parser.set_defaults(command="serve")
    parser.add_argument(
        "--config", default=None, metavar="FILE", help="Path to YAML config file"
    )
    SubscriberConfig.add_cli_args(parser)
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="Run the subscriber service")
    serve.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        metavar="FILE",
        help="Path to YAML config file",
    )
    SubscriberConfig.add_cli_args(serve, default=argparse.SUPPRESS)
    return parser


def cli() -> None:
    """Entry point for the ``subscriber`` console script."""
    parser = _build_cli_parser()
    args = parser.parse_args()
    config = SubscriberConfig.from_args(args)
    logger.init("tair-kvcache-subscriber", level=config.log_level)
    for key in config.unknown_config_keys:
        logger.warning(
            "ignoring unknown config file key",
            step="config",
            tags={"key": key},
        )
    try:
        asyncio.run(run(config))
    except Exception as exc:
        logger.critical(
            "subscriber exited with fatal error",
            step="lifecycle",
            tags={"error": exc.__class__.__name__, "message": str(exc)},
            exc_info=True,
        )
        raise
