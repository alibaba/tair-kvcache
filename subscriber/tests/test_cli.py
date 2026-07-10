from __future__ import annotations

from pathlib import Path

from subscriber.cli import _build_cli_parser
from subscriber.config import SubscriberConfig


def test_cli_parser_accepts_serve_subcommand_log_level() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(["serve", "--log-level", "debug"])

    config = SubscriberConfig.from_args(args)

    assert args.command == "serve"
    assert config.log_level == "debug"


def test_cli_parser_accepts_legacy_top_level_args() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(["--log-level", "debug"])

    config = SubscriberConfig.from_args(args)

    assert config.log_level == "debug"


def test_cli_parser_preserves_top_level_log_level_before_serve() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args(["--log-level", "debug", "serve"])

    config = SubscriberConfig.from_args(args)

    assert args.command == "serve"
    assert config.log_level == "debug"


def test_cli_parser_preserves_top_level_config_before_serve(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("log_level: 'debug'\n")
    parser = _build_cli_parser()
    args = parser.parse_args(["--config", str(config_file), "serve"])

    config = SubscriberConfig.from_args(args)

    assert args.command == "serve"
    assert config.log_level == "debug"
