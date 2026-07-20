from __future__ import annotations

import subscriber.__main__
from subscriber.cli import cli as real_cli


def test_main_module_exposes_cli() -> None:
    assert hasattr(subscriber.__main__, "cli")


def test_main_module_cli_is_the_real_cli() -> None:
    assert subscriber.__main__.cli is real_cli


def test_main_module_cli_is_callable() -> None:
    assert callable(subscriber.__main__.cli)
