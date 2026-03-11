#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys
from typing import Optional, Sequence
import runpy
import time

import aiohttp


_ORIG_AIOHTTP_REQUEST = None


def install_aiohttp_json_hijack(
    *,
    hijack_url_regex: Optional[str],
) -> None:
    global _ORIG_AIOHTTP_REQUEST
    if _ORIG_AIOHTTP_REQUEST is not None:
        return

    pattern = re.compile(hijack_url_regex) if hijack_url_regex else None
    _ORIG_AIOHTTP_REQUEST = aiohttp.ClientSession._request

    async def _patched_request(self, method, url, **kwargs):
        if pattern.search(url):
            payload = kwargs.get("json", None)
            if isinstance(payload, dict):
                if "sampling_params" not in payload:
                    payload["sample_params"] = {}
                if "custom_params" not in payload["sampling_params"]:
                    payload["sampling_params"]["custom_params"] = {}
                payload["sampling_params"]["custom_params"]["client_created_time"] = (
                    time.time()
                )
                kwargs["json"] = payload

        return await _ORIG_AIOHTTP_REQUEST(self, method, url, **kwargs)

    aiohttp.ClientSession._request = _patched_request


def main(argv: Sequence[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hijack-module", type=str, default="hisim.simulation.bench_serving"
    )
    args, bench_args = p.parse_known_args(argv)

    install_aiohttp_json_hijack(hijack_url_regex=r"/v1/(chat/)?completions|generate$")

    try:
        sys.argv = [sys.argv[0]] + list(bench_args)
        runpy.run_module(args.hijack_module, run_name="__main__")
    finally:
        sys.argv = sys.argv

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
