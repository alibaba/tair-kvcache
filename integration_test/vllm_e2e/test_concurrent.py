"""test_concurrent: multiple concurrent requests save/load KV verification.

Sends several distinct prompts concurrently (all prefill + save -> reference
captures), then the same prompts each with their own suffix concurrently (all
load + prefill -> loaded captures), and verifies each request's prefix KV data
matches. This exercises ReqState tracking, per-request block attribution and
async task races.

Works for both full-attention and hybrid models (selected via KVCM_E2E_MODEL).
"""

import unittest

from e2e_lib import run_e2e


class TestConcurrent(unittest.TestCase):
    def test_concurrent(self):
        run_e2e(
            scenario="concurrent",
            tp_size=1,
            num_prompts=4,
            preferred_block_size=0,
        )


if __name__ == "__main__":
    unittest.main()
