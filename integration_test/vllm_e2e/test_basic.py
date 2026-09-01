"""test_basic: single-request save/load KV verification (TP=1).

Sends one prompt (prefill + save -> reference capture), then the same prompt
with a suffix (load + prefill -> loaded capture), and verifies the prefix KV
data matches bit-exactly.

Works for both full-attention and hybrid models (selected via KVCM_E2E_MODEL);
see e2e_lib.run_e2e for the per-model orchestration differences.
"""

import unittest

from e2e_lib import run_e2e


class TestBasic(unittest.TestCase):
    def test_basic(self):
        run_e2e(
            scenario="basic",
            tp_size=1,
            num_prompts=1,
            preferred_block_size=0,  # manager block size == vllm block size
        )


if __name__ == "__main__":
    unittest.main()
