"""test_tp: TP=2 save/load KV verification with a non-trivial block translation.

Runs the save/load verification under tensor parallelism (TP=2), where each rank
has an independent forward context, slot mapping and capture, and the connector's
ZMQ-based TP coordination is fully exercised.

For full-attention models it also sets ``preferred_block_size=32`` while vLLM
uses its default block size (16), forcing the connector's manager-block <->
group-block translation (``_attn_token_indices``) to do real cross-block
mapping -- the code path most prone to symmetric save/load bugs. For hybrid
models the manager block size is pinned to the scheduler block size (mamba state
is per scheduler block), so run_e2e ignores preferred_block_size there.

Works for both full-attention and hybrid models (selected via KVCM_E2E_MODEL).
"""

import unittest

from e2e_lib import run_e2e


class TestTp(unittest.TestCase):
    def test_tp(self):
        run_e2e(
            scenario="tp",
            tp_size=2,
            num_prompts=2,
            preferred_block_size=32,  # != vllm block size (16) for full-attn models
        )


if __name__ == "__main__":
    unittest.main()
