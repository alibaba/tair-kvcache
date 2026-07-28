"""test_mutation: meta-test proving the e2e KV verification is not vacuous.

Runs the standard basic scenario with ``MutatedConnector`` (defined in
test_connector.py, injected only through the test-side
``kv_connector_module_path``), which shifts every attention slot produced by
``_attn_token_indices`` by one -- a symmetric off-by-one: save gathers token
t's KV from the shifted slot and load scatters it back there, so with
contiguous block tables a transport round trip cancels the bug in the interior
of the loaded range. It cannot cancel at the range boundary: one loaded
token's true slot is never written and keeps stale uninitialized data. The
capture comparison reads the cache through vLLM's own slot mapping and must
observe the divergence; run_e2e(expect_verification_failure=True) asserts the
verification FAILS. If the mutated run verifies clean, the harness is blind
and this test fails.
"""

import unittest

from e2e_lib import run_e2e


class TestMutation(unittest.TestCase):
    def test_mutated_connector_is_caught(self):
        run_e2e(
            scenario="mutation",
            tp_size=1,
            num_prompts=1,
            preferred_block_size=0,
            connector_name="MutatedConnector",
            expect_verification_failure=True,
        )


if __name__ == "__main__":
    unittest.main()
