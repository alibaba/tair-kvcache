import re
import subprocess
import sys
import unittest


class SdkSymbolsTest(unittest.TestCase):
    def test_public_api_is_exported_without_dependency_symbols(self):
        output = subprocess.check_output(
            ["nm", "-D", "--defined-only", SDK_PATH], text=True
        )
        symbols = [line.split()[-1] for line in output.splitlines() if line.strip()]
        allowed = re.compile(
            r"(?:_ZNK?16kv_cache_manager.*|_ZT[VISTC]N16kv_cache_manager.*|"
            r"kvcm_client_additional_memory_registration_version)\Z"
        )
        self.assertTrue(symbols)
        self.assertEqual([], [symbol for symbol in symbols if not allowed.fullmatch(symbol)])
        self.assertIn("kvcm_client_additional_memory_registration_version", symbols)
        demangled = subprocess.check_output(
            ["nm", "-D", "--defined-only", "-C", SDK_PATH], text=True
        )
        for client in ("ManagerClient", "MetaClient", "TransferClient", "RTPLLMClient"):
            self.assertIn(f"kv_cache_manager::{client}::Create(", demangled)


if __name__ == "__main__":
    SDK_PATH = sys.argv.pop(1)
    unittest.main()
