import unittest

import kv_cache_manager.manager_client as manager_client
from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient


class PublicApiTest(unittest.TestCase):
    def test_exports_manager_client_and_version(self):
        self.assertIs(manager_client.KvCacheManagerClient, KvCacheManagerClient)
        self.assertIsInstance(manager_client.__version__, str)
        self.assertTrue(manager_client.__version__)


if __name__ == "__main__":
    unittest.main()
