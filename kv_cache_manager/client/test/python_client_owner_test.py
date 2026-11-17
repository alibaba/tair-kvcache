import gc
import unittest
import weakref

from kv_cache_manager.client.pybind import kvcm_owner_lifetime_test as native


class Owner:
    pass


class PythonClientOwnerTest(unittest.TestCase):
    def check_lifetime(self, cycle):
        owner = Owner()
        reference = weakref.ref(owner)
        events = []
        client = native.create(owner, events)
        client.registration_owner = owner
        if cycle:
            client.cycle = client
        else:
            # Model GC clearing attributes before destroying the native holder.
            client.__dict__.clear()
        del owner
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertEqual(events, [])
        del client
        gc.collect()
        self.assertEqual(events, [("close", True), ("destroy", True)])
        self.assertIsNone(reference())

    def test_host_owner_survives_cleared_dynamic_attributes(self):
        self.check_lifetime(cycle=False)

    def test_host_owner_survives_native_holder_cyclic_gc(self):
        self.check_lifetime(cycle=True)


if __name__ == "__main__":
    unittest.main()
