"""
py_tpu_client_test.py — Python-level TPU client test using unittest + py_tpu_client binding.

Usage:
    bazel test --config=client_with_tpu //kv_cache_manager/client/src/internal/sdk/test:py_tpu_client_test
"""

import ctypes
import ctypes.util
import unittest

import numpy as np

import py_tpu_client as tc


def _aligned_alloc(size: int, alignment: int = 32):
    """Allocate aligned memory via libc posix_memalign. Returns (ptr, free_fn)."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t]
    libc.posix_memalign.restype = ctypes.c_int
    libc.free.argtypes = [ctypes.c_void_p]

    aligned_size = (size + alignment - 1) & ~(alignment - 1)
    ptr = ctypes.c_void_p()
    ret = libc.posix_memalign(ctypes.byref(ptr), alignment, aligned_size)
    if ret != 0:
        raise OSError(f"posix_memalign failed with {ret}")
    return ptr.value, libc.free


class TpuClientTest(unittest.TestCase):
    """Base test class — creates and initializes TpuClient in setUp."""

    # Set to True by test_0_jax_interaction to skip setUp init in later tests
    _jax_loaded_first = False

    def setUp(self):
        self.client = tc.TpuClient()
        # setUp runs before each test method.
        # For test_0_jax_interaction, init is deferred until after JAX import.
        if self._testMethodName != "test_0_jax_interaction":
            ec = self.client.init()
            self.assertEqual(ec, tc.ER_OK, "TpuClient.init() failed")

    def tearDown(self):
        self.client.destroy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assert_ok(self, ec, msg=""):
        self.assertEqual(ec, tc.ER_OK, f"{msg}: got {ec}")

    def _buffer_roundtrip(self, data: bytes) -> bytes:
        """Helper: H2D → D2H round-trip, return received bytes."""
        size = len(data)
        src = (ctypes.c_char * size)(*data)
        ec, buf = self.client.buffer_from_host(ctypes.addressof(src), size)
        self._assert_ok(ec, "buffer_from_host")
        self.assertNotEqual(buf, 0, "buffer ptr should not be null")

        dst = (ctypes.c_char * size)()
        ec = self.client.buffer_to_host(buf, ctypes.addressof(dst), size)
        self._assert_ok(ec, "buffer_to_host")

        self.client.destroy_buffer(buf)
        return bytes(dst)

    # ------------------------------------------------------------------
    # Tests — alphabetical order; test_0_jax_interaction runs first
    # to import JAX before TpuClient loads libtpu.
    # ------------------------------------------------------------------

    def test_0_jax_interaction(self):
        """JAX + TpuClient coexist: JAX loads libtpu first, TpuClient reuses via RTLD_NOLOAD."""
        # Import JAX BEFORE client.init() so JAX loads libtpu first
        import jax
        import jax.numpy as jnp

        devices = jax.devices()
        self.assertGreater(len(devices), 0, "No JAX devices found")

        x = jnp.arange(8, dtype=jnp.float32)
        self.assertEqual(len(x), 8)

        # Now init TpuClient — should detect libtpu already loaded via RTLD_NOLOAD
        ec = self.client.init()
        self.assertEqual(ec, tc.ER_OK, "TpuClient.init() after JAX failed")
        TpuClientTest._jax_loaded_first = True
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        ec, buf = self.client.buffer_from_host(data.ctypes.data, data.nbytes)
        self._assert_ok(ec, "buffer_from_host (with JAX)")

        dst = np.zeros(4, dtype=np.float32)
        ec = self.client.buffer_to_host(buf, dst.ctypes.data, data.nbytes)
        self._assert_ok(ec, "buffer_to_host (with JAX)")
        np.testing.assert_array_equal(data, dst)

        self.client.destroy_buffer(buf)

    def test_basic_roundtrip(self):
        """Synchronous BufferFromHost / BufferToHost with bytes."""
        data = b"TpuClient Python round-trip test!!!"
        result = self._buffer_roundtrip(data)
        self.assertEqual(result, data)

    def test_numpy_roundtrip(self):
        """Round-trip with numpy float32 array."""
        arr = np.arange(128, dtype=np.float32)
        ec, buf = self.client.buffer_from_host(arr.ctypes.data, arr.nbytes)
        self._assert_ok(ec, "buffer_from_host (numpy)")

        dst = np.zeros(128, dtype=np.float32)
        ec = self.client.buffer_to_host(buf, dst.ctypes.data, arr.nbytes)
        self._assert_ok(ec, "buffer_to_host (numpy)")
        np.testing.assert_array_equal(arr, dst)

        self.client.destroy_buffer(buf)

    def test_async_h2d(self):
        """Async BufferFromHostAsync + WaitEvent."""
        data = b"Async round-trip test data!!!!!!!!!"
        size = len(data)
        src = (ctypes.c_char * size)(*data)

        ec, buf, ev = self.client.buffer_from_host_async(ctypes.addressof(src), size)
        self._assert_ok(ec, "buffer_from_host_async")
        self._assert_ok(self.client.wait_event(ev), "wait_event (H2D)")

        dst = (ctypes.c_char * size)()
        ec = self.client.buffer_to_host(buf, ctypes.addressof(dst), size)
        self._assert_ok(ec, "buffer_to_host")
        self.assertEqual(bytes(dst), data)

        self.client.destroy_buffer(buf)

    def test_async_d2h(self):
        """Async BufferToHostAsync + WaitEvent."""
        data = b"Async D2H test data!!!!!!!!!!!!!!!!"
        size = len(data)
        src = (ctypes.c_char * size)(*data)

        ec, buf = self.client.buffer_from_host(ctypes.addressof(src), size)
        self._assert_ok(ec, "buffer_from_host")

        dst = (ctypes.c_char * size)()
        ec, ev = self.client.buffer_to_host_async(buf, ctypes.addressof(dst), size)
        self._assert_ok(ec, "buffer_to_host_async")
        self._assert_ok(self.client.wait_event(ev), "wait_event (D2H)")
        self.assertEqual(bytes(dst), data)

        self.client.destroy_buffer(buf)

    def test_large_buffer(self):
        """64KB numpy buffer round-trip."""
        arr = np.arange(16384, dtype=np.float32)  # 64KB
        ec, buf = self.client.buffer_from_host(arr.ctypes.data, arr.nbytes)
        self._assert_ok(ec, "buffer_from_host (64KB)")

        dst = np.zeros_like(arr)
        ec = self.client.buffer_to_host(buf, dst.ctypes.data, arr.nbytes)
        self._assert_ok(ec, "buffer_to_host (64KB)")
        np.testing.assert_array_equal(arr, dst)

        self.client.destroy_buffer(buf)

    def test_dma_map(self):
        """DmaMap / DmaUnmap."""
        buf_ptr, free_fn = _aligned_alloc(4096)
        try:
            self._assert_ok(self.client.dma_map(buf_ptr, 4096), "dma_map")
            self._assert_ok(self.client.dma_unmap(buf_ptr), "dma_unmap")
        finally:
            free_fn(buf_ptr)

    def test_raw_buffer(self):
        """RawBuffer extension: CreateRawAlias + RawBufferToHost."""
        if not self.client.has_raw_buffer_extension():
            self.skipTest("RawBuffer extension not available")

        arr = np.arange(256, dtype=np.uint8)
        ec, buf = self.client.buffer_from_host(arr.ctypes.data, arr.nbytes)
        self._assert_ok(ec, "buffer_from_host")

        ec, raw = self.client.create_raw_alias(buf)
        self._assert_ok(ec, "create_raw_alias")

        ec, dev_size = self.client.raw_buffer_get_device_size(raw)
        self._assert_ok(ec, "raw_buffer_get_device_size")
        self.assertGreaterEqual(dev_size, arr.nbytes,
                                "device size should be >= logical size")

        # Raw D2H — requires 32-byte aligned host buffer
        aligned_ptr, free_fn = _aligned_alloc(dev_size)
        try:
            ec, ev = self.client.raw_buffer_to_host(raw, aligned_ptr, 0, dev_size)
            self._assert_ok(ec, "raw_buffer_to_host")
            self._assert_ok(self.client.wait_event(ev), "wait_event (raw D2H)")

            raw_bytes = (ctypes.c_char * dev_size).from_address(aligned_ptr)
            has_nonzero = any(raw_bytes[i] != 0 for i in range(arr.nbytes))
            self.assertTrue(has_nonzero, "RawBuffer D2H returned all zeros")
        finally:
            free_fn(aligned_ptr)

        self.client.destroy_raw_buffer(raw)
        self.client.destroy_buffer(buf)

    def test_jax_array_extract_buffer(self):
        """Extract PJRT_Buffer* handle from a jax.Array."""
        import jax
        import jax.numpy as jnp

        x = jnp.arange(8, dtype=jnp.float32)
        ec, buf_ptr = self.client.extract_buffer_from_jax_array(x)
        self._assert_ok(ec, "extract_buffer_from_jax_array")
        self.assertNotEqual(buf_ptr, 0, "buffer ptr should not be null")

    def test_jax_array_d2h(self):
        """D2H transfer from jax.Array device buffer to host via TpuClient."""
        import jax
        import jax.numpy as jnp

        expected = np.arange(128, dtype=np.float32)
        x = jnp.array(expected)

        dst = np.zeros(128, dtype=np.float32)
        ec = self.client.buffer_to_host_from_jax(x, dst.ctypes.data, dst.nbytes)
        self._assert_ok(ec, "buffer_to_host_from_jax")
        np.testing.assert_array_equal(expected, dst)

    def test_jax_array_h2d_d2h_roundtrip(self):
        """H2D via TpuClient + D2H from jax.Array to verify data on device."""
        import jax
        import jax.numpy as jnp

        # Create data on host
        src_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                            dtype=np.float32)

        # H2D: create a new PJRT_Buffer on TPU via TpuClient
        ec, buf = self.client.buffer_from_host(src_data.ctypes.data, src_data.nbytes)
        self._assert_ok(ec, "buffer_from_host")

        # D2H: read back via TpuClient's BufferToHost
        dst = np.zeros(8, dtype=np.float32)
        ec = self.client.buffer_to_host(buf, dst.ctypes.data, dst.nbytes)
        self._assert_ok(ec, "buffer_to_host")
        np.testing.assert_array_equal(src_data, dst)

        self.client.destroy_buffer(buf)


def _ordered_suite():
    """Build a test suite with test_0_jax_interaction running first."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    # Add tests in a fixed order; test_0_jax_interaction must be first
    # so JAX loads libtpu before subsequent TpuClient tests.
    test_names = [
        "test_0_jax_interaction",
        "test_basic_roundtrip",
        "test_numpy_roundtrip",
        "test_async_h2d",
        "test_async_d2h",
        "test_large_buffer",
        "test_dma_map",
        "test_raw_buffer",
        "test_jax_array_extract_buffer",
        "test_jax_array_d2h",
        "test_jax_array_h2d_d2h_roundtrip",
    ]
    suite.addTests(loader.loadTestsFromNames(test_names, TpuClientTest))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(_ordered_suite())
