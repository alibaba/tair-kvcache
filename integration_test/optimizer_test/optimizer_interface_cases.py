"""Optimizer Service integration test cases and client abstraction.

This module defines:
- OptimizerServiceClientBase: abstract client interface for optimizer service
- OptimizerTestBase: test base class that manages optimizer server lifecycle
- OptimizerServiceTestCases: shared test cases for both HTTP and gRPC clients
"""

import abc
import json
import logging
import os
import signal
import socket
import subprocess
import time
import unittest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# --- Optimizer Service Client Abstract Base ---


class OptimizerServiceClientBase(abc.ABC):
    """Abstract base class for optimizer service client (HTTP or gRPC)."""

    @abc.abstractmethod
    def create_instance_group(self, data, check_response=True) -> dict:
        pass

    @abc.abstractmethod
    def update_instance_group(self, data, check_response=True) -> dict:
        pass

    @abc.abstractmethod
    def remove_instance_group(self, data, check_response=True) -> dict:
        pass

    @abc.abstractmethod
    def get_instance_group(self, data, check_response=True) -> dict:
        pass

    @abc.abstractmethod
    def list_instance_groups(self, data, check_response=True) -> dict:
        pass

    @abc.abstractmethod
    def register_instance(self, data, check_response=True) -> dict:
        pass

    @abc.abstractmethod
    def remove_instance(self, data, check_response=True) -> dict:
        pass

    @abc.abstractmethod
    def get_instance(self, data, check_response=True) -> dict:
        pass

    @abc.abstractmethod
    def trace_query(self, data, check_response=True) -> dict:
        pass

    @abc.abstractmethod
    def list_instances(self, data, check_response=True) -> dict:
        pass

    @abc.abstractmethod
    def reset_stats(self, data, check_response=True) -> dict:
        pass

    @abc.abstractmethod
    def close(self):
        pass


# --- Optimizer Server Manager ---


def _find_free_ports(n=2):
    """Allocate *n* distinct free ports by holding all sockets open simultaneously."""
    sockets = []
    ports = []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('127.0.0.1', 0))
        sockets.append(s)
        ports.append(s.getsockname()[1])
    for s in sockets:
        s.close()
    return ports


def _find_binary():
    """Find the optimizer server binary."""
    # Check common Bazel output paths
    candidates = [
        os.path.join(os.environ.get('TEST_SRCDIR', ''),
                     os.environ.get('TEST_WORKSPACE', ''),
                     'kv_cache_manager/optimizer/online_optimizer_server_main'),
        # Runfiles path
        'kv_cache_manager/optimizer/online_optimizer_server_main',
    ]

    # Also check relative to this test file for local dev
    test_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(test_dir, '../../'))
    candidates.append(os.path.join(
        workspace_root, 'bazel-bin/kv_cache_manager/optimizer/online_optimizer_server_main'))

    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    # Try to find via runfiles
    runfiles_dir = os.environ.get('RUNFILES_DIR', '')
    if runfiles_dir:
        workspace = os.environ.get('TEST_WORKSPACE', '')
        path = os.path.join(runfiles_dir, workspace,
                            'kv_cache_manager/optimizer/online_optimizer_server_main')
        if os.path.isfile(path):
            return path

    return None


class OptimizerServerManager:
    """Manages the lifecycle of an optimizer server process for testing."""

    def __init__(self):
        self.rpc_port = 0
        self.http_port = 0
        self._process = None
        self._config_path = None
        self._workdir = None

    def start(self, binary_path=None):
        """Start the optimizer server with dynamic ports."""
        self.rpc_port, self.http_port = _find_free_ports(2)

        if binary_path is None:
            binary_path = _find_binary()
        if binary_path is None:
            raise RuntimeError("Cannot find online_optimizer_server_main binary")

        # Create workdir
        tmpdir = os.environ.get('TEST_TMPDIR', '/tmp')
        self._workdir = os.path.join(tmpdir, f'optimizer_integ_{os.getpid()}')
        os.makedirs(self._workdir, exist_ok=True)

        # Write config
        self._config_path = os.path.join(self._workdir, 'config.json')
        config = {
            "rpc_port": self.rpc_port,
            "http_port": self.http_port,
            "registry_storage_uri": "",
            "metrics_report_interval_ms": 0,
            "enable_prometheus": False,
            "io_thread_num": 2,
        }
        with open(self._config_path, 'w') as f:
            json.dump(config, f)

        logging.info(f"Starting optimizer server: rpc_port={self.rpc_port}, "
                     f"http_port={self.http_port}, binary={binary_path}")

        # Start server process
        self._process = subprocess.Popen(
            [binary_path, '-c', self._config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self._workdir,
        )

        # Wait for server to be ready
        if not self._wait_for_ready(timeout=10):
            self.stop()
            raise RuntimeError("Optimizer server failed to start within 10 seconds")

        logging.info(f"Optimizer server started, pid={self._process.pid}")

    def _wait_for_ready(self, timeout=10):
        """Wait for the server to accept connections."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process.poll() is not None:
                stdout = self._process.stdout.read().decode() if self._process.stdout else ""
                stderr = self._process.stderr.read().decode() if self._process.stderr else ""
                logging.error(f"Server process exited: code={self._process.returncode}")
                logging.error(f"stdout: {stdout[:2000]}")
                logging.error(f"stderr: {stderr[:2000]}")
                return False
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                sock.connect(('127.0.0.1', self.http_port))
                sock.close()
                return True
            except (ConnectionRefusedError, socket.timeout, OSError):
                time.sleep(0.2)
        return False

    def stop(self):
        """Stop the optimizer server."""
        if self._process and self._process.poll() is None:
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            logging.info(f"Optimizer server stopped, pid={self._process.pid}")
        self._process = None


# --- Test Base Class ---


class OptimizerTestBase(abc.ABC, unittest.TestCase):
    """Base class for optimizer service integration tests.

    Manages server lifecycle and provides common test utilities.
    """

    _server_manager = None  # Shared across all tests in the suite

    @classmethod
    def setUpClass(cls):
        """Start the optimizer server once for all tests."""
        cls._server_manager = OptimizerServerManager()
        cls._server_manager.start()

    @classmethod
    def tearDownClass(cls):
        """Stop the optimizer server after all tests."""
        if cls._server_manager:
            cls._server_manager.stop()
            cls._server_manager = None

    @abc.abstractmethod
    def _create_client(self) -> OptimizerServiceClientBase:
        """Create a client instance (HTTP or gRPC)."""
        pass

    def setUp(self):
        self._client = self._create_client()
        self._trace_id = f"test_{self._testMethodName}_{int(time.time())}"

    def tearDown(self):
        self._client.close()


# --- Shared Test Cases ---


class OptimizerServiceTestCases(OptimizerTestBase):
    """Shared test cases for optimizer service, to be mixed into HTTP/gRPC tests."""

    # --- InstanceGroup CRUD Tests ---

    def test_create_and_get_instance_group(self):
        """Test creating and retrieving an instance group."""
        group_name = f"test_group_{self._trace_id}"
        create_req = {
            "trace_id": self._trace_id,
            "instance_group": {
                "name": group_name,
                "eviction_policy": "OPTIMIZER_EVICTION_POLICY_LRU",
                "capacity_gb": [1.0, 2.0],
            }
        }
        resp = self._client.create_instance_group(create_req)
        self.assertIn("header", resp)
        self.assertEqual(resp["header"]["status"]["code"], "OK")

        # Get and verify
        get_resp = self._client.get_instance_group({
            "trace_id": self._trace_id,
            "name": group_name,
        })
        self.assertEqual(get_resp["header"]["status"]["code"], "OK")
        self.assertEqual(get_resp["instance_group"]["name"], group_name)
        self.assertEqual(get_resp["instance_group"]["eviction_policy"], "OPTIMIZER_EVICTION_POLICY_LRU")
        self.assertEqual(len(get_resp["instance_group"]["capacity_gb"]), 2)

    def test_update_instance_group(self):
        """Test updating an instance group."""
        group_name = f"test_update_grp_{self._trace_id}"
        # Create
        self._client.create_instance_group({
            "trace_id": self._trace_id,
            "instance_group": {
                "name": group_name,
                "eviction_policy": "OPTIMIZER_EVICTION_POLICY_LRU",
                "capacity_gb": [1.0],
            }
        })
        # Update
        resp = self._client.update_instance_group({
            "trace_id": self._trace_id,
            "instance_group": {
                "name": group_name,
                "eviction_policy": "OPTIMIZER_EVICTION_POLICY_LRU",
                "capacity_gb": [2.0, 4.0],
            }
        })
        self.assertEqual(resp["header"]["status"]["code"], "OK")

        # Verify
        get_resp = self._client.get_instance_group({
            "trace_id": self._trace_id,
            "name": group_name,
        })
        self.assertEqual(get_resp["instance_group"]["eviction_policy"], "OPTIMIZER_EVICTION_POLICY_LRU")
        self.assertEqual(len(get_resp["instance_group"]["capacity_gb"]), 2)

    def test_remove_instance_group(self):
        """Test removing an instance group."""
        group_name = f"test_rm_grp_{self._trace_id}"
        # Create
        self._client.create_instance_group({
            "trace_id": self._trace_id,
            "instance_group": {
                "name": group_name,
                "eviction_policy": "OPTIMIZER_EVICTION_POLICY_LRU",
                "capacity_gb": [1.0],
            }
        })
        # Remove
        resp = self._client.remove_instance_group({
            "trace_id": self._trace_id,
            "name": group_name,
        })
        self.assertEqual(resp["header"]["status"]["code"], "OK")

        # Verify removed
        get_resp = self._client.get_instance_group({
            "trace_id": self._trace_id,
            "name": group_name,
        }, check_response=False)
        self.assertNotEqual(get_resp["header"]["status"]["code"], "OK")

    def test_list_instance_groups(self):
        """Test listing instance groups."""
        group_name = f"test_list_grp_{self._trace_id}"
        self._client.create_instance_group({
            "trace_id": self._trace_id,
            "instance_group": {
                "name": group_name,
                "eviction_policy": "OPTIMIZER_EVICTION_POLICY_LRU",
                "capacity_gb": [1.0],
            }
        })

        resp = self._client.list_instance_groups({"trace_id": self._trace_id})
        self.assertEqual(resp["header"]["status"]["code"], "OK")
        self.assertIn("instance_groups", resp)
        found = any(g["name"] == group_name for g in resp["instance_groups"])
        self.assertTrue(found, f"Group {group_name} not found in list")

    def test_create_group_with_empty_name_fails(self):
        """Empty group name should be rejected."""
        resp = self._client.create_instance_group({
            "trace_id": self._trace_id,
            "instance_group": {
                "name": "",
                "eviction_policy": "OPTIMIZER_EVICTION_POLICY_LRU",
                "capacity_gb": [1.0],
            }
        }, check_response=False)
        self.assertNotEqual(resp["header"]["status"]["code"], "OK")

    def test_get_nonexistent_group(self):
        """Getting a non-existent group should fail."""
        resp = self._client.get_instance_group({
            "trace_id": self._trace_id,
            "name": "totally_nonexistent_group_xyz",
        }, check_response=False)
        self.assertNotEqual(resp["header"]["status"]["code"], "OK")

    # --- Instance Registration Tests ---

    def _create_group_and_register(self, group_name, instance_id,
                                   block_size=1024, capacity_gb=1.0,
                                   linear_step=1):
        """Helper: create group then register instance."""
        self._client.create_instance_group({
            "trace_id": self._trace_id,
            "instance_group": {
                "name": group_name,
                "eviction_policy": "OPTIMIZER_EVICTION_POLICY_LRU",
                "capacity_gb": [capacity_gb],
            }
        })
        reg_resp = self._client.register_instance({
            "trace_id": self._trace_id,
            "instance_group": group_name,
            "instance_id": instance_id,
            "block_size": block_size,
            "linear_step": linear_step,
            "location_spec_infos": [{"name": "full", "size": block_size}],
            "location_spec_groups": [{"name": "full_group", "spec_names": ["full"]}],
            "optimizer_state_info": {"full_location_spec_group_name": "full_group"},
        })
        return reg_resp

    def test_register_instance(self):
        """Test registering an instance."""
        group_name = f"test_reg_grp_{self._trace_id}"
        inst_id = f"test_reg_inst_{self._trace_id}"
        resp = self._create_group_and_register(group_name, inst_id)
        self.assertEqual(resp["header"]["status"]["code"], "OK")
        self.assertIn("estimated_capacity_blocks", resp)
        self.assertGreater(len(resp["estimated_capacity_blocks"]), 0)

    def test_register_without_group_fails(self):
        """Registering in a non-existent group should fail."""
        resp = self._client.register_instance({
            "trace_id": self._trace_id,
            "instance_group": "nonexistent_group_xyz",
            "instance_id": "some_inst",
            "block_size": 1024,
            "location_spec_infos": [{"name": "full", "size": 1024}],
        }, check_response=False)
        self.assertNotEqual(resp["header"]["status"]["code"], "OK")

    def test_register_with_empty_instance_id_fails(self):
        """Empty instance_id should be rejected."""
        group_name = f"test_empty_id_grp_{self._trace_id}"
        self._client.create_instance_group({
            "trace_id": self._trace_id,
            "instance_group": {
                "name": group_name,
                "eviction_policy": "OPTIMIZER_EVICTION_POLICY_LRU",
                "capacity_gb": [1.0],
            }
        })
        resp = self._client.register_instance({
            "trace_id": self._trace_id,
            "instance_group": group_name,
            "instance_id": "",
            "block_size": 1024,
            "location_spec_infos": [{"name": "full", "size": 1024}],
        }, check_response=False)
        self.assertNotEqual(resp["header"]["status"]["code"], "OK")

    def test_register_with_empty_specs_fails(self):
        """Empty location_spec_infos should be rejected."""
        group_name = f"test_empty_specs_grp_{self._trace_id}"
        self._client.create_instance_group({
            "trace_id": self._trace_id,
            "instance_group": {
                "name": group_name,
                "eviction_policy": "OPTIMIZER_EVICTION_POLICY_LRU",
                "capacity_gb": [1.0],
            }
        })
        resp = self._client.register_instance({
            "trace_id": self._trace_id,
            "instance_group": group_name,
            "instance_id": "inst_no_specs",
            "block_size": 1024,
            "location_spec_infos": [],
        }, check_response=False)
        self.assertNotEqual(resp["header"]["status"]["code"], "OK")

    def test_get_instance(self):
        """Test GetInstance returns correct details."""
        group_name = f"test_getinst_grp_{self._trace_id}"
        inst_id = f"test_getinst_{self._trace_id}"
        self._create_group_and_register(group_name, inst_id, block_size=2048, linear_step=3)

        resp = self._client.get_instance({
            "trace_id": self._trace_id,
            "instance_id": inst_id,
        })
        self.assertEqual(resp["header"]["status"]["code"], "OK")
        self.assertEqual(resp["instance_id"], inst_id)
        self.assertEqual(resp["instance_group"], group_name)
        self.assertEqual(resp["block_size"], 2048)
        self.assertEqual(resp["linear_step"], 3)

    def test_remove_instance(self):
        """Test removing an instance."""
        group_name = f"test_rminst_grp_{self._trace_id}"
        inst_id = f"test_rminst_{self._trace_id}"
        self._create_group_and_register(group_name, inst_id)

        resp = self._client.remove_instance({
            "trace_id": self._trace_id,
            "instance_id": inst_id,
        })
        self.assertEqual(resp["header"]["status"]["code"], "OK")

        # Verify removed
        get_resp = self._client.get_instance({
            "trace_id": self._trace_id,
            "instance_id": inst_id,
        }, check_response=False)
        self.assertNotEqual(get_resp["header"]["status"]["code"], "OK")

    def test_remove_nonexistent_instance_fails(self):
        """Removing non-existent instance should fail."""
        resp = self._client.remove_instance({
            "trace_id": self._trace_id,
            "instance_id": "totally_nonexistent_inst",
        }, check_response=False)
        self.assertNotEqual(resp["header"]["status"]["code"], "OK")

    def test_list_instances(self):
        """Test listing instances by group."""
        group_name = f"test_listinst_grp_{self._trace_id}"
        inst_id = f"test_listinst_{self._trace_id}"
        self._create_group_and_register(group_name, inst_id)

        resp = self._client.list_instances({
            "trace_id": self._trace_id,
            "instance_group": group_name,
        })
        self.assertEqual(resp["header"]["status"]["code"], "OK")
        self.assertIn("instances", resp)
        found = any(i["instance_id"] == inst_id for i in resp["instances"])
        self.assertTrue(found, f"Instance {inst_id} not found in list")

    # --- TraceQuery Tests ---

    def test_trace_query_miss_and_hit(self):
        """Test TraceQuery: first query misses, second query hits."""
        group_name = f"test_tq_grp_{self._trace_id}"
        inst_id = f"test_tq_{self._trace_id}"
        self._create_group_and_register(group_name, inst_id)

        # First query - all miss
        resp1 = self._client.trace_query({
            "trace_id": self._trace_id,
            "instance_id": inst_id,
            "block_keys": [100, 200, 300],
        })
        self.assertEqual(resp1["header"]["status"]["code"], "OK")
        self.assertEqual(int(resp1["capacity_results"][0]["cache_hit_count"]), 0)
        self.assertEqual(int(resp1["total_blocks"]), 3)

        # Second query - same keys, all hit
        resp2 = self._client.trace_query({
            "trace_id": self._trace_id,
            "instance_id": inst_id,
            "block_keys": [100, 200, 300],
        })
        self.assertEqual(resp2["header"]["status"]["code"], "OK")
        self.assertEqual(int(resp2["capacity_results"][0]["cache_hit_count"]), 3)
        self.assertEqual(int(resp2["total_blocks"]), 3)

    def test_trace_query_mixed(self):
        """Test TraceQuery with mixed hit/miss."""
        group_name = f"test_tqm_grp_{self._trace_id}"
        inst_id = f"test_tqm_{self._trace_id}"
        self._create_group_and_register(group_name, inst_id)

        # Populate cache
        self._client.trace_query({
            "trace_id": self._trace_id,
            "instance_id": inst_id,
            "block_keys": [1, 2, 3],
        })

        # Mixed: key 1 hits, keys 4,5 miss
        resp = self._client.trace_query({
            "trace_id": self._trace_id,
            "instance_id": inst_id,
            "block_keys": [1, 4, 5],
        })
        self.assertEqual(resp["header"]["status"]["code"], "OK")
        self.assertEqual(int(resp["capacity_results"][0]["cache_hit_count"]), 1)
        self.assertEqual(int(resp["total_blocks"]), 3)

    def test_trace_query_nonexistent_instance(self):
        """TraceQuery on non-existent instance should fail."""
        resp = self._client.trace_query({
            "trace_id": self._trace_id,
            "instance_id": "nonexistent_inst_xyz",
            "block_keys": [1],
        }, check_response=False)
        self.assertNotEqual(resp["header"]["status"]["code"], "OK")

    def test_trace_query_unique_keys(self):
        """Test that unique keys are tracked correctly."""
        group_name = f"test_uk_grp_{self._trace_id}"
        inst_id = f"test_uk_{self._trace_id}"
        self._create_group_and_register(group_name, inst_id)

        resp = self._client.trace_query({
            "trace_id": self._trace_id,
            "instance_id": inst_id,
            "block_keys": [10, 20, 30],
        })
        self.assertEqual(int(resp["capacity_results"][0]["current_unique_keys"]), 3)

        # Add overlapping keys
        resp = self._client.trace_query({
            "trace_id": self._trace_id,
            "instance_id": inst_id,
            "block_keys": [20, 30, 40],
        })
        self.assertEqual(int(resp["capacity_results"][0]["current_unique_keys"]), 4)

    # --- ResetStats Tests ---

    def test_reset_stats(self):
        """Test resetting instance stats."""
        group_name = f"test_reset_grp_{self._trace_id}"
        inst_id = f"test_reset_{self._trace_id}"
        self._create_group_and_register(group_name, inst_id)

        # Generate some stats
        self._client.trace_query({
            "trace_id": self._trace_id,
            "instance_id": inst_id,
            "block_keys": [1, 2, 3],
        })

        # Reset
        resp = self._client.reset_stats({
            "trace_id": self._trace_id,
            "instance_id": inst_id,
        })
        self.assertEqual(resp["header"]["status"]["code"], "OK")

        # Verify stats cleared via list
        list_resp = self._client.list_instances({
            "trace_id": self._trace_id,
            "instance_group": group_name,
        })
        for inst in list_resp.get("instances", []):
            if inst["instance_id"] == inst_id:
                self.assertEqual(int(inst["total_queries"]), 0)
                self.assertEqual(int(inst["total_blocks_queried"]), 0)

    def test_reset_stats_nonexistent_fails(self):
        """Resetting stats for non-existent instance should fail."""
        resp = self._client.reset_stats({
            "trace_id": self._trace_id,
            "instance_id": "nonexistent_reset_inst",
        }, check_response=False)
        self.assertNotEqual(resp["header"]["status"]["code"], "OK")

    # --- Instance Isolation Tests ---

    def test_instance_isolation(self):
        """Data in one instance should not affect another."""
        grp_a = f"test_iso_grpA_{self._trace_id}"
        grp_b = f"test_iso_grpB_{self._trace_id}"
        inst_a = f"test_iso_A_{self._trace_id}"
        inst_b = f"test_iso_B_{self._trace_id}"

        self._create_group_and_register(grp_a, inst_a)
        self._create_group_and_register(grp_b, inst_b)

        # Populate instance A
        self._client.trace_query({
            "trace_id": self._trace_id,
            "instance_id": inst_a,
            "block_keys": [1000, 2000],
        })

        # Query instance B with same keys - should miss
        resp = self._client.trace_query({
            "trace_id": self._trace_id,
            "instance_id": inst_b,
            "block_keys": [1000, 2000],
        })
        self.assertEqual(int(resp["capacity_results"][0]["cache_hit_count"]), 0)

    def test_list_instances_filter_by_group(self):
        """ListInstances should filter by group."""
        grp_a = f"test_filter_grpA_{self._trace_id}"
        grp_b = f"test_filter_grpB_{self._trace_id}"
        inst_a = f"test_filter_A_{self._trace_id}"
        inst_b = f"test_filter_B_{self._trace_id}"

        self._create_group_and_register(grp_a, inst_a)
        self._create_group_and_register(grp_b, inst_b)

        # List group A only
        resp_a = self._client.list_instances({
            "trace_id": self._trace_id,
            "instance_group": grp_a,
        })
        for inst in resp_a.get("instances", []):
            self.assertEqual(inst["instance_group"], grp_a)

        # List all (empty filter)
        resp_all = self._client.list_instances({
            "trace_id": self._trace_id,
            "instance_group": "",
        })
        ids = [i["instance_id"] for i in resp_all.get("instances", [])]
        self.assertIn(inst_a, ids)
        self.assertIn(inst_b, ids)

    # --- Full Lifecycle Test ---

    def test_full_lifecycle(self):
        """Test complete lifecycle: register -> query -> list -> reset -> remove."""
        group_name = f"test_lc_grp_{self._trace_id}"
        inst_id = f"test_lc_{self._trace_id}"

        # 1. Register
        reg_resp = self._create_group_and_register(group_name, inst_id)
        self.assertEqual(reg_resp["header"]["status"]["code"], "OK")

        # 2. TraceQuery (miss)
        tq1 = self._client.trace_query({
            "trace_id": self._trace_id,
            "instance_id": inst_id,
            "block_keys": [1, 2, 3, 4, 5],
        })
        self.assertEqual(int(tq1["capacity_results"][0]["cache_hit_count"]), 0)
        self.assertEqual(int(tq1["total_blocks"]), 5)

        # 3. TraceQuery (hit)
        tq2 = self._client.trace_query({
            "trace_id": self._trace_id,
            "instance_id": inst_id,
            "block_keys": [1, 2, 3, 4, 5],
        })
        self.assertEqual(int(tq2["capacity_results"][0]["cache_hit_count"]), 5)

        # 4. List and verify stats
        list_resp = self._client.list_instances({
            "trace_id": self._trace_id,
            "instance_group": group_name,
        })
        found = False
        for inst in list_resp.get("instances", []):
            if inst["instance_id"] == inst_id:
                found = True
                self.assertEqual(int(inst["total_queries"]), 2)
                self.assertEqual(int(inst["total_blocks_queried"]), 10)
                self.assertEqual(int(inst["debug_info"]["unique_keys"]), 5)
                # Check capacity_summaries
                self.assertGreater(len(inst.get("capacity_summaries", [])), 0)
                self.assertEqual(int(inst["capacity_summaries"][0]["total_hits"]), 5)
        self.assertTrue(found, "Instance not found in list")

        # 5. Reset stats
        self._client.reset_stats({
            "trace_id": self._trace_id,
            "instance_id": inst_id,
        })

        # 6. Verify cleared
        list_resp2 = self._client.list_instances({
            "trace_id": self._trace_id,
            "instance_group": group_name,
        })
        for inst in list_resp2.get("instances", []):
            if inst["instance_id"] == inst_id:
                self.assertEqual(int(inst["total_queries"]), 0)

        # 7. Remove
        rm_resp = self._client.remove_instance({
            "trace_id": self._trace_id,
            "instance_id": inst_id,
        })
        self.assertEqual(rm_resp["header"]["status"]["code"], "OK")

        # 8. Verify removed
        tq3 = self._client.trace_query({
            "trace_id": self._trace_id,
            "instance_id": inst_id,
            "block_keys": [1],
        }, check_response=False)
        self.assertNotEqual(tq3["header"]["status"]["code"], "OK")

    # --- LinearStep Tests ---

    def test_register_with_linear_step(self):
        """Test instance with linear_step > 1."""
        group_name = f"test_ls_grp_{self._trace_id}"
        inst_id = f"test_ls_{self._trace_id}"
        self._create_group_and_register(group_name, inst_id, linear_step=4)

        resp = self._client.list_instances({
            "trace_id": self._trace_id,
            "instance_group": group_name,
        })
        for inst in resp.get("instances", []):
            if inst["instance_id"] == inst_id:
                self.assertEqual(int(inst["debug_info"]["linear_step"]), 4)

    # --- Duplicate Registration Tests ---

    def test_duplicate_register_overwrites(self):
        """Re-registering same instance should overwrite (reset state)."""
        group_name = f"test_dup_grp_{self._trace_id}"
        inst_id = f"test_dup_{self._trace_id}"
        self._create_group_and_register(group_name, inst_id)

        # Generate stats
        self._client.trace_query({
            "trace_id": self._trace_id,
            "instance_id": inst_id,
            "block_keys": [1, 2, 3],
        })

        # Re-register
        resp = self._client.register_instance({
            "trace_id": self._trace_id,
            "instance_group": group_name,
            "instance_id": inst_id,
            "block_size": 1024,
            "location_spec_infos": [{"name": "full", "size": 1024}],
            "location_spec_groups": [{"name": "full_group", "spec_names": ["full"]}],
            "optimizer_state_info": {"full_location_spec_group_name": "full_group"},
        })
        self.assertEqual(resp["header"]["status"]["code"], "OK")

        # Stats should be reset (new indexer)
        list_resp = self._client.list_instances({
            "trace_id": self._trace_id,
            "instance_group": group_name,
        })
        count = 0
        for inst in list_resp.get("instances", []):
            if inst["instance_id"] == inst_id:
                count += 1
                self.assertEqual(int(inst["total_queries"]), 0)
        self.assertEqual(count, 1, "Should have exactly one instance entry")
