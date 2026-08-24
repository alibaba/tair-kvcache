import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from kv_cache_manager.py_connector.dashtrace.instance_bootstrap import (
    InstanceBootstrapConfig,
    ensure_instance_registered,
)


class _Handler(BaseHTTPRequestHandler):
    registered = False
    register_requests = 0

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        if self.path == "/api/getInstanceInfo":
            code = "OK" if self.registered else "INSTANCE_NOT_EXIST"
        elif self.path == "/api/registerInstance":
            self.__class__.registered = True
            self.__class__.register_requests += 1
            code = "OK" if request.get("instance_id") == "instance-1" else "ERROR"
        else:
            self.send_error(404)
            return
        payload = json.dumps({"header": {"status": {"code": code}}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_):
        pass


class InstanceBootstrapTest(unittest.TestCase):
    def setUp(self):
        _Handler.registered = False
        _Handler.register_requests = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_registers_missing_instance_and_sets_readiness(self):
        ready = threading.Event()
        stopping = threading.Event()
        ensure_instance_registered(
            InstanceBootstrapConfig(
                base_url=f"http://127.0.0.1:{self.server.server_port}",
                registration={"instance_id": "instance-1", "instance_group": "default"},
                timeout_seconds=2,
                retry_interval_seconds=0.01,
            ),
            ready,
            stopping,
        )
        self.assertTrue(ready.is_set())
        self.assertEqual(1, _Handler.register_requests)

    def test_existing_instance_does_not_register_again(self):
        _Handler.registered = True
        ready = threading.Event()
        ensure_instance_registered(
            InstanceBootstrapConfig(
                base_url=f"http://127.0.0.1:{self.server.server_port}",
                registration={"instance_id": "instance-1"},
                timeout_seconds=2,
                retry_interval_seconds=0.01,
            ),
            ready,
            threading.Event(),
        )
        self.assertTrue(ready.is_set())
        self.assertEqual(0, _Handler.register_requests)


if __name__ == "__main__":
    unittest.main()
