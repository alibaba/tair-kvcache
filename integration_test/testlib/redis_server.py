import os
import shutil
import socket
import subprocess
import time


class RedisServer(object):
    def __init__(self, workdir):
        self.workdir = workdir
        self.port_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.port_socket.bind(("127.0.0.1", 0))
        self.port = self.port_socket.getsockname()[1]
        self.process = None
        self.stdout = None
        self.stderr = None

    @property
    def uri(self):
        return f"redis://127.0.0.1:{self.port}/"

    def start(self):
        redis_server = os.environ.get("REDIS_SERVER_BIN") or shutil.which("redis-server")
        if redis_server is None:
            raise RuntimeError("redis-server is required for this integration test; set REDIS_SERVER_BIN or PATH")

        redis_dir = os.path.join(self.workdir, "redis")
        os.makedirs(redis_dir, exist_ok=True)
        conf_path = os.path.join(redis_dir, "redis.conf")
        with open(conf_path, "w") as f:
            f.write("\n".join([
                f"port {self.port}",
                "bind 127.0.0.1",
                "protected-mode no",
                "save \"\"",
                "appendonly no",
                "daemonize no",
                f"dir {redis_dir}",
                "loglevel warning",
                "",
            ]))

        self.port_socket.close()
        self.port_socket = None
        self.stdout = open(os.path.join(redis_dir, "stdout"), "w")
        self.stderr = open(os.path.join(redis_dir, "stderr"), "w")
        self.process = subprocess.Popen([redis_server, conf_path], stdout=self.stdout, stderr=self.stderr)
        self._wait_until_ready()
        self.command("FLUSHALL")

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.process = None
        if self.stdout is not None:
            self.stdout.close()
            self.stdout = None
        if self.stderr is not None:
            self.stderr.close()
            self.stderr = None
        if self.port_socket is not None:
            self.port_socket.close()
            self.port_socket = None

    def _wait_until_ready(self):
        deadline = time.time() + 10
        last_error = None
        while time.time() < deadline:
            try:
                if self.command("PING") == "PONG":
                    return
            except Exception as e:
                last_error = e
                time.sleep(0.1)
        raise RuntimeError(f"redis-server did not become ready: {last_error}")

    def command(self, *args):
        with socket.create_connection(("127.0.0.1", self.port), timeout=2) as sock:
            payload = self._encode(args)
            sock.sendall(payload)
            reader = sock.makefile("rb")
            return self._read_reply(reader)

    def keys(self, pattern):
        result = self.command("KEYS", pattern)
        return sorted(result or [])

    def hgetall(self, key):
        result = self.command("HGETALL", key)
        if not result:
            return {}
        return dict(zip(result[0::2], result[1::2]))

    @staticmethod
    def _encode(args):
        parts = [f"*{len(args)}\r\n".encode()]
        for arg in args:
            value = str(arg).encode()
            parts.append(f"${len(value)}\r\n".encode())
            parts.append(value)
            parts.append(b"\r\n")
        return b"".join(parts)

    def _read_reply(self, reader):
        prefix = reader.read(1)
        if prefix == b"+":
            return reader.readline().rstrip(b"\r\n").decode()
        if prefix == b"-":
            raise RuntimeError(reader.readline().rstrip(b"\r\n").decode())
        if prefix == b":":
            return int(reader.readline().rstrip(b"\r\n"))
        if prefix == b"$":
            length = int(reader.readline().rstrip(b"\r\n"))
            if length < 0:
                return None
            data = reader.read(length)
            reader.read(2)
            return data.decode()
        if prefix == b"*":
            length = int(reader.readline().rstrip(b"\r\n"))
            if length < 0:
                return None
            return [self._read_reply(reader) for _ in range(length)]
        raise RuntimeError(f"unknown redis reply prefix: {prefix!r}")
