"""Fault injection on real Redis TCP traffic; forwards complete RESP frames."""

import socket
import socketserver
import threading


def _read_frame(reader):
    prefix = reader.read(1)
    if not prefix:
        raise EOFError()
    line = reader.readline()
    if not line.endswith(b"\r\n"):
        raise EOFError()
    raw = prefix + line
    value = line[:-2]
    if prefix == b"$":
        length = int(value)
        if length < 0:
            return raw, None
        body = reader.read(length + 2)
        if len(body) != length + 2:
            raise EOFError()
        return raw + body, body[:-2]
    if prefix == b"*":
        values = []
        for _ in range(int(value)):
            child_raw, child_value = _read_frame(reader)
            raw += child_raw
            values.append(child_value)
        return raw, values
    return raw, value


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.server.proxy._forward(self.request)


class RedisFaultProxy:
    def __init__(self, redis_port):
        self.redis_port = redis_port
        self._mutex = threading.Lock()
        self._sockets = set()
        self._disconnected = False
        self._next_connection = 0
        self._script_kinds = {}
        self._action = None
        self.fault_entered = threading.Event()
        self.reply_replayed = threading.Event()
        self.unlock_mismatch = threading.Event()
        self._release_reply = threading.Event()
        self._release_reply.set()
        self.fault_request = None
        self.fault_connection = None
        self.replay_connection = None
        self.noscript_count = 0
        self._server = _Server(("127.0.0.1", 0), _Handler)
        self._server.proxy = self
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05})

    @property
    def uri(self):
        return f"redis://127.0.0.1:{self.port}/"

    def start(self):
        self._thread.start()

    def disconnect(self):
        # Close established connections and reject new ones. Each retry goes
        # through the real RedisClient reconnect path and receives socket EOF.
        with self._mutex:
            self._disconnected = True
            sockets = list(self._sockets)
        self._close_sockets(sockets)

    def restore(self):
        with self._mutex:
            self._disconnected = False
            self._action = None
        self._release_reply.set()

    def stop(self):
        self._release_reply.set()
        self.disconnect()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def arm_renew_reply(self, action):
        if action not in ("drop", "hold"):
            raise ValueError(action)
        with self._mutex:
            self._action = action
            self.fault_request = None
            self.fault_connection = None
            self.replay_connection = None
            self.fault_entered.clear()
            self.reply_replayed.clear()
            self.unlock_mismatch.clear()
            self._release_reply.clear()

    @staticmethod
    def _close_sockets(sockets):
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def _observe_reply(self, connection, args, reply, value):
        with self._mutex:
            command = args[0].upper()
            if command == b"SCRIPT" and args[1].upper() == b"LOAD" and reply.startswith(b"$"):
                if b"'PTTL'" in args[2]:
                    self._script_kinds[value] = "renew"
                elif b"'DEL'" in args[2]:
                    self._script_kinds[value] = "unlock"
            if reply.startswith(b"-NOSCRIPT"):
                self.noscript_count += 1
            kind = self._script_kinds.get(args[1]) if command == b"EVALSHA" else None
            if kind == "unlock" and reply == b":-1\r\n":
                self.unlock_mismatch.set()
            if kind != "renew" or reply != b":1\r\n":
                return None
            if self.fault_request == tuple(args) and connection != self.fault_connection:
                self.replay_connection = connection
                self.reply_replayed.set()
            action = self._action
            if action is not None:
                self._action = None
                self.fault_request = tuple(args)
                self.fault_connection = connection
                self.fault_entered.set()
            return action

    def _forward(self, client):
        upstream = None
        with self._mutex:
            if self._disconnected:
                return
            self._next_connection += 1
            connection = self._next_connection
            self._sockets.add(client)
        try:
            upstream = socket.create_connection(("127.0.0.1", self.redis_port), timeout=2)
            upstream.settimeout(None)
            with self._mutex:
                self._sockets.add(upstream)
                if self._disconnected:
                    return
            with client.makefile("rb") as requests, upstream.makefile("rb") as responses:
                while True:
                    raw, args = _read_frame(requests)
                    upstream.sendall(raw)
                    reply, value = _read_frame(responses)
                    # The fault happens after Redis has executed the renewal.
                    # Dropping it forces replay; holding it models lost return
                    # traffic while LeaseCheck must revoke independently.
                    action = self._observe_reply(connection, args, reply, value)
                    if action == "drop":
                        return
                    if action == "hold" and not self._release_reply.wait(timeout=30):
                        raise TimeoutError("held Redis reply was not released")
                    client.sendall(reply)
        except (EOFError, OSError):
            pass
        finally:
            sockets = [client] + ([upstream] if upstream is not None else [])
            with self._mutex:
                self._sockets.difference_update(sockets)
            self._close_sockets(sockets)
