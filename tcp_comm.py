"""Best-effort non-blocking TCP server for robot telemetry.

Each message is encoded as one UTF-8 JSON object followed by '\\n'.
"""

from __future__ import annotations

import json
import socket


class RobotServer:
    def __init__(self, port, host="0.0.0.0"):
        self.host = host
        self.port = port
        self._server = None
        self._client = None
        self._client_address = None

    @property
    def has_client(self):
        return self._client is not None

    @property
    def client_address(self):
        return self._client_address

    def start(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((self.host, self.port))
        self._server.listen(1)
        self._server.setblocking(False)
        print(f"[tcp] listening on {self.host}:{self.port}", flush=True)

    def try_accept(self):
        if self._server is None or self.has_client:
            return False

        try:
            client, address = self._server.accept()
        except BlockingIOError:
            return False
        except OSError:
            return False

        client.setblocking(False)
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._client = client
        self._client_address = address

        print(
            f"[tcp] client connected: {address[0]}:{address[1]}",
            flush=True,
        )
        return True

    def try_send(self, payload):
        """Best-effort send of one newline-delimited JSON message."""
        if self._client is None:
            return False

        data = (json.dumps(payload, separators=(",", ":")) + "\\n").encode("utf-8")

        try:
            sent = self._client.send(data)
            if sent != len(data):
                self._disconnect_client()
                return False
            return True
        except BlockingIOError:
            return False
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._disconnect_client()
            return False

    def _disconnect_client(self):
        if self._client is not None:
            try:
                self._client.close()
            except OSError:
                pass

        if self._client_address is not None:
            print(
                f"[tcp] client disconnected: "
                f"{self._client_address[0]}:{self._client_address[1]}",
                flush=True,
            )

        self._client = None
        self._client_address = None

    def close(self):
        self._disconnect_client()

        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass

        self._server = None
