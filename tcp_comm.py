"""TCP communication module for robot → laptop sign detection data.

Protocol
--------
8-byte big-endian payload per detection event:
  bytes 0-3  uint32  class_id    (0-43, GTSRB class index)
  bytes 4-7  uint32  confidence  max(0, int(score_sum))

score_sum is the sum of raw FINN output logits for the winning class
over the temporal-filter window (default 10 frames).  Observed range:
0 – ~5000 for typical indoor/outdoor scenes.  Clamped to [0, 2^32-1]
before packing so the field is always a valid uint32.

Server = robot side (RobotServer).
Client = laptop side — see laptop_recv() helper.

Differences from the original Communication.py
-----------------------------------------------
  - SO_REUSEADDR on the server socket.
  - accept() uses a short timeout (0.1 s) so callers can check a
    stop_event without blocking.
  - Reconnection: try_send() clears the stale client and the next
    call to try_accept() / try_send() will accept a new one.
  - BrokenPipeError, ConnectionResetError and OSError are caught
    so a disconnected client never crashes the server.
  - All sockets are closed in a finally block.
  - recv_exact() reads exactly n bytes, handling fragmentation.
  - Port is configurable (default: 50007 from original).
  - encode_detection() / decode_detection() are explicit helpers.
"""
from __future__ import annotations

import socket
import struct
from collections.abc import Generator

DEFAULT_PORT = 50007
_FORMAT = "!II"          # big-endian, two uint32
PAYLOAD_SIZE = struct.calcsize(_FORMAT)   # 8 bytes


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------

def encode_detection(class_id: int, confidence: int) -> bytes:
    """Pack (class_id, confidence) into an 8-byte big-endian payload."""
    return struct.pack(_FORMAT, max(0, class_id), max(0, confidence))


def decode_detection(data: bytes) -> tuple[int, int]:
    """Unpack 8-byte payload -> (class_id, confidence).

    Raises ValueError if len(data) != PAYLOAD_SIZE.
    """
    if len(data) != PAYLOAD_SIZE:
        raise ValueError(f"Expected {PAYLOAD_SIZE} bytes, got {len(data)}")
    return struct.unpack(_FORMAT, data)


def recv_exact(sock: socket.socket, n: int, timeout_s: float = 5.0) -> None | bytes:
    """Receive exactly n bytes from sock, handling fragmentation.

    Returns None if the connection is closed before n bytes arrive or
    if the timeout fires.  Does not raise on normal disconnections.
    """
    orig_timeout = sock.gettimeout()
    sock.settimeout(timeout_s)
    buf = bytearray()
    try:
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
    except (TimeoutError, OSError):
        return None
    finally:
        sock.settimeout(orig_timeout)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Robot-side server
# ---------------------------------------------------------------------------

class RobotServer:
    """TCP server running on the robot; sends detections to a laptop client.

    Designed for use inside comm_process (single-threaded, event-loop style):

        srv = RobotServer(port=50007)
        srv.start()
        try:
            while not stop_event.is_set():
                srv.try_accept()          # non-blocking, 0.1 s window
                if have_event:
                    srv.try_send(cls, conf)   # no-op if no client
        finally:
            srv.close()
    """

    def __init__(self, port: int = DEFAULT_PORT):
        self._port = port
        self._server: socket.socket | None = None
        self._client: socket.socket | None = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Bind and listen.  Call once before the event loop."""
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("", self._port))
        self._server.listen(1)
        self._server.settimeout(0.1)    # makes accept() non-blocking enough
        print(f"[tcp] Listening on :{self._port}", flush=True)

    # ------------------------------------------------------------------
    def try_accept(self) -> bool:
        """Try to accept one pending connection without blocking long.

        Returns True if a new client was accepted.  If a client was
        already connected, it is closed before the new one is stored.
        """
        if self._server is None:
            return False
        try:
            conn, addr = self._server.accept()
            # Close old client cleanly
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass
            self._client = conn
            self._client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[tcp] Client connected from {addr}", flush=True)
            return True
        except socket.timeout:
            return False
        except OSError:
            return False

    # ------------------------------------------------------------------
    def try_send(self, class_id: int, confidence: int) -> bool:
        """Send one detection payload.

        If no client is connected, tries try_accept() first.
        Returns True on success, False otherwise.
        Silently drops the payload if no client is available.
        """
        if self._client is None:
            self.try_accept()
        if self._client is None:
            return False

        payload = encode_detection(class_id, confidence)
        try:
            self._client.sendall(payload)
            return True
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            print(f"[tcp] Client disconnected: {exc}", flush=True)
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            return False

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close all sockets.  Safe to call multiple times."""
        for s in (self._client, self._server):
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
        self._client = None
        self._server = None

    @property
    def has_client(self) -> bool:
        return self._client is not None


# ---------------------------------------------------------------------------
# Laptop-side helpers
# ---------------------------------------------------------------------------

def laptop_recv(
    robot_ip: str,
    port: int = DEFAULT_PORT,
    n_events: int = 0,
) -> Generator[tuple[int, int], None, None]:
    """Connect to the robot and yield (class_id, confidence) tuples.

    Parameters
    ----------
    robot_ip : IP address of the robot
    port     : TCP port (default 50007)
    n_events : stop after this many events; 0 = run until connection closes
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((robot_ip, port))
    count = 0
    try:
        while n_events == 0 or count < n_events:
            data = recv_exact(s, PAYLOAD_SIZE)
            if data is None:
                break
            yield decode_detection(data)
            count += 1
    finally:
        s.close()
