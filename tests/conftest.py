from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator

import pytest

EXTERNAL_NETWORK_ERROR = "External network access is disabled during tests."


def _is_loopback_address(sock: socket.socket, address: object) -> bool:
    unix_family = getattr(socket, "AF_UNIX", None)
    if unix_family is not None and sock.family == unix_family:
        return True
    if not isinstance(address, tuple) or not address:
        return False
    host = address[0]
    if isinstance(host, bytes):
        try:
            host = host.decode("ascii")
        except UnicodeDecodeError:
            return False
    if not isinstance(host, str):
        return False
    if host.casefold() == "localhost":
        return True
    normalized = host.split("%", 1)[0]
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def prohibit_real_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Permit local test infrastructure while rejecting every external connection."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def guarded_connect(sock: socket.socket, address: object) -> None:
        if not _is_loopback_address(sock, address):
            raise AssertionError(EXTERNAL_NETWORK_ERROR)
        original_connect(sock, address)  # type: ignore[arg-type]

    def guarded_connect_ex(sock: socket.socket, address: object) -> int:
        if not _is_loopback_address(sock, address):
            raise AssertionError(EXTERNAL_NETWORK_ERROR)
        return original_connect_ex(sock, address)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    yield
