from __future__ import annotations

import asyncio
import socket

import pytest


def test_external_ipv4_is_rejected_by_connect_and_connect_ex() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        with pytest.raises(AssertionError, match="External network access is disabled"):
            client.connect(("198.51.100.1", 443))
        with pytest.raises(AssertionError, match="External network access is disabled"):
            client.connect_ex(("198.51.100.1", 443))


def test_external_ipv6_is_rejected() -> None:
    try:
        client = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    except OSError:
        pytest.skip("IPv6 sockets are unavailable in this environment")
    with client, pytest.raises(AssertionError, match="External network access is disabled"):
        client.connect(("2001:db8::1", 443))


def test_localhost_connection_is_not_rejected_by_network_guard() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.connect(("localhost", port))
            connection, _address = server.accept()
            connection.close()


def test_asyncio_event_loop_can_be_created_and_closed() -> None:
    async def run_once() -> str:
        await asyncio.sleep(0)
        return "ok"

    assert asyncio.run(run_once()) == "ok"
