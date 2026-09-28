"""Guards on the test harness itself: the default run must be offline."""

import asyncio
import socket

import pytest
import pytest_socket


@pytest.mark.filterwarnings("ignore:A test tried to use socket.socket")
def test_inet_sockets_are_blocked_by_default() -> None:
    with pytest.raises(pytest_socket.SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


async def test_async_tests_run_with_sockets_blocked() -> None:
    await asyncio.sleep(0)
    assert asyncio.get_running_loop().is_running()


def _open_inet_socket() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.close()


@pytest.mark.live
def test_live_tests_get_network_access() -> None:
    _open_inet_socket()  # only runs with --run-live; would raise if sockets stayed blocked


@pytest.mark.neo4j
def test_neo4j_tests_get_network_access() -> None:
    _open_inet_socket()  # only runs with --run-neo4j
