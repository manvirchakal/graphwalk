"""Smoke-test a running graphwalk container: health check, auth, and one MCP call.

    python scripts/container_smoke.py http://127.0.0.1:8080 <bearer token>

Exits non-zero on any failure. Used by CI after ``docker run``.
"""

import asyncio
import json
import sys
import urllib.error
import urllib.request
from typing import NoReturn

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client


def fail(message: str) -> NoReturn:
    sys.exit(f"container smoke test failed: {message}")


def check_health(base: str) -> None:
    with urllib.request.urlopen(f"{base}/healthz", timeout=5) as response:  # noqa: S310
        body = json.loads(response.read())
    if body.get("status") != "ok":
        fail(f"health check returned {body}")


def check_auth_required(base: str) -> None:
    request = urllib.request.Request(  # noqa: S310
        f"{base}/mcp", data=b"{}", headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        urllib.request.urlopen(request, timeout=5)  # noqa: S310
    except urllib.error.HTTPError as error:
        if error.code != 401:  # noqa: PLR2004
            fail(f"an unauthenticated request got {error.code}, expected 401")
        return
    fail("an unauthenticated request was accepted")


async def check_mcp(base: str, token: str) -> None:
    http = httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=30)
    async with Client(streamable_http_client(f"{base}/mcp", http_client=http)) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        if not {"locate", "read", "status", "ingest"} <= tools:
            fail(f"missing tools: {tools}")
        result = await client.call_tool("status", {})
        if result.is_error:
            fail(f"status failed: {result.content}")


def main() -> None:
    base, token = sys.argv[1].rstrip("/"), sys.argv[2]
    check_health(base)
    check_auth_required(base)
    asyncio.run(check_mcp(base, token))
    print("container smoke test passed")  # noqa: T201


if __name__ == "__main__":
    main()
