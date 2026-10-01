"""The MCP server, end to end in-process: every tool, header handling, key hygiene, and
each auth mode. HTTP runs through the real Starlette app via an ASGI transport, so no
sockets are opened."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

import httpx2
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from pydantic import SecretStr
from starlette.applications import Starlette
from test_index import CURIE, EINSTEIN, GOLD, PIERRE, QUERY, extract, walker

from graphwalk.config import ResolvedConfig
from graphwalk.embeddings import FakeEmbedder
from graphwalk.index import Index
from graphwalk.ingest.pipeline import IngestConfig
from graphwalk.llm import FakeLLM
from graphwalk.server.app import build_server, http_app
from graphwalk.server.auth import IntrospectionVerifier, JWTVerifier, discover_jwks_url
from graphwalk.server.service import Service
from graphwalk.server.settings import ServerSettings
from graphwalk.stores.sqlite_store import SQLiteStore

SECRET = "sk-client-SECRET-abcdef0123456789"  # noqa: S105 - a fake key
TOKEN = "t" * 40
BASE = "http://127.0.0.1:8080"
DOCS = [
    {"id": "curie", "text": CURIE, "title": "Marie Curie"},
    {"id": "pierre", "text": PIERRE, "title": "Pierre Curie"},
    {"id": "einstein", "text": EINSTEIN, "title": "Albert Einstein"},
]


class Recorder:
    """An index factory with fake backends that records each configuration it gets."""

    def __init__(self) -> None:
        self.configs: list[ResolvedConfig] = []

    def __call__(self, store: SQLiteStore, config: ResolvedConfig) -> Index:
        self.configs.append(config)
        return Index(
            store,
            config=config,
            owns_store=False,
            decider=walker(),
            llm=FakeLLM(extract),
            embedder=FakeEmbedder(),
            ingest=IngestConfig(routing="exact"),
        )


def make_service(mode: str = "stdio", **settings: Any) -> tuple[Service, Recorder]:
    recorder = Recorder()
    service = Service(
        store=SQLiteStore(":memory:"),
        settings=ServerSettings(**settings),
        mode=mode,  # type: ignore[arg-type]
        index_factory=recorder,
    )
    return service, recorder


def data(result: Any) -> Any:
    assert not result.is_error, result.content
    if result.structured_content is not None:
        content = result.structured_content
        return content.get("result", content) if isinstance(content, dict) else content
    return json.loads(result.content[0].text)


def error_text(result: Any) -> str:
    assert result.is_error
    return result.content[0].text


async def ingest_and_wait(client: Client, **arguments: Any) -> dict[str, Any]:
    job = data(await client.call_tool("ingest", arguments))
    for _ in range(200):
        status = data(await client.call_tool("ingest_status", {"job_id": job["job_id"]}))
        if status["state"] != "running":
            return status
        await asyncio.sleep(0.01)
    raise AssertionError("ingest did not finish")


# -- every tool, in process ------------------------------------------------------------


async def test_every_tool_end_to_end() -> None:
    service, _ = make_service()
    async with Client(build_server(service)) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert names == {
            "locate", "walk", "read", "neighbors", "get_node", "ingest", "ingest_status",
            "status",
        }  # fmt: skip
        done = await ingest_and_wait(client, documents=DOCS, source_id="wiki")
        assert done["state"] == "done"
        assert done["chunks_done"] == done["chunks_total"] == 3
        assert done["report"]["new"] == 3

        status = data(await client.call_tool("status", {}))
        assert status["documents"] == 3
        assert status["sources"] == {"wiki": 3}
        assert status["nodes"] > 0

        located = data(await client.call_tool("locate", {"query": QUERY, "k": 5}))
        assert located["entries"] == ["Marie Curie"]
        assert located["decision_model"] == "fake-decider-1"
        texts = []
        for loc in located["locations"]:
            args = {k: loc[k] for k in ("key", "start", "end", "doc_hash")}
            texts.append(data(await client.call_tool("read", args))["text"])
        for gold in GOLD:
            assert any(gold in t for t in texts)

        walked = data(await client.call_tool("walk", {"query": QUERY}))
        assert [e["name"] for e in walked["entries"]] == ["Marie Curie"]
        assert walked["answers"]
        best = walked["answers"][0]
        assert walked["confidence"] == pytest.approx(best["confidence"])
        assert best["path"]
        assert not walked["escalated"]
        assert walked["decision_model"] == "fake-decider-1"
        assert walked["decision_calls"] > 0
        assert "query is empty" in error_text(await client.call_tool("walk", {"query": " "}))
        first = located["locations"][0]
        assert first["path"] == ["Marie Curie --spouse_of--> Pierre Curie"]

        whole = data(await client.call_tool("read", {"key": "wiki/pierre"}))
        assert whole["text"] == PIERRE

        around = data(await client.call_tool("neighbors", {"node": "pierre curie"}))
        assert {e["relation"] for e in around["edges"]} == {"spouse_of", "born_in"}
        assert all(e["sources"] for e in around["edges"])
        (node,) = data(await client.call_tool("get_node", {"node": "Pierre Curie"}))
        assert node["type"] == "person"
        assert node["degree"] == 2


async def test_tool_errors_are_readable() -> None:
    service, _ = make_service(max_k=10, max_read_chars=100)
    async with Client(build_server(service)) as client:
        await ingest_and_wait(client, documents=DOCS, source_id="wiki")
        assert "k must be between 1 and 10" in error_text(
            await client.call_tool("locate", {"query": QUERY, "k": 11})
        )
        assert "no node" in error_text(await client.call_tool("neighbors", {"node": "Nobody"}))
        assert "no ingest job" in error_text(
            await client.call_tool("ingest_status", {"job_id": "x"})
        )
        assert "either documents or path" in error_text(await client.call_tool("ingest", {}))
        assert "not in the store" in error_text(
            await client.call_tool("read", {"key": "wiki/missing"})
        )
        long = data(await client.call_tool("read", {"key": "wiki/curie"}))
        assert long["truncated"]
        assert len(long["text"]) == 100
        dupes = [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]
        assert "unique" in error_text(await client.call_tool("ingest", {"documents": dupes}))


async def test_ingest_paths_stdio_versus_remote(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "pierre.txt").write_text(PIERRE, encoding="utf-8")
    stdio, _ = make_service()
    async with Client(build_server(stdio)) as client:
        done = await ingest_and_wait(client, path=str(folder), source_id="files")
        assert done["state"] == "done"
    remote, _ = make_service("http")
    job = remote.ingest
    with pytest.raises(Exception, match="does not ingest server paths"):
        await job({}, None, str(folder), None)
    rooted, _ = make_service("http", ingest_root=tmp_path)
    with pytest.raises(Exception, match="outside GRAPHWALK_INGEST_ROOT"):
        await rooted.ingest({}, None, "../../etc", None)
    started = await rooted.ingest({}, None, "docs", "files")
    assert started.state == "running"
    await rooted.close()


# -- HTTP: headers, keys, auth ---------------------------------------------------------


@asynccontextmanager
async def http_client(
    app: Starlette, headers: dict[str, str], *, start: bool = True
) -> AsyncIterator[Client]:
    """An MCP client over HTTP to ``app``; ``start=False`` if its lifespan already runs
    (the session manager can be started only once per app)."""
    http = httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url=BASE, headers=headers
    )
    transport = streamable_http_client(f"{BASE}/mcp", http_client=http)
    if not start:
        async with Client(transport) as client:
            yield client
        return
    async with app.router.lifespan_context(app), Client(transport) as client:
        yield client


def remote(**settings: Any) -> tuple[Starlette, Service, Recorder]:
    service, recorder = make_service("http", **settings)
    return http_app(build_server(service), service.settings), service, recorder


async def test_remote_keys_come_from_headers_only(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("OPENROUTER_API_KEY", "server-env-key")
    app, _, recorder = remote()
    headers = {
        "X-Anthropic-API-Key": SECRET,
        "X-Graphwalk-LLM-Provider": "anthropic",
        "X-OpenAI-Base-URL": "http://169.254.169.254/latest",
    }
    async with http_client(app, headers) as client:
        await ingest_and_wait(client, documents=DOCS)
        status = data(await client.call_tool("status", {}))
        located = data(await client.call_tool("locate", {"query": QUERY}))
        assert located["locations"]
    config = recorder.configs[0]
    s = config.settings
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == SECRET
    assert s.llm_provider == "anthropic"
    assert s.openrouter_api_key is None  # the server's key is not lent out by default
    assert s.openai_base_url == "https://api.openai.com/v1"  # not on the allowlist
    assert config.warnings
    assert status["config"]["keys"]["anthropic"] is True
    assert status["jobs"] == []  # remote clients do not see other clients' jobs
    assert len(recorder.configs) == 1  # one index per distinct configuration
    assert SECRET not in json.dumps(status)
    assert SECRET not in caplog.text
    assert SECRET[-8:] not in caplog.text


async def test_server_keys_are_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "server-env-key")
    app, _, recorder = remote(server_keys=True, allowed_base_urls="https://gw.example")
    headers = {"openai-base-url": "https://gw.example/v1"}
    async with http_client(app, headers) as client:
        data(await client.call_tool("status", {}))
        await client.call_tool("locate", {"query": QUERY})
    s = recorder.configs[0].settings
    assert s.openrouter_api_key is not None
    assert s.openai_base_url == "https://gw.example/v1"


async def test_different_client_keys_get_different_indexes() -> None:
    app, _, recorder = remote()
    async with app.router.lifespan_context(app):
        for key in ("sk-client-one-0123456789", "sk-client-two-0123456789"):
            http = httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app),
                base_url=BASE,
                headers={"x-openrouter-api-key": key},
            )
            transport = streamable_http_client(f"{BASE}/mcp", http_client=http)
            async with Client(transport) as client:
                await client.call_tool("locate", {"query": QUERY})
    keys = [c.settings.openrouter_api_key for c in recorder.configs]
    assert [k.get_secret_value() for k in keys if k] == [
        "sk-client-one-0123456789",
        "sk-client-two-0123456789",
    ]


async def test_bearer_auth_and_open_health_check() -> None:
    app, _, _ = remote(auth="bearer", auth_token=TOKEN)
    async with app.router.lifespan_context(app):
        http = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=BASE)
        assert (await http.get("/healthz")).json()["status"] == "ok"
        body = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
        assert (await http.post("/mcp", json=body)).status_code == 401
        wrong = {"Authorization": "Bearer " + "x" * 40}
        assert (await http.post("/mcp", json=body, headers=wrong)).status_code == 401
        auth = {"Authorization": f"Bearer {TOKEN}"}
        async with http_client(app, auth, start=False) as client:
            assert data(await client.call_tool("status", {}))["nodes"] == 0


def test_insecure_remote_configurations_are_refused() -> None:
    with pytest.raises(ValueError, match="only allowed on localhost"):
        ServerSettings(host="0.0.0.0").check_remote()  # noqa: S104
    ServerSettings(host="127.0.0.1").check_remote()
    with pytest.raises(ValueError, match="32"):
        ServerSettings(auth="bearer", auth_token=SecretStr("short")).check_remote()
    with pytest.raises(ValueError, match="OAUTH_ISSUER"):
        ServerSettings(auth="oauth").check_remote()
    with pytest.raises(ValueError, match="CLIENT_ID"):
        ServerSettings(oauth_introspection_url="https://as.example/introspect")


async def test_dns_rebinding_protection_rejects_unknown_hosts() -> None:
    app, _, _ = remote()
    async with app.router.lifespan_context(app):
        http = httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app), base_url="http://evil.example:8080"
        )
        response = await http.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        assert response.status_code == 421


# -- OAuth -----------------------------------------------------------------------------

ISSUER = "https://as.example"
RESOURCE = "http://127.0.0.1:8080/mcp"


class StubAuthServer:
    """An authorization server's public face: RFC 8414 metadata, a JWKS, introspection."""

    def __init__(self) -> None:
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        jwk = json.loads(RSAAlgorithm.to_jwk(self.key.public_key()))
        self.jwks = {"keys": [{**jwk, "kid": "k1", "use": "sig", "alg": "RS256"}]}
        self.active: dict[str, dict[str, Any]] = {}
        self.requests: list[str] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request.url.path)
        if request.url.path.startswith("/.well-known/oauth-authorization-server"):
            return httpx2.Response(200, json={"issuer": ISSUER, "jwks_uri": f"{ISSUER}/jwks"})
        if request.url.path == "/jwks":
            return httpx2.Response(200, json=self.jwks)
        if request.url.path == "/introspect":
            token = dict(x.split("=", 1) for x in request.content.decode().split("&"))["token"]
            return httpx2.Response(200, json=self.active.get(token, {"active": False}))
        return httpx2.Response(404)

    def client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler))

    def token(self, **claims: Any) -> str:
        now = int(time.time())
        body = {"iss": ISSUER, "aud": RESOURCE, "sub": "user-1", "client_id": "agent",
                "scope": "graphwalk", "iat": now, "exp": now + 300, **claims}  # fmt: skip
        body = {k: v for k, v in body.items() if v is not None}
        return jwt.encode(body, self.key, algorithm="RS256", headers={"kid": "k1"})


async def test_jwt_verifier_accepts_valid_and_rejects_bad_tokens() -> None:
    stub = StubAuthServer()
    verifier = JWTVerifier(issuer=ISSUER, audience=RESOURCE, client=stub.client())
    good = await verifier.verify_token(stub.token())
    assert good is not None
    assert good.scopes == ["graphwalk"]
    assert good.client_id == "agent"
    assert stub.requests[:2] == ["/.well-known/oauth-authorization-server", "/jwks"]
    now = int(time.time())
    for bad in (
        stub.token(exp=now - 3600),
        stub.token(aud="https://other.example/mcp"),
        stub.token(iss="https://evil.example"),
        stub.token(exp=None),
        jwt.encode({"iss": ISSUER, "aud": RESOURCE, "exp": now + 60}, "k" * 32, algorithm="HS256"),
        "not-a-jwt",
    ):
        assert await verifier.verify_token(bad) is None


async def test_jwks_discovery_checks_the_issuer() -> None:
    stub = StubAuthServer()
    assert await discover_jwks_url(ISSUER, stub.client()) == f"{ISSUER}/jwks"
    with pytest.raises(ValueError, match="names issuer"):
        await discover_jwks_url("https://as.example/other", stub.client())


async def test_introspection_verifier() -> None:
    stub = StubAuthServer()
    verifier = IntrospectionVerifier(
        url=f"{ISSUER}/introspect",
        client_id="rs",
        client_secret="s",  # noqa: S106 - a fake secret
        audience=RESOURCE,
        client=stub.client(),
    )
    now = int(time.time())
    stub.active = {
        "good": {"active": True, "aud": RESOURCE, "scope": "graphwalk", "exp": now + 60},
        "expired": {"active": True, "aud": RESOURCE, "exp": now - 60},
        "elsewhere": {"active": True, "aud": ["https://other.example"], "exp": now + 60},
    }
    assert (await verifier.verify_token("good")) is not None
    for token in ("expired", "elsewhere", "unknown"):
        assert await verifier.verify_token(token) is None


async def test_oauth_resource_server_end_to_end() -> None:
    stub = StubAuthServer()
    service, _ = make_service(
        "http", auth="oauth", oauth_issuer=ISSUER, resource_url=RESOURCE, oauth_scopes="graphwalk"
    )
    server = build_server(service)
    server._token_verifier = JWTVerifier(issuer=ISSUER, audience=RESOURCE, client=stub.client())
    app = http_app(server, service.settings)
    async with app.router.lifespan_context(app):
        http = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url=BASE)
        metadata = (await http.get("/.well-known/oauth-protected-resource/mcp")).json()
        assert metadata["resource"] == RESOURCE
        assert metadata["authorization_servers"] == [ISSUER]
        body = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
        missing = await http.post("/mcp", json=body)
        assert missing.status_code == 401
        assert "resource_metadata" in missing.headers["www-authenticate"]
        expired = stub.token(exp=int(time.time()) - 3600)
        assert (await http.post("/mcp", json=body, headers=_bearer(expired))).status_code == 401
        other = stub.token(aud="https://other.example/mcp")
        assert (await http.post("/mcp", json=body, headers=_bearer(other))).status_code == 401
        unscoped = stub.token(scope="read")
        assert (await http.post("/mcp", json=body, headers=_bearer(unscoped))).status_code == 403
        async with http_client(app, _bearer(stub.token()), start=False) as client:
            assert data(await client.call_tool("status", {}))["documents"] == 0


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_stdio_subprocess(tmp_path: Path) -> None:
    """``graphwalk mcp`` as a real subprocess over stdio (pipes, no sockets)."""
    import os
    import sys

    from mcp import StdioServerParameters

    db = tmp_path / "g.db"
    store = SQLiteStore(db)
    await store.close()
    env = {k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_BASE_URL"))}
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from graphwalk.cli import app; app()", "mcp", "--db", str(db)],
        env={**env, "OPENAI_API_KEY": SECRET, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
    )
    async with Client(params) as client:
        status = data(await client.call_tool("status", {}))
    assert status["nodes"] == 0
    assert status["config"]["keys"]["openai"] is True  # stdio: keys from the environment
    assert status["config"]["keys"]["openrouter"] is False
