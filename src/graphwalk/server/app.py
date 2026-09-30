"""The MCP server: tool definitions, the stdio runner, and the HTTP app with auth.

``graphwalk mcp`` (stdio) and ``graphwalk mcp --http`` both build the same tools; they
differ in where provider keys come from (see :mod:`graphwalk.server.service`) and in
who may connect (HTTP only; see :mod:`graphwalk.server.auth`).
"""

import functools
from collections.abc import Awaitable, Callable
from typing import Annotated, Literal

from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError as MCPToolError
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from graphwalk import __version__
from graphwalk.core.errors import GraphwalkError
from graphwalk.server.auth import BearerTokenMiddleware, IntrospectionVerifier, JWTVerifier
from graphwalk.server.service import (
    InlineDocument,
    JobOut,
    LocateOut,
    Mode,
    NeighborsOut,
    NodeOut,
    ReadOut,
    Service,
    StatusOut,
)
from graphwalk.server.settings import ServerSettings

MCP_PATH = "/mcp"
HEALTH_PATH = "/healthz"

INSTRUCTIONS = """\
graphwalk indexes a set of documents as a knowledge graph and finds where in those
documents a question is answered. Typical use: call `locate` with the question, then
`read` the most promising locations (pass key, start, end, doc_hash exactly as returned)
and answer from that text. Locations come with the graph path that found them. Use
`neighbors`/`get_node` to explore entities, `status` to see what is indexed, and
`ingest` + `ingest_status` to add documents."""


def _headers(ctx: Context) -> dict[str, str]:
    return {k.lower(): v for k, v in (ctx.headers or {}).items()}


def _session(ctx: Context) -> str:
    return _headers(ctx).get("mcp-session-id", "default")


def _tool_errors[**P, R](fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Expected failures become tool errors the model can read."""

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await fn(*args, **kwargs)
        except (GraphwalkError, ValueError, KeyError, FileNotFoundError) as error:
            raise MCPToolError(str(error)) from None

    return wrapper


def build_server(service: Service) -> MCPServer:
    """An MCP server exposing ``service`` as tools (no transport or auth yet)."""
    settings = service.settings
    server = MCPServer(
        name="graphwalk",
        title="graphwalk",
        instructions=INSTRUCTIONS,
        version=__version__,
        log_level=settings.log_level,
        token_verifier=_verifier(settings) if service.mode == "http" else None,
        auth=_auth_settings(settings) if service.mode == "http" else None,
    )

    async def guarded[R](ctx: Context, call: Callable[[], Awaitable[R]]) -> R:
        async with service.session_slot(_session(ctx)):
            return await call()

    @server.tool()
    @_tool_errors
    async def locate(
        query: Annotated[str, Field(description="The question, in natural language.")],
        ctx: Context,
        k: Annotated[int, Field(description="How many locations to return.")] = 5,
        mode: Annotated[
            Literal["graph", "dense", "hybrid"],
            Field(
                description="graph: walk the entity graph; dense: embedding search; "
                "hybrid: both, fused."
            ),
        ] = "graph",
    ) -> LocateOut:
        """Find the passages in the indexed documents that answer a question. Returns
        document spans (key, start, end, doc_hash) with a snippet and the graph path
        that led there, best first. Read the full text with `read`."""
        headers = _headers(ctx)
        return await guarded(ctx, lambda: service.locate(headers, query, k, mode))

    @server.tool()
    @_tool_errors
    async def read(  # noqa: PLR0917 - MCP tool arguments
        key: Annotated[str, Field(description="Document key from a location.")],
        ctx: Context,
        start: Annotated[
            int | None, Field(description="Span start; omit for the whole document.")
        ] = None,
        end: Annotated[
            int | None, Field(description="Span end; omit for the whole document.")
        ] = None,
        doc_hash: Annotated[
            str | None, Field(description="From the location; detects edited documents.")
        ] = None,
        context: Annotated[int, Field(description="Extra characters on each side.")] = 0,
    ) -> ReadOut:
        """Read the text at a location returned by `locate` (or a whole document).
        `stale` means the document changed since it was located."""
        headers = _headers(ctx)
        return await guarded(ctx, lambda: service.read(headers, key, start, end, doc_hash, context))

    @server.tool()
    @_tool_errors
    async def neighbors(
        node: Annotated[str, Field(description="Entity name (any alias) or node id.")],
        ctx: Context,
        direction: Annotated[Literal["out", "in", "both"], Field()] = "both",
        limit: Annotated[int, Field(description="Maximum edges to return.")] = 25,
    ) -> NeighborsOut:
        """The relations around an entity, each with the document spans that state it."""
        return await guarded(ctx, lambda: service.neighbors(node, direction, limit))

    @server.tool()
    @_tool_errors
    async def get_node(
        node: Annotated[str, Field(description="Entity name (any alias) or node id.")],
        ctx: Context,
    ) -> list[NodeOut]:
        """An entity's type, summary, aliases, attributes, and where it is mentioned."""
        return await guarded(ctx, lambda: service.get_node(node))

    @server.tool()
    @_tool_errors
    async def ingest(
        ctx: Context,
        documents: Annotated[
            list[InlineDocument] | None,
            Field(description="Documents to index: [{id, text, title?}]."),
        ] = None,
        path: Annotated[
            str | None,
            Field(description="A file or folder to index (txt, md, json, jsonl, csv)."),
        ] = None,
        source_id: Annotated[
            str | None,
            Field(description="Names this batch; re-ingesting the same source updates it."),
        ] = None,
    ) -> JobOut:
        """Add or update documents in the index, in the background. Unchanged documents
        are skipped. Poll `ingest_status` with the returned job id."""
        headers = _headers(ctx)
        return await guarded(ctx, lambda: service.ingest(headers, documents, path, source_id))

    @server.tool()
    @_tool_errors
    async def ingest_status(
        job_id: Annotated[str, Field(description="From `ingest`.")],
    ) -> JobOut:
        """Progress of an ingest job (chunks done of total), and its report when done."""
        return service.ingest_status(job_id)

    @server.tool()
    @_tool_errors
    async def status(ctx: Context) -> StatusOut:
        """What is indexed (entities, relations, documents per source), which providers
        and models requests from this client use, and the server's limits."""
        return await service.status(_headers(ctx))

    @server.custom_route(HEALTH_PATH, methods=["GET"])
    async def health(request: Request) -> Response:  # pyright: ignore[reportUnusedFunction]
        del request
        nodes, edges = await service.store.counts()
        return JSONResponse(
            {"status": "ok", "version": __version__, "nodes": nodes, "edges": edges}
        )

    return server


def _auth_settings(settings: ServerSettings) -> AuthSettings | None:
    if settings.auth != "oauth":
        return None
    assert settings.oauth_issuer is not None  # noqa: S101 - check_remote ran
    return AuthSettings(
        issuer_url=settings.oauth_issuer,  # pyright: ignore[reportArgumentType]
        resource_server_url=settings.resource_url,  # pyright: ignore[reportArgumentType]
        required_scopes=list(settings.scopes) or None,
        validate_token_resource=False,  # the verifiers check the audience themselves
    )


def _verifier(settings: ServerSettings) -> JWTVerifier | IntrospectionVerifier | None:
    if settings.auth != "oauth":
        return None
    assert settings.oauth_issuer is not None  # noqa: S101 - check_remote ran
    audience = settings.oauth_audience or settings.resource_url or ""
    if settings.oauth_introspection_url:
        secret = settings.oauth_client_secret
        return IntrospectionVerifier(
            url=settings.oauth_introspection_url,
            client_id=settings.oauth_client_id or "",
            client_secret=None if secret is None else secret.get_secret_value(),
            audience=audience,
        )
    return JWTVerifier(
        issuer=settings.oauth_issuer, audience=audience, jwks_url=settings.oauth_jwks_url
    )


def http_app(server: MCPServer, settings: ServerSettings) -> Starlette:
    """The streamable-HTTP ASGI app with DNS-rebinding protection and the configured
    auth. Raises ``ValueError`` for an insecure configuration."""
    settings.check_remote()
    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    for host in settings.extra_hosts:
        hosts += [host, f"{host}:*"]
    app = server.streamable_http_app(
        streamable_http_path=MCP_PATH,
        json_response=True,
        transport_security=TransportSecuritySettings(allowed_hosts=hosts),
        host=settings.host,
    )
    if settings.auth == "bearer":
        assert settings.auth_token is not None  # noqa: S101 - check_remote ran
        app.add_middleware(
            BearerTokenMiddleware,  # pyright: ignore[reportArgumentType]
            token=settings.auth_token.get_secret_value(),
            open_paths=frozenset({HEALTH_PATH}),
        )
    return app


def service_for(settings: ServerSettings, mode: Mode) -> Service:
    from graphwalk.stores.sqlite_store import SQLiteStore  # noqa: PLC0415

    return Service(store=SQLiteStore(settings.db), settings=settings, mode=mode)


async def run_stdio(settings: ServerSettings) -> None:
    service = service_for(settings, "stdio")
    try:
        await build_server(service).run_stdio_async()
    finally:
        await service.close()


async def run_http(settings: ServerSettings) -> None:
    import uvicorn  # noqa: PLC0415 - the mcp extra

    service = service_for(settings, "http")
    app = http_app(build_server(service), settings)
    config = uvicorn.Config(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        server_header=False,
    )
    try:
        await uvicorn.Server(config).serve()
    finally:
        await service.close()
