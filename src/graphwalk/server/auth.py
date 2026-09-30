"""Who may use a remote server: a static bearer token, or OAuth 2.1 access tokens.

In ``oauth`` mode graphwalk is a *resource server* per the MCP authorization spec: it
never issues tokens. The MCP SDK serves the RFC 9728 protected-resource metadata
(pointing clients at the operator's authorization server), answers 401/403 with the
right ``WWW-Authenticate`` headers, and enforces required scopes. This module supplies
the token check, as either

* :class:`JWTVerifier`: signature against the issuer's JWKS (discovered from its RFC
  8414 / OpenID metadata unless given), plus ``iss``, ``aud``, and ``exp``; or
* :class:`IntrospectionVerifier`: RFC 7662 introspection for opaque tokens.

Both return ``None`` for any invalid token (the SDK then answers 401) and never log the
token.
"""

import asyncio
import hmac
import logging
import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any, cast
from urllib.parse import urlsplit

import httpx2
import jwt
from mcp.server.auth.provider import AccessToken
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

JWKS_TTL_S = 3600.0
ALGORITHMS = ("RS256", "RS384", "RS512", "PS256", "ES256", "ES384", "EdDSA")
"""Asymmetric only: a shared-secret (HS*) JWT from an issuer would let anyone who can
read the secret mint tokens."""


def _scopes(claims: dict[str, Any]) -> list[str]:
    raw = claims.get("scope", claims.get("scp", ""))
    if isinstance(raw, str):
        return raw.split()
    if isinstance(raw, list):
        return [str(s) for s in cast("list[object]", raw)]
    return []


def _access_token(token: str, claims: dict[str, Any], audience: str) -> AccessToken:
    exp = claims.get("exp")
    return AccessToken(
        token=token,
        client_id=str(claims.get("client_id") or claims.get("azp") or claims.get("sub") or ""),
        scopes=_scopes(claims),
        expires_at=int(exp) if isinstance(exp, int | float) else None,
        resource=audience,
        subject=str(claims["sub"]) if "sub" in claims else None,
        claims={k: v for k, v in claims.items() if k in ("iss", "sub", "aud", "act")},
    )


def _audiences(claims: dict[str, Any]) -> list[str]:
    aud = claims.get("aud")
    if isinstance(aud, str):
        return [aud]
    if isinstance(aud, list):
        return [str(a) for a in cast("list[object]", aud)]
    return []


async def discover_jwks_url(issuer: str, client: httpx2.AsyncClient) -> str:
    """The issuer's ``jwks_uri``, from RFC 8414 metadata or OpenID discovery."""
    base = issuer.rstrip("/")
    parts = urlsplit(base)
    origin, path = f"{parts.scheme}://{parts.netloc}", parts.path
    candidates = (
        f"{origin}/.well-known/oauth-authorization-server{path}",  # RFC 8414: before the path
        f"{base}/.well-known/openid-configuration",  # OpenID Connect: after it
    )
    for url in candidates:
        try:
            response = await client.get(url)
        except httpx2.HTTPError:
            continue
        if response.status_code != 200:  # noqa: PLR2004
            continue
        metadata = cast("dict[str, Any]", response.json())
        if metadata.get("issuer") not in (issuer, base):
            msg = f"issuer metadata names issuer {metadata.get('issuer')!r}, expected {issuer!r}"
            raise ValueError(msg)
        uri = metadata.get("jwks_uri")
        if isinstance(uri, str):
            return uri
    msg = f"could not discover jwks_uri for {issuer!r}; set GRAPHWALK_OAUTH_JWKS_URL"
    raise ValueError(msg)


class JWTVerifier:
    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_url: str | None = None,
        client: httpx2.AsyncClient | None = None,
        leeway_s: float = 30.0,
    ) -> None:
        self._issuer = issuer
        self._audience = audience
        self._jwks_url = jwks_url
        self._client = client or httpx2.AsyncClient(timeout=10.0)
        self._leeway = leeway_s
        self._keys: jwt.PyJWKSet | None = None
        self._fetched_at = 0.0
        self._lock = asyncio.Lock()

    async def _key_set(self, *, refresh: bool = False) -> jwt.PyJWKSet:
        async with self._lock:
            stale = time.monotonic() - self._fetched_at > JWKS_TTL_S
            if self._keys is None or stale or refresh:
                if self._jwks_url is None:
                    self._jwks_url = await discover_jwks_url(self._issuer, self._client)
                response = await self._client.get(self._jwks_url)
                response.raise_for_status()
                self._keys = jwt.PyJWKSet.from_dict(response.json())
                self._fetched_at = time.monotonic()
            return self._keys

    async def _signing_key(self, kid: str | None) -> jwt.PyJWK | None:
        for refresh in (False, True):  # an unknown kid may mean the issuer rotated keys
            keys = await self._key_set(refresh=refresh)
            for key in keys.keys:
                if kid is None or key.key_id == kid:
                    return key
            if kid is None:
                return None
        return None

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            header = jwt.get_unverified_header(token)
            key = await self._signing_key(header.get("kid"))
            if key is None:
                return None
            claims: dict[str, Any] = jwt.decode(
                token,
                key=key,
                algorithms=list(ALGORITHMS),
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway,
                options={"require": ["exp", "iss", "aud"]},
            )
        except (jwt.PyJWTError, httpx2.HTTPError, ValueError) as error:
            logger.info("rejected access token: %s", type(error).__name__)
            return None
        return _access_token(token, claims, self._audience)


class IntrospectionVerifier:
    def __init__(
        self,
        *,
        url: str,
        client_id: str,
        client_secret: str | None,
        audience: str,
        client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._url = url
        self._auth = (client_id, client_secret or "")
        self._audience = audience
        self._client = client or httpx2.AsyncClient(timeout=10.0)

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            response = await self._client.post(
                self._url,
                data={"token": token, "token_type_hint": "access_token"},
                auth=self._auth,
            )
            response.raise_for_status()
            claims = cast("dict[str, Any]", response.json())
        except (httpx2.HTTPError, ValueError) as error:
            logger.warning("token introspection failed: %s", type(error).__name__)
            return None
        if claims.get("active") is not True:
            return None
        exp = claims.get("exp")
        if isinstance(exp, int | float) and exp < time.time():
            return None
        audiences = _audiences(claims)
        if audiences and self._audience not in audiences:
            return None
        return _access_token(token, claims, self._audience)


class BearerTokenMiddleware:
    """Requires ``Authorization: Bearer <token>`` on every HTTP request except the
    health check. Compares in constant time."""

    def __init__(self, app: ASGIApp, token: str, *, open_paths: frozenset[str]) -> None:
        self._app = app
        self._token = token.encode()
        self._open = open_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in self._open:
            await self._app(scope, receive, send)
            return
        headers = cast("list[tuple[bytes, bytes]]", scope.get("headers", []))
        given = next((v for k, v in headers if k == b"authorization"), b"")
        scheme, _, token = given.partition(b" ")
        if scheme.lower() == b"bearer" and hmac.compare_digest(token.strip(), self._token):
            await self._app(scope, receive, send)
            return
        await _reject(send)


async def _reject(send: Callable[[MutableMapping[str, Any]], Awaitable[None]]) -> None:
    body = b'{"error": "invalid_token", "error_description": "missing or invalid bearer token"}'
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b'Bearer error="invalid_token"'),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
