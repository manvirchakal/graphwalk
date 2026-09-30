"""Server settings, from ``GRAPHWALK_*`` environment variables.

Provider keys are not here: they come from :mod:`graphwalk.config` (stdio: the
environment; remote: each client's headers, plus the server's environment only when
``GRAPHWALK_SERVER_KEYS=true``).
"""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

type AuthMode = Literal["none", "bearer", "oauth"]

LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
MIN_TOKEN_CHARS = 32


def _split(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GRAPHWALK_", extra="ignore")

    db: Path = Path("graphwalk.db")
    """The SQLite graph the server serves (created if missing)."""
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "WARNING"
    """Server log level. INFO logs every provider request (URLs only, never keys)."""

    # -- who may connect (remote mode)
    auth: AuthMode = "none"
    """``none`` (localhost only), ``bearer`` (a static token), or ``oauth``."""
    auth_token: SecretStr | None = None
    resource_url: str | None = None
    """The public URL of the MCP endpoint (``https://host/mcp``); required for oauth."""
    oauth_issuer: str | None = None
    oauth_audience: str | None = None
    """Expected ``aud``; defaults to ``resource_url``."""
    oauth_scopes: str = ""
    """Comma-separated scopes every token must carry."""
    oauth_jwks_url: str | None = None
    """Validate JWTs with these keys; default: discovered from the issuer's metadata."""
    oauth_introspection_url: str | None = None
    """Validate opaque tokens by RFC 7662 introspection instead of as JWTs."""
    oauth_client_id: str | None = None
    oauth_client_secret: SecretStr | None = None
    allowed_hosts: str = ""
    """Extra ``Host`` header values to accept (DNS-rebinding protection), comma-separated,
    e.g. ``graphwalk.example.com``. Localhost is always accepted."""

    # -- provider keys and base URLs in remote mode
    server_keys: bool = False
    """Let clients without their own keys use the server's environment keys."""
    allowed_base_urls: str = ""
    """Comma-separated base URLs clients may point providers at via headers."""

    # -- limits
    ingest_root: Path | None = None
    """Remote ``ingest`` may read server paths only under this directory (else only
    documents sent inline). stdio mode reads any path the user can."""
    max_k: int = Field(default=50, ge=1)
    max_read_chars: int = Field(default=20_000, ge=100)
    session_concurrency: int = Field(default=4, ge=1)
    """Concurrent tool calls per session."""
    max_ingest_chars: int = Field(default=5_000_000, ge=1)
    """Characters one inline ``ingest`` call may send."""

    @property
    def scopes(self) -> tuple[str, ...]:
        return _split(self.oauth_scopes)

    @property
    def base_url_allowlist(self) -> tuple[str, ...]:
        return _split(self.allowed_base_urls)

    @property
    def extra_hosts(self) -> tuple[str, ...]:
        return _split(self.allowed_hosts)

    def check_remote(self) -> Self:
        """Refuse insecure remote configurations before serving."""
        if self.auth == "none" and self.host not in LOCAL_HOSTS:
            msg = (
                f"GRAPHWALK_AUTH=none is only allowed on localhost, not {self.host!r}; "
                "use bearer or oauth"
            )
            raise ValueError(msg)
        if self.auth == "bearer":
            token = self.auth_token
            if token is None or len(token.get_secret_value()) < MIN_TOKEN_CHARS:
                msg = (
                    "GRAPHWALK_AUTH=bearer needs GRAPHWALK_AUTH_TOKEN of "
                    f"{MIN_TOKEN_CHARS}+ characters"
                )
                raise ValueError(msg)
        if self.auth == "oauth" and (self.oauth_issuer is None or self.resource_url is None):
            msg = "GRAPHWALK_AUTH=oauth needs GRAPHWALK_OAUTH_ISSUER and GRAPHWALK_RESOURCE_URL"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _introspection_needs_client(self) -> Self:
        if self.oauth_introspection_url and not self.oauth_client_id:
            msg = "GRAPHWALK_OAUTH_INTROSPECTION_URL needs GRAPHWALK_OAUTH_CLIENT_ID (and secret)"
            raise ValueError(msg)
        return self
