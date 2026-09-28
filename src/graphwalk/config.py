"""Runtime configuration, loaded from the environment and an optional ``.env`` file."""

from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DecisionProvider = Literal["openrouter", "typesafe"]

# Pinned model ids per provider. Never use a moving alias such as ``jev-latest``.
JEV_MODEL_TYPESAFE = "jev-1.13.0"
JEV_MODEL_OPENROUTER = "typesafe/jev-1.13"

TYPESAFE_BASE_URL = "https://api.typesafe.ai"
OPENROUTER_BASE_URL = "https://openrouter.ai/api"


class GraphwalkSettings(BaseSettings):
    """Top-level settings.

    Provider API keys use their conventional unprefixed names (``OPENROUTER_API_KEY``,
    ``TYPESAFE_API_KEY``); everything else uses the ``GRAPHWALK_`` prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="GRAPHWALK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    decision_provider: DecisionProvider = "openrouter"
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "GRAPHWALK_OPENROUTER_API_KEY"),
    )
    typesafe_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("TYPESAFE_API_KEY", "GRAPHWALK_TYPESAFE_API_KEY"),
    )
    jev_model_openrouter: str = JEV_MODEL_OPENROUTER
    jev_model_typesafe: str = JEV_MODEL_TYPESAFE
    jev_timeout_s: float = Field(default=10.0, gt=0)
    jev_max_retries: int = Field(default=2, ge=0)

    @property
    def jev_model(self) -> str:
        """The pinned Jev model id for the selected provider."""
        if self.decision_provider == "openrouter":
            return self.jev_model_openrouter
        return self.jev_model_typesafe

    @property
    def decision_base_url(self) -> str:
        """The API root for the selected provider."""
        if self.decision_provider == "openrouter":
            return OPENROUTER_BASE_URL
        return TYPESAFE_BASE_URL

    @property
    def decision_api_key(self) -> SecretStr | None:
        """The API key for the selected provider, if configured."""
        if self.decision_provider == "openrouter":
            return self.openrouter_api_key
        return self.typesafe_api_key
