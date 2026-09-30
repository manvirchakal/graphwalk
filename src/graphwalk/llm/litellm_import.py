"""Import LiteLLM without its network call on import.

By default ``import litellm`` downloads the model price map from GitHub. graphwalk
never estimates cost from price tables (it records only provider-reported cost), so
that request is pure overhead, and it fails or hangs offline. Setting
``LITELLM_LOCAL_MODEL_COST_MAP`` (unless the user already set it) uses the bundled copy.
"""

import os
from typing import Any


def import_litellm(extra_hint: str = "uv sync --extra llm") -> Any:
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    try:
        import litellm  # noqa: PLC0415 - optional, heavy dependency
    except ImportError as error:  # pragma: no cover - depends on the environment
        msg = f"this backend needs the 'llm' extra: {extra_hint}"
        raise ImportError(msg) from error
    litellm.suppress_debug_info = True
    return litellm
