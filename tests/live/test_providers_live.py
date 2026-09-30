"""Live smoke tests: one per provider and role. Skipped by default.

Run with ``uv run pytest --run-live tests/live/test_providers_live.py``. Each test
skips itself if its provider's key is not set. Together they cost well under a cent.
x.ai has no default model: set ``GRAPHWALK_LLM_MODEL`` to run its test.
"""

import os

import pytest

from graphwalk.config import resolve_config
from graphwalk.decisions import ChoiceQuestion, DecisionRequest
from graphwalk.decisions.jev import JevBackend
from graphwalk.decisions.llm_decider import LLMDecider
from graphwalk.llm import Message
from graphwalk.providers import make_decider, make_embedder, make_llm

pytestmark = pytest.mark.live

REQUEST = DecisionRequest(
    state={"query": "Who directed the film Inception?"},
    questions=(
        ChoiceQuestion(
            key="q",
            instructions="Which edge from Inception leads to the answer?",
            options={
                "o1": "Inception --directed_by--> Christopher Nolan",
                "o2": "Inception --release_year--> 2010",
                "STOP": "Inception itself is the answer.",
            },
        ),
    ),
)


def need(var: str) -> None:
    if not os.environ.get(var):
        pytest.skip(f"{var} is not set")


@pytest.mark.parametrize("provider", ["openrouter", "openai", "anthropic", "xai"])
async def test_chat(provider: str) -> None:
    need(f"{provider.upper()}_API_KEY")
    if provider == "xai":
        need("GRAPHWALK_LLM_MODEL")
    llm = make_llm(resolve_config({"llm_provider": provider}), max_tokens=64)
    result = await llm.complete([Message(role="user", content="Reply with the word OK.")])
    assert "ok" in result.text.lower()
    assert result.model


@pytest.mark.parametrize("provider", ["openrouter", "typesafe"])
async def test_jev_decision(provider: str) -> None:
    need(f"{provider.upper()}_API_KEY")
    decider = make_decider(resolve_config({"decision_provider": provider}))
    assert isinstance(decider, JevBackend)
    try:
        response = await decider.decide(REQUEST)
    finally:
        await decider.aclose()
    assert response.results["q"].top == "o1"


async def test_llm_decider_fallback() -> None:
    need("OPENROUTER_API_KEY")
    decider = LLMDecider(make_llm(resolve_config({"llm_provider": "openrouter"}), max_tokens=256))
    response = await decider.decide(REQUEST)
    assert response.results["q"].top == "o1"
    assert response.model.startswith("llm-decider:")


@pytest.mark.parametrize("provider", ["openai", "openrouter"])
async def test_remote_embeddings(provider: str) -> None:
    need(f"{provider.upper()}_API_KEY")
    embedder = make_embedder(resolve_config({"embedding_provider": provider}))
    vectors = await embedder.embed(["Christopher Nolan directed Inception.", "Paris is in France."])
    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 100
