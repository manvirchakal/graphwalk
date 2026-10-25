import json
from typing import Any

import httpx2
import pytest

from graphwalk.decisions.base import ChoiceQuestion, DecisionRequest, JSONContent
from graphwalk.eval.stated_decider import (
    StatedDecider,
    parse_scores,
    parse_top1,
    stated_prompt,
    vote_distribution,
)


def question(n: int = 3) -> ChoiceQuestion:
    options: dict[str, JSONContent | None] = {f"rel_{i}": None for i in range(n - 1)}
    options["STOP"] = "done"
    return ChoiceQuestion(key="move", instructions="Pick.", options=options)


def reply(text: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "cost": 0.001},
    }


def test_prompt_asks_for_the_mode_format() -> None:
    text = stated_prompt("s", question(), "top1")
    assert "C. STOP" in text
    assert text.endswith("(for example: B 85). Nothing else.")
    assert "Answer with one letter." not in text


def test_parse_scores_and_top1() -> None:
    scores = parse_scores(question(), "A=80\nB = 10\nC: 30")
    assert scores == {"rel_0": 80.0, "rel_1": 10.0, "STOP": 30.0}
    assert parse_scores(question(), "no idea") is None
    top = parse_top1(question(), "B 70")
    assert top is not None
    assert top["rel_1"] == pytest.approx(0.7)
    assert top["rel_0"] == pytest.approx(0.15)
    assert parse_top1(question(), "Z 70") is None


def test_votes_are_smoothed() -> None:
    dist = vote_distribution(question(), ["A", "A", "B", "x", "a"])
    assert dist == {"rel_0": 3.1, "rel_1": 1.1, "STOP": 0.1}


async def test_vote_mode_samples_and_sums_usage() -> None:
    bodies: list[dict[str, Any]] = []
    answers = iter(["A", "A", "B"])

    def handle(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=reply(next(answers)))

    backend = StatedDecider(
        "m", "vote", samples=3, base_url="https://openrouter.ai/api/v1",
        transport=httpx2.MockTransport(handle),
    )  # fmt: skip
    response = await backend.decide(DecisionRequest(state="s", questions=(question(),)))
    assert response.results["move"].top == "rel_0"
    assert response.usage.cost_usd == pytest.approx(0.003)
    assert all(b["temperature"] == 1.0 and b["max_tokens"] == 1 for b in bodies)
    assert bodies[0]["reasoning"] == {"enabled": False}


async def test_unparsed_reply_is_uniform_and_counted() -> None:
    backend = StatedDecider(
        "m", "top1", base_url="http://localhost:1/v1",
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=reply("hmm"))),
    )  # fmt: skip
    response = await backend.decide(DecisionRequest(state="s", questions=(question(),)))
    assert list(response.results["move"].probabilities.values()) == pytest.approx([1 / 3] * 3)
    assert backend.diagnostics() == {"asked": 1, "unparsed": 1}
