import math

import pytest

from graphwalk.decisions.base import ChoiceQuestion, JSONContent
from graphwalk.eval.logprob_decider import (
    FLOOR,
    first_token_alternatives,
    letter_distribution,
    prompt,
)


def question(n: int = 3) -> ChoiceQuestion:
    options: dict[str, JSONContent | None] = {f"rel_{i}": None for i in range(n - 1)}
    options["STOP"] = "the current nodes answer the question"
    return ChoiceQuestion(key="move", instructions="Pick the next relation.", options=options)


def test_prompt_letters_every_option() -> None:
    text = prompt({"question": "q"}, question())
    assert "A. rel_0" in text
    assert "C. STOP: " in text


def test_letter_variants_are_summed_and_missing_letters_floored() -> None:
    top = [
        {"token": "B", "logprob": math.log(0.6)},
        {"token": " b", "logprob": math.log(0.2)},
        {"token": "A", "logprob": math.log(0.2)},
        {"token": "<eos>", "logprob": math.log(0.01)},
    ]
    result = letter_distribution(question(), top)
    assert result.top == "rel_1"
    assert result.probabilities["rel_1"] == pytest.approx(0.8, abs=1e-4)
    assert result.probabilities["STOP"] == pytest.approx(FLOOR, rel=0.01)


def test_no_letter_gives_uniform() -> None:
    result = letter_distribution(question(), [{"token": "Hello", "logprob": 0.0}])
    assert list(result.probabilities.values()) == pytest.approx([1 / 3] * 3)


def test_first_token_alternatives() -> None:
    data = {"choices": [{"logprobs": {"content": [{"top_logprobs": [{"token": "A"}]}]}}]}
    assert first_token_alternatives(data) == [{"token": "A"}]
    assert first_token_alternatives({"choices": [{"logprobs": None}]}) is None
    assert first_token_alternatives({}) is None
