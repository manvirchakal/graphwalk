import logging
import math

import pytest
from pydantic import ValidationError

from graphwalk.decisions import (
    ChoiceQuestion,
    DecisionBackendError,
    DecisionRequest,
    normalize_distribution,
)


def q(*labels: str, key: str = "q") -> ChoiceQuestion:
    return ChoiceQuestion(key=key, instructions="pick", options=dict.fromkeys(labels))


def test_normalizes_exactly() -> None:
    result = normalize_distribution(q("a", "b", "c"), {"a": 2.0, "b": 1.0, "c": 1.0})
    assert result.probabilities == {"a": 0.5, "b": 0.25, "c": 0.25}
    assert result.top == "a"
    assert math.fsum(result.probabilities.values()) == 1.0


def test_preserves_offered_order_and_breaks_ties_by_it() -> None:
    result = normalize_distribution(q("z", "a"), {"a": 0.5, "z": 0.5})
    assert list(result.probabilities) == ["z", "a"]
    assert result.top == "z"


def test_unknown_label_is_an_error() -> None:
    with pytest.raises(DecisionBackendError, match="not offered"):
        normalize_distribution(q("a", "b"), {"a": 0.5, "b": 0.3, "c": 0.2})


def test_missing_label_gets_zero_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        result = normalize_distribution(q("a", "b", "c"), {"a": 3.0, "b": 1.0})
    assert result.probabilities == {"a": 0.75, "b": 0.25, "c": 0.0}
    assert "no probability for ['c']" in caplog.text


@pytest.mark.parametrize(
    "bad", [{"a": -0.1, "b": 1.0}, {"a": math.nan, "b": 1.0}, {"a": 0, "b": 0}]
)
def test_invalid_mass_is_an_error(bad: dict[str, float]) -> None:
    with pytest.raises(DecisionBackendError):
        normalize_distribution(q("a", "b"), bad)


def test_question_validation() -> None:
    with pytest.raises(ValidationError):
        q("only")  # need at least two options
    with pytest.raises(ValidationError, match="question key"):
        q("a", "b", key="has space")
    with pytest.raises(ValidationError, match="non-empty"):
        q("a", "")


def test_request_needs_unique_keys() -> None:
    with pytest.raises(ValidationError, match="unique"):
        DecisionRequest(state="s", questions=(q("a", "b", key="x"), q("c", "d", key="x")))
    with pytest.raises(ValidationError):
        DecisionRequest(state="s", questions=())
