"""Pure pieces of traversal: scoring, sampling, batching, labels, confidence."""

import math
import random

import pytest

from graphwalk.decisions import ChoiceQuestion, DecisionBackendError, choice_confidence
from graphwalk.decisions.base import normalize_distribution
from graphwalk.traversal import TraversalConfig
from graphwalk.traversal.batching import estimate_tokens, split_questions
from graphwalk.traversal.prompts import STOP, Move, edge_text, make_labels, names_text
from graphwalk.traversal.sampling import draw, reshape
from graphwalk.traversal.scoring import normalized_score, step_logp

# -- scoring ---------------------------------------------------------------------------


def test_step_logp_floors_zero() -> None:
    assert step_logp(0.0, 1e-3) == math.log(1e-3)
    assert step_logp(0.5, 1e-3) == math.log(0.5)


@pytest.mark.parametrize(
    ("alpha", "expected"), [(0.0, -3.0), (1.0, -1.0), (0.5, -3.0 / math.sqrt(3))]
)
def test_normalized_score(alpha: float, expected: float) -> None:
    assert math.isclose(normalized_score(-3.0, 3, alpha), expected)


def test_normalized_score_without_decisions() -> None:
    assert normalized_score(0.0, 0, 1.0) == 0.0


def test_mean_logp_is_log_of_typesafe_geometric_mean() -> None:
    probs = [0.9, 0.5, 0.7]
    ours = normalized_score(sum(math.log(p) for p in probs), 3, 1.0)
    assert math.isclose(math.exp(ours), math.prod(probs) ** (1 / 3))


# -- sampling --------------------------------------------------------------------------


def test_reshape_identity() -> None:
    dist = {"a": 0.5, "b": 0.3, "c": 0.2}
    assert reshape(dist, temperature=1.0, top_p=1.0) == pytest.approx(dist)


def test_reshape_zero_temperature_is_argmax_first_on_ties() -> None:
    assert reshape({"a": 0.4, "b": 0.4, "c": 0.2}, temperature=0, top_p=1.0) == {
        "a": 1.0,
        "b": 0.0,
        "c": 0.0,
    }


def test_reshape_temperature_sharpens_and_flattens() -> None:
    dist = {"a": 0.6, "b": 0.4}
    assert reshape(dist, temperature=0.5, top_p=1.0)["a"] > 0.6
    assert reshape(dist, temperature=2.0, top_p=1.0)["a"] < 0.6


def test_top_p_keeps_smallest_prefix() -> None:
    out = reshape({"a": 0.1, "b": 0.6, "c": 0.3}, temperature=1.0, top_p=0.8)
    assert out == pytest.approx({"a": 0.0, "b": 2 / 3, "c": 1 / 3})
    assert list(out) == ["a", "b", "c"]  # label order preserved


def test_top_p_ties_break_by_label_order() -> None:
    out = reshape({"a": 0.5, "b": 0.5}, temperature=1.0, top_p=0.5)
    assert out == {"a": 1.0, "b": 0.0}


def test_draw_is_seeded_and_skips_zero_mass() -> None:
    dist = {"a": 0.0, "b": 0.5, "c": 0.5}
    first = [draw(dist, random.Random(3)) for _ in range(5)]
    assert first == [draw(dist, random.Random(3)) for _ in range(5)]
    rng = random.Random(0)
    assert "a" not in {draw(dist, rng) for _ in range(200)}


def test_sampling_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        reshape({}, temperature=1.0, top_p=1.0)
    with pytest.raises(ValueError, match="no mass"):
        draw({"a": 0.0}, random.Random(0))


# -- batching --------------------------------------------------------------------------


def _q(key: str, size: int) -> ChoiceQuestion:
    return ChoiceQuestion(key=key, instructions="x" * size, options={"a": None, "b": None})


def test_estimate_tokens_is_conservative() -> None:
    assert estimate_tokens("abc") == 2  # '"abc"' = 5 chars / 3


def test_split_packs_in_order_and_splits_when_full() -> None:
    questions = [_q(f"q{i}", 300) for i in range(5)]  # ~110 tokens each
    groups = split_questions(
        {"query": "q"},
        questions,
        max_request_tokens=300,
        max_state_plus_question_tokens=300,
        safety_margin=0.0,
    )
    assert [[q.key for q in g] for g in groups] == [["q0", "q1"], ["q2", "q3"], ["q4"]]


def test_split_keeps_everything_together_when_it_fits() -> None:
    questions = [_q(f"q{i}", 10) for i in range(4)]
    groups = split_questions(
        "s",
        questions,
        max_request_tokens=64_000,
        max_state_plus_question_tokens=32_000,
        safety_margin=0.2,
    )
    assert len(groups) == 1


def test_split_rejects_oversized_question() -> None:
    with pytest.raises(DecisionBackendError, match="per-question budget"):
        split_questions(
            "s",
            [_q("big", 3000)],
            max_request_tokens=10_000,
            max_state_plus_question_tokens=500,
            safety_margin=0.0,
        )


# -- labels and prompt text ------------------------------------------------------------


def test_opaque_labels() -> None:
    moves = [Move("r", "out", ("a",)), Move("r", "in", ("b",))]
    assert make_labels("opaque", moves, ["A", "B"]) == ["o1", "o2"]


def test_readable_labels_are_unique_and_never_stop() -> None:
    moves = [Move("r", "out", ("a",)), Move("r", "out", ("a2",)), Move("x", "in", ("s",))]
    labels = make_labels("readable", moves, ["A", "A", "S"])
    assert labels == ["r -> A", "r -> A #2", "x <- S"]
    assert STOP not in make_labels("readable", [Move("STOP", "out", ("n",))], [None])


def test_readable_relation_labels() -> None:
    moves = [Move("directed_by", "out", ("a",)), Move("directed_by", "in", ("b",))]
    assert make_labels("readable", moves, [None, None]) == [
        "directed_by",
        "directed_by (inverse)",
    ]


def test_edge_text_follows_stored_direction() -> None:
    assert edge_text("Nolan", "directed_by", "in", "Inception") == (
        "Inception --directed_by--> Nolan"
    )
    assert edge_text("Inception", "directed_by", "out", "Nolan") == (
        "Inception --directed_by--> Nolan"
    )


def test_names_text() -> None:
    assert names_text(["a", "b"], 3) == "{a, b}"
    assert names_text(["a", "b", "c", "d"], 2) == "{a, b, +2 more}"


# -- confidence ------------------------------------------------------------------------


def test_choice_confidence_matches_live_jev() -> None:
    # Observed live on 2026-09-28: probabilities mixed 0.93 / negative 0.07 / positive 0
    # came back with confidence 0.89.
    assert choice_confidence({"m": 0.93, "n": 0.07, "p": 0.0}) == pytest.approx(0.895)
    assert choice_confidence({"a": 0.5, "b": 0.5}) == 0.0
    assert choice_confidence({"a": 1.0, "b": 0.0}) == 1.0
    assert choice_confidence({"a": 1.0}) == 1.0


def test_normalize_prefers_reported_confidence() -> None:
    question = ChoiceQuestion(key="k", instructions="i", options={"a": None, "b": None})
    assert normalize_distribution(question, {"a": 0.9, "b": 0.1}).confidence == pytest.approx(0.8)
    reported = normalize_distribution(question, {"a": 0.9, "b": 0.1}, confidence=0.7)
    assert reported.confidence == 0.7
    assert normalize_distribution(question, {"a": 1, "b": 0}, confidence=1.2).confidence == 1.0


# -- config ----------------------------------------------------------------------------


def test_config_width_and_validation() -> None:
    assert TraversalConfig(strategy="greedy", beam_width=5).width == 1
    assert TraversalConfig(strategy="sample", n_samples=7).width == 7
    assert TraversalConfig(beam_width=4).width == 4
    with pytest.raises(ValueError, match="prefilter_top_n"):
        TraversalConfig(prefilter_threshold=10, prefilter_top_n=20)
