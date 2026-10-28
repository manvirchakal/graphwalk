"""The ``no_decision`` flag of committed score rows (paper P9)."""

from graphwalk.eval.runner import no_decision


def test_no_decision_reads_n_choices_when_recorded() -> None:
    assert no_decision({"n_choices": 0, "min_confidence": None}, -6.9) is True
    assert no_decision({"n_choices": 2, "min_confidence": 0.4}, -0.1) is False


def test_no_decision_falls_back_to_min_confidence() -> None:
    assert no_decision({"min_confidence": None}, 0.0) is True
    assert no_decision({"min_confidence": 0.7}, -0.2) is False


def test_no_decision_is_unknown_without_a_scored_walk() -> None:
    assert no_decision({"min_confidence": None}, None) is None
    assert no_decision({}, -0.2) is None
