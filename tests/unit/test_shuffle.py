from graphwalk.decisions.base import ChoiceQuestion, DecisionRequest, JSONContent
from graphwalk.decisions.fake import FakeDecisionBackend
from graphwalk.eval.shuffle import ShuffledDecider, shuffled


def question(key: str = "b0") -> ChoiceQuestion:
    options: dict[str, JSONContent | None] = {f"r{i}": f"relation {i}" for i in range(8)}
    options["STOP"] = None
    return ChoiceQuestion(key=key, instructions="pick", options=options)


def test_shuffle_permutes_and_is_deterministic() -> None:
    q = question()
    a, b = shuffled(q, 1), shuffled(q, 1)
    assert list(a.options) == list(b.options)
    assert sorted(a.options) == sorted(q.options)
    assert a.options == q.options  # same label -> description pairs
    assert list(a.options) != list(q.options)
    assert list(shuffled(q, 2).options) != list(a.options)


async def test_wrapper_passes_shuffled_questions_through() -> None:
    inner = FakeDecisionBackend()
    backend = ShuffledDecider(inner, seed=3)
    response = await backend.decide(DecisionRequest(state="s", questions=(question(),)))
    assert list(inner.requests[0].questions[0].options) == list(shuffled(question(), 3).options)
    assert set(response.results["b0"].probabilities) == set(question().options)
    assert backend.model_id == inner.model_id
