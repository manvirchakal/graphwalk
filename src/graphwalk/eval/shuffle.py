"""Option-order control: a decider wrapper that presents each question's options in a
random order.

A decider that reads option letters (the ``logprob`` decider) can prefer letters or
positions regardless of content (Zheng et al., ICLR 2024). The walk presents options in
a fixed order (by direction and relation name, STOP last), so a run with shuffled
options, compared with the original, measures how much order alone moves accuracy and
confidence. Results are keyed by label, so nothing needs mapping back.

The permutation depends on the seed, the question key and the option labels, so the same
question is shuffled the same way every time it is asked within a run.
"""

import random

from graphwalk.decisions.base import (
    ChoiceQuestion,
    DecisionBackend,
    DecisionRequest,
    DecisionResponse,
)


def shuffled(question: ChoiceQuestion, seed: int) -> ChoiceQuestion:
    labels = list(question.options)
    random.Random(f"{seed}:{question.key}:{'|'.join(labels)}").shuffle(labels)  # noqa: S311
    options = {label: question.options[label] for label in labels}
    return question.model_copy(update={"options": options})


class ShuffledDecider:
    """Wraps a decider; every question reaches it with its options permuted."""

    def __init__(self, inner: DecisionBackend, seed: int) -> None:
        self.inner = inner
        self.seed = seed

    @property
    def model_id(self) -> str:
        return self.inner.model_id

    @property
    def max_options(self) -> int:
        return self.inner.max_options

    async def decide(self, request: DecisionRequest) -> DecisionResponse:
        questions = tuple(shuffled(q, self.seed) for q in request.questions)
        return await self.inner.decide(request.model_copy(update={"questions": questions}))

    async def aclose(self) -> None:
        await self.inner.aclose()
