"""Path scoring: floored log-probabilities with length normalization.

``score = sum(log max(p_i, floor)) / L ** alpha`` where ``L`` counts decisions
(including STOP). ``alpha = 1`` is the mean log-prob, i.e. the log of TypeSafe's
hierarchical-classification path score ``prod(p_i) ** (1 / L)``. Forced moves (a
single legal option) are not decisions: they add nothing and do not increase ``L``.
"""

import math


def step_logp(probability: float, floor: float) -> float:
    """``log(max(p, floor))``. Jev rounds to 2 decimals, so exact zeros are common."""
    return math.log(max(probability, floor))


def normalized_score(sum_logp: float, n_decisions: int, alpha: float) -> float:
    """Length-normalized path score; 0.0 for a path with no decisions."""
    if n_decisions == 0:
        return 0.0
    return sum_logp / (n_decisions**alpha)
