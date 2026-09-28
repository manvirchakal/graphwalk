"""Client-side temperature and nucleus (top-p) sampling over a returned distribution."""

import math
import random
from collections.abc import Mapping


def reshape(
    probabilities: Mapping[str, float], *, temperature: float, top_p: float
) -> dict[str, float]:
    """Apply ``p ** (1/T)`` then top-p, renormalize. Keeps the input's label order.

    * ``T = 0`` puts all mass on the argmax (ties: first label in order).
    * top-p keeps the smallest set, by descending probability (ties: label order), whose
      mass reaches ``top_p``; the rest get 0.
    """
    labels = list(probabilities)
    if not labels:
        msg = "empty distribution"
        raise ValueError(msg)
    if temperature == 0:
        top = max(labels, key=lambda label: probabilities[label])
        return {label: 1.0 if label == top else 0.0 for label in labels}
    powered = {label: max(probabilities[label], 0.0) ** (1.0 / temperature) for label in labels}
    total = math.fsum(powered.values())
    if total <= 0:
        msg = "distribution has no mass"
        raise ValueError(msg)
    tempered = {label: p / total for label, p in powered.items()}
    if top_p >= 1.0:
        return tempered
    ranked = sorted(labels, key=lambda label: -tempered[label])  # stable: label order on ties
    kept: set[str] = set()
    mass = 0.0
    for label in ranked:
        kept.add(label)
        mass += tempered[label]
        if mass >= top_p - 1e-12:
            break
    kept_total = math.fsum(tempered[label] for label in kept)
    return {label: (tempered[label] / kept_total if label in kept else 0.0) for label in labels}


def draw(probabilities: Mapping[str, float], rng: random.Random) -> str:
    """Draw one label. Deterministic for a given ``rng`` state and label order."""
    labels = [label for label, p in probabilities.items() if p > 0]
    if not labels:
        msg = "distribution has no mass"
        raise ValueError(msg)
    return rng.choices(labels, weights=[probabilities[label] for label in labels], k=1)[0]
