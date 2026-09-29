"""Call, token, and cost accounting for ingestion."""

from dataclasses import dataclass


@dataclass
class Spend:
    llm_calls: int = 0
    decision_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0
    """``None`` once any call did not report its cost (costs are never estimated)."""

    def add_cost(self, cost: float | None) -> None:
        self.cost_usd = None if self.cost_usd is None or cost is None else self.cost_usd + cost

    def add(self, other: "Spend") -> None:
        self.llm_calls += other.llm_calls
        self.decision_calls += other.decision_calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.add_cost(other.cost_usd)
