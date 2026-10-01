"""Answer questions over a small curated knowledge graph.

    pip install graphwalk          # needs OPENROUTER_API_KEY (or TYPESAFE_API_KEY)
    python examples/kg/walk.py

Imports ``movies.csv`` into a SQLite graph next to this file, then walks it.
"""

import asyncio
from pathlib import Path

from graphwalk import Index

HERE = Path(__file__).parent
TYPES = ["film", "person", "place", "year"]  # for the answer-type hint
QUESTIONS = (
    "Where was the director of Inception born?",
    "Which films did Christopher Nolan direct?",
    "Who starred in The Revenant?",
)


async def main() -> None:
    async with Index.open(HERE / "movies.db", node_types=TYPES) as index:
        await index.import_triples(HERE / "movies.csv")  # idempotent
        for question in QUESTIONS:
            result = await index.walk(question)
            best = result.best
            if best is None:
                print(f"{question} -> no answer")
                continue
            path = " / ".join(f"{h.relation}:{h.direction}" for h in best.path)
            print(f"{question} -> {sorted(best.names)} (p={best.confidence:.2f}; {path})")


if __name__ == "__main__":
    asyncio.run(main())
