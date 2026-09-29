import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from graphwalk.decisions import ChoiceQuestion, FakeDecisionBackend, JSONContent
from graphwalk.eval.ingest_eval import (
    documents,
    norm_name,
    norm_value,
    run_dir,
    run_variants,
    sample_records,
    write_results,
)
from graphwalk.ingest import IngestConfig
from graphwalk.llm import FakeLLM, Message

RECORDS: list[dict[str, Any]] = [
    {
        "type": "compositional",
        "question": "Who is the mother of the director of Film A?",
        "evidences": [["Film A", "director", "Jo Doe"], ["Jo Doe", "mother", "Ann Doe"]],
        "context": [
            ["Film A (film)", ["Film A is a film.", "It was directed by Jo Doe."]],
            ["Jo Doe", ["Jo Doe is a director.", "Jo's mother is Ann Doe."]],
        ],
    },
    {
        "type": "inference",
        "question": "When was Jo Doe's mother born?",
        "evidences": [["Jo Doe", "mother", "Ann Doe"], ["Ann Doe", "date of birth", "5 May 1950"]],
        "context": [
            ["Jo Doe", ["duplicate title, ignored"]],
            ["Ann Doe", ["Ann Doe was born on 5 May 1950."]],
        ],
    },
    {"type": "comparison", "evidences": [["x", "y", "z"]], "context": []},
]

REPLIES = {
    "Film A (film)": {
        "entities": [{"name": "Film A", "type": "film"}, {"name": "Jo Doe", "type": "person"}],
        "relations": [{"source": "Film A", "type": "directed_by", "target": "Jo Doe"}],
    },
    "Jo Doe": {
        "entities": [{"name": "Jo Doe", "type": "person"}, {"name": "Ann Doe", "type": "person"}],
        "relations": [{"source": "Jo Doe", "type": "mother", "target": "Ann Doe"}],
    },
    "Ann Doe": {
        "entities": [
            {"name": "Ann Doe", "type": "person", "attributes": {"born": "1950-05-05"}},
        ],
        "relations": [],
    },
}


def respond(messages: Sequence[Message]) -> str:
    title = messages[1].content.split("\n", 1)[0].removeprefix("Title: ")
    return json.dumps(REPLIES[title])


def by_name(question: ChoiceQuestion, _state: JSONContent) -> dict[str, float]:
    """Pick the candidate named exactly like the mention, else NEW."""
    assert isinstance(question.instructions, dict)
    mention = question.instructions["mention"]
    assert isinstance(mention, dict)
    return {
        label: float(
            label == "NEW"
            if not any(isinstance(o, dict) and o["name"] == mention["name"]
                       for o in question.options.values())
            else isinstance(option, dict) and option["name"] == mention["name"]
        )
        for label, option in question.options.items()
    }  # fmt: skip


def test_normalization() -> None:
    assert norm_name("Polish-Russian War (film)") == "polish russian war"
    assert norm_name("  Małgorzata  Braunek ") == "malgorzata braunek"
    assert norm_value("12 June 1516") == norm_value("1516-06-12") == "1516-06-12"
    assert norm_value("August 8, 1975") == "1975-08-08"
    assert norm_value("Nice") == "nice"


def test_sampling_and_documents() -> None:
    assert len(sample_records(RECORDS, 10, 0)) == 2  # comparison questions are skipped
    assert len(sample_records(RECORDS, 1, 0)) == 1
    docs = documents(RECORDS[:2])
    assert [d.doc_id for d in docs] == ["Film A (film)", "Jo Doe", "Ann Doe"]
    assert docs[1].text == "Jo Doe is a director. Jo's mother is Ann Doe."


async def test_run_variants_scores_and_writes(tmp_path: Path) -> None:
    cache: dict[str, Any] = {}
    seen: list[str] = []
    results = await run_variants(
        RECORDS[:2],
        {"exact": IngestConfig(routing="exact"), "jev": IngestConfig(escalate=False)},
        llm=FakeLLM(respond),
        decider=FakeDecisionBackend(script=by_name),
        embedder=None,
        cache=cache,
        on_variant=lambda r: seen.append(r.name),
        graph_dir=tmp_path,
    )
    assert seen == ["exact", "jev"]
    assert (tmp_path / "jev.graph.json").exists()
    assert results[1].report.extraction_cache_hits == 3
    for result in results:
        score = result.score
        assert (score.questions, score.gold_entities, score.entities_found) == (2, 3, 3)
        assert score.entities_split == 0
        assert score.overmerged_nodes == 0
        assert (score.triples, score.triples_recalled) == (3, 3)
        assert (score.literal_triples, score.literal_recalled) == (1, 1)
        assert score.chains_complete == 2
        rates = score.rates()
        assert rates["chain_recall"] == 1.0
        assert rates["entity_triple_recall"] == 1.0
    out = write_results(run_dir(tmp_path, "t"), results, {"n": 2})
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "| exact | 3 | 2 |" in summary
    assert "3 triples (1 with a literal object)" in summary
    assert json.loads((out / "results.json").read_text(encoding="utf-8"))["params"] == {"n": 2}


async def test_score_counts_splits_and_overmerges(tmp_path: Path) -> None:
    replies = {
        **REPLIES,
        # Ann's own paragraph names her with a different type: the NEW-only router keeps
        # a second node. Jo's paragraph folds "Ann Doe" into "Jo Doe" (a wrong merge).
        "Jo Doe": {"entities": [{"name": "Jo Doe", "type": "person"}], "relations": []},
    }
    llm = FakeLLM(lambda m: json.dumps(replies[m[1].content.split("\n", 1)[0][7:]]))
    decider = FakeDecisionBackend(script=lambda q, _s: {o: float(o == "NEW") for o in q.options})
    (result,) = await run_variants(
        RECORDS[:2], {"new": IngestConfig(escalate=False)}, llm=llm, decider=decider,
        embedder=None, cache={},
    )  # fmt: skip
    score = result.score
    assert score.entities_split == 1  # "Jo Doe": one node per paragraph
    assert score.triples_recalled == 2  # Film A -> Jo Doe, and Ann's birth date
    assert score.chains_complete == 0
    assert math.isnan(
        type(score)(**{**score.model_dump(), "literal_triples": 0}).rates()["literal_recall"]
    )
