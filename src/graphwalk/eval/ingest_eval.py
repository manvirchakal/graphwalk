"""Ingestion quality on 2WikiMultiHopQA: build a graph from the questions' paragraphs
and check it against their gold evidence triples.

For a seeded sample of walkable dev questions, every context paragraph (gold and
distractor, de-duplicated by title) is ingested as one document. Then:

* **entity recall**: gold entities (triple subjects and non-literal objects) that match
  a node's name or alias after normalization (case, accents, punctuation, and a
  trailing parenthetical are ignored);
* **split entities**: gold entities matched by more than one node (missed merges);
* **over-merged nodes**: nodes whose names and aliases match both ends of a gold triple,
  two entities known to be different (e.g. a person and their father). A lower bound on
  wrong merges: most involve non-gold names, and 2Wiki's gold sometimes spells one
  entity two ways, so plain "matches two gold names" would also count correct merges;
* **triple recall**: gold triples with an edge, in either direction and of any type,
  between matching nodes, or (for a literal object such as a date) a matching attribute
  value on the subject's node;
* **complete chains**: questions all of whose evidence triples are recalled, i.e. the
  graph holds the whole reasoning path.

Several routing variants run on the same extractions (cached per chunk and model),
so they differ only in routing.
"""

import json
import random
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, JsonValue

from graphwalk.core.model import Node, NodeId
from graphwalk.decisions.base import DecisionBackend
from graphwalk.embeddings.base import Embedder
from graphwalk.eval.datasets.twowiki import WALKABLE_TYPES
from graphwalk.eval.runner import environment
from graphwalk.ingest.pipeline import ExtractionCache, IngestConfig, IngestPipeline, IngestReport
from graphwalk.ingest.sources import SourceDocument, TextSource
from graphwalk.llm.base import LLMBackend
from graphwalk.stores.base import GraphStore
from graphwalk.stores.networkx_store import NetworkXStore

_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_TRANSLIT = str.maketrans(
    {
        "ł": "l",
        "Ł": "L",
        "ø": "o",
        "Ø": "O",
        "đ": "d",
        "Đ": "D",
        "ß": "ss",
        "æ": "ae",
        "Æ": "AE",
        "œ": "oe",
        "Œ": "OE",
    }
)
_DATE_FORMATS = ("%d %B %Y", "%B %d, %Y", "%B %d %Y", "%Y-%m-%d", "%d %b %Y", "%b %d, %Y")


def norm_name(name: str) -> str:
    """Case-, accent-, and punctuation-insensitive; drops a trailing ``(...)``."""
    text = _PAREN.sub("", name.strip()).translate(_TRANSLIT)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return _NON_ALNUM.sub(" ", text.casefold()).strip()


def norm_value(value: object) -> str:
    """Dates to ISO ``YYYY-MM-DD`` when they parse, else :func:`norm_name`."""
    text = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=UTC).date().isoformat()
        except ValueError:
            continue
    return norm_name(text)


def is_literal(relation: str) -> bool:
    return relation.startswith("date of")


def sample_records(records: Sequence[Mapping[str, Any]], n: int, seed: int) -> list[Any]:
    walkable = [r for r in records if r.get("type") in WALKABLE_TYPES and r.get("evidences")]
    if n >= len(walkable):
        return list(walkable)
    indices = sorted(random.Random(seed).sample(range(len(walkable)), n))  # noqa: S311
    return [walkable[i] for i in indices]


def documents(records: Sequence[Mapping[str, Any]]) -> list[SourceDocument]:
    """One document per context paragraph title (first occurrence wins)."""
    docs: dict[str, SourceDocument] = {}
    for record in records:
        for title, sentences in record["context"]:
            if title not in docs:
                text = " ".join(s.strip() for s in sentences)
                docs[title] = SourceDocument(doc_id=title, text=text, title=title)
    return list(docs.values())


class IngestScore(BaseModel):
    questions: int
    nodes: int
    edges: int
    gold_entities: int
    entities_found: int
    entities_split: int
    overmerged_nodes: int
    triples: int
    triples_recalled: int
    literal_triples: int
    literal_recalled: int
    chains_complete: int

    def rates(self) -> dict[str, float]:
        def rate(a: int, b: int) -> float:
            return a / b if b else float("nan")

        entity_triples = self.triples - self.literal_triples
        return {
            "entity_recall": rate(self.entities_found, self.gold_entities),
            "split_rate": rate(self.entities_split, self.entities_found),
            "triple_recall": rate(self.triples_recalled, self.triples),
            "entity_triple_recall": rate(
                self.triples_recalled - self.literal_recalled, entity_triples
            ),
            "literal_recall": rate(self.literal_recalled, self.literal_triples),
            "chain_recall": rate(self.chains_complete, self.questions),
        }


@dataclass
class _Lookup:
    nodes: dict[NodeId, Node]
    by_name: dict[str, set[NodeId]]
    linked: set[frozenset[NodeId]]


async def _lookup(store: GraphStore) -> _Lookup:
    nodes: dict[NodeId, Node] = {}
    by_name: dict[str, set[NodeId]] = {}
    async for node in store.iter_nodes():
        nodes[node.id] = node
        for name in {norm_name(n) for n in (node.name, *node.aliases)}:
            by_name.setdefault(name, set()).add(node.id)
    linked = {frozenset((e.source, e.target)) async for e in store.iter_edges()}
    return _Lookup(nodes, by_name, linked)


def _recalled(lookup: _Lookup, subject: str, relation: str, obj: str) -> bool:
    subjects = lookup.by_name.get(norm_name(subject), set())
    if is_literal(relation) or norm_name(obj) not in lookup.by_name:
        wanted = norm_value(obj)
        for node_id in subjects:
            values = lookup.nodes[node_id].attributes.values()
            if any(norm_value(v) == wanted for v in values):
                return True
        if is_literal(relation):
            return False
    objects = lookup.by_name.get(norm_name(obj), set())
    return any(frozenset((s, o)) in lookup.linked for s in subjects for o in objects)


async def score(store: GraphStore, records: Sequence[Mapping[str, Any]]) -> IngestScore:
    lookup = await _lookup(store)
    gold: set[str] = set()
    triples: set[tuple[str, str, str]] = set()
    complete = 0
    for record in records:
        chain = [(str(s), str(r), str(o)) for s, r, o in record["evidences"]]
        triples.update(chain)
        complete += all(_recalled(lookup, *t) for t in chain)
        for s, r, o in chain:
            gold.add(norm_name(s))
            if not is_literal(r):
                gold.add(norm_name(o))
    found = [g for g in gold if g in lookup.by_name]
    distinct = {frozenset((norm_name(s), norm_name(o))) for s, r, o in triples if not is_literal(r)}
    overmerged = 0
    for node in lookup.nodes.values():
        names = {norm_name(n) for n in (node.name, *node.aliases)}
        overmerged += any(pair <= names for pair in distinct if len(pair) == 2)  # noqa: PLR2004
    recalled = [t for t in triples if _recalled(lookup, *t)]
    nodes, edges = await store.counts()
    return IngestScore(
        questions=len(records),
        nodes=nodes,
        edges=edges,
        gold_entities=len(gold),
        entities_found=len(found),
        entities_split=sum(len(lookup.by_name[g]) > 1 for g in found),
        overmerged_nodes=overmerged,
        triples=len(triples),
        triples_recalled=len(recalled),
        literal_triples=sum(is_literal(r) for _, r, _ in triples),
        literal_recalled=sum(is_literal(r) for _, r, _ in recalled),
        chains_complete=complete,
    )


class VariantResult(BaseModel):
    name: str
    config: dict[str, JsonValue]
    score: IngestScore
    report: IngestReport


async def run_variants(
    records: Sequence[Mapping[str, Any]],
    variants: Mapping[str, IngestConfig],
    *,
    llm: LLMBackend,
    decider: DecisionBackend | None,
    embedder: Embedder | None,
    cache: ExtractionCache,
    on_variant: Callable[[VariantResult], None] | None = None,
    graph_dir: Path | None = None,
    escalation_llm: LLMBackend | None = None,
) -> list[VariantResult]:
    """Ingest the records' paragraphs once per variant (fresh graph each) and score."""
    source = TextSource("2wiki:dev:context", documents(records))
    results: list[VariantResult] = []
    for name, config in variants.items():
        store = NetworkXStore()
        pipeline = IngestPipeline(
            store,
            llm,
            decider,
            embedder=embedder,
            config=config,
            extraction_cache=cache,
            escalation_llm=escalation_llm,
        )
        report = await pipeline.ingest(source)
        result = VariantResult(
            name=name,
            config=config.model_dump(mode="json"),
            score=await score(store, records),
            report=report,
        )
        if graph_dir is not None:
            await store.save(graph_dir / f"{name}.graph.json")
        results.append(result)
        if on_variant is not None:
            on_variant(result)
    return results


def _pct(value: float) -> str:
    return "n/a" if value != value else f"{value:.3f}"  # noqa: PLR0124 - NaN check


def summary_table(results: Sequence[VariantResult]) -> str:
    header = (
        "| variant | nodes | edges | entity recall | split | over-merged | triple recall "
        "(entity / literal) | complete chains | decision calls | escalations | LLM calls "
        "| cost $ |\n|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = [header]
    for r in results:
        s, rep, rates = r.score, r.report, r.score.rates()
        cost = "n/a" if rep.cost_usd is None else f"{rep.cost_usd:.4f}"
        rows.append(
            f"| {r.name} | {s.nodes} | {s.edges} | {_pct(rates['entity_recall'])} "
            f"| {_pct(rates['split_rate'])} | {s.overmerged_nodes} "
            f"| {_pct(rates['triple_recall'])} ({_pct(rates['entity_triple_recall'])} / "
            f"{_pct(rates['literal_recall'])}) | {_pct(rates['chain_recall'])} "
            f"| {rep.decision_calls} | {rep.escalations} | {rep.llm_calls} | {cost} |"
        )
    return "\n".join(rows)


def write_results(
    out_dir: Path, results: Sequence[VariantResult], params: dict[str, JsonValue]
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = environment()
    document: dict[str, JsonValue] = {
        "params": params,
        "environment": env,
        "variants": [cast("JsonValue", r.model_dump(mode="json")) for r in results],
    }
    (out_dir / "results.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    first = results[0].score if results else None
    lines = [
        "# 2Wiki ingestion quality",
        "",
        f"- params: `{json.dumps(params, sort_keys=True)}`",
        f"- git: `{env['git_sha']}`{' (dirty)' if env['git_dirty'] else ''}",
        f"- run at: {env['timestamp']}",
    ]
    if first is not None:
        lines.append(
            f"- gold: {first.questions} questions, {first.gold_entities} entities, "
            f"{first.triples} triples ({first.literal_triples} with a literal object)"
        )
    lines += ["", summary_table(results), ""]
    for r in results:
        rep = r.report
        lines.append(
            f"- **{r.name}**: {rep.documents} documents, {rep.chunks} chunks, "
            f"{rep.entities} entities / {rep.relations} relations extracted, "
            f"{rep.nodes_created} nodes created, {rep.nodes_merged} merged, "
            f"{rep.extraction_cache_hits} cached extractions, {len(rep.errors)} errors, "
            f"{rep.elapsed_s:.0f}s"
        )
        if rep.escalation_model:
            esc = rep.escalation_cost_usd
            lines.append(
                f"  - escalation to `{rep.escalation_model}`: {rep.escalation_calls} calls, "
                f"{rep.escalation_input_tokens} input / {rep.escalation_output_tokens} output "
                f"tokens, {'n/a' if esc is None else f'${esc:.4f}'}"
            )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir


def run_dir(root: Path, name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return root / f"{stamp}-{name}"
