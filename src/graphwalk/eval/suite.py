"""Named system presets and one-call dataset runs (used by ``graphwalk eval``)."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import JsonValue

from graphwalk.decisions.base import DecisionBackend
from graphwalk.embeddings.base import Embedder, Vectors
from graphwalk.eval.datasets import metaqa, twowiki
from graphwalk.eval.datasets.cache import cache_dir
from graphwalk.eval.runner import SystemRun, run_system, sample_questions, write_results
from graphwalk.eval.systems import (
    DocIndex,
    GraphwalkSystem,
    Linking,
    VectorRAGSystem,
    entity_documents,
)
from graphwalk.eval.types import EvalQuestion, QASystem
from graphwalk.llm.base import LLMBackend
from graphwalk.stores.networkx_store import NetworkXStore
from graphwalk.traversal import NameEntryResolver, TraversalConfig, Traverser
from graphwalk.traversal.prompts import node_text

type Dataset = Literal["metaqa", "2wiki"]

COMMON: dict[str, JsonValue] = {
    # Both datasets' answers are never the topic entity itself.
    "allow_stop_at_start": False,
    # Relation-mode answer sets can be large; the frontier size never reaches the prompt
    # (options show a preview and a count), so don't cut recall here.
    "max_frontier": 2000,
    "budget": {"max_depth": 4, "max_decision_calls": 12},
}
PRESETS: dict[str, dict[str, JsonValue]] = {
    "greedy": {"strategy": "greedy"},
    "beam": {"strategy": "beam", "beam_width": 3},
    "relation": {"strategy": "greedy", "hop_mode": "relation"},
    "relation-beam": {"strategy": "beam", "beam_width": 3, "hop_mode": "relation"},
    "sample": {"strategy": "sample", "n_samples": 5, "temperature": 1.0},
}
GLOSSES = "dataset"
"""Sentinel value for ``relation_glosses``: use the dataset's own relation glosses."""
TUNED_V2: dict[str, JsonValue] = {
    # Chosen on MetaQA dev (scripts/tune_metaqa_dev.py; docs/results-m5.md).
    "show_types": True,
    "stop_style": "literal",
    "relation_glosses": GLOSSES,
    "answer_type": "hint",
}
PRESETS.update({f"{name}-v2": {**PRESETS[name], **TUNED_V2} for name in list(PRESETS)})
DATASET_GLOSSES: dict[str, dict[str, str]] = {"metaqa": metaqa.RELATION_GLOSSES}
RAG = "rag"
SYSTEMS = (*PRESETS, RAG)


def preset_config(
    name: str, *, glosses: dict[str, str] | None = None, **overrides: JsonValue
) -> TraversalConfig:
    if name not in PRESETS:
        msg = f"unknown preset {name!r}; choose from {sorted(PRESETS)}"
        raise ValueError(msg)
    merged: dict[str, JsonValue] = {**COMMON, **PRESETS[name], **overrides}
    if merged.get("relation_glosses") == GLOSSES:
        merged["relation_glosses"] = dict(glosses or {})
    return TraversalConfig.model_validate(merged)


@dataclass
class Factories:
    """How to build backends; tests swap in fakes."""

    decider: Callable[[], DecisionBackend]
    embedder: Callable[[], Embedder]
    llm: Callable[[], LLMBackend]
    _embedder: Embedder | None = field(default=None, init=False)

    def shared_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = self.embedder()
        return self._embedder


async def load_dataset(
    dataset: Dataset, hops: int = 1, split: str = "test"
) -> tuple[NetworkXStore, list[EvalQuestion], str]:
    """Returns ``(graph, questions, index key)``; the key names the RAG index cache.
    ``split`` applies to MetaQA (``dev`` for tuning, ``test`` for reporting)."""
    if dataset == "metaqa":
        kb, qa = metaqa.download(hops, split)
        store = await metaqa.build_store(metaqa.read_triples(kb), source_id="metaqa:kb")
        return store, metaqa.read_questions(qa, hops, split), f"metaqa-{metaqa.REVISION[:8]}"
    if split != "test":
        msg = "2wiki has one split here (its dev set, used as test)"
        raise ValueError(msg)
    records = twowiki.read_records(twowiki.download())
    store = await twowiki.build_graph(records)
    return store, twowiki.read_questions(records), f"2wiki-{twowiki.REVISION[:8]}"


async def doc_index(
    store: NetworkXStore, embedder: Embedder, key: str, *, cache: bool = True
) -> DocIndex:
    safe = embedder.model_id.replace("/", "_").replace(":", "_")
    path = cache_dir().parent / "indexes" / f"{key}-{safe}.npz"
    if cache and path.exists():
        data = np.load(path, allow_pickle=False)
        return DocIndex(
            ids=tuple(str(x) for x in data["ids"]),
            texts=tuple(str(x) for x in data["texts"]),
            vectors=data["vectors"].astype(np.float32),
            embedder_model=embedder.model_id,
        )
    index = await DocIndex.build(await entity_documents(store), embedder)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, ids=np.array(index.ids), texts=np.array(index.texts), vectors=index.vectors)
    return index


async def node_vectors(
    store: NetworkXStore, embedder: Embedder, key: str, *, cache: bool = True
) -> tuple[list[str], Vectors]:
    """``prompts.node_text`` of every node and its vector, cached like the doc index."""
    texts = sorted({node_text(node) async for node in store.iter_nodes()})
    safe = embedder.model_id.replace("/", "_").replace(":", "_")
    path = cache_dir().parent / "indexes" / f"{key}-nodes-{safe}.npz"
    if cache and path.exists():
        data = np.load(path, allow_pickle=False)
        if [str(t) for t in data["texts"]] == texts:
            return texts, data["vectors"].astype(np.float32)
    vectors = await embedder.embed(texts) if texts else np.zeros((0, 0), dtype=np.float32)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, texts=np.array(texts), vectors=vectors)
    return texts, vectors


async def build_systems(
    names: Sequence[str],
    store: NetworkXStore,
    index_key: str,
    factories: Factories,
    *,
    linking: Linking,
    rag_k: int = 5,
    cache_index: bool = True,
    overrides: dict[str, JsonValue] | None = None,
    glosses: dict[str, str] | None = None,
    label: str = "",
) -> list[QASystem]:
    """Build systems by preset name. ``overrides`` apply to every graphwalk preset;
    ``label`` is appended to system names (e.g. to tell tuning variants apart)."""
    systems: list[QASystem] = []
    decider: DecisionBackend | None = None
    preloaded: tuple[list[str], Vectors] | None = None
    node_types = sorted({node.type async for node in store.iter_nodes()})
    for name in names:
        if name == RAG:
            embedder = factories.shared_embedder()
            index = await doc_index(store, embedder, index_key, cache=cache_index)
            systems.append(VectorRAGSystem(index, embedder, factories.llm(), k=rag_k))
            continue
        decider = decider or factories.decider()
        embedder = factories.shared_embedder()
        if preloaded is None:
            preloaded = await node_vectors(store, embedder, index_key, cache=cache_index)
        traverser = Traverser(
            store,
            decider,
            embedder=factories.shared_embedder(),
            config=preset_config(name, glosses=glosses, **(overrides or {})),
            node_types=node_types,
        )
        traverser.preload_embeddings(*preloaded)
        resolver = NameEntryResolver(store) if linking == "resolve" else None
        systems.append(
            GraphwalkSystem(
                store,
                traverser,
                name=f"graphwalk-{name}{label}",
                linking=linking,
                resolver=resolver,
            )
        )
    return systems


async def run_dataset(
    dataset: Dataset,
    systems: Sequence[str],
    factories: Factories,
    *,
    hops: int = 1,
    n: int | None = 100,
    seed: int = 0,
    linking: Linking | None = None,
    concurrency: int = 8,
    out_root: Path = Path("results"),
    rag_k: int = 5,
    cache_index: bool = True,
    on_progress: Callable[[str, int, int], None] | None = None,
    split: str = "test",
    variants: Sequence[tuple[str, dict[str, JsonValue]]] = (("", {}),),
) -> tuple[Path, list[SystemRun]]:
    """Run ``systems`` on a seeded subset. Each ``(label, overrides)`` in ``variants``
    builds the graphwalk presets once more with those config overrides (RAG is built
    only for the first variant), so ablations share one question subset."""
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        msg = f"unknown systems {unknown}; choose from {list(SYSTEMS)}"
        raise ValueError(msg)
    store, questions, key = await load_dataset(dataset, hops, split)
    subset = sample_questions(questions, n, seed)
    link: Linking = linking or ("given" if dataset == "metaqa" else "gold")
    built: list[QASystem] = []
    for index, (variant, overrides) in enumerate(variants):
        names = systems if index == 0 else [s for s in systems if s != RAG]
        built += await build_systems(
            names,
            store,
            key,
            factories,
            linking=link,
            rag_k=rag_k,
            cache_index=cache_index,
            overrides=overrides,
            glosses=DATASET_GLOSSES.get(dataset),
            label=variant,
        )
    runs: list[SystemRun] = []
    for system in built:
        progress = None if on_progress is None else partial(on_progress, system.name)
        runs.append(await run_system(system, subset, concurrency=concurrency, on_done=progress))
    label = f"{dataset}-{hops}hop" if dataset == "metaqa" else f"{dataset}-{link}"
    if split != "test":
        label += f"-{split}"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = write_results(
        out_root / f"{stamp}-{label}",
        dataset=label,
        runs=runs,
        params={
            "dataset": dataset,
            "hops": hops if dataset == "metaqa" else None,
            "n": len(subset),
            "seed": seed,
            "split": split,
            "variants": dict(variants),
            "linking": link,
            "concurrency": concurrency,
            "rag_k": rag_k,
            "total_questions": len(questions),
        },
    )
    return out, runs
