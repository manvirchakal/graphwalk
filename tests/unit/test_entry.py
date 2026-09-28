from collections.abc import Mapping

from factories import edge, node
from graphwalk.decisions import ChoiceQuestion, FakeDecisionBackend, JSONContent
from graphwalk.embeddings import FakeEmbedder
from graphwalk.stores.networkx_store import NetworkXStore
from graphwalk.traversal import (
    ChoiceEntryResolver,
    EntryResolver,
    NameEntryResolver,
)
from graphwalk.traversal.entry import ENTRY_KEY
from kg_fixtures import movie_store


async def test_is_an_entry_resolver() -> None:
    assert isinstance(NameEntryResolver(NetworkXStore()), EntryResolver)


async def test_finds_names_case_insensitively_in_order() -> None:
    resolver = NameEntryResolver(await movie_store())
    assert await resolver.resolve("who directed inception?") == ["inception"]
    found = await resolver.resolve("Did Titanic come out before Memento?")
    assert found == ["titanic", "memento"]


async def test_longest_mention_wins_and_aliases_match() -> None:
    resolver = NameEntryResolver(await movie_store())
    assert await resolver.resolve("Where was Christopher Nolan born?") == ["nolan"]
    assert await resolver.resolve("films by Nolan") == ["nolan"]


async def test_whole_words_only_and_short_names_ignored() -> None:
    store = NetworkXStore()
    await store.upsert_node(node("cat", name="Cat"))
    await store.upsert_node(node("x", name="X"))
    resolver = NameEntryResolver(store)
    assert await resolver.resolve("a category of X things") == []


async def test_same_name_returns_every_node() -> None:
    store = NetworkXStore()
    await store.upsert_node(node("paris-fr", name="Paris"))
    await store.upsert_node(node("paris-tx", name="Paris"))
    assert await NameEntryResolver(store).resolve("Paris population") == ["paris-fr", "paris-tx"]


async def test_possessives_match_exactly() -> None:
    resolver = NameEntryResolver(await movie_store())  # no embedder: must be a name match
    assert await resolver.resolve("james cameron's films") == ["cameron"]


async def test_embedding_fallback_and_refresh() -> None:
    store = await movie_store()
    resolver = NameEntryResolver(store, embedder=FakeEmbedder())
    assert await resolver.resolve("that cameron guy") == ["cameron"]  # nearest by embedding
    await store.upsert_node(node("dune", name="Dune"))
    resolver.refresh()
    assert await resolver.resolve("Dune") == ["dune"]


async def test_no_match_without_embedder_is_empty() -> None:
    resolver = NameEntryResolver(await movie_store())
    assert await resolver.resolve("completely unrelated") == []


async def test_stopword_only_names_are_not_mentions() -> None:
    store = await movie_store()
    await store.upsert_node(node("who-song", name="Who..."))
    resolver = NameEntryResolver(store)
    assert await resolver.resolve("Who directed Inception?") == ["inception"]


async def _royals() -> NetworkXStore:
    store = NetworkXStore()
    await store.upsert_node(node("reg", name="Reginald II of Bar"))
    await store.upsert_node(node("reg3", name="Reginald III of Bar"))
    await store.upsert_node(node("bar", name="Bar"))
    await store.upsert_node(node("wal", name="Prince Waldemar of Schaumburg-Lippe"))
    await store.upsert_node(node("wal2", name="Prince Waldemar Stephen of Schaumburg-Lippe"))
    await store.upsert_node(node("dad", name="Adolf"))
    await store.upsert_edge(edge("wal2", "father", "dad"))
    return store


async def test_candidates_rank_exact_then_fuzzy() -> None:
    resolver = NameEntryResolver(await _royals())
    cands = await resolver.candidates("Where did Reginald Ii, Count Of Bar's father die?")
    assert cands[0].node_id == "bar"  # the only exact mention
    assert cands[0].exact
    fuzzy = [c.node_id for c in cands if not c.exact]
    assert fuzzy[0] == "reg"  # all of its content words appear; "reg3" misses "iii"
    assert len(await resolver.candidates("Reginald of Bar", limit=2)) == 2


async def test_best_only_returns_the_top_candidate() -> None:
    resolver = NameEntryResolver(await _royals(), best_only=True)
    assert await resolver.resolve("Who is Reginald Ii Of Bar?") == ["reg"]
    assert await resolver.resolve("nothing here") == []


def _pick(name: str):
    def script(question: ChoiceQuestion, state: JSONContent) -> Mapping[str, float] | None:
        del state
        assert question.key == ENTRY_KEY
        options = question.options
        for label, card in options.items():
            if isinstance(card, dict) and card.get("name") == name:
                return {label: 1.0}
        return None

    return script


async def test_choice_resolver_picks_and_accounts_the_call() -> None:
    store = await _royals()
    decider = FakeDecisionBackend(
        script=_pick("Prince Waldemar Stephen of Schaumburg-Lippe"), cost_per_call=0.001
    )
    resolver = ChoiceEntryResolver(store, decider)
    assert isinstance(resolver, EntryResolver)
    link = await resolver.link("Who is Prince Waldemar Of Schaumburg-Lippe's grandmother?")
    assert link.nodes == ("wal2",)
    assert link.decision_calls == 1
    assert link.cost_usd == 0.001
    assert link.input_tokens > 0
    assert "wal" in link.detail["candidates"]  # type: ignore[operator]
    options = decider.requests[0].questions[0].options
    cards = [c for c in options.values() if isinstance(c, dict)]
    card = next(c for c in cards if str(c["name"]).startswith("Prince Waldemar S"))
    assert card["relations"] == ["Prince Waldemar Stephen of Schaumburg-Lippe --father--> …"]


async def test_choice_resolver_skips_the_call_for_one_candidate() -> None:
    decider = FakeDecisionBackend()
    resolver = ChoiceEntryResolver(await movie_store(), decider)
    assert await resolver.resolve("Who directed Inception?") == ["inception"]
    assert await resolver.resolve("zzz qqq") == []
    assert decider.calls == 0


async def test_choice_resolver_falls_back_to_top_candidate_on_error() -> None:
    decider = FakeDecisionBackend(fail_on_calls={1})
    resolver = ChoiceEntryResolver(await _royals(), decider, max_candidates=3)
    link = await resolver.link("Reginald of Bar")
    assert link.nodes == (("bar",))
    assert link.decision_calls == 1
    assert "error" in link.detail
