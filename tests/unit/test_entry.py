from factories import node
from graphwalk.embeddings import FakeEmbedder
from graphwalk.stores.networkx_store import NetworkXStore
from graphwalk.traversal import EntryResolver, NameEntryResolver
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
