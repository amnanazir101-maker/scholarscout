from scholarscout.papers import PaperRegistry, merge_papers, rank_papers
from scholarscout.reader import chunk_text
from scholarscout.store import PaperStore
from tests.fakes import PAPERS, FakeEmbedder, make_paper


def test_merge_combines_arxiv_and_openalex_records():
    from_arxiv = make_paper(
        "Mamba: Linear-Time Sequence Modeling",
        "Short.",
        arxiv_id="2312.00752",
        venue="arXiv (cs.LG)",
        source="arxiv",
    )
    from_openalex = make_paper(
        "Mamba: linear-time sequence modeling",
        "A longer abstract.",
        doi="10.1/x",
        citations=1500,
        venue="COLM",
        source="openalex",
    )
    [merged] = merge_papers([from_arxiv, from_openalex])
    assert merged.arxiv_id == "2312.00752"
    assert merged.doi == "10.1/x"
    assert merged.citations == 1500
    assert merged.abstract == "A longer abstract."
    assert merged.venue == "COLM"


def test_rank_prefers_relevant_papers():
    ranked = rank_papers(
        "benchmark for efficient transformers on long sequences",
        PAPERS,
        FakeEmbedder(),
        current_year=2026,
    )
    assert ranked[0].title.startswith("Long Range Arena")
    assert all(0 <= paper.score <= 1 for paper in ranked)


def test_registry_assigns_ids_once():
    registry = PaperRegistry()
    first, new = registry.add(PAPERS[0])
    again, new_again = registry.add(PAPERS[0].model_copy(update={"title": "Different title"}))
    assert (first.id, new) == ("P1", True)
    assert (again.id, new_again) == ("P1", False)  # same arXiv ID -> same paper
    assert registry.add(PAPERS[1])[0].id == "P2"
    assert [paper.id for paper in registry.all()] == ["P1", "P2"]


def test_chunk_text_overlaps():
    chunks = chunk_text(" ".join(str(i) for i in range(300)), words_per_chunk=180, overlap=30)
    assert len(chunks) == 2
    assert chunks[1].split()[0] == "150"


def test_store_filters_by_paper_and_kind():
    store = PaperStore(FakeEmbedder())
    registry = PaperRegistry()
    first, _ = registry.add(PAPERS[0])
    second, _ = registry.add(PAPERS[1])
    store.add_paper(first, full_text="Selective scans run on GPUs. " * 50)
    store.add_paper(second)

    hits = store.search("copying from long context", k=1)
    assert hits[0].paper_id == second.id
    full_text = store.search("GPU scans", paper_ids=[first.id], kind="full_text")
    assert full_text and all(c.paper_id == first.id and c.kind == "full_text" for c in full_text)
